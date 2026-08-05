from __future__ import annotations

import asyncio
import copy
import json
from typing import Any, NoReturn, cast
from uuid import UUID

import pytest
from agent_hooks import (
    ALLOW,
    AgentContext,
    Decision,
    InterceptionPoint,
    Transform,
    Verdict,
)
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputRefusal,
    ResponseOutputText,
)
from openai.types.responses.response_output_text import AnnotationURLCitation

import agents.util._approvals as approvals_module
from agents import Agent, ModelSettings, RunConfig, Runner, Tool, ToolExecutionConfig, function_tool
from agents.agent_output import AgentOutputSchemaBase
from agents.exceptions import UserError
from agents.extensions.agent_hooks import (
    AgentHooksBlockedError,
    AgentHooksConfig,
    AgentHooksExecutionError,
    AgentHooksLimits,
    RecordSink,
)
from agents.handoffs import Handoff
from agents.items import ModelResponse, TResponseInputItem
from agents.models.interface import ModelTracing
from agents.retry import ModelRetrySettings
from agents.usage import RequestUsage, Usage
from tests.fake_model import FakeModel

_BOUNDED_CONTEXT_BYTES = 80 * 1024
_OVERSIZED_CONTEXT_TEXT = "x" * (96 * 1024)


class RecordingInterceptor:
    def __init__(self, verdicts: dict[InterceptionPoint, Verdict] | None = None) -> None:
        self.verdicts = verdicts or {}
        self.contexts: list[AgentContext] = []

    async def intercept(self, context: AgentContext, /) -> Verdict:
        self.contexts.append(copy.deepcopy(context))
        return self.verdicts.get(InterceptionPoint(context["interception_point"]), ALLOW)


class CancellingInterceptor:
    def __init__(self, point: InterceptionPoint) -> None:
        self.point = point
        self.started = asyncio.Event()
        self.drained = asyncio.Event()

    async def intercept(self, context: AgentContext, /) -> Verdict:
        if context["interception_point"] == self.point.value:
            self.started.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.drained.set()
        return ALLOW


class LabelInterceptor:
    def __init__(self) -> None:
        self.pre_source_labels: list[str] | None = None

    async def intercept(self, context: AgentContext, /) -> Verdict:
        point = InterceptionPoint(context["interception_point"])
        if point is InterceptionPoint.INPUT:
            return Verdict(decision=Decision.ALLOW, result_labels=("input-approved",))
        if point is InterceptionPoint.PRE_MODEL_CALL:
            extensions = context.get("extensions")
            if isinstance(extensions, dict):
                openai_agents = extensions.get("openai_agents")
                if isinstance(openai_agents, dict):
                    labels = openai_agents.get("source_labels")
                    if isinstance(labels, list) and all(isinstance(label, str) for label in labels):
                        self.pre_source_labels = labels
        return ALLOW


class DeniedLabelInterceptor:
    def __init__(self) -> None:
        self.shutdown_source_labels: object = None

    async def intercept(self, context: AgentContext, /) -> Verdict:
        point = InterceptionPoint(context["interception_point"])
        if point is InterceptionPoint.INPUT:
            return Verdict(
                decision=Decision.DENY,
                reason="input_denied",
                result_labels=("must-not-flow",),
            )
        if point is InterceptionPoint.AGENT_SHUTDOWN:
            extensions = context.get("extensions")
            if isinstance(extensions, dict):
                openai_agents = extensions.get("openai_agents")
                if isinstance(openai_agents, dict):
                    self.shutdown_source_labels = openai_agents.get("source_labels")
        return ALLOW


class OversizedLabelInterceptor:
    async def intercept(self, context: AgentContext, /) -> Verdict:
        if context["interception_point"] == InterceptionPoint.INPUT.value:
            return Verdict(decision=Decision.ALLOW, result_labels=("x" * 257,))
        return ALLOW


class FreeformMetadataInterceptor:
    async def intercept(self, context: AgentContext, /) -> Verdict:
        if context["interception_point"] == InterceptionPoint.INPUT.value:
            return Verdict(
                decision=Decision.ALLOW,
                message="sensitive free-form policy explanation",
            )
        return ALLOW


def _message(text: str) -> ResponseOutputMessage:
    return ResponseOutputMessage(
        id="message-1",
        content=[ResponseOutputText(annotations=[], text=text, type="output_text")],
        role="assistant",
        status="completed",
        type="message",
    )


def _tool_call(call_id: str, *, arguments: str = "{}") -> ResponseFunctionToolCall:
    return ResponseFunctionToolCall(
        id=f"item-{call_id}",
        call_id=call_id,
        type="function_call",
        name="lookup",
        arguments=arguments,
        status="completed",
    )


def _run_config(
    interceptor: object,
    records: RecordSink,
    *,
    with_tools: bool = False,
    max_tool_calls: int = 32,
    sink: RecordSink | None = None,
) -> RunConfig:
    return RunConfig(
        agent_hooks=AgentHooksConfig(
            agent_id="governed-agent-v1",
            interceptors=(interceptor,),  # type: ignore[arg-type]
            record_sink=records if sink is None else sink,
            limits=AgentHooksLimits(max_tool_calls_per_turn=max_tool_calls),
        ),
        trace_include_sensitive_data=False,
        model_settings=ModelSettings(
            parallel_tool_calls=False if with_tools else None,
            retry=ModelRetrySettings(max_retries=0),
        ),
        tool_execution=(
            ToolExecutionConfig(max_function_tool_concurrency=1) if with_tools else None
        ),
    )


def _context_for(
    interceptor: RecordingInterceptor,
    point: InterceptionPoint,
) -> AgentContext:
    return next(
        context for context in interceptor.contexts if context["interception_point"] == point.value
    )


def _assert_secret_absent_from_sdk_traceback(
    error: BaseException,
    secret: str,
) -> None:
    pending: list[BaseException] = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        assert secret not in repr(current)
        traceback = current.__traceback__
        while traceback is not None:
            if "/src/agents/" in traceback.tb_frame.f_code.co_filename:
                assert secret not in repr(traceback.tb_frame.f_locals)
            traceback = traceback.tb_next
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)


async def test_model_allow_order_and_request_id_pairing() -> None:
    interceptor = RecordingInterceptor()
    records = RecordSink(max_records=1000)
    model = FakeModel(initial_output=[_message("done")])
    agent = Agent(name="governed-agent", model=model, instructions="be concise")

    result = await Runner.run(
        agent,
        "hello",
        max_turns=1,
        run_config=_run_config(interceptor, records),
    )

    assert result.final_output == "done"
    assert [record.interception_point for record in records] == [
        InterceptionPoint.AGENT_STARTUP,
        InterceptionPoint.INPUT,
        InterceptionPoint.PRE_MODEL_CALL,
        InterceptionPoint.POST_MODEL_CALL,
        InterceptionPoint.OUTPUT,
        InterceptionPoint.AGENT_SHUTDOWN,
    ]
    pre = _context_for(interceptor, InterceptionPoint.PRE_MODEL_CALL)
    post = _context_for(interceptor, InterceptionPoint.POST_MODEL_CALL)
    assert pre["request_id"] == post["request_id"]
    UUID(pre["request_id"])
    assert pre["model"] == {"id": "FakeModel"}
    assert post["model"] == {"id": "FakeModel"}
    assert all(record.interceptors_registered == 2 for record in records)


async def test_sdk_provider_data_is_dropped_without_reading_its_value() -> None:
    class UnreadableProviderData:
        def __deepcopy__(self, memo: object) -> NoReturn:
            raise AssertionError("provider_data was copied")

        def __bool__(self) -> NoReturn:
            raise AssertionError("provider_data was evaluated")

    message = ResponseOutputMessage.model_validate(
        {
            "id": "message-1",
            "content": [ResponseOutputText(annotations=[], text="done", type="output_text")],
            "role": "assistant",
            "status": "completed",
            "type": "message",
            "provider_data": UnreadableProviderData(),
        }
    )
    records = RecordSink(max_records=1000)

    result = await Runner.run(
        Agent(name="governed-agent", model=FakeModel(initial_output=[message])),
        "hello",
        max_turns=1,
        run_config=_run_config(RecordingInterceptor(), records),
    )

    assert result.final_output == "done"
    raw_message = result.new_items[0].raw_item
    assert isinstance(raw_message, ResponseOutputMessage)
    assert raw_message.model_extra == {}


async def test_sdk_provider_data_is_dropped_from_governed_tool_call() -> None:
    side_effects: list[str] = []

    @function_tool(name_override="lookup")
    async def lookup(value: str) -> str:
        side_effects.append(value)
        return "recorded"

    tool_call = ResponseFunctionToolCall.model_validate(
        {
            "id": "item-call-1",
            "call_id": "call-1",
            "type": "function_call",
            "name": "lookup",
            "arguments": '{"value":"approved"}',
            "status": None,
            "provider_data": object(),
        }
    )
    model = FakeModel(initial_output=[tool_call])
    model.set_next_output([_message("done")])
    records = RecordSink(max_records=1000)

    result = await Runner.run(
        Agent(name="governed-agent", model=model, tools=[lookup]),
        "hello",
        max_turns=2,
        run_config=_run_config(RecordingInterceptor(), records, with_tools=True),
    )

    assert result.final_output == "done"
    assert side_effects == ["approved"]
    raw_tool_call = result.new_items[0].raw_item
    assert isinstance(raw_tool_call, ResponseFunctionToolCall)
    assert raw_tool_call.model_extra == {}


async def test_model_transforms_reach_governed_dispatch_and_processing() -> None:
    interceptor = RecordingInterceptor(
        {
            InterceptionPoint.PRE_MODEL_CALL: Verdict(
                decision=Decision.TRANSFORM,
                transform=Transform(path="$target[1].content", value="governed input"),
            ),
            InterceptionPoint.POST_MODEL_CALL: Verdict(
                decision=Decision.TRANSFORM,
                transform=Transform(path="$target.content", value="governed output"),
            ),
        }
    )
    records = RecordSink(max_records=1000)
    model = FakeModel(initial_output=[_message("raw output")])
    agent: Agent[None] = Agent(
        name="governed-agent",
        model=model,
        instructions="system text",
    )

    result = await Runner.run(
        agent,
        "raw input",
        max_turns=1,
        run_config=_run_config(interceptor, records),
    )

    assert model.first_turn_args is not None
    assert model.first_turn_args["input"] == [{"content": "governed input", "role": "user"}]
    assert result.final_output == "governed output"


async def test_model_callback_is_snapshotted_before_pre_model_await(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class PausingInterceptor:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def intercept(self, context: AgentContext, /) -> Verdict:
            if context["interception_point"] == InterceptionPoint.PRE_MODEL_CALL.value:
                self.started.set()
                await self.release.wait()
            return ALLOW

    interceptor = PausingInterceptor()
    records = RecordSink(max_records=1000)
    replacement_calls: list[str] = []
    model = FakeModel(initial_output=[_message("done")])
    agent: Agent[None] = Agent(name="governed-agent", model=model)

    async def replacement(*_args: object, **_kwargs: object) -> NoReturn:
        replacement_calls.append("replacement")
        raise AssertionError("replacement model callback was invoked")

    task = asyncio.create_task(
        Runner.run(
            agent,
            "approved input",
            max_turns=1,
            run_config=_run_config(interceptor, records),
        )
    )
    await asyncio.wait_for(interceptor.started.wait(), timeout=5.0)
    monkeypatch.setattr(model, "get_response", replacement)
    interceptor.release.set()
    result = await task

    assert result.final_output == "done"
    assert replacement_calls == []
    assert model.first_turn_args is not None
    assert model.first_turn_args["input"] == [{"content": "approved input", "role": "user"}]


async def test_parent_cancellation_waits_for_suppressing_governed_model() -> None:
    started = asyncio.Event()
    cancellation_caught = asyncio.Event()
    release = asyncio.Event()

    class SuppressingModel(FakeModel):
        async def get_response(self, *args: Any, **kwargs: Any) -> ModelResponse:
            started.set()
            try:
                await asyncio.Event().wait()
                raise AssertionError("model wait unexpectedly completed")
            except asyncio.CancelledError:
                cancellation_caught.set()
                await release.wait()
                return ModelResponse(
                    output=[_message("must not complete")],
                    usage=Usage(),
                    response_id="suppressed-cancellation",
                )

    records = RecordSink(max_records=1000)
    interceptor = RecordingInterceptor()
    agent: Agent[None] = Agent(name="governed-agent", model=SuppressingModel())
    task = asyncio.create_task(
        Runner.run(
            agent,
            "hello",
            max_turns=1,
            run_config=_run_config(interceptor, records),
        )
    )
    await started.wait()
    task.cancel()
    await asyncio.wait_for(cancellation_caught.wait(), timeout=1)

    try:
        assert not task.done()
        assert all(
            record.interception_point is not InterceptionPoint.AGENT_SHUTDOWN for record in records
        )
    finally:
        release.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    post_contexts = [
        context
        for context in interceptor.contexts
        if context["interception_point"] == InterceptionPoint.POST_MODEL_CALL.value
    ]
    points = [record.interception_point for record in records]
    assert points.count(InterceptionPoint.PRE_MODEL_CALL) == 1
    assert points.count(InterceptionPoint.POST_MODEL_CALL) == 1
    assert points[-1] is InterceptionPoint.AGENT_SHUTDOWN
    assert len(post_contexts) == 1
    assert post_contexts[0]["response"]["finish_reason"] == "cancelled"


async def test_post_model_uses_detached_usage_and_identifiers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class PausingPostInterceptor:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def intercept(self, context: AgentContext, /) -> Verdict:
            if context["interception_point"] == InterceptionPoint.POST_MODEL_CALL.value:
                self.started.set()
                await self.release.wait()
            return ALLOW

    interceptor = PausingPostInterceptor()
    records = RecordSink(max_records=1000)
    model = FakeModel(initial_output=[_message("done")])
    model.set_hardcoded_usage(Usage(requests=1, input_tokens=3, output_tokens=2, total_tokens=5))
    captured_responses: list[ModelResponse] = []
    original_get_response = model.get_response

    async def capture_response(*args: Any, **kwargs: Any) -> ModelResponse:
        response = await original_get_response(*args, **kwargs)
        captured_responses.append(response)
        return response

    monkeypatch.setattr(model, "get_response", capture_response)
    agent: Agent[None] = Agent(name="governed-agent", model=model)
    task = asyncio.create_task(
        Runner.run(
            agent,
            "hello",
            max_turns=1,
            run_config=_run_config(interceptor, records),
        )
    )

    await asyncio.wait_for(interceptor.started.wait(), timeout=1)
    response = captured_responses[0]
    response.usage.input_tokens = 400
    response.usage.output_tokens = 500
    response.usage.total_tokens = 900
    response.response_id = "mutated-response"
    response.request_id = "mutated-request"
    interceptor.release.set()
    result = await task

    returned_response = result.raw_responses[0]
    assert returned_response.response_id == "resp-789"
    assert returned_response.request_id is None
    assert returned_response.usage.input_tokens == 3
    assert returned_response.usage.output_tokens == 2
    assert returned_response.usage.total_tokens == 5
    assert result.context_wrapper.usage.total_tokens == 5


async def test_model_retry_mutation_is_isolated_across_turns_and_failure() -> None:
    failure_text = "unique model retry mutation failure 5b8f94"
    primary_error = RuntimeError(failure_text)

    class RetryMutatingModel(FakeModel):
        def __init__(self) -> None:
            super().__init__(initial_output=[_tool_call("call-1")])
            self.observed_max_retries: list[int] = []

        async def get_response(
            self,
            system_instructions: str | None,
            input: str | list[TResponseInputItem],
            model_settings: ModelSettings,
            tools: list[Tool],
            output_schema: AgentOutputSchemaBase | None,
            handoffs: list[Handoff],
            tracing: ModelTracing,
            *,
            previous_response_id: str | None,
            conversation_id: str | None,
            prompt: Any | None,
        ) -> ModelResponse:
            retry = model_settings.retry
            assert retry is not None
            assert retry.max_retries is not None
            self.observed_max_retries.append(retry.max_retries)
            retry.max_retries = 3
            if len(self.observed_max_retries) == 2:
                raise primary_error
            if len(self.observed_max_retries) > 2:
                raise AssertionError("unexpected third model dispatch")
            return await super().get_response(
                system_instructions=system_instructions,
                input=input,
                model_settings=model_settings,
                tools=tools,
                output_schema=output_schema,
                handoffs=handoffs,
                tracing=tracing,
                previous_response_id=previous_response_id,
                conversation_id=conversation_id,
                prompt=prompt,
            )

    interceptor = RecordingInterceptor()
    records = RecordSink(max_records=1000)
    model = RetryMutatingModel()
    tool_invocations = 0

    @function_tool(name_override="lookup")
    async def lookup() -> str:
        nonlocal tool_invocations
        tool_invocations += 1
        return "tool result"

    agent = Agent(name="governed-agent", model=model, tools=[lookup])

    with pytest.raises(AgentHooksExecutionError) as error_info:
        await Runner.run(
            agent,
            "hello",
            max_turns=3,
            run_config=_run_config(interceptor, records, with_tools=True),
        )

    assert id(error_info.value) != id(primary_error)
    assert primary_error.__traceback__ is None
    assert primary_error.__cause__ is None
    assert primary_error.__context__ is None
    assert model.observed_max_retries == [0, 0]
    assert tool_invocations == 1
    points = [record.interception_point for record in records]
    assert points == [
        InterceptionPoint.AGENT_STARTUP,
        InterceptionPoint.INPUT,
        InterceptionPoint.PRE_MODEL_CALL,
        InterceptionPoint.POST_MODEL_CALL,
        InterceptionPoint.PRE_TOOL_CALL,
        InterceptionPoint.POST_TOOL_CALL,
        InterceptionPoint.PRE_MODEL_CALL,
        InterceptionPoint.POST_MODEL_CALL,
        InterceptionPoint.AGENT_SHUTDOWN,
    ]
    pre_contexts = [
        context
        for context in interceptor.contexts
        if context["interception_point"] == InterceptionPoint.PRE_MODEL_CALL.value
    ]
    post_contexts = [
        context
        for context in interceptor.contexts
        if context["interception_point"] == InterceptionPoint.POST_MODEL_CALL.value
    ]
    assert len(pre_contexts) == 2
    assert len(post_contexts) == 2
    pre_request_ids = [context["request_id"] for context in pre_contexts]
    post_request_ids = [context["request_id"] for context in post_contexts]
    assert pre_request_ids == post_request_ids
    assert len(set(pre_request_ids)) == 2
    for request_id in pre_request_ids:
        UUID(request_id)
    assert post_contexts[1]["response"] == {
        "content": None,
        "tool_calls": [],
        "finish_reason": "error",
    }
    assert "usage" not in post_contexts[1]
    post_records = [
        record
        for record in records
        if record.interception_point is InterceptionPoint.POST_MODEL_CALL
    ]
    assert len(post_records) == 2
    assert failure_text not in repr(post_contexts[1])
    assert failure_text not in repr(post_records[1])


async def test_pre_model_deny_prevents_dispatch_and_post_record() -> None:
    interceptor = RecordingInterceptor(
        {
            InterceptionPoint.PRE_MODEL_CALL: Verdict(
                decision=Decision.DENY,
                reason="model_request_denied",
            )
        }
    )
    records = RecordSink(max_records=1000)
    model = FakeModel(initial_output=[_message("unused")])
    agent: Agent[None] = Agent(name="governed-agent", model=model)

    with pytest.raises(AgentHooksBlockedError) as error_info:
        await Runner.run(
            agent,
            "hello",
            max_turns=1,
            run_config=_run_config(interceptor, records),
        )

    assert error_info.value.point == InterceptionPoint.PRE_MODEL_CALL.value
    assert model.first_turn_args is None
    assert all(
        record.interception_point is not InterceptionPoint.POST_MODEL_CALL for record in records
    )


async def test_post_model_deny_counts_usage_and_discards_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "raw-model-secret-7d1c7a"
    interceptor = RecordingInterceptor(
        {
            InterceptionPoint.POST_MODEL_CALL: Verdict(
                decision=Decision.DENY,
                reason="model_response_denied",
            )
        }
    )
    records = RecordSink(max_records=1000)
    model = FakeModel(initial_output=[_message(secret)])
    model.set_hardcoded_usage(Usage(requests=1, input_tokens=7, output_tokens=3, total_tokens=10))
    agent: Agent[None] = Agent(name="governed-agent", model=model)
    accounted: list[Usage] = []
    original_add = Usage.add

    def track_add(self: Usage, other: Usage) -> None:
        assert all(
            context["interception_point"] != InterceptionPoint.POST_MODEL_CALL.value
            for context in interceptor.contexts
        )
        accounted.append(copy.deepcopy(other))
        original_add(self, other)

    monkeypatch.setattr(Usage, "add", track_add)

    with pytest.raises(AgentHooksBlockedError) as error_info:
        await Runner.run(
            agent,
            "hello",
            max_turns=1,
            run_config=_run_config(interceptor, records),
        )

    assert error_info.value.point == InterceptionPoint.POST_MODEL_CALL.value
    assert len(accounted) == 1
    assert accounted[0].requests == 1
    assert accounted[0].input_tokens == 7
    assert accounted[0].output_tokens == 3
    assert accounted[0].total_tokens == 10
    assert all(record.interception_point is not InterceptionPoint.OUTPUT for record in records)
    assert error_info.value.run_data is None
    traceback = error_info.value.__traceback__
    while traceback is not None:
        if "/src/agents/" in traceback.tb_frame.f_code.co_filename:
            assert secret not in repr(traceback.tb_frame.f_locals)
        traceback = traceback.tb_next


async def test_invalid_model_usage_is_rejected_before_accounting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interceptor = RecordingInterceptor()
    records = RecordSink(max_records=1000)
    malformed_usage = Usage(requests=1, total_tokens=1)
    malformed_usage.input_tokens = cast(Any, "not-an-integer")
    model = FakeModel(initial_output=[_message("unused")])
    model.set_hardcoded_usage(malformed_usage)
    agent: Agent[None] = Agent(name="governed-agent", model=model)
    accounted: list[Usage] = []
    original_add = Usage.add

    def track_add(self: Usage, other: Usage) -> None:
        accounted.append(self)
        original_add(self, other)

    monkeypatch.setattr(Usage, "add", track_add)

    with pytest.raises(AgentHooksBlockedError) as error_info:
        await Runner.run(
            agent,
            "hello",
            max_turns=1,
            run_config=_run_config(interceptor, records),
        )

    assert error_info.value.reason == "host_error:context_invalid"
    assert accounted == []
    post_records = [
        record
        for record in records
        if record.interception_point is InterceptionPoint.POST_MODEL_CALL
    ]
    assert len(post_records) == 1
    assert post_records[0].verdict.reason == "host_error:context_invalid"


@pytest.mark.parametrize(
    "location",
    [
        pytest.param("aggregate", id="aggregate-input-tokens"),
        pytest.param("request", id="request-input-tokens"),
    ],
)
async def test_copy_attacking_integer_usage_is_rejected_before_copy_and_accounting(
    location: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = f"copy-attacking-usage-secret-{location}-65bd2e"
    deepcopy_calls = 0

    class CopyAttackingInt(int):
        def __deepcopy__(self, _memo: dict[int, object]) -> object:
            nonlocal deepcopy_calls
            deepcopy_calls += 1
            return -1 if location == "aggregate" else "not-an-integer"

        def __repr__(self) -> str:
            return f"CopyAttackingInt({secret!r})"

    attack_value = CopyAttackingInt(3)
    malformed_usage = Usage(
        requests=1,
        input_tokens=3,
        output_tokens=2,
        total_tokens=5,
    )
    if location == "aggregate":
        malformed_usage.input_tokens = attack_value
    else:
        assert location == "request"
        request_usage = RequestUsage(
            input_tokens=3,
            output_tokens=2,
            total_tokens=5,
            input_tokens_details=copy.deepcopy(malformed_usage.input_tokens_details),
            output_tokens_details=copy.deepcopy(malformed_usage.output_tokens_details),
        )
        request_usage.input_tokens = attack_value
        malformed_usage.request_usage_entries = [request_usage]

    interceptor = RecordingInterceptor()
    records = RecordSink(max_records=1000)
    model = FakeModel(initial_output=[_message("unused")])
    model.set_hardcoded_usage(malformed_usage)
    agent: Agent[None] = Agent(name="governed-agent", model=model)
    add_calls = 0

    def track_add(_self: Usage, _other: Usage) -> None:
        nonlocal add_calls
        add_calls += 1

    monkeypatch.setattr(Usage, "add", track_add)

    with pytest.raises(AgentHooksBlockedError) as error_info:
        await Runner.run(
            agent,
            "hello",
            max_turns=1,
            run_config=_run_config(interceptor, records),
        )

    assert error_info.value.reason == "host_error:context_invalid"
    assert deepcopy_calls == 0
    assert add_calls == 0
    points = [record.interception_point for record in records]
    assert points.count(InterceptionPoint.POST_MODEL_CALL) == 1
    assert points.count(InterceptionPoint.AGENT_SHUTDOWN) == 1
    assert points[-2:] == [
        InterceptionPoint.POST_MODEL_CALL,
        InterceptionPoint.AGENT_SHUTDOWN,
    ]
    _assert_secret_absent_from_sdk_traceback(error_info.value, secret)


@pytest.mark.parametrize(
    "malformed_shape",
    [
        pytest.param("aggregate_input_details", id="aggregate-input-details"),
        pytest.param("aggregate_output_details", id="aggregate-output-details"),
        pytest.param("request_entries_not_list", id="request-entries-not-list"),
        pytest.param("request_entry_wrong_type", id="request-entry-wrong-type"),
        pytest.param("request_input_details", id="request-input-details"),
        pytest.param("request_output_details", id="request-output-details"),
    ],
)
async def test_nested_model_usage_shape_is_rejected_and_redacted(
    malformed_shape: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = f"nested-usage-secret-{malformed_shape}-4f81d2"

    class SecretUsageValue:
        def __repr__(self) -> str:
            return f"SecretUsageValue({secret!r})"

    malformed_value = SecretUsageValue()
    malformed_usage = Usage(
        requests=1,
        input_tokens=3,
        output_tokens=2,
        total_tokens=5,
    )
    valid_entry = RequestUsage(
        input_tokens=malformed_usage.input_tokens,
        output_tokens=malformed_usage.output_tokens,
        total_tokens=malformed_usage.total_tokens,
        input_tokens_details=copy.deepcopy(malformed_usage.input_tokens_details),
        output_tokens_details=copy.deepcopy(malformed_usage.output_tokens_details),
    )
    malformed_usage.request_usage_entries = [valid_entry]

    if malformed_shape == "aggregate_input_details":
        malformed_usage.input_tokens_details = cast(Any, malformed_value)
    elif malformed_shape == "aggregate_output_details":
        malformed_usage.output_tokens_details = cast(Any, malformed_value)
    elif malformed_shape == "request_entries_not_list":
        malformed_usage.request_usage_entries = cast(Any, (malformed_value,))
    elif malformed_shape == "request_entry_wrong_type":
        malformed_usage.request_usage_entries = cast(Any, [malformed_value])
    elif malformed_shape == "request_input_details":
        valid_entry.input_tokens_details = cast(Any, malformed_value)
    else:
        assert malformed_shape == "request_output_details"
        valid_entry.output_tokens_details = cast(Any, malformed_value)

    interceptor = RecordingInterceptor()
    records = RecordSink(max_records=1000)
    model = FakeModel(initial_output=[_message("unused")])
    model.set_hardcoded_usage(malformed_usage)
    agent: Agent[None] = Agent(name="governed-agent", model=model)
    add_calls: list[tuple[Usage, Usage]] = []

    def track_add(self: Usage, other: Usage) -> None:
        add_calls.append((self, other))

    monkeypatch.setattr(Usage, "add", track_add)

    with pytest.raises(AgentHooksBlockedError) as error_info:
        await Runner.run(
            agent,
            "hello",
            max_turns=1,
            run_config=_run_config(interceptor, records),
        )

    assert error_info.value.reason == "host_error:context_invalid"
    assert add_calls == []
    emitted_records = list(records)
    post_records = [
        record
        for record in emitted_records
        if record.interception_point is InterceptionPoint.POST_MODEL_CALL
    ]
    assert len(post_records) == 1
    assert post_records[0].verdict.reason == "host_error:context_invalid"
    assert [record.interception_point for record in emitted_records[-2:]] == [
        InterceptionPoint.POST_MODEL_CALL,
        InterceptionPoint.AGENT_SHUTDOWN,
    ]
    assert (
        sum(
            record.interception_point is InterceptionPoint.AGENT_SHUTDOWN
            for record in emitted_records
        )
        == 1
    )
    _assert_secret_absent_from_sdk_traceback(error_info.value, secret)


@pytest.mark.parametrize(
    "output",
    [
        [
            ResponseOutputMessage(
                id="message-1",
                content=[ResponseOutputRefusal(refusal="no", type="refusal")],
                role="assistant",
                status="completed",
                type="message",
            )
        ],
        [
            ResponseOutputMessage(
                id="message-with-citation",
                content=[
                    ResponseOutputText(
                        annotations=[
                            AnnotationURLCitation(
                                end_index=4,
                                start_index=0,
                                title="unreviewed title",
                                type="url_citation",
                                url="https://unreviewed.example/secret",
                            )
                        ],
                        text="done",
                        type="output_text",
                    )
                ],
                role="assistant",
                status="completed",
                type="message",
            )
        ],
        [_tool_call("call-1", arguments="not-json")],
    ],
)
async def test_unsupported_or_malformed_model_output_has_truthful_post_deny(
    output: list[Any],
) -> None:
    interceptor = RecordingInterceptor()
    records = RecordSink(max_records=1000)
    model = FakeModel(initial_output=output)
    model.set_hardcoded_usage(Usage(input_tokens=9, output_tokens=4, total_tokens=13))
    agent: Agent[None] = Agent(name="governed-agent", model=model)

    with pytest.raises(AgentHooksBlockedError) as error_info:
        await Runner.run(
            agent,
            "hello",
            max_turns=1,
            run_config=_run_config(interceptor, records),
        )

    assert error_info.value.reason == "host_error:adapter_unsupported"
    post_record = next(
        record
        for record in records
        if record.interception_point is InterceptionPoint.POST_MODEL_CALL
    )
    assert not post_record.proceeds
    assert post_record.verdict.reason == "host_error:adapter_unsupported"
    post_context = _context_for(interceptor, InterceptionPoint.POST_MODEL_CALL)
    assert post_context["usage"] == {"prompt_tokens": 9, "completion_tokens": 4}
    assert model.first_turn_args is not None
    assert all(record.interception_point is not InterceptionPoint.OUTPUT for record in records)


async def test_post_model_enforces_max_tool_calls_before_copy_or_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interceptor = RecordingInterceptor()
    records = RecordSink(max_records=1000)
    side_effects = 0
    copy_calls = 0

    original_model_copy = ResponseFunctionToolCall.model_copy

    def tracked_model_copy(
        self: ResponseFunctionToolCall,
        *args: Any,
        **kwargs: Any,
    ) -> ResponseFunctionToolCall:
        nonlocal copy_calls
        copy_calls += 1
        return original_model_copy(self, *args, **kwargs)

    monkeypatch.setattr(ResponseFunctionToolCall, "model_copy", tracked_model_copy)

    @function_tool(name_override="lookup")
    async def lookup() -> str:
        nonlocal side_effects
        side_effects += 1
        return "unused"

    model = FakeModel(initial_output=[_tool_call("call-1"), _tool_call("call-2")])
    agent = Agent(name="governed-agent", model=model, tools=[lookup])

    with pytest.raises(AgentHooksBlockedError) as error_info:
        await Runner.run(
            agent,
            "hello",
            max_turns=1,
            run_config=_run_config(interceptor, records, with_tools=True, max_tool_calls=1),
        )

    assert error_info.value.point == InterceptionPoint.POST_MODEL_CALL.value
    assert error_info.value.reason == "host_error:adapter_unsupported"
    assert copy_calls == 0
    assert side_effects == 0


async def test_oversized_model_result_emits_one_terminal_post_record() -> None:
    interceptor = RecordingInterceptor()
    records = RecordSink(max_records=1000)
    model = FakeModel(initial_output=[_message(_OVERSIZED_CONTEXT_TEXT)])
    agent: Agent[None] = Agent(name="governed-agent", model=model)
    run_config = _run_config(interceptor, records)
    run_config.agent_hooks = AgentHooksConfig(
        agent_id="governed-agent-v1",
        interceptors=(interceptor,),
        record_sink=records,
        limits=AgentHooksLimits(max_context_bytes=_BOUNDED_CONTEXT_BYTES),
    )

    with pytest.raises(AgentHooksBlockedError) as error_info:
        await Runner.run(
            agent,
            "hello",
            max_turns=1,
            run_config=run_config,
        )

    assert error_info.value.point == InterceptionPoint.POST_MODEL_CALL.value
    post_records = [
        record
        for record in records
        if record.interception_point is InterceptionPoint.POST_MODEL_CALL
    ]
    assert len(post_records) == 1
    assert post_records[0].verdict.reason == "host_error:context_invalid"
    assert model.first_turn_args is not None
    assert all(record.interception_point is not InterceptionPoint.OUTPUT for record in records)


async def test_oversized_tool_arguments_reject_before_json_parse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_parse(arguments: str | None) -> dict[str, Any] | None:
        raise AssertionError("oversized arguments reached JSON parsing")

    monkeypatch.setattr(approvals_module, "parse_function_tool_arguments", unexpected_parse)
    interceptor = RecordingInterceptor()
    records = RecordSink(max_records=1000)
    model = FakeModel(
        initial_output=[
            _tool_call(
                "call-1",
                arguments=json.dumps({"value": _OVERSIZED_CONTEXT_TEXT}),
            )
        ]
    )

    @function_tool(name_override="lookup")
    async def lookup(value: str) -> str:
        return value

    agent = Agent(
        name="governed-agent",
        model=model,
        tools=[lookup],
    )
    run_config = _run_config(interceptor, records, with_tools=True)
    run_config.agent_hooks = AgentHooksConfig(
        agent_id="governed-agent-v1",
        interceptors=(interceptor,),
        record_sink=records,
        limits=AgentHooksLimits(max_context_bytes=_BOUNDED_CONTEXT_BYTES),
    )

    with pytest.raises(AgentHooksBlockedError) as error_info:
        await Runner.run(agent, "hello", max_turns=1, run_config=run_config)

    assert error_info.value.reason == "host_error:context_invalid"
    post_records = [
        record
        for record in records
        if record.interception_point is InterceptionPoint.POST_MODEL_CALL
    ]
    assert len(post_records) == 1
    assert post_records[0].verdict.reason == "host_error:context_invalid"


async def test_aggregate_tool_arguments_reject_before_json_parse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parse_calls = 0
    tool_invocations = 0

    def unexpected_parse(arguments: str | None) -> dict[str, Any] | None:
        nonlocal parse_calls
        parse_calls += 1
        raise AssertionError("aggregate oversized arguments reached JSON parsing")

    monkeypatch.setattr(approvals_module, "parse_function_tool_arguments", unexpected_parse)
    arguments = json.dumps({"value": "x" * (48 * 1024)})
    assert len(arguments) < _BOUNDED_CONTEXT_BYTES
    assert len(arguments) * 2 > _BOUNDED_CONTEXT_BYTES
    interceptor = RecordingInterceptor()
    records = RecordSink(max_records=1000)
    model = FakeModel(
        initial_output=[
            _tool_call("call-1", arguments=arguments),
            _tool_call("call-2", arguments=arguments),
        ]
    )

    @function_tool(name_override="lookup")
    async def lookup(value: str) -> str:
        nonlocal tool_invocations
        tool_invocations += 1
        return value

    agent = Agent(name="governed-agent", model=model, tools=[lookup])
    run_config = _run_config(interceptor, records, with_tools=True)
    run_config.agent_hooks = AgentHooksConfig(
        agent_id="governed-agent-v1",
        interceptors=(interceptor,),
        record_sink=records,
        limits=AgentHooksLimits(max_context_bytes=_BOUNDED_CONTEXT_BYTES),
    )

    with pytest.raises(AgentHooksBlockedError) as error_info:
        await Runner.run(agent, "hello", max_turns=1, run_config=run_config)

    assert error_info.value.point == InterceptionPoint.POST_MODEL_CALL.value
    assert error_info.value.reason == "host_error:context_invalid"
    assert parse_calls == 0
    assert tool_invocations == 0
    post_records = [
        record
        for record in records
        if record.interception_point is InterceptionPoint.POST_MODEL_CALL
    ]
    assert len(post_records) == 1
    assert post_records[0].verdict.reason == "host_error:context_invalid"
    points = [record.interception_point for record in records]
    assert points.count(InterceptionPoint.AGENT_SHUTDOWN) == 1
    assert points[-1] is InterceptionPoint.AGENT_SHUTDOWN


async def test_deep_tool_arguments_reject_before_json_parse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_parse(arguments: str | None) -> dict[str, Any] | None:
        raise AssertionError("deep arguments reached JSON parsing")

    monkeypatch.setattr(approvals_module, "parse_function_tool_arguments", unexpected_parse)
    nested_value = "[" * 129 + "0" + "]" * 129
    arguments = '{"value":' + nested_value + "}"
    interceptor = RecordingInterceptor()
    records = RecordSink(max_records=1000)
    model = FakeModel(initial_output=[_tool_call("call-1", arguments=arguments)])

    @function_tool(name_override="lookup")
    async def lookup(value: object) -> str:
        return str(value)

    agent = Agent(name="governed-agent", model=model, tools=[lookup])

    with pytest.raises(AgentHooksBlockedError) as error_info:
        await Runner.run(
            agent,
            "hello",
            max_turns=1,
            run_config=_run_config(interceptor, records, with_tools=True),
        )

    assert error_info.value.reason == "host_error:context_invalid"
    post_records = [
        record
        for record in records
        if record.interception_point is InterceptionPoint.POST_MODEL_CALL
    ]
    assert len(post_records) == 1
    assert post_records[0].verdict.reason == "host_error:context_invalid"


async def test_freeform_verdict_metadata_fails_closed_without_record_disclosure() -> None:
    secret = "sensitive free-form policy explanation"
    records = RecordSink(max_records=1000)
    model = FakeModel(initial_output=[_message("unused")])
    agent = Agent(name="governed-agent", model=model)

    with pytest.raises(AgentHooksBlockedError):
        await Runner.run(
            agent,
            "hello",
            max_turns=1,
            run_config=_run_config(FreeformMetadataInterceptor(), records),
        )

    assert secret not in repr(records.snapshot())
    assert model.first_turn_args is None


@pytest.mark.parametrize(
    ("point", "path", "value"),
    [
        (InterceptionPoint.PRE_MODEL_CALL, "$target[0].role", "assistant"),
        (InterceptionPoint.POST_MODEL_CALL, "$target.finish_reason", "tool_calls"),
    ],
)
async def test_identity_changing_model_transform_has_truthful_deny(
    point: InterceptionPoint,
    path: str,
    value: object,
) -> None:
    interceptor = RecordingInterceptor(
        {
            point: Verdict(
                decision=Decision.TRANSFORM,
                transform=Transform(path=path, value=value),
            )
        }
    )
    records = RecordSink(max_records=1000)
    model = FakeModel(initial_output=[_message("unused")])
    agent: Agent[None] = Agent(name="governed-agent", model=model)

    with pytest.raises(AgentHooksBlockedError) as error_info:
        await Runner.run(
            agent,
            "hello",
            max_turns=1,
            run_config=_run_config(interceptor, records),
        )

    assert error_info.value.reason == "host_error:transform_invalid"
    point_record = next(record for record in records if record.interception_point is point)
    assert not point_record.proceeds
    assert point_record.verdict.reason == "host_error:transform_invalid"
    if point is InterceptionPoint.PRE_MODEL_CALL:
        assert model.first_turn_args is None
    else:
        assert model.first_turn_args is not None
    assert all(record.interception_point is not InterceptionPoint.OUTPUT for record in records)


async def test_folded_model_context_bound_fails_inside_truthful_record() -> None:
    interceptor = RecordingInterceptor(
        {
            InterceptionPoint.PRE_MODEL_CALL: Verdict(
                decision=Decision.TRANSFORM,
                transform=Transform(
                    path="$target[0].content",
                    value=_OVERSIZED_CONTEXT_TEXT,
                ),
            )
        }
    )
    records = RecordSink(max_records=1000)
    model = FakeModel(initial_output=[_message("unused")])
    agent = Agent(name="governed-agent", model=model)
    run_config = _run_config(interceptor, records)
    assert run_config.agent_hooks is not None
    run_config.agent_hooks = AgentHooksConfig(
        agent_id="governed-agent-v1",
        interceptors=(interceptor,),
        record_sink=records,
        limits=AgentHooksLimits(
            max_context_bytes=_BOUNDED_CONTEXT_BYTES,
            max_verdict_bytes=128 * 1024,
        ),
    )

    with pytest.raises(AgentHooksBlockedError) as error_info:
        await Runner.run(agent, "hello", max_turns=1, run_config=run_config)

    assert error_info.value.reason == "host_error:transform_invalid"
    pre_record = next(
        record
        for record in records
        if record.interception_point is InterceptionPoint.PRE_MODEL_CALL
    )
    assert pre_record.verdict.reason == "host_error:transform_invalid"
    assert model.first_turn_args is None


async def test_duplicate_model_tool_call_id_is_rejected_before_execution() -> None:
    interceptor = RecordingInterceptor()
    records = RecordSink(max_records=1000)
    model = FakeModel(initial_output=[_tool_call("duplicate"), _tool_call("duplicate")])

    @function_tool(name_override="lookup")
    async def lookup() -> str:
        return "unused"

    agent = Agent(
        name="governed-agent",
        model=model,
        tools=[lookup],
    )

    with pytest.raises(AgentHooksBlockedError) as error_info:
        await Runner.run(
            agent,
            "hello",
            max_turns=1,
            run_config=_run_config(interceptor, records, with_tools=True),
        )

    assert error_info.value.reason == "host_error:adapter_unsupported"
    assert all(
        record.interception_point is not InterceptionPoint.PRE_TOOL_CALL for record in records
    )


async def test_permit_labels_flow_to_later_model_context() -> None:
    interceptor = LabelInterceptor()
    records = RecordSink(max_records=1000)
    model = FakeModel(initial_output=[_message("done")])
    agent = Agent(name="governed-agent", model=model)

    await Runner.run(
        agent,
        "hello",
        max_turns=1,
        run_config=_run_config(interceptor, records),
    )

    assert interceptor.pre_source_labels == ["input-approved"]
    input_record = next(
        record for record in records if record.interception_point is InterceptionPoint.INPUT
    )
    assert input_record.verdict.result_labels == ("input-approved",)


async def test_nonproceeding_labels_do_not_flow_to_shutdown() -> None:
    interceptor = DeniedLabelInterceptor()
    records = RecordSink(max_records=1000)
    model = FakeModel(initial_output=[_message("unused")])
    agent = Agent(name="governed-agent", model=model)

    with pytest.raises(AgentHooksBlockedError):
        await Runner.run(
            agent,
            "hello",
            max_turns=1,
            run_config=_run_config(interceptor, records),
        )

    input_record = next(
        record for record in records if record.interception_point is InterceptionPoint.INPUT
    )
    assert not input_record.proceeds
    assert interceptor.shutdown_source_labels is None


async def test_oversized_label_fails_through_final_validator() -> None:
    interceptor = OversizedLabelInterceptor()
    records = RecordSink(max_records=1000)
    model = FakeModel(initial_output=[_message("unused")])
    agent = Agent(name="governed-agent", model=model)

    with pytest.raises(AgentHooksBlockedError) as error_info:
        await Runner.run(
            agent,
            "hello",
            max_turns=1,
            run_config=_run_config(interceptor, records),
        )

    assert error_info.value.reason == "host_error:transform_invalid"
    input_record = next(
        record for record in records if record.interception_point is InterceptionPoint.INPUT
    )
    assert input_record.verdict.reason == "host_error:transform_invalid"


async def test_admission_reserves_maximum_label_ledger_capacity() -> None:
    interceptor = RecordingInterceptor()
    records = RecordSink(max_records=1000)
    model = FakeModel(initial_output=[_message("unused")])
    agent = Agent(name="governed-agent", model=model)
    run_config = _run_config(interceptor, records)
    run_config.agent_hooks = AgentHooksConfig(
        agent_id="governed-agent-v1",
        interceptors=(interceptor,),
        record_sink=records,
        limits=AgentHooksLimits(max_context_bytes=64 * 1024),
    )

    with pytest.raises(UserError, match="trusted lifecycle envelopes"):
        await Runner.run(agent, "hello", max_turns=1, run_config=run_config)

    assert len(records) == 0
    assert model.first_turn_args is None


@pytest.mark.parametrize(
    "point",
    [InterceptionPoint.PRE_MODEL_CALL, InterceptionPoint.POST_MODEL_CALL],
)
async def test_model_emission_cancellation_drains_and_skips_downstream(
    point: InterceptionPoint,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interceptor = CancellingInterceptor(point)
    records = RecordSink(max_records=1000)
    model = FakeModel(initial_output=[_message("unused")])
    model.set_hardcoded_usage(Usage(input_tokens=4, output_tokens=1, total_tokens=5))
    agent: Agent[None] = Agent(name="governed-agent", model=model)
    accounted: list[Usage] = []
    original_add = Usage.add

    def track_add(self: Usage, other: Usage) -> None:
        accounted.append(copy.deepcopy(other))
        original_add(self, other)

    monkeypatch.setattr(Usage, "add", track_add)
    task = asyncio.create_task(
        Runner.run(
            agent,
            "hello",
            max_turns=1,
            run_config=_run_config(interceptor, records),
        )
    )
    await asyncio.wait_for(interceptor.started.wait(), timeout=5.0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert interceptor.drained.is_set()
    if point is InterceptionPoint.PRE_MODEL_CALL:
        assert model.first_turn_args is None
        assert accounted == []
    else:
        assert model.first_turn_args is not None
        assert len(accounted) == 1
        assert accounted[0].input_tokens == 4
        assert accounted[0].output_tokens == 1
        assert accounted[0].total_tokens == 5
    post_records = [
        record
        for record in records
        if record.interception_point is InterceptionPoint.POST_MODEL_CALL
    ]
    assert len(post_records) == (0 if point is InterceptionPoint.PRE_MODEL_CALL else 1)
    assert all(record.interception_point is not InterceptionPoint.OUTPUT for record in records)
    shutdown_records = [
        record
        for record in records
        if record.interception_point is InterceptionPoint.AGENT_SHUTDOWN
    ]
    assert len(shutdown_records) == 1
