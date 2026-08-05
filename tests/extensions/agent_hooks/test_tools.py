from __future__ import annotations

import asyncio
import copy
import json
from typing import Any, cast
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
    ResponseOutputText,
)

from agents import (
    Agent,
    FunctionTool,
    ModelSettings,
    RunConfig,
    Runner,
    ToolExecutionConfig,
    ToolGuardrailFunctionOutput,
    ToolInputGuardrailData,
    ToolOutputGuardrailData,
    function_tool,
    tool_input_guardrail,
    tool_output_guardrail,
)
from agents.agent_output import AgentOutputSchemaBase
from agents.extensions.agent_hooks import (
    AgentHooksBlockedError,
    AgentHooksConfig,
    AgentHooksExecutionError,
    AgentHooksLimits,
    RecordSink,
)
from agents.handoffs import Handoff
from agents.items import ModelResponse, ToolCallOutputItem, TResponseInputItem
from agents.models.interface import ModelTracing
from agents.retry import ModelRetrySettings
from agents.tool import Tool
from agents.tool_context import ToolContext
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


def _message(text: str) -> ResponseOutputMessage:
    return ResponseOutputMessage(
        id=f"message-{text}",
        content=[ResponseOutputText(annotations=[], text=text, type="output_text")],
        role="assistant",
        status="completed",
        type="message",
    )


def _tool_call(
    call_id: str = "model-call-1",
    *,
    value: str = "raw",
    arguments: str | None = None,
) -> ResponseFunctionToolCall:
    return ResponseFunctionToolCall(
        id=f"item-{call_id}",
        call_id=call_id,
        type="function_call",
        name="lookup",
        arguments=arguments if arguments is not None else json.dumps({"value": value}),
        status="completed",
    )


def _run_config(
    interceptor: object,
    records: RecordSink,
    *,
    sink: RecordSink | None = None,
    max_tool_calls: int = 32,
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
            parallel_tool_calls=False,
            retry=ModelRetrySettings(max_retries=0),
        ),
        tool_execution=ToolExecutionConfig(max_function_tool_concurrency=1),
    )


def _make_tool(
    events: list[tuple[str, object]],
    *,
    result: object = "real result",
) -> FunctionTool:
    @function_tool(name_override="lookup")
    async def lookup(value: str) -> str:
        events.append(("invoke", value))
        return cast(str, result)

    return lookup


def _two_turn_model(first_output: list[Any]) -> FakeModel:
    model = FakeModel(initial_output=first_output)
    model.set_next_output([_message("done")])
    return model


def _output_from_second_model_input(model: FakeModel) -> str:
    model_input = model.last_turn_args["input"]
    assert isinstance(model_input, list)
    outputs = [
        item
        for item in model_input
        if isinstance(item, dict) and item.get("type") == "function_call_output"
    ]
    assert len(outputs) == 1
    output = outputs[0].get("output")
    assert isinstance(output, str)
    return output


async def test_tool_allow_transforms_and_exact_pairing() -> None:
    interceptor = RecordingInterceptor(
        {
            InterceptionPoint.PRE_TOOL_CALL: Verdict(
                decision=Decision.TRANSFORM,
                transform=Transform(path="$target.value", value="governed argument"),
            ),
            InterceptionPoint.POST_TOOL_CALL: Verdict(
                decision=Decision.TRANSFORM,
                transform=Transform(path="$target", value="governed result"),
            ),
        }
    )
    records = RecordSink(max_records=1000)
    events: list[tuple[str, object]] = []
    model = _two_turn_model([_tool_call()])
    agent: Agent[None] = Agent(
        name="governed-agent",
        model=model,
        tools=[_make_tool(events)],
    )

    result = await Runner.run(
        agent,
        "hello",
        max_turns=2,
        run_config=_run_config(interceptor, records),
    )

    assert result.final_output == "done"
    assert result.last_agent is agent
    assert all(item.agent is agent for item in result.new_items)
    assert events == [("invoke", "governed argument")]
    assert _output_from_second_model_input(model) == "governed result"
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
        InterceptionPoint.OUTPUT,
        InterceptionPoint.AGENT_SHUTDOWN,
    ]
    pre_context = next(
        context
        for context in interceptor.contexts
        if context["interception_point"] == InterceptionPoint.PRE_TOOL_CALL.value
    )
    post_context = next(
        context
        for context in interceptor.contexts
        if context["interception_point"] == InterceptionPoint.POST_TOOL_CALL.value
    )
    assert pre_context["tool_call"]["id"] == post_context["tool_call"]["id"]
    UUID(pre_context["tool_call"]["id"])
    assert pre_context["tool_call"]["id"] != "model-call-1"
    assert post_context["tool_call"]["args"] == {"value": "governed argument"}
    assert post_context["budgets"] == {"tool_call_count": 1}


async def test_manual_function_tool_arguments_are_schema_validated_before_side_effects() -> None:
    interceptor = RecordingInterceptor()
    records = RecordSink(max_records=1000)
    invocations = 0

    async def invoke(context: ToolContext[Any], arguments: str) -> str:
        nonlocal invocations
        invocations += 1
        return arguments

    tool = FunctionTool(
        name="lookup",
        description="Lookup a value.",
        params_json_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
            "additionalProperties": False,
        },
        on_invoke_tool=invoke,
    )
    model = _two_turn_model([_tool_call(arguments=json.dumps({"value": 7}))])
    agent: Agent[None] = Agent(name="governed-agent", model=model, tools=[tool])

    result = await Runner.run(
        agent,
        "hello",
        max_turns=2,
        run_config=_run_config(interceptor, records),
    )

    assert result.final_output == "done"
    assert invocations == 0
    assert _output_from_second_model_input(model) == "blocked: governance_error"
    pre_tool_record = next(
        record for record in records if record.interception_point is InterceptionPoint.PRE_TOOL_CALL
    )
    assert not pre_tool_record.proceeds
    assert pre_tool_record.verdict.reason == "host_error:transform_invalid"


async def test_manual_function_tool_local_schema_reference_is_supported() -> None:
    interceptor = RecordingInterceptor()
    records = RecordSink(max_records=1000)
    invocations = 0

    async def invoke(_context: ToolContext[Any], arguments: str) -> str:
        nonlocal invocations
        invocations += 1
        return arguments

    tool = FunctionTool(
        name="lookup",
        description="Lookup a value.",
        params_json_schema={
            "type": "object",
            "properties": {"value": {"$ref": "#/$defs/value"}},
            "required": ["value"],
            "additionalProperties": False,
            "$defs": {"value": {"type": "string"}},
        },
        on_invoke_tool=invoke,
    )
    model = _two_turn_model([_tool_call(value="approved")])
    agent: Agent[None] = Agent(name="governed-agent", model=model, tools=[tool])

    result = await Runner.run(
        agent,
        "hello",
        max_turns=2,
        run_config=_run_config(interceptor, records),
    )

    assert result.final_output == "done"
    assert invocations == 1


async def test_pre_tool_deny_is_local_and_skips_tool_invocation() -> None:
    interceptor = RecordingInterceptor(
        {
            InterceptionPoint.PRE_TOOL_CALL: Verdict(
                decision=Decision.DENY,
                reason="tool_denied",
            )
        }
    )
    records = RecordSink(max_records=1000)
    events: list[tuple[str, object]] = []
    model = _two_turn_model([_tool_call()])
    agent: Agent[None] = Agent(
        name="governed-agent",
        model=model,
        tools=[_make_tool(events)],
    )

    result = await Runner.run(
        agent,
        "hello",
        max_turns=2,
        run_config=_run_config(interceptor, records),
    )

    assert result.final_output == "done"
    assert events == []
    assert _output_from_second_model_input(model) == "blocked: policy_denied"
    assert [record.interception_point for record in records].count(
        InterceptionPoint.PRE_TOOL_CALL
    ) == 1
    assert all(
        record.interception_point is not InterceptionPoint.POST_TOOL_CALL for record in records
    )
    second_pre_model = [
        context
        for context in interceptor.contexts
        if context["interception_point"] == InterceptionPoint.PRE_MODEL_CALL.value
    ][1]
    assert second_pre_model["budgets"] == {"tool_call_count": 0}


async def test_post_tool_deny_keeps_one_side_effect_and_discards_real_result() -> None:
    interceptor = RecordingInterceptor(
        {
            InterceptionPoint.POST_TOOL_CALL: Verdict(
                decision=Decision.DENY,
                reason="result_denied",
            )
        }
    )
    records = RecordSink(max_records=1000)
    events: list[tuple[str, object]] = []
    model = _two_turn_model([_tool_call()])
    agent: Agent[None] = Agent(
        name="governed-agent",
        model=model,
        tools=[_make_tool(events)],
    )

    result = await Runner.run(
        agent,
        "hello",
        max_turns=2,
        run_config=_run_config(interceptor, records),
    )

    assert result.final_output == "done"
    assert events == [("invoke", "raw")]
    blocked_output = _output_from_second_model_input(model)
    assert blocked_output == "blocked: policy_denied"
    assert "real result" not in blocked_output
    assert [record.interception_point for record in records].count(
        InterceptionPoint.POST_TOOL_CALL
    ) == 1
    second_pre_model = [
        context
        for context in interceptor.contexts
        if context["interception_point"] == InterceptionPoint.PRE_MODEL_CALL.value
    ][1]
    assert second_pre_model["budgets"] == {"tool_call_count": 1}


async def test_post_model_argument_transform_reaches_tool_pipeline() -> None:
    class TransformToolArguments(RecordingInterceptor):
        async def intercept(self, context: AgentContext, /) -> Verdict:
            self.contexts.append(copy.deepcopy(context))
            target = context.get("target")
            if (
                context["interception_point"] == InterceptionPoint.POST_MODEL_CALL.value
                and isinstance(target, dict)
                and target.get("tool_calls")
            ):
                return Verdict(
                    decision=Decision.TRANSFORM,
                    transform=Transform(
                        path="$target.tool_calls[0].args.value",
                        value="from post model",
                    ),
                )
            return ALLOW

    interceptor = TransformToolArguments()
    records = RecordSink(max_records=1000)
    events: list[tuple[str, object]] = []
    model = _two_turn_model([_tool_call()])
    agent: Agent[None] = Agent(
        name="governed-agent",
        model=model,
        tools=[_make_tool(events)],
    )

    result = await Runner.run(
        agent,
        "hello",
        max_turns=2,
        run_config=_run_config(interceptor, records),
    )

    assert result.final_output == "done"
    assert events == [("invoke", "from post model")]
    pre_tool = next(
        context
        for context in interceptor.contexts
        if context["interception_point"] == InterceptionPoint.PRE_TOOL_CALL.value
    )
    assert pre_tool["target"] == {"value": "from post model"}


@pytest.mark.parametrize(
    ("point", "value", "expected_invocations"),
    [
        (InterceptionPoint.PRE_TOOL_CALL, [], 0),
        (InterceptionPoint.POST_TOOL_CALL, {"not": "a string"}, 1),
    ],
)
async def test_invalid_tool_transform_becomes_local_truthful_deny(
    point: InterceptionPoint,
    value: object,
    expected_invocations: int,
) -> None:
    interceptor = RecordingInterceptor(
        {
            point: Verdict(
                decision=Decision.TRANSFORM,
                transform=Transform(path="$target", value=value),
            )
        }
    )
    records = RecordSink(max_records=1000)
    events: list[tuple[str, object]] = []
    model = _two_turn_model([_tool_call()])
    agent: Agent[None] = Agent(
        name="governed-agent",
        model=model,
        tools=[_make_tool(events)],
    )

    result = await Runner.run(
        agent,
        "hello",
        max_turns=2,
        run_config=_run_config(interceptor, records),
    )

    assert result.final_output == "done"
    assert [name for name, _ in events].count("invoke") == expected_invocations
    assert _output_from_second_model_input(model) == "blocked: governance_error"
    point_record = next(record for record in records if record.interception_point is point)
    assert not point_record.proceeds
    assert point_record.verdict.reason == "host_error:transform_invalid"
    if point is InterceptionPoint.PRE_TOOL_CALL:
        assert all(
            record.interception_point is not InterceptionPoint.POST_TOOL_CALL for record in records
        )


@pytest.mark.parametrize("arguments", ["not-json", "[]"])
async def test_malformed_tool_arguments_fail_closed_without_side_effect(
    arguments: str,
) -> None:
    interceptor = RecordingInterceptor()
    records = RecordSink(max_records=1000)
    events: list[tuple[str, object]] = []
    model = FakeModel(initial_output=[_tool_call(arguments=arguments)])
    agent = Agent(name="governed-agent", model=model, tools=[_make_tool(events)])

    with pytest.raises(Exception) as error_info:
        await Runner.run(
            agent,
            "hello",
            max_turns=1,
            run_config=_run_config(interceptor, records),
        )

    assert type(error_info.value).__name__ == "AgentHooksBlockedError"
    assert events == []
    post_model = next(
        record
        for record in records
        if record.interception_point is InterceptionPoint.POST_MODEL_CALL
    )
    assert post_model.verdict.reason == "host_error:adapter_unsupported"


async def test_non_string_tool_result_is_blocked_before_model_input() -> None:
    interceptor = RecordingInterceptor()
    records = RecordSink(max_records=1000)
    events: list[tuple[str, object]] = []
    tool = _make_tool(events, result={"secret": "not incorporated"})
    model = _two_turn_model([_tool_call()])
    agent: Agent[None] = Agent(name="governed-agent", model=model, tools=[tool])

    result = await Runner.run(
        agent,
        "hello",
        max_turns=2,
        run_config=_run_config(interceptor, records),
    )

    assert result.final_output == "done"
    assert events == [("invoke", "raw")]
    assert _output_from_second_model_input(model) == "blocked: governance_error"
    post_record = next(
        record
        for record in records
        if record.interception_point is InterceptionPoint.POST_TOOL_CALL
    )
    assert post_record.verdict.reason == "host_error:adapter_unsupported"


async def test_oversized_tool_result_emits_one_terminal_post_record() -> None:
    interceptor = RecordingInterceptor()
    records = RecordSink(max_records=1000)
    events: list[tuple[str, object]] = []
    model = _two_turn_model([_tool_call()])
    agent: Agent[None] = Agent(
        name="governed-agent",
        model=model,
        tools=[_make_tool(events, result=_OVERSIZED_CONTEXT_TEXT)],
    )
    run_config = _run_config(interceptor, records)
    run_config.agent_hooks = AgentHooksConfig(
        agent_id="governed-agent-v1",
        interceptors=(interceptor,),
        record_sink=records,
        limits=AgentHooksLimits(max_context_bytes=_BOUNDED_CONTEXT_BYTES),
    )

    result = await Runner.run(
        agent,
        "hello",
        max_turns=2,
        run_config=run_config,
    )

    assert result.final_output == "done"
    post_records = [
        record
        for record in records
        if record.interception_point is InterceptionPoint.POST_TOOL_CALL
    ]
    assert len(post_records) == 1
    assert post_records[0].verdict.reason == "host_error:context_invalid"
    assert events == [("invoke", "raw")]
    assert _output_from_second_model_input(model) == "blocked: governance_error"


async def test_sdk_generated_string_error_is_governed_as_error_result() -> None:
    interceptor = RecordingInterceptor()
    records = RecordSink(max_records=1000)
    events: list[tuple[str, object]] = []

    @function_tool(name_override="lookup")
    async def lookup(value: str) -> str:
        events.append(("invoke", value))
        raise ValueError("sensitive tool failure")

    model = _two_turn_model([_tool_call()])
    agent: Agent[None] = Agent(name="governed-agent", model=model, tools=[lookup])

    result = await Runner.run(
        agent,
        "hello",
        max_turns=2,
        run_config=_run_config(interceptor, records),
    )

    assert result.final_output == "done"
    assert events == [("invoke", "raw")]
    post_context = next(
        context
        for context in interceptor.contexts
        if context["interception_point"] == InterceptionPoint.POST_TOOL_CALL.value
    )
    assert post_context["tool_result"]["is_error"] is True
    assert isinstance(post_context["target"], str)
    assert _output_from_second_model_input(model) == post_context["target"]


async def test_unhandled_tool_error_emits_post_before_error_shutdown() -> None:
    interceptor = RecordingInterceptor()
    records = RecordSink(max_records=1000)
    events: list[tuple[str, object]] = []

    @function_tool(name_override="lookup", failure_error_function=None)
    async def lookup(value: str) -> str:
        events.append(("invoke", value))
        raise RuntimeError("sensitive tool failure")

    model = FakeModel(initial_output=[_tool_call()])
    agent = Agent(name="governed-agent", model=model, tools=[lookup])

    with pytest.raises(AgentHooksExecutionError):
        await Runner.run(
            agent,
            "hello",
            max_turns=1,
            run_config=_run_config(interceptor, records),
        )

    assert events == [("invoke", "raw")]
    points = [record.interception_point for record in records]
    assert points.count(InterceptionPoint.PRE_TOOL_CALL) == 1
    assert points.count(InterceptionPoint.POST_TOOL_CALL) == 1
    assert points[-1] is InterceptionPoint.AGENT_SHUTDOWN
    post_context = next(
        context
        for context in interceptor.contexts
        if context["interception_point"] == InterceptionPoint.POST_TOOL_CALL.value
    )
    assert post_context["tool_result"] == {
        "value": "tool_error:RuntimeError",
        "is_error": True,
        "duration_ms": post_context["tool_result"]["duration_ms"],
    }
    assert "sensitive tool failure" not in repr(post_context)


async def test_tool_base_exception_emits_post_before_error_shutdown() -> None:
    class FatalToolError(BaseException):
        pass

    primary_error = FatalToolError()
    interceptor = RecordingInterceptor()
    records = RecordSink(max_records=1000)

    @function_tool(name_override="lookup", failure_error_function=None)
    async def lookup(value: str) -> str:
        raise primary_error

    model = FakeModel(initial_output=[_tool_call()])
    agent = Agent(name="governed-agent", model=model, tools=[lookup])

    with pytest.raises(AgentHooksExecutionError):
        await Runner.run(
            agent,
            "hello",
            max_turns=1,
            run_config=_run_config(interceptor, records),
        )

    assert primary_error.__traceback__ is None
    assert primary_error.__cause__ is None
    assert primary_error.__context__ is None
    points = [record.interception_point for record in records]
    assert points.count(InterceptionPoint.PRE_TOOL_CALL) == 1
    assert points.count(InterceptionPoint.POST_TOOL_CALL) == 1
    assert points[-1] is InterceptionPoint.AGENT_SHUTDOWN


async def test_governed_tool_cannot_replay_prior_host_error() -> None:
    denied_records = RecordSink(max_records=1000)
    with pytest.raises(AgentHooksBlockedError) as denied_info:
        await Runner.run(
            Agent(name="denied-agent", model=FakeModel()),
            "hello",
            max_turns=1,
            run_config=_run_config(
                RecordingInterceptor(
                    {
                        InterceptionPoint.INPUT: Verdict(
                            decision=Decision.DENY,
                            reason="policy_denied",
                        )
                    }
                ),
                denied_records,
            ),
        )

    replayed_error = denied_info.value

    @function_tool(name_override="lookup", failure_error_function=None)
    async def lookup(value: str) -> str:
        raise replayed_error

    records = RecordSink(max_records=1000)
    model = FakeModel(initial_output=[_tool_call(value="approved")])
    agent = Agent(name="governed-agent", model=model, tools=[lookup])

    with pytest.raises(AgentHooksExecutionError):
        await Runner.run(
            agent,
            "hello",
            max_turns=1,
            run_config=_run_config(RecordingInterceptor(), records),
        )


@pytest.mark.parametrize(
    ("point", "expected_invocations"),
    [
        (InterceptionPoint.PRE_TOOL_CALL, 0),
        (InterceptionPoint.POST_TOOL_CALL, 1),
    ],
)
async def test_tool_emission_cancellation_drains_before_shutdown(
    point: InterceptionPoint,
    expected_invocations: int,
) -> None:
    interceptor = CancellingInterceptor(point)
    records = RecordSink(max_records=1000)
    events: list[tuple[str, object]] = []
    model = _two_turn_model([_tool_call()])
    agent = Agent(name="governed-agent", model=model, tools=[_make_tool(events)])
    task = asyncio.create_task(
        Runner.run(
            agent,
            "hello",
            max_turns=2,
            run_config=_run_config(interceptor, records),
        )
    )
    await interceptor.started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert interceptor.drained.is_set()
    assert [name for name, _ in events].count("invoke") == expected_invocations
    assert [record.interception_point for record in records].count(
        InterceptionPoint.POST_TOOL_CALL
    ) == (0 if point is InterceptionPoint.PRE_TOOL_CALL else 1)
    shutdown_records = [
        record
        for record in records
        if record.interception_point is InterceptionPoint.AGENT_SHUTDOWN
    ]
    assert len(shutdown_records) == 1


async def test_multiple_tool_calls_execute_serially_in_model_order() -> None:
    active = 0
    invocation_order: list[str] = []
    first_entered = asyncio.Event()
    release_first = asyncio.Event()

    @function_tool(name_override="lookup")
    async def lookup(value: str) -> str:
        nonlocal active
        active += 1
        assert active == 1
        invocation_order.append(value)
        if value == "first":
            first_entered.set()
            await release_first.wait()
        active -= 1
        return value

    interceptor = RecordingInterceptor()
    records = RecordSink(max_records=1000)
    model = _two_turn_model(
        [
            _tool_call("call-first", value="first"),
            _tool_call("call-second", value="second"),
        ]
    )
    agent = Agent(name="governed-agent", model=model, tools=[lookup])
    task = asyncio.create_task(
        Runner.run(
            agent,
            "hello",
            max_turns=2,
            run_config=_run_config(interceptor, records, max_tool_calls=2),
        )
    )
    await first_entered.wait()
    assert invocation_order == ["first"]
    release_first.set()
    result = await task

    assert result.final_output == "done"
    assert invocation_order == ["first", "second"]
    assert [record.interception_point for record in records].count(
        InterceptionPoint.PRE_TOOL_CALL
    ) == 2
    assert [record.interception_point for record in records].count(
        InterceptionPoint.POST_TOOL_CALL
    ) == 2


async def test_original_tool_mutation_after_start_does_not_change_execution() -> None:
    class PausingInterceptor(RecordingInterceptor):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def intercept(self, context: AgentContext, /) -> Verdict:
            self.contexts.append(copy.deepcopy(context))
            if context["interception_point"] == InterceptionPoint.PRE_MODEL_CALL.value:
                self.started.set()
                await self.release.wait()
            return ALLOW

    interceptor = PausingInterceptor()
    records = RecordSink(max_records=1000)
    invocations: list[str] = []

    @function_tool(name_override="lookup")
    async def lookup(value: str) -> str:
        invocations.append("admitted")
        return value

    async def replacement(context: ToolContext[Any], arguments: str) -> str:
        invocations.append("replacement")
        return arguments

    model = _two_turn_model([_tool_call()])
    agent = Agent(name="governed-agent", model=model, tools=[lookup])
    task = asyncio.create_task(
        Runner.run(
            agent,
            "hello",
            max_turns=2,
            run_config=_run_config(interceptor, records),
        )
    )
    await interceptor.started.wait()
    lookup.on_invoke_tool = replacement
    interceptor.release.set()
    result = await task

    assert result.final_output == "done"
    assert invocations == ["admitted"]


async def test_manual_tool_schema_and_guardrail_mutations_do_not_change_snapshot() -> None:
    class PausingInterceptor(RecordingInterceptor):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def intercept(self, context: AgentContext, /) -> Verdict:
            self.contexts.append(copy.deepcopy(context))
            if context["interception_point"] == InterceptionPoint.PRE_MODEL_CALL.value:
                self.started.set()
                await self.release.wait()
            return ALLOW

    side_effects: list[str] = []

    async def invoke(_context: ToolContext[Any], arguments: str) -> str:
        side_effects.append(f"invoke:{arguments}")
        return arguments

    @tool_input_guardrail
    def hostile_input_guardrail(
        _data: ToolInputGuardrailData,
    ) -> ToolGuardrailFunctionOutput:
        side_effects.append("input_guardrail")
        raise AssertionError("mutated input guardrail executed")

    @tool_output_guardrail
    def hostile_output_guardrail(
        _data: ToolOutputGuardrailData,
    ) -> ToolGuardrailFunctionOutput:
        side_effects.append("output_guardrail")
        raise AssertionError("mutated output guardrail executed")

    schema: dict[str, Any] = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    tool = FunctionTool(
        name="lookup",
        description="Lookup a value.",
        params_json_schema=schema,
        on_invoke_tool=invoke,
        strict_json_schema=False,
        tool_input_guardrails=[],
        tool_output_guardrails=[],
    )
    interceptor = PausingInterceptor()
    records = RecordSink(max_records=1000)
    model = _two_turn_model([_tool_call(arguments=json.dumps({"value": 7}))])
    agent: Agent[None] = Agent(name="governed-agent", model=model, tools=[tool])
    task = asyncio.create_task(
        Runner.run(
            agent,
            "hello",
            max_turns=2,
            run_config=_run_config(interceptor, records),
        )
    )

    try:
        await asyncio.wait_for(interceptor.started.wait(), timeout=5.0)
        schema["properties"]["value"]["type"] = "integer"
        assert tool.tool_input_guardrails is not None
        assert tool.tool_output_guardrails is not None
        tool.tool_input_guardrails.append(hostile_input_guardrail)
        tool.tool_output_guardrails.append(hostile_output_guardrail)
        interceptor.release.set()
        result = await asyncio.wait_for(task, timeout=5.0)
    finally:
        interceptor.release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert result.final_output == "done"
    assert side_effects == []
    assert model.first_turn_args is not None
    snapshot_tools = cast(list[FunctionTool], model.first_turn_args["tools"])
    assert len(snapshot_tools) == 1
    snapshot_tool = snapshot_tools[0]
    assert type(snapshot_tool) is FunctionTool
    assert snapshot_tool.strict_json_schema is False
    assert snapshot_tool.params_json_schema["properties"]["value"]["type"] == "string"
    assert snapshot_tool.tool_input_guardrails is None
    assert snapshot_tool.tool_output_guardrails is None
    pre_tool_records = [
        record for record in records if record.interception_point is InterceptionPoint.PRE_TOOL_CALL
    ]
    assert len(pre_tool_records) == 1
    assert not pre_tool_records[0].proceeds
    assert pre_tool_records[0].verdict.reason == "host_error:transform_invalid"
    assert _output_from_second_model_input(model) == "blocked: governance_error"


async def test_model_tool_definition_mutation_cannot_change_governed_execution() -> None:
    admitted_invocations: list[str] = []
    hostile_callbacks: list[str] = []

    async def hostile_invoke(_context: ToolContext[Any], _arguments: str) -> str:
        hostile_callbacks.append("invoke")
        return "hostile result"

    @tool_output_guardrail
    def hostile_output_guardrail(
        _data: ToolOutputGuardrailData,
    ) -> ToolGuardrailFunctionOutput:
        hostile_callbacks.append("output_guardrail")
        raise AssertionError("model-installed output guardrail executed")

    def hostile_custom_data_extractor(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        hostile_callbacks.append("custom_data_extractor")
        raise AssertionError("model-installed custom data extractor executed")

    class MutatingFakeModel(FakeModel):
        def __init__(self) -> None:
            super().__init__(initial_output=[_tool_call(value="approved")])
            self.set_next_output([_message("done")])
            self.calls = 0

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
            if self.calls == 0:
                assert len(tools) == 1
                model_tool = tools[0]
                assert isinstance(model_tool, FunctionTool)
                model_tool.on_invoke_tool = hostile_invoke
                model_tool.tool_output_guardrails = [hostile_output_guardrail]
                model_tool.custom_data_extractor = cast(Any, hostile_custom_data_extractor)
                model_tool._output_type_adapter = cast(Any, object())
                model_tool.params_json_schema["properties"] = {"hostile": {"type": "integer"}}
            self.calls += 1
            return await super().get_response(
                system_instructions,
                input,
                model_settings,
                tools,
                output_schema,
                handoffs,
                tracing,
                previous_response_id=previous_response_id,
                conversation_id=conversation_id,
                prompt=prompt,
            )

    @function_tool(name_override="lookup")
    async def lookup(value: str) -> str:
        admitted_invocations.append(value)
        return "governed result"

    interceptor = RecordingInterceptor()
    records = RecordSink(max_records=1000)
    model = MutatingFakeModel()
    agent: Agent[None] = Agent(name="governed-agent", model=model, tools=[lookup])

    result = await Runner.run(
        agent,
        "hello",
        max_turns=2,
        run_config=_run_config(interceptor, records),
    )

    assert result.final_output == "done"
    assert admitted_invocations == ["approved"]
    assert hostile_callbacks == []
    assert _output_from_second_model_input(model) == "governed result"
    points = [record.interception_point for record in records]
    assert points.count(InterceptionPoint.PRE_TOOL_CALL) == 1
    assert points.count(InterceptionPoint.POST_TOOL_CALL) == 1


async def test_tool_context_mutations_do_not_escape_governed_post_processing() -> None:
    @function_tool(name_override="lookup")
    async def lookup(context: ToolContext[None], value: str) -> str:
        assert context.agent is None
        assert context.run_config is None
        assert context.tool_call is not None
        context.tool_call.arguments = json.dumps({"value": "hostile"})
        context.tool_call.call_id = "hostile-model-call"
        context.tool_arguments = json.dumps({"value": "hostile"})
        context._custom_data = {"hostile": "callback-local"}
        context.usage.requests = 999
        return f"governed result:{value}"

    interceptor = RecordingInterceptor()
    records = RecordSink(max_records=1000)
    model = _two_turn_model([_tool_call(value="approved")])
    agent: Agent[None] = Agent(name="governed-agent", model=model, tools=[lookup])

    result = await Runner.run(
        agent,
        "hello",
        max_turns=2,
        run_config=_run_config(interceptor, records),
    )

    assert result.final_output == "done"
    post_context = next(
        context
        for context in interceptor.contexts
        if context["interception_point"] == InterceptionPoint.POST_TOOL_CALL.value
    )
    assert post_context["tool_call"]["args"] == {"value": "approved"}
    assert post_context["target"] == "governed result:approved"
    model_input = model.last_turn_args["input"]
    assert isinstance(model_input, list)
    model_outputs = [
        item
        for item in model_input
        if isinstance(item, dict) and item.get("type") == "function_call_output"
    ]
    assert len(model_outputs) == 1
    assert model_outputs[0].get("call_id") == "model-call-1"
    assert model_outputs[0].get("output") == "governed result:approved"
    generated_outputs = [item for item in result.new_items if isinstance(item, ToolCallOutputItem)]
    assert len(generated_outputs) == 1
    assert generated_outputs[0].custom_data is None
    points = [record.interception_point for record in records]
    assert points.count(InterceptionPoint.PRE_TOOL_CALL) == 1
    assert points.count(InterceptionPoint.POST_TOOL_CALL) == 1


async def test_parent_cancellation_pairs_and_drains_started_tool_invocation() -> None:
    started = asyncio.Event()
    drained = asyncio.Event()

    @function_tool(name_override="lookup")
    async def lookup(value: str) -> str:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            drained.set()
        return value

    interceptor = RecordingInterceptor()
    records = RecordSink(max_records=1000)
    model = _two_turn_model([_tool_call(value="approved")])
    agent = Agent(name="governed-agent", model=model, tools=[lookup])
    task = asyncio.create_task(
        Runner.run(
            agent,
            "hello",
            max_turns=2,
            run_config=_run_config(interceptor, records),
        )
    )
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert drained.is_set()
    assert [record.interception_point for record in records].count(
        InterceptionPoint.PRE_TOOL_CALL
    ) == 1
    assert [record.interception_point for record in records].count(
        InterceptionPoint.POST_TOOL_CALL
    ) == 1
    assert [record.interception_point for record in records].count(
        InterceptionPoint.AGENT_SHUTDOWN
    ) == 1


async def test_parent_cancellation_waits_for_suppressing_governed_tool() -> None:
    started = asyncio.Event()
    cancellation_caught = asyncio.Event()
    release = asyncio.Event()

    @function_tool(name_override="lookup")
    async def lookup(value: str) -> str:
        started.set()
        try:
            await asyncio.Event().wait()
            raise AssertionError("tool wait unexpectedly completed")
        except asyncio.CancelledError:
            cancellation_caught.set()
            await release.wait()
            return value

    records = RecordSink(max_records=1000)
    model = _two_turn_model([_tool_call(value="approved")])
    agent = Agent(name="governed-agent", model=model, tools=[lookup])
    task = asyncio.create_task(
        Runner.run(
            agent,
            "hello",
            max_turns=2,
            run_config=_run_config(RecordingInterceptor(), records),
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

    points = [record.interception_point for record in records]
    assert points.count(InterceptionPoint.PRE_TOOL_CALL) == 1
    assert points.count(InterceptionPoint.POST_TOOL_CALL) == 1
    assert points[-1] is InterceptionPoint.AGENT_SHUTDOWN
