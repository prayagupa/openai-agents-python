from __future__ import annotations

import asyncio
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
from openai.types.responses import ResponseOutputMessage, ResponseOutputText

import agents.run as run_module
from agents import (
    Agent,
    FunctionTool,
    ModelSettings,
    RunConfig,
    Runner,
    Tool,
    ToolExecutionConfig,
    function_tool,
)
from agents.exceptions import UserError
from agents.extensions.agent_hooks import (
    AgentHooksBlockedError,
    AgentHooksConfig,
    AgentHooksExecutionError,
    AgentHooksLimits,
    RecordSink,
)
from agents.items import ItemHelpers
from agents.lifecycle import RunHooks
from agents.models.interface import ModelProvider
from agents.result import RunResultStreaming
from agents.retry import ModelRetrySettings
from agents.run_config import SandboxRunConfig
from agents.run_context import AgentHookContext
from agents.run_internal.agent_hooks import (
    AgentHooksRunSession,
    bind_agent_hooks_session,
    create_agent_hooks_session,
    get_current_agent_hooks_session,
    reset_agent_hooks_session,
)
from agents.sandbox.runtime import SandboxRuntime
from tests.fake_model import FakeModel
from tests.test_responses import get_function_tool

_BOUNDED_CONTEXT_BYTES = 80 * 1024
_OVERSIZED_CONTEXT_TEXT = "x" * (96 * 1024)


class AllowInterceptor:
    async def intercept(self, context: AgentContext, /) -> Verdict:
        return ALLOW


class RecordingInterceptor:
    def __init__(self, verdicts: dict[InterceptionPoint, Verdict] | None = None) -> None:
        self.verdicts = verdicts or {}
        self.points: list[InterceptionPoint] = []
        self.shutdown_reasons: list[str] = []

    async def intercept(self, context: AgentContext, /) -> Verdict:
        point = InterceptionPoint(context["interception_point"])
        self.points.append(point)
        if point is InterceptionPoint.AGENT_SHUTDOWN:
            summary = context.get("summary")
            if isinstance(summary, dict) and isinstance(summary.get("reason"), str):
                self.shutdown_reasons.append(summary["reason"])
        return self.verdicts.get(point, ALLOW)


class RecordingRunHooks(RunHooks[None]):
    def __init__(self) -> None:
        self.started = 0
        self.outputs: list[object] = []

    async def on_agent_start(
        self,
        context: AgentHookContext[None],
        agent: Agent[None],
    ) -> None:
        self.started += 1

    async def on_agent_end(
        self,
        context: AgentHookContext[None],
        agent: Agent[None],
        output: Any,
    ) -> None:
        self.outputs.append(output)


class ContextRecordingRunHooks(RecordingRunHooks):
    def __init__(self) -> None:
        super().__init__()
        self.sessions: list[AgentHooksRunSession | None] = []

    async def on_agent_end(
        self,
        context: AgentHookContext[None],
        agent: Agent[None],
        output: Any,
    ) -> None:
        await super().on_agent_end(context, agent, output)
        self.sessions.append(get_current_agent_hooks_session())


class MetadataInterceptor:
    def __init__(self) -> None:
        self.agent_ids: list[str] = []
        self.frameworks: list[str] = []
        self.session_ids: list[str] = []
        self.caller_session_ids: list[str] = []
        self.correlation_ids: list[str] = []
        self.trace_ids: list[str] = []
        self.span_ids: list[str] = []
        self.startup_tools: list[list[str]] = []
        self.child_session_bound: list[bool] = []

    async def intercept(self, context: AgentContext, /) -> Verdict:
        agent = context.get("agent")
        session = context.get("session")
        extensions = context.get("extensions")
        if isinstance(agent, dict):
            if isinstance(agent.get("id"), str):
                self.agent_ids.append(agent["id"])
            if isinstance(agent.get("framework"), str):
                self.frameworks.append(agent["framework"])
        if isinstance(session, dict) and isinstance(session.get("id"), str):
            self.session_ids.append(session["id"])
        trace = context.get("trace")
        if isinstance(trace, dict):
            if isinstance(trace.get("trace_id"), str):
                self.trace_ids.append(trace["trace_id"])
            if isinstance(trace.get("span_id"), str):
                self.span_ids.append(trace["span_id"])
        if isinstance(extensions, dict):
            openai_agents = extensions.get("openai_agents")
            if isinstance(openai_agents, dict):
                if isinstance(openai_agents.get("caller_session_id"), str):
                    self.caller_session_ids.append(openai_agents["caller_session_id"])
                if isinstance(openai_agents.get("correlation_id"), str):
                    self.correlation_ids.append(openai_agents["correlation_id"])
        if context.get("interception_point") == InterceptionPoint.AGENT_STARTUP.value:
            agent_init = context.get("agent_init")
            if isinstance(agent_init, dict):
                tools = agent_init.get("tools_registered")
                if isinstance(tools, list) and all(isinstance(tool, str) for tool in tools):
                    self.startup_tools.append(tools)

        async def child_has_session() -> bool:
            return get_current_agent_hooks_session() is not None

        child = asyncio.create_task(child_has_session())
        self.child_session_bound.append(await child)
        return ALLOW


class CancellingInputInterceptor:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.drained = asyncio.Event()
        self.shutdown_reasons: list[str] = []

    async def intercept(self, context: AgentContext, /) -> Verdict:
        point = InterceptionPoint(context["interception_point"])
        if point is InterceptionPoint.INPUT:
            self.started.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.drained.set()
        if point is InterceptionPoint.AGENT_SHUTDOWN:
            summary = context.get("summary")
            if isinstance(summary, dict) and isinstance(summary.get("reason"), str):
                self.shutdown_reasons.append(summary["reason"])
        return ALLOW


class InvalidTransformInterceptor:
    def __init__(self, point: InterceptionPoint, value: object) -> None:
        self.point = point
        self.value = value
        self.shutdown_reasons: list[str] = []

    async def intercept(self, context: AgentContext, /) -> Verdict:
        point = InterceptionPoint(context["interception_point"])
        if point is InterceptionPoint.AGENT_SHUTDOWN:
            summary = context.get("summary")
            if isinstance(summary, dict) and isinstance(summary.get("reason"), str):
                self.shutdown_reasons.append(summary["reason"])
        if point is self.point:
            return Verdict(
                decision=Decision.TRANSFORM,
                transform=Transform(path="$target.content", value=self.value),
            )
        return ALLOW


class TimingOutInputInterceptor:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.drained = asyncio.Event()
        self.shutdown_reasons: list[str] = []

    async def intercept(self, context: AgentContext, /) -> Verdict:
        point = InterceptionPoint(context["interception_point"])
        if point is InterceptionPoint.INPUT:
            self.started.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.drained.set()
        if point is InterceptionPoint.AGENT_SHUTDOWN:
            summary = context.get("summary")
            if isinstance(summary, dict) and isinstance(summary.get("reason"), str):
                self.shutdown_reasons.append(summary["reason"])
        return ALLOW


class ExcessiveLabelsInterceptor:
    async def intercept(self, context: AgentContext, /) -> Verdict:
        if context.get("interception_point") == InterceptionPoint.INPUT.value:
            return Verdict(
                decision=Decision.ALLOW,
                result_labels=tuple("label" for _ in range(257)),
            )
        return ALLOW


def _text_message(text: str) -> ResponseOutputMessage:
    return ResponseOutputMessage(
        id="message-1",
        content=[ResponseOutputText(annotations=[], text=text, type="output_text")],
        role="assistant",
        status="completed",
        type="message",
    )


def _governed_run_config(
    *,
    agent_hooks: AgentHooksConfig,
    with_tools: bool = False,
) -> RunConfig:
    return RunConfig(
        agent_hooks=agent_hooks,
        trace_include_sensitive_data=False,
        model_settings=ModelSettings(
            parallel_tool_calls=False if with_tools else None,
            retry=ModelRetrySettings(max_retries=0),
        ),
        tool_execution=(
            ToolExecutionConfig(max_function_tool_concurrency=1) if with_tools else None
        ),
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


@pytest.mark.parametrize(
    "case",
    [
        "run_hooks",
        "agent_hooks",
        "agent_input_guardrail",
        "agent_output_guardrail",
        "run_input_guardrail",
        "run_output_guardrail",
        "tool_input_guardrail",
        "tool_output_guardrail",
        "sensitive_tracing",
    ],
)
async def test_native_callbacks_and_sensitive_tracing_fail_admission(case: str) -> None:
    records = RecordSink(max_records=1000)
    model = FakeModel(initial_output=[_text_message("unused")])

    @function_tool(name_override="lookup")
    async def lookup(value: str) -> str:
        return value

    tool = lookup
    agent: Agent[None] = Agent(name="governed-agent", model=model, tools=[tool])
    run_config = _governed_run_config(
        agent_hooks=AgentHooksConfig(
            agent_id="governed-agent-v1",
            interceptors=(AllowInterceptor(),),
            record_sink=records,
        ),
        with_tools=True,
    )
    hooks: RunHooks[None] | None = None

    if case == "run_hooks":
        hooks = RecordingRunHooks()
    elif case == "agent_hooks":
        agent.hooks = cast(Any, object())
    elif case == "agent_input_guardrail":
        agent.input_guardrails = [cast(Any, object())]
    elif case == "agent_output_guardrail":
        agent.output_guardrails = [cast(Any, object())]
    elif case == "run_input_guardrail":
        run_config.input_guardrails = [cast(Any, object())]
    elif case == "run_output_guardrail":
        run_config.output_guardrails = [cast(Any, object())]
    elif case == "tool_input_guardrail":
        tool.tool_input_guardrails = [cast(Any, object())]
    elif case == "tool_output_guardrail":
        tool.tool_output_guardrails = [cast(Any, object())]
    else:
        run_config.trace_include_sensitive_data = True

    with pytest.raises(UserError, match="does not support native callbacks|redacted tracing"):
        await Runner.run(
            agent,
            "hello",
            max_turns=1,
            hooks=hooks,
            run_config=run_config,
        )

    assert len(records) == 0
    assert model.first_turn_args is None


async def test_insufficient_record_capacity_fails_before_startup() -> None:
    records = RecordSink(max_records=5)
    model = FakeModel(initial_output=[_text_message("unused")])
    agent = Agent(name="governed-agent", model=model)
    run_config = _governed_run_config(
        agent_hooks=AgentHooksConfig(
            agent_id="governed-agent-v1",
            interceptors=(AllowInterceptor(),),
            record_sink=records,
        )
    )

    with pytest.raises(UserError, match="RecordSink capacity"):
        await Runner.run(agent, "hello", max_turns=1, run_config=run_config)

    assert len(records) == 0
    assert model.first_turn_args is None


async def test_failed_run_releases_unused_record_reservation() -> None:
    records = RecordSink(max_records=8)
    limits = AgentHooksLimits(max_tool_calls_per_turn=1)
    denying_interceptor = RecordingInterceptor(
        {
            InterceptionPoint.INPUT: Verdict(
                decision=Decision.DENY,
                reason="policy_denied",
            )
        }
    )
    denied_model = FakeModel(initial_output=[_text_message("unused")])
    denied_agent = Agent(name="governed-agent", model=denied_model)
    denied_run_config = _governed_run_config(
        agent_hooks=AgentHooksConfig(
            agent_id="governed-agent-v1",
            interceptors=(denying_interceptor,),
            record_sink=records,
            limits=limits,
        )
    )

    with pytest.raises(AgentHooksBlockedError) as error_info:
        await Runner.run(
            denied_agent,
            "hello",
            max_turns=1,
            run_config=denied_run_config,
        )

    assert error_info.value.point == InterceptionPoint.INPUT.value
    assert error_info.value.reason == "policy_denied"
    assert denied_model.first_turn_args is None
    assert [record.interception_point for record in records.drain()] == [
        InterceptionPoint.AGENT_STARTUP,
        InterceptionPoint.INPUT,
        InterceptionPoint.AGENT_SHUTDOWN,
    ]

    succeeding_model = FakeModel(initial_output=[_text_message("done")])
    succeeding_agent = Agent(name="governed-agent", model=succeeding_model)
    succeeding_run_config = _governed_run_config(
        agent_hooks=AgentHooksConfig(
            agent_id="governed-agent-v1",
            interceptors=(AllowInterceptor(),),
            record_sink=records,
            limits=limits,
        )
    )

    result = await Runner.run(
        succeeding_agent,
        "hello",
        max_turns=1,
        run_config=succeeding_run_config,
    )

    assert result.final_output == "done"
    assert succeeding_model.first_turn_args is not None
    assert [record.interception_point for record in records] == [
        InterceptionPoint.AGENT_STARTUP,
        InterceptionPoint.INPUT,
        InterceptionPoint.PRE_MODEL_CALL,
        InterceptionPoint.POST_MODEL_CALL,
        InterceptionPoint.OUTPUT,
        InterceptionPoint.AGENT_SHUTDOWN,
    ]


async def test_success_emits_model_lifecycle_in_order() -> None:
    records = RecordSink(max_records=1000)
    model = FakeModel(initial_output=[_text_message("done")])
    agent = Agent(name="governed-agent", model=model)
    run_config = _governed_run_config(
        agent_hooks=AgentHooksConfig(
            agent_id="governed-agent-v1",
            interceptors=(AllowInterceptor(),),
            record_sink=records,
        )
    )

    result = await Runner.run(agent, "hello", max_turns=1, run_config=run_config)

    assert result.final_output == "done"
    assert [record.interception_point for record in records] == [
        InterceptionPoint.AGENT_STARTUP,
        InterceptionPoint.INPUT,
        InterceptionPoint.PRE_MODEL_CALL,
        InterceptionPoint.POST_MODEL_CALL,
        InterceptionPoint.OUTPUT,
        InterceptionPoint.AGENT_SHUTDOWN,
    ]
    assert [record.sequence for record in records] == list(range(6))
    assert len({record.session_id for record in records}) == 1
    UUID(records[0].session_id)


async def test_record_sink_snapshots_do_not_mutate_retained_trace() -> None:
    records = RecordSink(max_records=1000)
    model = FakeModel(initial_output=[_text_message("done")])
    agent = Agent(name="governed-agent", model=model)
    run_config = _governed_run_config(
        agent_hooks=AgentHooksConfig(
            agent_id="governed-agent-v1",
            interceptors=(AllowInterceptor(),),
            record_sink=records,
        )
    )

    await Runner.run(agent, "hello", max_turns=1, run_config=run_config)

    exposed_record = records.snapshot()[0]
    exposed_trace = exposed_record.trace
    assert exposed_trace is not None
    original_trace = dict(exposed_trace)
    exposed_trace["trace_id"] = "mutated"

    assert records.snapshot()[0].trace == original_trace
    assert records[0].trace == original_trace


async def test_transformed_input_and_output_reach_native_boundaries() -> None:
    transformed_input = "governed input"
    transformed_output = "governed output"
    interceptor = RecordingInterceptor(
        {
            InterceptionPoint.INPUT: Verdict(
                decision=Decision.TRANSFORM,
                transform=Transform(path="$target.content", value=transformed_input),
            ),
            InterceptionPoint.OUTPUT: Verdict(
                decision=Decision.TRANSFORM,
                transform=Transform(path="$target.content", value=transformed_output),
            ),
        }
    )
    records = RecordSink(max_records=1000)
    model = FakeModel(initial_output=[_text_message("raw output")])
    agent: Agent[None] = Agent(name="governed-agent", model=model)
    run_config = _governed_run_config(
        agent_hooks=AgentHooksConfig(
            agent_id="governed-agent-v1",
            interceptors=(interceptor,),
            record_sink=records,
        )
    )

    result = await Runner.run(
        agent,
        "raw input",
        max_turns=1,
        run_config=run_config,
    )

    assert model.first_turn_args is not None
    assert model.first_turn_args["input"] == ItemHelpers.input_to_new_input_list(transformed_input)
    assert result.final_output == transformed_output
    expected_points = [
        InterceptionPoint.AGENT_STARTUP,
        InterceptionPoint.INPUT,
        InterceptionPoint.PRE_MODEL_CALL,
        InterceptionPoint.POST_MODEL_CALL,
        InterceptionPoint.OUTPUT,
        InterceptionPoint.AGENT_SHUTDOWN,
    ]
    assert interceptor.points == expected_points
    assert [record.interception_point for record in records] == expected_points


@pytest.mark.parametrize(
    "denied_point",
    [
        InterceptionPoint.AGENT_STARTUP,
        InterceptionPoint.INPUT,
        InterceptionPoint.OUTPUT,
    ],
)
async def test_deny_blocks_downstream_side_effect_and_closes_once(
    denied_point: InterceptionPoint,
) -> None:
    interceptor = RecordingInterceptor(
        {denied_point: Verdict(decision=Decision.DENY, reason="policy_denied")}
    )
    records = RecordSink(max_records=1000)
    model = FakeModel(initial_output=[_text_message("raw output")])
    agent: Agent[None] = Agent(name="governed-agent", model=model)
    run_config = _governed_run_config(
        agent_hooks=AgentHooksConfig(
            agent_id="governed-agent-v1",
            interceptors=(interceptor,),
            record_sink=records,
        )
    )

    with pytest.raises(AgentHooksBlockedError) as error_info:
        await Runner.run(
            agent,
            "raw input",
            max_turns=1,
            run_config=run_config,
        )

    assert error_info.value.point == denied_point.value
    assert error_info.value.reason == "policy_denied"
    assert interceptor.shutdown_reasons == ["error"]
    assert interceptor.points.count(InterceptionPoint.AGENT_SHUTDOWN) == 1
    assert (
        sum(record.interception_point is InterceptionPoint.AGENT_SHUTDOWN for record in records)
        == 1
    )
    if denied_point in {InterceptionPoint.AGENT_STARTUP, InterceptionPoint.INPUT}:
        assert model.first_turn_args is None
    else:
        assert model.first_turn_args is not None


async def test_shutdown_deny_is_audit_only_and_does_not_repeat_work() -> None:
    interceptor = RecordingInterceptor(
        {
            InterceptionPoint.AGENT_SHUTDOWN: Verdict(
                decision=Decision.DENY,
                reason="shutdown_observation",
            )
        }
    )
    records = RecordSink(max_records=1000)
    model = FakeModel(initial_output=[_text_message("done")])
    agent = Agent(name="governed-agent", model=model)
    run_config = _governed_run_config(
        agent_hooks=AgentHooksConfig(
            agent_id="governed-agent-v1",
            interceptors=(interceptor,),
            record_sink=records,
        )
    )

    result = await Runner.run(agent, "hello", max_turns=1, run_config=run_config)

    assert result.final_output == "done"
    assert model.first_turn_args is not None
    assert interceptor.shutdown_reasons == ["completed"]
    assert interceptor.points.count(InterceptionPoint.AGENT_SHUTDOWN) == 1
    assert records[-1].verdict.decision is Decision.DENY


async def test_trusted_envelope_static_tools_and_child_tasks_share_one_session() -> None:
    interceptor = MetadataInterceptor()
    records = RecordSink(max_records=1000)
    model = FakeModel(initial_output=[_text_message("done")])

    @function_tool(name_override="lookup")
    async def lookup() -> str:
        return "unused"

    agent = Agent(
        name="governed-agent",
        model=model,
        tools=[lookup],
    )
    run_config = _governed_run_config(
        agent_hooks=AgentHooksConfig(
            agent_id="trusted-agent-id",
            session_id="trusted-session-id",
            correlation_id="trusted-correlation-id",
            interceptors=(interceptor,),
            record_sink=records,
        ),
        with_tools=True,
    )

    result = await Runner.run(agent, "hello", max_turns=1, run_config=run_config)

    assert result.final_output == "done"
    assert interceptor.agent_ids == ["trusted-agent-id"] * 6
    assert interceptor.frameworks == ["openai-agents"] * 6
    assert len(set(interceptor.session_ids)) == 1
    UUID(interceptor.session_ids[0])
    assert interceptor.caller_session_ids == ["trusted-session-id"] * 6
    assert interceptor.correlation_ids == ["trusted-correlation-id"] * 6
    assert len(set(interceptor.trace_ids)) == 1
    assert len(interceptor.trace_ids[0]) == 32
    int(interceptor.trace_ids[0], 16)
    assert len(set(interceptor.span_ids)) == 1
    assert len(interceptor.span_ids[0]) == 16
    int(interceptor.span_ids[0], 16)
    assert interceptor.startup_tools == [["lookup"]]
    assert interceptor.child_session_bound == [True] * 6
    assert [record.sequence for record in records] == list(range(6))
    assert all(record.identity_provider == "jcs-sha256" for record in records)
    assert all(
        record.trace
        == {
            "trace_id": interceptor.trace_ids[0],
            "span_id": interceptor.span_ids[0],
        }
        for record in records
    )
    assert all(record.composition.profile.value == "sequential/run_all" for record in records)


async def test_cancellation_during_emission_drains_callback_and_closes_once() -> None:
    interceptor = CancellingInputInterceptor()
    records = RecordSink(max_records=1000)
    model = FakeModel(initial_output=[_text_message("unused")])
    agent = Agent(name="governed-agent", model=model)
    run_config = _governed_run_config(
        agent_hooks=AgentHooksConfig(
            agent_id="governed-agent-v1",
            interceptors=(interceptor,),
            record_sink=records,
        )
    )

    task = asyncio.create_task(Runner.run(agent, "hello", max_turns=1, run_config=run_config))
    await interceptor.started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert interceptor.drained.is_set()
    assert interceptor.shutdown_reasons == ["cancelled"]
    assert model.first_turn_args is None
    assert [record.interception_point for record in records].count(
        InterceptionPoint.AGENT_SHUTDOWN
    ) == 1
    assert all(record.verdict.message is None for record in records)
    assert all(record.verdict.evidence is None for record in records)
    assert all(record.verdict.approval is None for record in records)
    assert get_current_agent_hooks_session() is None


async def test_disabled_nested_run_masks_and_restores_outer_session() -> None:
    outer_session = cast(AgentHooksRunSession, object())
    outer_token = bind_agent_hooks_session(outer_session)
    try:
        model = FakeModel(initial_output=[_text_message("done")])
        hooks = ContextRecordingRunHooks()
        agent: Agent[None] = Agent(name="disabled-agent", model=model)

        result = await Runner.run(agent, "hello", max_turns=1, hooks=hooks)

        assert result.final_output == "done"
        assert hooks.sessions == [None]
        assert get_current_agent_hooks_session() is outer_session
    finally:
        reset_agent_hooks_session(outer_token)

    assert get_current_agent_hooks_session() is None


async def test_disabled_nested_stream_masks_and_restores_outer_session() -> None:
    outer_session = cast(AgentHooksRunSession, object())
    outer_token = bind_agent_hooks_session(outer_session)
    try:
        model = FakeModel(initial_output=[_text_message("done")])
        agent: Agent[None] = Agent(name="disabled-agent", model=model)
        result = Runner.run_streamed(agent, "hello", max_turns=1)

        async for _event in result.stream_events():
            pass

        assert result.final_output == "done"
        assert get_current_agent_hooks_session() is outer_session
    finally:
        reset_agent_hooks_session(outer_token)

    assert get_current_agent_hooks_session() is None


async def test_nested_stream_sandbox_cleanup_registration_uses_empty_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_sessions: list[AgentHooksRunSession | None] = []
    original_ensure = RunResultStreaming.ensure_sandbox_cleanup_on_completion

    def record_cleanup_registration(result: RunResultStreaming) -> None:
        observed_sessions.append(get_current_agent_hooks_session())
        original_ensure(result)

    monkeypatch.setattr(SandboxRuntime, "enabled", property(lambda _runtime: True))
    monkeypatch.setattr(
        RunResultStreaming,
        "ensure_sandbox_cleanup_on_completion",
        record_cleanup_registration,
    )
    outer_session = cast(AgentHooksRunSession, object())
    outer_token = bind_agent_hooks_session(outer_session)
    try:
        model = FakeModel(initial_output=[_text_message("done")])
        result = Runner.run_streamed(Agent(name="disabled-agent", model=model), "hello")
        async for _event in result.stream_events():
            pass
        assert get_current_agent_hooks_session() is outer_session
    finally:
        reset_agent_hooks_session(outer_token)

    assert observed_sessions == [None]


async def test_disabled_model_public_agent_hooks_error_remains_ordinary_error() -> None:
    forged_error = AgentHooksBlockedError(
        point=InterceptionPoint.PRE_MODEL_CALL.value,
        reason="host_error:forged",
        sequence=99,
    )
    model = FakeModel(initial_output=forged_error)
    agent = Agent(name="disabled-agent", model=model)

    with pytest.raises(AgentHooksBlockedError) as error_info:
        await Runner.run(agent, "hello", max_turns=1)

    assert error_info.value is forged_error
    assert error_info.value.run_data is not None


async def test_disabled_model_replayed_host_error_remains_ordinary_error() -> None:
    records = RecordSink(max_records=1000)
    denied_config = _governed_run_config(
        agent_hooks=AgentHooksConfig(
            agent_id="governed-agent-v1",
            interceptors=(
                RecordingInterceptor(
                    {
                        InterceptionPoint.INPUT: Verdict(
                            decision=Decision.DENY,
                            reason="policy_denied",
                        )
                    }
                ),
            ),
            record_sink=records,
        )
    )
    with pytest.raises(AgentHooksBlockedError) as denied_info:
        await Runner.run(
            Agent(name="denied-agent", model=FakeModel()),
            "hello",
            max_turns=1,
            run_config=denied_config,
        )

    replayed_error = denied_info.value
    model = FakeModel(initial_output=replayed_error)
    agent = Agent(name="disabled-agent", model=model)
    with pytest.raises(AgentHooksBlockedError) as replay_info:
        await Runner.run(agent, "hello", max_turns=1)

    assert replay_info.value is replayed_error
    assert replay_info.value.run_data is not None


async def test_governed_model_cannot_replay_prior_host_error() -> None:
    records = RecordSink(max_records=1000)
    denied_config = _governed_run_config(
        agent_hooks=AgentHooksConfig(
            agent_id="governed-agent-v1",
            interceptors=(
                RecordingInterceptor(
                    {
                        InterceptionPoint.INPUT: Verdict(
                            decision=Decision.DENY,
                            reason="policy_denied",
                        )
                    }
                ),
            ),
            record_sink=records,
        )
    )
    with pytest.raises(AgentHooksBlockedError) as denied_info:
        await Runner.run(
            Agent(name="denied-agent", model=FakeModel()),
            "hello",
            max_turns=1,
            run_config=denied_config,
        )

    replayed_error = denied_info.value
    replay_records = RecordSink(max_records=1000)
    model = FakeModel(initial_output=replayed_error)
    agent = Agent(name="governed-agent", model=model)
    run_config = _governed_run_config(
        agent_hooks=AgentHooksConfig(
            agent_id="governed-agent-v1",
            interceptors=(AllowInterceptor(),),
            record_sink=replay_records,
        )
    )

    with pytest.raises(AgentHooksExecutionError):
        await Runner.run(agent, "hello", max_turns=1, run_config=run_config)


async def test_explicit_task_local_none_restores_outer_binding() -> None:
    outer_session = cast(AgentHooksRunSession, object())
    outer_token = bind_agent_hooks_session(outer_session)
    try:
        inner_token = bind_agent_hooks_session(None)
        try:
            assert get_current_agent_hooks_session() is None
        finally:
            reset_agent_hooks_session(inner_token)
        assert get_current_agent_hooks_session() is outer_session
    finally:
        reset_agent_hooks_session(outer_token)


@pytest.mark.parametrize(
    ("input_value", "max_turns"),
    [
        ([], 1),
        ("hello", 0),
        ("hello", -1),
        ("hello", True),
        ("hello", 1.5),
    ],
)
async def test_input_and_turn_admission_fail_before_side_effects(
    input_value: object,
    max_turns: object,
) -> None:
    records = RecordSink(max_records=1000)
    model = FakeModel(initial_output=[_text_message("unused")])
    hooks = RecordingRunHooks()
    agent: Agent[None] = Agent(name="governed-agent", model=model)
    run_config = _governed_run_config(
        agent_hooks=AgentHooksConfig(
            agent_id="governed-agent-v1",
            interceptors=(AllowInterceptor(),),
            record_sink=records,
        )
    )

    with pytest.raises(UserError):
        await Runner.run(
            agent,
            cast(Any, input_value),
            max_turns=cast(Any, max_turns),
            hooks=hooks,
            run_config=run_config,
        )

    assert len(records) == 0
    assert model.first_turn_args is None
    assert hooks.started == 0


@pytest.mark.parametrize(
    "case",
    [
        "agent_subclass",
        "handoff",
        "mcp_server",
        "prompt",
        "dynamic_instructions",
        "structured_output",
        "tool_as_final",
        "unsupported_tool",
        "dynamic_tool",
        "approval_tool",
        "deferred_tool",
        "agent_as_tool",
    ],
)
async def test_agent_shape_admission_fails_before_side_effects(case: str) -> None:
    records = RecordSink(max_records=1000)
    model = FakeModel(initial_output=[_text_message("unused")])
    hooks = RecordingRunHooks()
    agent: Agent[None] = Agent(name="governed-agent", model=model)

    if case == "agent_subclass":

        class AgentSubclass(Agent[None]):
            pass

        agent = AgentSubclass(name="governed-agent", model=model)
    elif case == "handoff":
        agent.handoffs.append(Agent(name="other", model=FakeModel()))
    elif case == "mcp_server":
        agent.mcp_servers.append(cast(Any, object()))
    elif case == "prompt":
        agent.prompt = cast(Any, lambda: None)
    elif case == "dynamic_instructions":
        agent.instructions = lambda context, current_agent: "dynamic"
    elif case == "structured_output":
        agent.output_type = dict[str, str]
    elif case == "tool_as_final":
        agent.tool_use_behavior = "stop_on_first_tool"
    elif case == "unsupported_tool":
        agent.tools.append(cast(Any, object()))
    elif case in {"dynamic_tool", "approval_tool", "deferred_tool"}:
        tool = get_function_tool("lookup", "unused")
        if case == "dynamic_tool":
            tool.is_enabled = lambda context, current_agent: True
        elif case == "approval_tool":
            tool.needs_approval = True
        else:
            tool.defer_loading = True
        agent.tools.append(tool)
    else:
        nested_agent = Agent(name="nested", model=FakeModel())
        agent.tools.append(nested_agent.as_tool("nested", "nested agent"))

    run_config = _governed_run_config(
        agent_hooks=AgentHooksConfig(
            agent_id="governed-agent-v1",
            interceptors=(AllowInterceptor(),),
            record_sink=records,
        )
    )

    with pytest.raises(UserError):
        await Runner.run(
            agent,
            "hello",
            max_turns=1,
            hooks=hooks,
            run_config=run_config,
        )

    assert len(records) == 0
    assert model.first_turn_args is None
    assert hooks.started == 0


@pytest.mark.parametrize(
    "case",
    [
        "handoff_filter",
        "handoff_history",
        "session_callback",
        "session_settings",
        "model_filter",
        "sandbox",
        "tool_error_formatter",
        "tool_not_found_behavior",
    ],
)
async def test_run_config_admission_fails_before_side_effects(case: str) -> None:
    records = RecordSink(max_records=1000)
    model = FakeModel(initial_output=[_text_message("unused")])
    hooks = RecordingRunHooks()
    agent: Agent[None] = Agent(name="governed-agent", model=model)
    run_config = _governed_run_config(
        agent_hooks=AgentHooksConfig(
            agent_id="governed-agent-v1",
            interceptors=(AllowInterceptor(),),
            record_sink=records,
        )
    )

    if case == "handoff_filter":
        run_config.handoff_input_filter = cast(Any, object())
    elif case == "handoff_history":
        run_config.nest_handoff_history = True
    elif case == "session_callback":
        run_config.session_input_callback = cast(Any, object())
    elif case == "session_settings":
        run_config.session_settings = cast(Any, object())
    elif case == "model_filter":
        run_config.call_model_input_filter = cast(Any, object())
    elif case == "sandbox":
        run_config.sandbox = SandboxRunConfig()
    elif case == "tool_error_formatter":
        run_config.tool_error_formatter = cast(Any, object())
    else:
        run_config.tool_not_found_behavior = "return_error_to_model"

    with pytest.raises(UserError):
        await Runner.run(
            agent,
            "hello",
            max_turns=1,
            hooks=hooks,
            run_config=run_config,
        )

    assert len(records) == 0
    assert model.first_turn_args is None
    assert hooks.started == 0


async def test_original_model_provider_is_not_read_after_snapshot() -> None:
    class Registration:
        harness_id = "captured-harness"

    class SingleReadProvider(ModelProvider):
        def __init__(self, model: FakeModel) -> None:
            self.model = model
            self.registration_reads = 0

        @property
        def agent_registration(self) -> Registration:
            self.registration_reads += 1
            if len(records) > 0:
                raise AssertionError("original provider was read after Agent Hooks startup")
            if self.registration_reads > 1:
                raise AssertionError("original provider was read after snapshot")
            return Registration()

        def get_model(self, model_name: str | None) -> FakeModel:
            assert model_name == "captured-model"
            return self.model

    records = RecordSink(max_records=1000)
    model = FakeModel(initial_output=[_text_message("done")])
    provider = SingleReadProvider(model)
    agent = Agent(name="governed-agent", model="captured-model")
    run_config = _governed_run_config(
        agent_hooks=AgentHooksConfig(
            agent_id="governed-agent-v1",
            interceptors=(AllowInterceptor(),),
            record_sink=records,
        )
    )
    run_config.model_provider = provider

    result = await Runner.run(agent, "hello", max_turns=1, run_config=run_config)

    assert result.final_output == "done"
    assert provider.registration_reads == 1


async def test_callback_capable_tracing_disabled_fails_before_records() -> None:
    class HostileBool:
        def __init__(self) -> None:
            self.bool_calls = 0

        def __bool__(self) -> bool:
            self.bool_calls += 1
            raise AssertionError("hostile tracing_disabled was evaluated")

    hostile_value = HostileBool()
    records = RecordSink(max_records=1000)
    model = FakeModel(initial_output=[_text_message("unused")])
    agent = Agent(name="governed-agent", model=model)
    run_config = _governed_run_config(
        agent_hooks=AgentHooksConfig(
            agent_id="governed-agent-v1",
            interceptors=(AllowInterceptor(),),
            record_sink=records,
        )
    )
    run_config.tracing_disabled = cast(Any, hostile_value)

    with pytest.raises(UserError, match="tracing_disabled"):
        await Runner.run(agent, "hello", max_turns=1, run_config=run_config)

    assert hostile_value.bool_calls == 0
    assert len(records) == 0
    assert model.first_turn_args is None


@pytest.mark.parametrize(
    "case",
    ["session", "previous_response", "auto_previous_response", "conversation", "errors"],
)
async def test_run_option_admission_fails_before_side_effects(case: str) -> None:
    records = RecordSink(max_records=1000)
    model = FakeModel(initial_output=[_text_message("unused")])
    hooks = RecordingRunHooks()
    agent: Agent[None] = Agent(name="governed-agent", model=model)
    run_config = _governed_run_config(
        agent_hooks=AgentHooksConfig(
            agent_id="governed-agent-v1",
            interceptors=(AllowInterceptor(),),
            record_sink=records,
        )
    )

    with pytest.raises(UserError):
        if case == "session":
            await Runner.run(
                agent,
                "hello",
                max_turns=1,
                hooks=hooks,
                run_config=run_config,
                session=cast(Any, object()),
            )
        elif case == "previous_response":
            await Runner.run(
                agent,
                "hello",
                max_turns=1,
                hooks=hooks,
                run_config=run_config,
                previous_response_id="response-1",
            )
        elif case == "auto_previous_response":
            await Runner.run(
                agent,
                "hello",
                max_turns=1,
                hooks=hooks,
                run_config=run_config,
                auto_previous_response_id=True,
            )
        elif case == "conversation":
            await Runner.run(
                agent,
                "hello",
                max_turns=1,
                hooks=hooks,
                run_config=run_config,
                conversation_id="conversation-1",
            )
        else:
            await Runner.run(
                agent,
                "hello",
                max_turns=1,
                hooks=hooks,
                run_config=run_config,
                error_handlers=cast(Any, object()),
            )

    assert len(records) == 0
    assert model.first_turn_args is None
    assert hooks.started == 0


def test_run_streamed_rejects_before_records_or_side_effects() -> None:
    records = RecordSink(max_records=1000)
    model = FakeModel(initial_output=[_text_message("unused")])
    hooks = RecordingRunHooks()
    agent: Agent[None] = Agent(name="governed-agent", model=model)
    run_config = _governed_run_config(
        agent_hooks=AgentHooksConfig(
            agent_id="governed-agent-v1",
            interceptors=(AllowInterceptor(),),
            record_sink=records,
        )
    )

    with pytest.raises(UserError, match="does not support streaming"):
        Runner.run_streamed(agent, "hello", hooks=hooks, run_config=run_config)

    assert len(records) == 0
    assert model.first_turn_args is None
    assert hooks.started == 0


@pytest.mark.parametrize("point", [InterceptionPoint.INPUT, InterceptionPoint.OUTPUT])
async def test_non_string_transform_fails_closed(point: InterceptionPoint) -> None:
    interceptor = InvalidTransformInterceptor(point, 42)
    records = RecordSink(max_records=1000)
    model = FakeModel(initial_output=[_text_message("raw output")])
    agent: Agent[None] = Agent(name="governed-agent", model=model)
    run_config = _governed_run_config(
        agent_hooks=AgentHooksConfig(
            agent_id="governed-agent-v1",
            interceptors=(interceptor,),
            record_sink=records,
        )
    )

    with pytest.raises(AgentHooksBlockedError) as error_info:
        await Runner.run(
            agent,
            "hello",
            max_turns=1,
            run_config=run_config,
        )

    assert error_info.value.point == point.value
    assert interceptor.shutdown_reasons == ["error"]
    assert [record.interception_point for record in records].count(
        InterceptionPoint.AGENT_SHUTDOWN
    ) == 1
    point_record = next(record for record in records if record.interception_point is point)
    assert not point_record.proceeds
    assert point_record.verdict.reason == "host_error:transform_invalid"
    if point is InterceptionPoint.INPUT:
        assert model.first_turn_args is None
    else:
        assert model.first_turn_args is not None


async def test_context_byte_limit_is_enforced_before_interceptor_dispatch() -> None:
    interceptor = RecordingInterceptor()
    records = RecordSink(max_records=1000)
    model = FakeModel(initial_output=[_text_message("unused")])
    agent = Agent(name="governed-agent", model=model)
    run_config = _governed_run_config(
        agent_hooks=AgentHooksConfig(
            agent_id="governed-agent-v1",
            interceptors=(interceptor,),
            record_sink=records,
            limits=AgentHooksLimits(max_context_bytes=_BOUNDED_CONTEXT_BYTES),
        )
    )

    with pytest.raises(AgentHooksBlockedError) as error_info:
        await Runner.run(agent, _OVERSIZED_CONTEXT_TEXT, max_turns=1, run_config=run_config)

    assert error_info.value.point == InterceptionPoint.INPUT.value
    assert error_info.value.reason == "host_error:context_invalid"
    assert interceptor.points == [
        InterceptionPoint.AGENT_STARTUP,
        InterceptionPoint.AGENT_SHUTDOWN,
    ]
    assert [record.sequence for record in records] == [0, 1]
    assert model.first_turn_args is None


async def test_input_deny_clears_payload_from_sdk_traceback_frames() -> None:
    secret = "sensitive-input-4a0f65"
    interceptor = RecordingInterceptor(
        {
            InterceptionPoint.INPUT: Verdict(
                decision=Decision.DENY,
                reason="policy_denied",
            )
        }
    )
    records = RecordSink(max_records=1000)
    model = FakeModel(initial_output=[_text_message("unused")])
    agent = Agent(name="governed-agent", model=model, instructions=secret)
    run_config = _governed_run_config(
        agent_hooks=AgentHooksConfig(
            agent_id="governed-agent-v1",
            correlation_id=secret,
            interceptors=(interceptor,),
            record_sink=records,
        )
    )

    with pytest.raises(AgentHooksBlockedError) as error_info:
        await Runner.run(
            agent,
            secret,
            max_turns=1,
            run_config=run_config,
        )

    _assert_secret_absent_from_sdk_traceback(error_info.value, secret)
    assert model.first_turn_args is None


async def test_direct_agent_runner_clears_request_graph_from_traceback() -> None:
    secret = "sensitive-direct-runner-1c7334"
    interceptor = RecordingInterceptor(
        {
            InterceptionPoint.INPUT: Verdict(
                decision=Decision.DENY,
                reason="policy_denied",
            )
        }
    )
    agent = Agent(name="governed-agent", model=FakeModel(), instructions=secret)
    run_config = _governed_run_config(
        agent_hooks=AgentHooksConfig(
            agent_id="governed-agent-v1",
            correlation_id=secret,
            interceptors=(interceptor,),
            record_sink=RecordSink(max_records=1000),
        )
    )

    with pytest.raises(AgentHooksBlockedError) as error_info:
        await run_module.AgentRunner().run(
            agent,
            secret,
            max_turns=1,
            run_config=run_config,
        )

    _assert_secret_absent_from_sdk_traceback(error_info.value, secret)


def test_run_sync_clears_request_graph_from_traceback() -> None:
    secret = "sensitive-sync-runner-e6fe18"
    interceptor = RecordingInterceptor(
        {
            InterceptionPoint.INPUT: Verdict(
                decision=Decision.DENY,
                reason="policy_denied",
            )
        }
    )
    agent = Agent(name="governed-agent", model=FakeModel(), instructions=secret)
    run_config = _governed_run_config(
        agent_hooks=AgentHooksConfig(
            agent_id="governed-agent-v1",
            correlation_id=secret,
            interceptors=(interceptor,),
            record_sink=RecordSink(max_records=1000),
        )
    )

    with pytest.raises(AgentHooksBlockedError) as error_info:
        Runner.run_sync(
            agent,
            secret,
            max_turns=1,
            run_config=run_config,
        )

    _assert_secret_absent_from_sdk_traceback(error_info.value, secret)


@pytest.mark.parametrize(
    "limits",
    [
        AgentHooksLimits(max_context_bytes=64),
        AgentHooksLimits(max_context_depth=2),
    ],
)
async def test_impossible_trusted_context_limits_fail_during_admission(
    limits: AgentHooksLimits,
) -> None:
    interceptor = RecordingInterceptor()
    records = RecordSink(max_records=1000)
    model = FakeModel(initial_output=[_text_message("unused")])
    agent = Agent(name="governed-agent", model=model)
    run_config = _governed_run_config(
        agent_hooks=AgentHooksConfig(
            agent_id="governed-agent-v1",
            interceptors=(interceptor,),
            record_sink=records,
            limits=limits,
        )
    )

    with pytest.raises(UserError, match="trusted lifecycle envelopes"):
        await Runner.run(agent, "hello", max_turns=1, run_config=run_config)

    assert interceptor.points == []
    assert len(records) == 0
    assert model.first_turn_args is None


async def test_cyclic_transform_is_rejected_before_sdk_write_back() -> None:
    cyclic_value: list[object] = []
    cyclic_value.append(cyclic_value)
    interceptor = InvalidTransformInterceptor(InterceptionPoint.INPUT, cyclic_value)
    records = RecordSink(max_records=1000)
    model = FakeModel(initial_output=[_text_message("unused")])
    agent = Agent(name="governed-agent", model=model)
    run_config = _governed_run_config(
        agent_hooks=AgentHooksConfig(
            agent_id="governed-agent-v1",
            interceptors=(interceptor,),
            record_sink=records,
        )
    )

    with pytest.raises(AgentHooksBlockedError) as error_info:
        await Runner.run(agent, "hello", max_turns=1, run_config=run_config)

    assert error_info.value.point == InterceptionPoint.INPUT.value
    assert model.first_turn_args is None
    input_record = next(
        record for record in records if record.interception_point is InterceptionPoint.INPUT
    )
    assert input_record.verdict.reason == "host_error:interceptor_failed"


async def test_model_failure_sanitizes_primary_error_and_emits_error_shutdown() -> None:
    primary_error = RuntimeError("model failed")
    interceptor = RecordingInterceptor()
    records = RecordSink(max_records=1000)
    model = FakeModel(initial_output=primary_error)
    agent = Agent(name="governed-agent", model=model)
    run_config = _governed_run_config(
        agent_hooks=AgentHooksConfig(
            agent_id="governed-agent-v1",
            interceptors=(interceptor,),
            record_sink=records,
        )
    )

    with pytest.raises(AgentHooksExecutionError) as error_info:
        await Runner.run(agent, "hello", max_turns=1, run_config=run_config)

    assert id(error_info.value) != id(primary_error)
    assert primary_error.__traceback__ is None
    assert primary_error.__cause__ is None
    assert primary_error.__context__ is None
    assert interceptor.shutdown_reasons == ["error"]
    assert interceptor.points.count(InterceptionPoint.PRE_MODEL_CALL) == 1
    assert interceptor.points.count(InterceptionPoint.POST_MODEL_CALL) == 1
    assert [record.interception_point for record in records].count(
        InterceptionPoint.AGENT_SHUTDOWN
    ) == 1


def test_run_sync_uses_one_complete_lifecycle() -> None:
    records = RecordSink(max_records=1000)
    model = FakeModel(initial_output=[_text_message("done")])
    agent = Agent(name="governed-agent", model=model)
    run_config = _governed_run_config(
        agent_hooks=AgentHooksConfig(
            agent_id="governed-agent-v1",
            interceptors=(AllowInterceptor(),),
            record_sink=records,
        )
    )

    result = Runner.run_sync(agent, "hello", max_turns=1, run_config=run_config)

    assert result.final_output == "done"
    assert [record.interception_point for record in records] == [
        InterceptionPoint.AGENT_STARTUP,
        InterceptionPoint.INPUT,
        InterceptionPoint.PRE_MODEL_CALL,
        InterceptionPoint.POST_MODEL_CALL,
        InterceptionPoint.OUTPUT,
        InterceptionPoint.AGENT_SHUTDOWN,
    ]


async def test_interceptor_timeout_drains_callback_and_fails_closed() -> None:
    interceptor = TimingOutInputInterceptor()
    records = RecordSink(max_records=1000)
    model = FakeModel(initial_output=[_text_message("unused")])
    agent = Agent(name="governed-agent", model=model)
    run_config = _governed_run_config(
        agent_hooks=AgentHooksConfig(
            agent_id="governed-agent-v1",
            interceptors=(interceptor,),
            record_sink=records,
            interceptor_timeout_seconds=0.01,
        )
    )

    with pytest.raises(AgentHooksBlockedError) as error_info:
        await Runner.run(agent, "hello", max_turns=1, run_config=run_config)

    assert error_info.value.point == InterceptionPoint.INPUT.value
    assert error_info.value.reason == "host_error:interceptor_timeout"
    assert interceptor.drained.is_set()
    assert interceptor.shutdown_reasons == ["error"]
    assert model.first_turn_args is None


async def test_verdict_label_limit_fails_closed_before_write_back() -> None:
    records = RecordSink(max_records=1000)
    model = FakeModel(initial_output=[_text_message("unused")])
    agent = Agent(name="governed-agent", model=model)
    run_config = _governed_run_config(
        agent_hooks=AgentHooksConfig(
            agent_id="governed-agent-v1",
            interceptors=(ExcessiveLabelsInterceptor(),),
            record_sink=records,
        )
    )

    with pytest.raises(AgentHooksBlockedError):
        await Runner.run(agent, "hello", max_turns=1, run_config=run_config)

    assert model.first_turn_args is None
    input_record = next(
        record for record in records if record.interception_point is InterceptionPoint.INPUT
    )
    assert input_record.verdict.reason == "host_error:transform_invalid"


async def test_completed_run_releases_per_run_session_references(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_sessions: list[AgentHooksRunSession] = []

    def capture_session(
        *,
        config: AgentHooksConfig,
        starting_agent: Agent[Any],
        max_turns: int,
    ) -> AgentHooksRunSession:
        session = create_agent_hooks_session(
            config=config,
            starting_agent=starting_agent,
            max_turns=max_turns,
        )
        captured_sessions.append(session)
        return session

    monkeypatch.setattr(run_module, "create_agent_hooks_session", capture_session)
    model = FakeModel(initial_output=[_text_message("done")])
    agent = Agent(name="governed-agent", model=model)
    run_config = _governed_run_config(
        agent_hooks=AgentHooksConfig(
            agent_id="governed-agent-v1",
            interceptors=(AllowInterceptor(),),
            record_sink=RecordSink(max_records=1000),
        )
    )

    result = await Runner.run(agent, "hello", max_turns=1, run_config=run_config)

    assert result.final_output == "done"
    assert len(captured_sessions) == 1
    session = captured_sessions[0]
    assert session.released


@pytest.mark.parametrize("failure_mode", ["cancellation", "timeout"])
async def test_repeated_failed_runs_release_sessions_and_shutdown_once(
    failure_mode: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_sessions: list[AgentHooksRunSession] = []

    def capture_session(
        *,
        config: AgentHooksConfig,
        starting_agent: Agent[Any],
        max_turns: int,
    ) -> AgentHooksRunSession:
        session = create_agent_hooks_session(
            config=config,
            starting_agent=starting_agent,
            max_turns=max_turns,
        )
        captured_sessions.append(session)
        return session

    monkeypatch.setattr(run_module, "create_agent_hooks_session", capture_session)

    for _ in range(3):
        records = RecordSink(max_records=1000)
        model = FakeModel(initial_output=[_text_message("unused")])
        agent = Agent(name="governed-agent", model=model)
        interceptor: CancellingInputInterceptor | TimingOutInputInterceptor
        if failure_mode == "cancellation":
            interceptor = CancellingInputInterceptor()
            timeout_seconds = 5.0
        else:
            interceptor = TimingOutInputInterceptor()
            timeout_seconds = 0.01
        run_config = _governed_run_config(
            agent_hooks=AgentHooksConfig(
                agent_id="governed-agent-v1",
                interceptors=(interceptor,),
                record_sink=records,
                interceptor_timeout_seconds=timeout_seconds,
            )
        )

        if failure_mode == "cancellation":
            task = asyncio.create_task(
                Runner.run(agent, "hello", max_turns=1, run_config=run_config)
            )
            await interceptor.started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            expected_reason = "cancelled"
        else:
            with pytest.raises(AgentHooksBlockedError) as error_info:
                await Runner.run(agent, "hello", max_turns=1, run_config=run_config)
            assert error_info.value.reason == "host_error:interceptor_timeout"
            assert interceptor.started.is_set()
            expected_reason = "error"

        assert interceptor.drained.is_set()
        assert interceptor.shutdown_reasons == [expected_reason]
        assert model.first_turn_args is None
        assert [record.interception_point for record in records].count(
            InterceptionPoint.AGENT_SHUTDOWN
        ) == 1
        assert captured_sessions[-1].released
        assert get_current_agent_hooks_session() is None

    assert len(captured_sessions) == 3
    assert all(session.released for session in captured_sessions)


@pytest.mark.parametrize("retry", [None, ModelRetrySettings(max_retries=1)])
async def test_retry_admission_requires_explicit_zero_retries(
    retry: ModelRetrySettings | None,
) -> None:
    records = RecordSink(max_records=1000)
    model = FakeModel(initial_output=[_text_message("unused")])
    agent = Agent(name="governed-agent", model=model)
    run_config = _governed_run_config(
        agent_hooks=AgentHooksConfig(
            agent_id="governed-agent-v1",
            interceptors=(AllowInterceptor(),),
            record_sink=records,
        )
    )
    assert run_config.model_settings is not None
    run_config.model_settings.retry = retry

    with pytest.raises(UserError, match="retry.max_retries=0"):
        await Runner.run(agent, "hello", max_turns=1, run_config=run_config)

    assert len(records) == 0
    assert model.first_turn_args is None


@pytest.mark.parametrize(
    ("parallel_tool_calls", "tool_execution"),
    [
        (None, ToolExecutionConfig(max_function_tool_concurrency=1)),
        (True, ToolExecutionConfig(max_function_tool_concurrency=1)),
        (False, None),
        (False, ToolExecutionConfig(max_function_tool_concurrency=2)),
        (
            False,
            ToolExecutionConfig(
                max_function_tool_concurrency=1,
                pre_approval_tool_input_guardrails=True,
            ),
        ),
    ],
)
async def test_tool_execution_admission_requires_complete_serial_contract(
    parallel_tool_calls: bool | None,
    tool_execution: ToolExecutionConfig | None,
) -> None:
    records = RecordSink(max_records=1000)
    model = FakeModel(initial_output=[_text_message("unused")])
    agent = Agent(
        name="governed-agent",
        model=model,
        tools=[get_function_tool("lookup", "unused")],
    )
    run_config = _governed_run_config(
        agent_hooks=AgentHooksConfig(
            agent_id="governed-agent-v1",
            interceptors=(AllowInterceptor(),),
            record_sink=records,
        ),
        with_tools=True,
    )
    assert run_config.model_settings is not None
    run_config.model_settings.parallel_tool_calls = parallel_tool_calls
    run_config.tool_execution = tool_execution

    with pytest.raises(UserError):
        await Runner.run(agent, "hello", max_turns=1, run_config=run_config)

    assert len(records) == 0
    assert model.first_turn_args is None


async def test_function_tool_subclass_fails_admission_before_records_and_dispatch() -> None:
    class DerivedFunctionTool(FunctionTool):
        pass

    records = RecordSink(max_records=1000)
    model = FakeModel(initial_output=[_text_message("unused")])
    source_tool = get_function_tool("lookup", "unused")
    tool = DerivedFunctionTool(
        name=source_tool.name,
        description=source_tool.description,
        params_json_schema=source_tool.params_json_schema,
        on_invoke_tool=source_tool.on_invoke_tool,
    )
    agent = Agent(name="governed-agent", model=model, tools=[tool])
    run_config = _governed_run_config(
        agent_hooks=AgentHooksConfig(
            agent_id="governed-agent-v1",
            interceptors=(AllowInterceptor(),),
            record_sink=records,
        ),
        with_tools=True,
    )

    with pytest.raises(
        UserError,
        match=r"^Agent Hooks requires exact FunctionTool entries$",
    ):
        await Runner.run(agent, "hello", max_turns=1, run_config=run_config)

    assert len(records) == 0
    assert model.first_turn_args is None


async def test_function_tool_schema_subclass_fails_before_copy_records_and_dispatch() -> None:
    deepcopy_calls = 0

    class HostileSchema(dict[str, object]):
        def __deepcopy__(self, _memo: dict[int, object]) -> HostileSchema:
            nonlocal deepcopy_calls
            deepcopy_calls += 1
            return self

    async def invoke_tool(_context: object, _arguments: str) -> str:
        return "unused"

    records = RecordSink(max_records=1000)
    model = FakeModel(initial_output=[_text_message("unused")])
    tool = FunctionTool(
        name="lookup",
        description="Look up a value",
        params_json_schema=HostileSchema({"type": "object"}),
        on_invoke_tool=invoke_tool,
        strict_json_schema=False,
    )
    assert type(tool) is FunctionTool
    agent = Agent(name="governed-agent", model=model, tools=[tool])
    run_config = _governed_run_config(
        agent_hooks=AgentHooksConfig(
            agent_id="governed-agent-v1",
            interceptors=(AllowInterceptor(),),
            record_sink=records,
        ),
        with_tools=True,
    )

    with pytest.raises(
        UserError,
        match=r"^Agent Hooks requires every FunctionTool parameter schema to be valid$",
    ) as error_info:
        await Runner.run(agent, "hello", max_turns=1, run_config=run_config)

    assert error_info.value.__context__ is None
    assert deepcopy_calls == 0
    assert len(records) == 0
    assert model.first_turn_args is None


@pytest.mark.parametrize(
    "schema",
    [
        {"$ref": "https://example.invalid/tool-schema.json"},
        {"type": "object", "patternProperties": {"(a+)+$": {"type": "string"}}},
        {"type": "object", "anyOf": [{"required": ["value"]}, {"required": ["other"]}]},
    ],
)
async def test_unsafe_function_tool_schema_fails_before_records_and_dispatch(
    schema: dict[str, object],
) -> None:
    async def invoke_tool(_context: object, _arguments: str) -> str:
        return "unused"

    records = RecordSink(max_records=1000)
    model = FakeModel(initial_output=[_text_message("unused")])
    tool = FunctionTool(
        name="lookup",
        description="Look up a value",
        params_json_schema=schema,
        on_invoke_tool=invoke_tool,
        strict_json_schema=False,
    )
    agent = Agent(name="governed-agent", model=model, tools=[tool])
    run_config = _governed_run_config(
        agent_hooks=AgentHooksConfig(
            agent_id="governed-agent-v1",
            interceptors=(AllowInterceptor(),),
            record_sink=records,
        ),
        with_tools=True,
    )

    with pytest.raises(UserError, match="FunctionTool parameter schema"):
        await Runner.run(agent, "hello", max_turns=1, run_config=run_config)

    assert len(records) == 0
    assert model.first_turn_args is None


async def test_custom_function_tool_bind_hook_fails_before_hook_records_and_dispatch() -> None:
    async def original_invoke_tool(_context: object, _arguments: str) -> str:
        return "unused"

    tool = FunctionTool(
        name="lookup",
        description="Look up a value",
        params_json_schema={"type": "object", "properties": {}},
        on_invoke_tool=original_invoke_tool,
    )
    bind_calls = 0

    def hostile_bind_hook(_tool: FunctionTool) -> None:
        nonlocal bind_calls
        bind_calls += 1
        raise AssertionError("custom FunctionTool bind hook was invoked")

    async def replacement_invoke_tool(_context: object, _arguments: str) -> str:
        return "unused"

    cast(Any, replacement_invoke_tool).__agents_bind_function_tool__ = hostile_bind_hook
    tool.on_invoke_tool = replacement_invoke_tool
    assert type(tool) is FunctionTool
    records = RecordSink(max_records=1000)
    model = FakeModel(initial_output=[_text_message("unused")])
    agent = Agent(name="governed-agent", model=model, tools=[tool])
    run_config = _governed_run_config(
        agent_hooks=AgentHooksConfig(
            agent_id="governed-agent-v1",
            interceptors=(AllowInterceptor(),),
            record_sink=records,
        ),
        with_tools=True,
    )

    with pytest.raises(
        UserError,
        match=r"^Agent Hooks could not snapshot the admitted run: UserError$",
    ) as error_info:
        await Runner.run(agent, "hello", max_turns=1, run_config=run_config)

    assert error_info.value.__context__ is None
    assert bind_calls == 0
    assert len(records) == 0
    assert model.first_turn_args is None


@pytest.mark.parametrize("formatter_kind", ["failure", "timeout"])
async def test_custom_function_tool_error_formatter_fails_before_records_and_dispatch(
    formatter_kind: str,
) -> None:
    def custom_error_formatter(_context: object, _error: Exception) -> str:
        return "custom error"

    if formatter_kind == "failure":

        @function_tool(
            name_override="lookup",
            failure_error_function=custom_error_formatter,
        )
        async def lookup_with_failure_formatter(value: str) -> str:
            return value

        tool = lookup_with_failure_formatter
    else:

        @function_tool(
            name_override="lookup",
            timeout=1.0,
            timeout_error_function=custom_error_formatter,
        )
        async def lookup_with_timeout_formatter(value: str) -> str:
            return value

        tool = lookup_with_timeout_formatter

    assert type(tool) is FunctionTool
    records = RecordSink(max_records=1000)
    model = FakeModel(initial_output=[_text_message("unused")])
    agent = Agent(name="governed-agent", model=model, tools=[tool])
    run_config = _governed_run_config(
        agent_hooks=AgentHooksConfig(
            agent_id="governed-agent-v1",
            interceptors=(AllowInterceptor(),),
            record_sink=records,
        ),
        with_tools=True,
    )

    with pytest.raises(
        UserError,
        match=r"^Agent Hooks does not support custom FunctionTool error formatters$",
    ):
        await Runner.run(agent, "hello", max_turns=1, run_config=run_config)

    assert len(records) == 0
    assert model.first_turn_args is None


@pytest.mark.parametrize(
    "case",
    [
        "namespace",
        "callers",
        "structured_output",
        "output_adapter",
        "failure_formatter_flag",
        "custom_extractor",
        "duplicate_names",
        "sync_tool",
    ],
)
async def test_unsupported_function_tool_shape_fails_before_records(case: str) -> None:
    records = RecordSink(max_records=1000)
    model = FakeModel(initial_output=[_text_message("unused")])
    tool = get_function_tool("lookup", "unused")
    if case == "namespace":
        tool._tool_namespace = "unsupported"
    elif case == "callers":
        tool.allowed_callers = []
    elif case == "structured_output":
        tool.output_json_schema = {"type": "object"}
    elif case == "output_adapter":
        tool._output_type_adapter = cast(Any, object())
    elif case == "failure_formatter_flag":
        tool._use_default_failure_error_function = cast(Any, object())
    elif case == "custom_extractor":

        async def extract_custom_data(context: object) -> dict[str, object]:
            return {}

        tool.custom_data_extractor = cast(Any, extract_custom_data)
    if case == "sync_tool":

        @function_tool(name_override="lookup")
        def sync_lookup(value: str) -> str:
            return value

        tools: list[Tool] = [sync_lookup]
    else:
        tools = (
            [tool, get_function_tool("lookup", "unused")] if case == "duplicate_names" else [tool]
        )
    agent = Agent(name="governed-agent", model=model, tools=tools)
    run_config = _governed_run_config(
        agent_hooks=AgentHooksConfig(
            agent_id="governed-agent-v1",
            interceptors=(AllowInterceptor(),),
            record_sink=records,
        ),
        with_tools=True,
    )

    expected_error = {
        "sync_tool": "asynchronous FunctionTool handlers",
        "output_adapter": "FunctionTool output adapters",
        "failure_formatter_flag": "boolean FunctionTool failure formatter flag",
    }.get(case)
    with pytest.raises(UserError, match=expected_error):
        await Runner.run(agent, "hello", max_turns=1, run_config=run_config)

    assert len(records) == 0
    assert model.first_turn_args is None
