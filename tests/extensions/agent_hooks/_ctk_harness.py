from __future__ import annotations

import copy
import json
from typing import Any, cast

from agent_hooks import AgentContext, EnforcementMode, Verdict
from agent_hooks.approval import ApprovalResolver
from agent_hooks.composition import CompositionConfig
from agent_hooks.ctk import Capability, RunOutcome, RunRecord, Scenario
from agent_hooks.ctk.harness import ToolSpec
from agent_hooks.interceptor import Interceptor
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)

from agents import Agent, FunctionTool, ModelSettings, RunConfig, Runner, Tool, ToolExecutionConfig
from agents.extensions.agent_hooks import AgentHooksBlockedError, AgentHooksConfig, RecordSink
from agents.items import TResponseOutputItem
from agents.retry import ModelRetrySettings
from agents.tool_context import ToolContext
from tests.fake_model import FakeModel


class _AsyncCTKInterceptor:
    def __init__(self, interceptor: Interceptor) -> None:
        self._interceptor = interceptor

    async def intercept(self, context: AgentContext, /) -> Verdict:
        result = self._interceptor.intercept(context)
        if isinstance(result, Verdict):
            return result
        try:
            return Verdict.from_wire(result)
        except ValueError:
            return cast(Verdict, result)


def project_vector_to_fixed_surface(vector: dict[str, Any]) -> dict[str, Any]:
    """Project a profile-independent CTK vector onto the adapter's fixed profile."""
    projected = copy.deepcopy(vector)
    composition = projected.get("composition")
    if composition not in (None, {"profile": "sequential/run_all"}):
        raise ValueError("CTK vector requires an unsupported composition profile")
    if projected.get("approval_script"):
        raise ValueError("CTK vector requires the unsupported approval resolver")
    if projected.get("identity_provider", "jcs-sha256") != "jcs-sha256":
        raise ValueError("CTK vector requires an unsupported identity provider")
    projected["composition"] = {"profile": "sequential/run_all"}
    _apply_openai_projection_expectations(projected)
    return projected


def _apply_openai_projection_expectations(vector: dict[str, Any]) -> None:
    vector_id = vector["id"]
    expected_interceptions = vector["expect"]["interceptions"]
    for expected_record in vector["expect"].get("records", []):
        absent = expected_record.get("absent")
        if isinstance(absent, list) and "trace" in absent:
            absent.remove("trace")
    if vector_id == "AH-CTK-001":
        for expected in expected_interceptions:
            if expected["interception_point"] in {"pre_tool_call", "post_tool_call"}:
                expected.get("context", {}).pop("tool_call.id", None)
    elif vector_id == "AH-CTK-100":
        pre_model = next(
            item
            for item in expected_interceptions
            if item["interception_point"] == "pre_model_call"
        )
        pre_model["context"] = {
            "messages[2].role": "tool",
            "messages[2].content.output": "blocked: policy_denied",
        }


class OpenAIAgentsCTKHarness:
    """Drive supported CTK scenarios through the production non-streaming runner."""

    name = "openai-agents-fixed-agent-hooks"
    capabilities: set[Capability] = {
        Capability.MODEL_CALLS,
        Capability.TOOL_CALLS,
    }

    def __init__(self) -> None:
        self._scenario: Scenario | None = None
        self._interceptors: tuple[_AsyncCTKInterceptor, ...] = ()
        self._mode = EnforcementMode.ENFORCE
        self._records = RecordSink(max_records=1000)
        self._tool_invocations: list[dict[str, Any]] = []

    def setup(
        self,
        scenario: Scenario,
        interceptors: list[Interceptor],
        resolver: ApprovalResolver | None,
        mode: EnforcementMode,
        composition: CompositionConfig,
        identity_provider: str | None,
        redact_for_approval: list[str] | None = None,
    ) -> None:
        if resolver is not None or redact_for_approval:
            raise ValueError("The fixed adapter surface has no approval resolver")
        if composition != CompositionConfig.run_all():
            raise ValueError("The fixed adapter surface requires sequential/run_all")
        if identity_provider != "jcs-sha256":
            raise ValueError("The fixed adapter surface requires jcs-sha256")
        self._scenario = scenario
        self._interceptors = tuple(_AsyncCTKInterceptor(item) for item in interceptors)
        self._mode = mode
        self._records = RecordSink(max_records=1000)
        self._tool_invocations = []
        for tool in scenario.tools.values():
            for behavior in tool.behavior:
                if behavior.is_error or not isinstance(behavior.return_, str):
                    raise ValueError("The fixed adapter surface requires string tool results")

    async def run(self) -> RunRecord:
        scenario = self._require_scenario()
        input_content = scenario.input.get("content")
        if scenario.input.get("role") != "user" or not isinstance(input_content, str):
            raise ValueError("The fixed adapter surface requires string user input")

        model_outputs: list[list[TResponseOutputItem] | Exception] = []
        for index, scripted_response in enumerate(scenario.model_script):
            has_tool_calls = bool(scripted_response.tool_calls)
            expected_finish_reason = "tool_calls" if has_tool_calls else "stop"
            if scripted_response.finish_reason != expected_finish_reason:
                raise ValueError("The fixed adapter requires semantic tool_calls/stop reasons")
            output: list[TResponseOutputItem] = []
            if scripted_response.content is not None:
                if not isinstance(scripted_response.content, str):
                    raise ValueError("The fixed adapter surface requires string model content")
                output.append(_message(index, scripted_response.content))
            for call_index, tool_call in enumerate(scripted_response.tool_calls):
                output.append(_tool_call(index, call_index, tool_call))
            model_outputs.append(output)
        model = FakeModel()
        model.add_multiple_turn_outputs(model_outputs)
        tools: list[Tool] = [self._make_tool(tool) for tool in scenario.tools.values()]
        agent: Agent[None] = Agent(name="ctk-agent", model=model, tools=tools)
        run_config = RunConfig(
            agent_hooks=AgentHooksConfig(
                agent_id="ctk-agent",
                interceptors=self._interceptors,
                record_sink=self._records,
                mode=self._mode,
            ),
            trace_include_sensitive_data=False,
            model_settings=ModelSettings(
                parallel_tool_calls=False if tools else None,
                retry=ModelRetrySettings(max_retries=0),
            ),
            tool_execution=(
                ToolExecutionConfig(max_function_tool_concurrency=1) if tools else None
            ),
        )

        final_output: object | None = None
        outcome = RunOutcome.COMPLETED
        try:
            result = await Runner.run(
                agent,
                input_content,
                max_turns=max(1, len(model_outputs)),
                run_config=run_config,
            )
            final_output = result.final_output
        except AgentHooksBlockedError:
            outcome = RunOutcome.BLOCKED

        return RunRecord(
            outcome=outcome,
            final_output=final_output,
            tool_invocations=list(self._tool_invocations),
            identities=[
                (record.input_identity, record.enforced_identity) for record in self._records
            ],
            records=[record.to_wire() for record in self._records],
        )

    def teardown(self) -> None:
        self._scenario = None
        self._interceptors = ()
        self._records = RecordSink(max_records=1000)
        self._tool_invocations = []

    def _require_scenario(self) -> Scenario:
        if self._scenario is None:
            raise RuntimeError("CTK harness setup was not called")
        return self._scenario

    def _make_tool(self, tool: ToolSpec) -> FunctionTool:
        async def invoke(_context: ToolContext[Any], arguments_json: str) -> str:
            arguments = json.loads(arguments_json)
            if not isinstance(arguments, dict) or not all(
                isinstance(key, str) for key in arguments
            ):
                raise ValueError("CTK tool arguments must be a JSON object")
            typed_arguments = cast(dict[str, Any], arguments)
            self._tool_invocations.append(
                {"name": tool.name, "args": copy.deepcopy(typed_arguments)}
            )
            result, is_error = tool.invoke(typed_arguments)
            if is_error or not isinstance(result, str):
                raise ValueError("The fixed adapter surface requires successful string results")
            return result

        return FunctionTool(
            name=tool.name,
            description=f"CTK scripted tool {tool.name}",
            params_json_schema=_params_schema(tool.schema),
            on_invoke_tool=invoke,
            strict_json_schema=False,
        )


def _message(index: int, text: str) -> ResponseOutputMessage:
    return ResponseOutputMessage(
        id=f"ctk-message-{index}",
        content=[ResponseOutputText(annotations=[], text=text, type="output_text")],
        role="assistant",
        status="completed",
        type="message",
    )


def _tool_call(
    turn_index: int,
    call_index: int,
    call: dict[str, Any],
) -> ResponseFunctionToolCall:
    call_id = call.get("id")
    name = call.get("name")
    arguments = call.get("args")
    if not isinstance(call_id, str) or not isinstance(name, str) or not isinstance(arguments, dict):
        raise ValueError("CTK model tool calls require string IDs/names and object arguments")
    return ResponseFunctionToolCall(
        id=f"ctk-tool-item-{turn_index}-{call_index}",
        call_id=call_id,
        type="function_call",
        name=name,
        arguments=json.dumps(
            arguments,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        status="completed",
    )


def _params_schema(shorthand: dict[str, Any]) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    for name, value in shorthand.items():
        properties[name] = {"type": value} if isinstance(value, str) else copy.deepcopy(value)
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": not bool(properties),
    }
    if properties:
        schema["required"] = list(properties)
    return schema
