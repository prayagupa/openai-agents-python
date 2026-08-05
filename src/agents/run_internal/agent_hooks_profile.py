"""Admission and immutable execution snapshots for the Agent Hooks profile."""

from __future__ import annotations

import inspect
import math
from collections.abc import AsyncIterator, Awaitable, Callable
from types import FunctionType
from typing import Any, cast

from openai.types.responses.response_prompt_param import ResponsePromptParam

from ..agent import Agent
from ..agent_output import AgentOutputSchemaBase
from ..exceptions import UserError
from ..handoffs import Handoff
from ..items import ModelResponse, TResponseInputItem, TResponseStreamEvent
from ..model_settings import ModelSettings, _coerce_model_settings
from ..models.interface import Model, ModelProvider, ModelTracing
from ..models.openai_agent_registration import add_openai_harness_id_to_metadata
from ..retry import ModelRetryAdvice, ModelRetryAdviceRequest
from ..run_config import RunConfig, ToolExecutionConfig
from ..run_context import TContext
from ..tool import (
    FunctionTool,
    Tool,
    ToolOrigin,
    ToolOriginType,
    _FailureHandlingFunctionToolInvoker,
    function_tool_has_custom_error_formatter,
    set_function_tool_failure_error_function,
)
from ..tool_context import ToolContext
from .agent_hooks import validate_agent_hooks_admission
from .turn_preparation import get_model, get_model_settings


class _CapturedModel(Model):
    """Capture model methods before Agent Hooks callbacks can run."""

    def __init__(self, model: Model) -> None:
        self.model = getattr(model, "model", type(model).__name__)
        self._get_response = model.get_response
        self._stream_response = model.stream_response
        self._get_retry_advice = model.get_retry_advice
        self._cleanup_on_end = model._cleanup_on_run_end
        self._close_model = model.close

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
        prompt: ResponsePromptParam | None,
    ) -> ModelResponse:
        return await self._get_response(
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

    def stream_response(
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
        prompt: ResponsePromptParam | None,
    ) -> AsyncIterator[TResponseStreamEvent]:
        return self._stream_response(
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

    def get_retry_advice(self, request: ModelRetryAdviceRequest) -> ModelRetryAdvice | None:
        return self._get_retry_advice(request)

    async def _cleanup_on_run_end(self, owner: object) -> None:
        await self._cleanup_on_end(owner)

    async def close(self) -> None:
        await self._close_model()


class _CapturedModelProvider(ModelProvider):
    """Expose only the captured model and resolved trace metadata marker."""

    _agent_hooks_trace_metadata_resolved = True

    def __init__(self, model: Model) -> None:
        self._model = model

    def get_model(self, model_name: str | None) -> Model:
        return self._model


def _copy_exact_json(value: object) -> object:
    """Reconstruct JSON data without invoking application copy protocols."""
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise UserError("Agent Hooks FunctionTool schemas must contain finite JSON values")
        return value
    if type(value) is list:
        return [_copy_exact_json(item) for item in cast(list[object], value)]
    if type(value) is dict:
        result: dict[str, object] = {}
        for key, item in cast(dict[object, object], value).items():
            if type(key) is not str:
                raise UserError("Agent Hooks FunctionTool schema keys must be exact strings")
            result[key] = _copy_exact_json(item)
        return result
    raise UserError("Agent Hooks FunctionTool schemas must use exact JSON value types")


async def _unavailable_model_tool(_context: object, _arguments: str) -> object:
    raise RuntimeError("Governed model tool definitions are not invokable")


def _snapshot_function_tool(source: FunctionTool, *, invokable: bool) -> FunctionTool:
    """Build one strict-profile tool without application copy or bind hooks."""
    if type(source) is not FunctionTool:
        raise UserError("Agent Hooks requires exact FunctionTool entries")
    if type(source.name) is not str or type(source.description) is not str:
        raise UserError("Agent Hooks FunctionTool names and descriptions must be exact strings")
    if type(source.strict_json_schema) is not bool:
        raise UserError("Agent Hooks FunctionTool strict_json_schema must be a bool")
    if type(source.timeout_behavior) is not str:
        raise UserError("Agent Hooks FunctionTool timeout behavior must be a string")
    if source.timeout_seconds is not None and type(source.timeout_seconds) not in {int, float}:
        raise UserError("Agent Hooks FunctionTool timeout must be a number or None")

    schema = _copy_exact_json(source.params_json_schema)
    if type(schema) is not dict:
        raise UserError("Agent Hooks FunctionTool schemas must be JSON objects")

    callback = source.on_invoke_tool
    snapshot_callback: Callable[[ToolContext[Any], str], Awaitable[Any]]
    if invokable:
        if type(callback) is _FailureHandlingFunctionToolInvoker:
            snapshot_callback = callback
        elif type(callback) is FunctionType and inspect.iscoroutinefunction(callback):
            if getattr(callback, "__agents_bind_function_tool__", None) is not None:
                raise UserError("Agent Hooks does not support custom FunctionTool bind hooks")
            snapshot_callback = callback
        else:
            raise UserError(
                "Agent Hooks requires SDK function_tool wrappers or exact async functions"
            )
    else:
        snapshot_callback = _unavailable_model_tool

    snapshot = FunctionTool(
        name=source.name,
        description=source.description,
        params_json_schema=cast(dict[str, object], schema),
        on_invoke_tool=snapshot_callback,
        strict_json_schema=source.strict_json_schema,
        is_enabled=True,
        tool_input_guardrails=None,
        tool_output_guardrails=None,
        needs_approval=False,
        timeout_seconds=source.timeout_seconds,
        timeout_behavior=source.timeout_behavior,
        timeout_error_function=None,
        defer_loading=False,
        custom_data_extractor=None,
        allowed_callers=None,
        output_json_schema=None,
        _output_type_adapter=None,
        _failure_error_function=None,
        _use_default_failure_error_function=(
            True if source._use_default_failure_error_function else False
        ),
        _is_agent_tool=False,
        _is_codex_tool=False,
        _agent_instance=None,
        _tool_namespace=None,
        _tool_namespace_description=None,
        _mcp_title=None,
        _tool_origin=ToolOrigin(type=ToolOriginType.FUNCTION),
        _emit_tool_origin=True,
    )
    if not source._use_default_failure_error_function:
        set_function_tool_failure_error_function(snapshot, None)
    return snapshot


def snapshot_agent_hooks_model_settings(settings: ModelSettings) -> ModelSettings:
    """Reconstruct model-visible settings without retaining mutable application objects."""
    if type(settings) is not ModelSettings:
        raise UserError("Agent Hooks requires exact ModelSettings")
    try:
        payload = settings.to_json_dict()
        snapshot = _coerce_model_settings(
            payload,
            parameter_name="Agent Hooks model settings",
            model_settings_type=ModelSettings,
        )
    except Exception as error:
        raise UserError(
            f"Agent Hooks could not snapshot model settings: {type(error).__name__}"
        ) from None
    if snapshot.retry is None or snapshot.retry.max_retries != 0:
        raise UserError("Agent Hooks requires model retry.max_retries=0")
    return snapshot


def validate_agent_hooks_profile(
    *,
    starting_agent: Agent[TContext],
    input: object,
    max_turns: object,
    run_config: RunConfig,
    error_handlers: object | None,
    previous_response_id: str | None,
    auto_previous_response_id: bool,
    conversation_id: str | None,
    session: object | None,
    context: object | None,
    run_hooks: object | None,
) -> None:
    """Reject mutable native callback and telemetry surfaces before snapshotting."""
    if type(starting_agent) is not Agent:
        raise UserError("Agent Hooks requires an exact plain Agent instance")
    if type(run_config) is not RunConfig:
        raise UserError("Agent Hooks requires an exact RunConfig instance")
    if type(run_config.tracing_disabled) is not bool:
        raise UserError("Agent Hooks tracing_disabled must be an exact bool")
    if type(run_config.trace_include_sensitive_data) is not bool:
        raise UserError("Agent Hooks trace_include_sensitive_data must be an exact bool")
    if type(run_config.workflow_name) is not str:
        raise UserError("Agent Hooks workflow_name must be an exact string")
    for field_name in ("trace_id", "group_id"):
        value = getattr(run_config, field_name)
        if value is not None and type(value) is not str:
            raise UserError(f"Agent Hooks {field_name} must be an exact string or None")
    if run_config.trace_metadata is not None:
        trace_metadata = _copy_exact_json(run_config.trace_metadata)
        if type(trace_metadata) is not dict:
            raise UserError("Agent Hooks trace_metadata must be an exact JSON object")
    if run_config.tracing is not None:
        tracing = _copy_exact_json(run_config.tracing)
        if type(tracing) is not dict:
            raise UserError("Agent Hooks tracing must be an exact dictionary")
        tracing = cast(dict[str, object], tracing)
        if set(tracing) - {"api_key", "include_task_and_turn_spans"}:
            raise UserError("Agent Hooks tracing contains unsupported fields")
        if tracing.get("api_key") is not None and type(tracing.get("api_key")) is not str:
            raise UserError("Agent Hooks tracing.api_key must be an exact string")
        if (
            tracing.get("include_task_and_turn_spans") is not None
            and type(tracing.get("include_task_and_turn_spans")) is not bool
        ):
            raise UserError("Agent Hooks tracing.include_task_and_turn_spans must be an exact bool")
    for tool in starting_agent.tools:
        if type(tool) is not FunctionTool:
            raise UserError("Agent Hooks requires exact FunctionTool entries")
        if type(tool._use_default_failure_error_function) is not bool:
            raise UserError("Agent Hooks requires a boolean FunctionTool failure formatter flag")
        if tool._output_type_adapter is not None:
            raise UserError("Agent Hooks does not support FunctionTool output adapters")
        if function_tool_has_custom_error_formatter(tool):
            raise UserError("Agent Hooks does not support custom FunctionTool error formatters")
    validate_agent_hooks_admission(
        starting_agent=starting_agent,
        input=input,
        max_turns=max_turns,
        run_config=run_config,
        error_handlers=error_handlers,
        previous_response_id=previous_response_id,
        auto_previous_response_id=auto_previous_response_id,
        conversation_id=conversation_id,
        session=session,
        context=context,
        run_hooks=run_hooks,
    )
    if context is not None:
        raise UserError("Agent Hooks does not support caller context in the fixed profile")
    if run_hooks is not None:
        raise UserError("Agent Hooks does not support native callbacks in the fixed profile")
    if (
        starting_agent.hooks is not None
        or starting_agent.input_guardrails
        or starting_agent.output_guardrails
        or run_config.input_guardrails
        or run_config.output_guardrails
    ):
        raise UserError("Agent Hooks does not support native callbacks in the fixed profile")
    if run_config.trace_include_sensitive_data:
        raise UserError(
            "Agent Hooks requires redacted tracing with trace_include_sensitive_data=False"
        )
    for tool in starting_agent.tools:
        if isinstance(tool, FunctionTool) and (
            tool.tool_input_guardrails or tool.tool_output_guardrails
        ):
            raise UserError("Agent Hooks does not support native callbacks in the fixed profile")


def snapshot_agent_hooks_run(
    *,
    starting_agent: Agent[TContext],
    run_config: RunConfig,
) -> tuple[Agent[TContext], RunConfig]:
    """Capture executable settings and callbacks synchronously before the first await."""
    try:
        resolved_model = get_model(starting_agent, run_config)
        effective_model_settings = snapshot_agent_hooks_model_settings(
            get_model_settings(starting_agent, run_config)
        )
        snapshot_tools = [
            _snapshot_function_tool(cast(FunctionTool, tool), invokable=True)
            for tool in starting_agent.tools
        ]
        snapshot_agent = starting_agent.clone(
            tools=snapshot_tools,
            mcp_servers=[],
            mcp_config={},
            handoffs=[],
            model=_CapturedModel(resolved_model),
            model_settings=effective_model_settings,
            input_guardrails=[],
            output_guardrails=[],
            hooks=None,
        )
        trace_metadata_value = (
            _copy_exact_json(run_config.trace_metadata)
            if run_config.trace_metadata is not None
            else None
        )
        trace_metadata = add_openai_harness_id_to_metadata(
            cast(dict[str, Any] | None, trace_metadata_value),
            model_provider=run_config.model_provider,
        )
        tracing_value = (
            _copy_exact_json(run_config.tracing) if run_config.tracing is not None else None
        )
        tool_execution = run_config.tool_execution
        snapshot_config = RunConfig(
            model=None,
            model_provider=_CapturedModelProvider(cast(Model, snapshot_agent.model)),
            model_settings=None,
            handoff_input_filter=None,
            nest_handoff_history=False,
            handoff_history_mapper=None,
            input_guardrails=None,
            output_guardrails=None,
            tracing_disabled=run_config.tracing_disabled,
            tracing=cast(Any, tracing_value),
            trace_include_sensitive_data=False,
            workflow_name=run_config.workflow_name,
            trace_id=run_config.trace_id,
            group_id=run_config.group_id,
            trace_metadata=trace_metadata,
            session_input_callback=None,
            call_model_input_filter=None,
            tool_error_formatter=None,
            session_settings=None,
            reasoning_item_id_policy=run_config.reasoning_item_id_policy,
            sandbox=None,
            tool_execution=(
                ToolExecutionConfig(
                    max_function_tool_concurrency=tool_execution.max_function_tool_concurrency,
                    pre_approval_tool_input_guardrails=False,
                )
                if tool_execution is not None
                else None
            ),
            tool_not_found_behavior="raise_error",
            agent_hooks=run_config.agent_hooks,
        )
    except Exception as error:
        raise UserError(
            f"Agent Hooks could not snapshot the admitted run: {type(error).__name__}"
        ) from None
    return snapshot_agent, snapshot_config


def snapshot_agent_hooks_model_tools(tools: list[Tool]) -> list[Tool]:
    """Return detached tool definitions that cannot mutate or invoke execution tools."""

    return [_snapshot_function_tool(cast(FunctionTool, tool), invokable=False) for tool in tools]
