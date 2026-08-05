"""Per-run Agent Hooks admission, lifecycle, and task-local state."""

from __future__ import annotations

import asyncio
import copy
import json
import math
import re
import sys
import time
from collections.abc import Callable
from contextvars import ContextVar, Token
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, TypeGuard, TypeVar, cast
from uuid import uuid4

from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails

from .._tool_identity import get_explicit_function_tool_namespace, get_function_tool_qualified_name
from ..agent import Agent
from ..exceptions import AgentsException, UserError
from ..items import ModelResponse, TResponseInputItem, TResponseOutputItem
from ..run_config import RunConfig
from ..tool import (
    FunctionTool,
    ToolOriginType,
    get_function_tool_origin,
    is_async_function_tool,
)
from ..usage import RequestUsage, Usage, _make_input_tokens_details
from .agent_hooks_errors import (
    AgentHooksAuditError,
    AgentHooksBlockedError,
    _AgentHooksSetupError,
    create_agent_hooks_audit_error,
    create_agent_hooks_blocked_error,
    create_agent_hooks_execution_error,
    create_agent_hooks_setup_error,
    is_host_agent_hooks_error,
)

if sys.version_info < (3, 11):
    from exceptiongroup import BaseExceptionGroup  # pyright: ignore[reportMissingImports]

if TYPE_CHECKING:
    from agent_hooks import (
        AgentContext,
        AgentContextBuilder,
        InterceptionEmitter,
        InterceptionRecord,
        Verdict,
    )
    from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage

    from ..extensions.agent_hooks import (
        AgentHooksConfig,
        AgentHooksLimits,
        AsyncInterceptor,
        _RecordSinkReservation,
    )
    from ..models.interface import Model

TContext = TypeVar("TContext")

_FRAMEWORK = "openai-agents"
_HOST_VALIDATOR_NAME = "openai-agents-host-validator"
_HOST_TRANSFORM_INVALID = "openai_agents_host:transform_invalid"
_HOST_ADAPTER_UNSUPPORTED = "openai_agents_host:adapter_unsupported"
_HOST_CONTEXT_INVALID = "openai_agents_host:context_invalid"
_MAX_IDENTIFIER_BYTES = 256
_MAX_LABEL_BYTES = 256
_MAX_VERDICT_COLLECTION_ITEMS = 256
_OPAQUE_CODE_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}")
_SAFE_TRANSFORM_PATH_PATTERN = re.compile(
    r"\$(?:target|policy_target)(?:(?:\.[A-Za-z_][A-Za-z0-9_]*)|(?:\[\d+\]))*"
)
_SAFE_SCHEMA_KEYWORDS = {
    "$comment",
    "$defs",
    "$ref",
    "additionalProperties",
    "const",
    "default",
    "deprecated",
    "description",
    "enum",
    "examples",
    "exclusiveMaximum",
    "exclusiveMinimum",
    "items",
    "maxItems",
    "maxLength",
    "maxProperties",
    "maximum",
    "minItems",
    "minLength",
    "minProperties",
    "minimum",
    "multipleOf",
    "prefixItems",
    "properties",
    "readOnly",
    "required",
    "title",
    "type",
    "writeOnly",
}
_SCHEMA_MAP_KEYWORDS = {"$defs", "properties"}
_SCHEMA_SINGLE_KEYWORDS = {"additionalProperties", "items"}


class _AgentHooksValidationError(Exception):
    pass


class _AgentHooksUnsupportedError(Exception):
    pass


class _AgentHooksContextInvalidError(Exception):
    pass


class _AgentHooksCancelledError(asyncio.CancelledError):
    pass


class _AgentHooksKeyboardInterrupt(KeyboardInterrupt):
    pass


class _AgentHooksSystemExit(SystemExit):
    pass


@dataclass(frozen=True, slots=True)
class _ModelCall:
    model_id: str
    request_id: str


@dataclass(frozen=True, slots=True)
class _ModelPreEntry:
    kind: str
    role: str
    original: object | None
    call_id: str | None = None
    name: str | None = None


@dataclass(frozen=True, slots=True)
class _ModelPreState:
    entries: tuple[_ModelPreEntry, ...]
    unsupported: bool = False


@dataclass(frozen=True, slots=True)
class _ModelPostEntry:
    kind: str
    item_id: str
    call_id: str | None = None
    name: str | None = None
    status_completed: bool = False


@dataclass(frozen=True, slots=True)
class _ModelPostState:
    usage: Usage
    response_id: str | None
    request_id: str | None
    entries: tuple[_ModelPostEntry, ...]
    finish_reason: str
    unsupported: bool = False
    context_invalid: bool = False


_ModelValidationState = _ModelPreState | _ModelPostState


@dataclass(frozen=True, slots=True)
class ToolInvocation:
    """Trusted host sidecar for one permitted static function invocation."""

    invocation_id: str
    name: str
    arguments: dict[str, object]
    tool_call: object


@dataclass(frozen=True, slots=True)
class _ToolPreState:
    invocation_id: str
    name: str
    params_json_schema: dict[str, object]
    unsupported: bool = False


@dataclass(frozen=True, slots=True)
class _ToolPostState:
    invocation_id: str
    name: str
    arguments: dict[str, object]
    is_error: bool
    unsupported: bool = False
    context_invalid: bool = False


@dataclass(frozen=True, slots=True)
class ToolPreDecision:
    """Internal result of one governed pre-tool emission."""

    invocation: ToolInvocation | None
    blocked_message: str | None = None


@dataclass(frozen=True, slots=True)
class ToolPostDecision:
    """Internal result of one governed post-tool emission."""

    result: str | None
    blocked_message: str | None = None


_ValidationState = _ModelValidationState | _ToolPreState | _ToolPostState


class _EmissionSidecar:
    """Hold trusted validation state only while one emission is in flight."""

    __slots__ = ("labels", "labels_invalid", "validation_state")

    def __init__(self) -> None:
        self.labels: list[str] = []
        self.labels_invalid = False
        self.validation_state: _ValidationState | None = None

    def begin(self, validation_state: _ValidationState | None) -> None:
        self.labels = []
        self.labels_invalid = False
        self.validation_state = validation_state

    def add_labels(self, labels: tuple[str, ...]) -> None:
        for label in labels:
            if label not in self.labels:
                self.labels.append(label)

    def clear(self) -> None:
        self.labels = []
        self.labels_invalid = False
        self.validation_state = None


def _validate_result_labels(labels: tuple[str, ...], *, max_items: int) -> None:
    if len(labels) > max_items:
        raise _AgentHooksValidationError
    for label in labels:
        if not label:
            raise _AgentHooksValidationError
        try:
            encoded = label.encode("utf-8")
        except UnicodeEncodeError:
            raise _AgentHooksValidationError from None
        if len(encoded) > _MAX_LABEL_BYTES:
            raise _AgentHooksValidationError


def _validate_model_pre_target(target: object, state: _ModelPreState) -> None:
    if state.unsupported:
        raise _AgentHooksUnsupportedError
    if not isinstance(target, list):
        raise _AgentHooksValidationError
    target_items = cast(list[object], target)
    if len(target_items) != len(state.entries):
        raise _AgentHooksValidationError

    for raw_semantic, entry in zip(target_items, state.entries, strict=True):
        if not isinstance(raw_semantic, dict):
            raise _AgentHooksValidationError
        semantic = cast(dict[str, object], raw_semantic)
        if set(semantic) != {"role", "content"} or semantic.get("role") != entry.role:
            raise _AgentHooksValidationError
        content = semantic.get("content")
        if entry.kind in {"instructions", "message", "assistant_message"}:
            if not isinstance(content, str):
                raise _AgentHooksValidationError
            continue
        if not isinstance(content, dict):
            raise _AgentHooksValidationError
        semantic_content = cast(dict[str, object], content)
        if entry.kind == "function_call":
            if set(semantic_content) != {"type", "call_id", "name", "arguments"}:
                raise _AgentHooksValidationError
            if (
                semantic_content.get("type") != "function_call"
                or semantic_content.get("call_id") != entry.call_id
                or semantic_content.get("name") != entry.name
                or not isinstance(semantic_content.get("arguments"), dict)
            ):
                raise _AgentHooksValidationError
            continue
        if set(semantic_content) != {"type", "call_id", "output"}:
            raise _AgentHooksValidationError
        if (
            semantic_content.get("type") != "function_call_output"
            or semantic_content.get("call_id") != entry.call_id
            or not isinstance(semantic_content.get("output"), str)
        ):
            raise _AgentHooksValidationError


def _validate_model_post_target(target: object, state: _ModelPostState) -> None:
    if state.context_invalid:
        raise _AgentHooksContextInvalidError
    if state.unsupported:
        raise _AgentHooksUnsupportedError
    if not isinstance(target, dict):
        raise _AgentHooksValidationError
    semantic = cast(dict[str, object], target)
    if set(semantic) != {"content", "tool_calls", "finish_reason"}:
        raise _AgentHooksValidationError
    if semantic.get("finish_reason") != state.finish_reason:
        raise _AgentHooksValidationError

    message_entries = [entry for entry in state.entries if entry.kind == "message"]
    content = semantic.get("content")
    if (message_entries and not isinstance(content, str)) or (
        not message_entries and content is not None
    ):
        raise _AgentHooksValidationError

    raw_tool_calls = semantic.get("tool_calls")
    tool_entries = [entry for entry in state.entries if entry.kind == "function_call"]
    if not isinstance(raw_tool_calls, list):
        raise _AgentHooksValidationError
    tool_call_items = cast(list[object], raw_tool_calls)
    if len(tool_call_items) != len(tool_entries):
        raise _AgentHooksValidationError
    for raw_call, entry in zip(tool_call_items, tool_entries, strict=True):
        if not isinstance(raw_call, dict):
            raise _AgentHooksValidationError
        call = cast(dict[str, object], raw_call)
        if set(call) != {"id", "name", "args"}:
            raise _AgentHooksValidationError
        if (
            call.get("id") != entry.call_id
            or call.get("name") != entry.name
            or not isinstance(call.get("args"), dict)
        ):
            raise _AgentHooksValidationError


def _project_tool_pre(
    *,
    tool: FunctionTool,
    tool_call: object,
    max_bytes: int,
    max_depth: int,
) -> tuple[dict[str, object], _ToolPreState]:
    from openai.types.responses import ResponseFunctionToolCall

    invocation_id = str(uuid4())
    params_json_schema = cast(dict[str, object], copy.deepcopy(tool.params_json_schema))
    if not isinstance(tool_call, ResponseFunctionToolCall):
        return {}, _ToolPreState(
            invocation_id=invocation_id,
            name=tool.name,
            params_json_schema=params_json_schema,
            unsupported=True,
        )
    arguments = _parse_bounded_json_object(
        tool_call.arguments,
        max_bytes=max_bytes,
        max_depth=max_depth,
    )
    if (
        arguments is None
        or tool_call.name != tool.name
        or not _is_bounded_identifier(tool_call.call_id)
    ):
        return {}, _ToolPreState(
            invocation_id=invocation_id,
            name=tool.name,
            params_json_schema=params_json_schema,
            unsupported=True,
        )
    return arguments, _ToolPreState(
        invocation_id=invocation_id,
        name=tool.name,
        params_json_schema=params_json_schema,
    )


def _parse_bounded_json_object(
    value: str,
    *,
    max_bytes: int,
    max_depth: int,
) -> dict[str, object]:
    from ..util._approvals import parse_function_tool_arguments

    try:
        _utf8_size(value, max_bytes=max_bytes)
    except _AgentHooksValidationError:
        raise _AgentHooksContextInvalidError from None

    depth = 0
    in_string = False
    escaped = False
    for character in value:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "[{":
            depth += 1
            if depth > max_depth:
                raise _AgentHooksContextInvalidError
        elif character in "]}" and depth > 0:
            depth -= 1

    try:
        parsed = parse_function_tool_arguments(value)
    except RecursionError:
        raise _AgentHooksContextInvalidError from None
    if parsed is None:
        raise _AgentHooksUnsupportedError
    try:
        _validate_json_value(parsed, max_bytes=max_bytes, max_depth=max_depth)
    except _AgentHooksValidationError:
        raise _AgentHooksContextInvalidError from None
    return cast(dict[str, object], parsed)


def _rebuild_tool_call(tool_call: object, arguments: dict[str, object]) -> object:
    from openai.types.responses import ResponseFunctionToolCall

    if not isinstance(tool_call, ResponseFunctionToolCall):
        raise _AgentHooksValidationError
    payload = tool_call.model_dump(exclude_unset=True)
    payload["arguments"] = json.dumps(
        arguments,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    try:
        return ResponseFunctionToolCall.model_validate(payload)
    except (TypeError, ValueError):
        raise _AgentHooksValidationError from None


def _validate_tool_pre_target(target: object, state: _ToolPreState) -> None:
    if state.unsupported:
        raise _AgentHooksUnsupportedError
    if not isinstance(target, dict):
        raise _AgentHooksValidationError
    _validate_tool_arguments_schema(
        cast(dict[str, object], target),
        schema=state.params_json_schema,
    )


def _validate_tool_arguments_schema(
    arguments: dict[str, object],
    *,
    schema: dict[str, object],
) -> None:
    from jsonschema import Draft202012Validator, SchemaError, ValidationError

    try:
        Draft202012Validator(schema).validate(arguments)
    except (SchemaError, ValidationError):
        raise _AgentHooksValidationError from None


def _resolve_local_schema_reference(root: dict[str, object], reference: str) -> object:
    if not reference.startswith("#/") or "%" in reference:
        raise _AgentHooksValidationError
    current: object = root
    for raw_token in reference[2:].split("/"):
        if re.search(r"~(?![01])", raw_token):
            raise _AgentHooksValidationError
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if type(current) is dict:
            mapping = cast(dict[str, object], current)
            if token not in mapping:
                raise _AgentHooksValidationError
            current = mapping[token]
        elif type(current) is list and token.isdigit():
            values = cast(list[object], current)
            index = int(token)
            if index >= len(values):
                raise _AgentHooksValidationError
            current = values[index]
        else:
            raise _AgentHooksValidationError
    return current


def _validate_schema_profile(schema: dict[str, object]) -> None:
    graph: dict[int, list[int]] = {}
    visited: set[int] = set()
    pending = [schema]
    while pending:
        node = pending.pop()
        node_id = id(node)
        if node_id in visited:
            continue
        visited.add(node_id)
        if set(node) - _SAFE_SCHEMA_KEYWORDS:
            raise _AgentHooksValidationError
        children: list[dict[str, object]] = []
        for keyword in _SCHEMA_MAP_KEYWORDS:
            raw_mapping = node.get(keyword)
            if raw_mapping is None:
                continue
            if type(raw_mapping) is not dict:
                raise _AgentHooksValidationError
            for child in cast(dict[str, object], raw_mapping).values():
                if type(child) is not dict:
                    raise _AgentHooksValidationError
                children.append(cast(dict[str, object], child))
        for keyword in _SCHEMA_SINGLE_KEYWORDS:
            child = node.get(keyword)
            if child is None or type(child) is bool:
                continue
            if type(child) is not dict:
                raise _AgentHooksValidationError
            children.append(cast(dict[str, object], child))
        prefix_items = node.get("prefixItems")
        if prefix_items is not None:
            if type(prefix_items) is not list:
                raise _AgentHooksValidationError
            for child in cast(list[object], prefix_items):
                if type(child) is not dict:
                    raise _AgentHooksValidationError
                children.append(cast(dict[str, object], child))
        reference = node.get("$ref")
        if reference is not None:
            if type(reference) is not str:
                raise _AgentHooksValidationError
            target = _resolve_local_schema_reference(schema, reference)
            if type(target) is not dict:
                raise _AgentHooksValidationError
            children.append(cast(dict[str, object], target))
        graph[node_id] = [id(child) for child in children]
        pending.extend(children)

    colors: dict[int, int] = {}
    for root_id in graph:
        if colors.get(root_id, 0) != 0:
            continue
        stack: list[tuple[int, bool]] = [(root_id, False)]
        while stack:
            node_id, exiting = stack.pop()
            if exiting:
                colors[node_id] = 2
                continue
            if colors.get(node_id, 0) == 1:
                raise _AgentHooksValidationError
            if colors.get(node_id, 0) == 2:
                continue
            colors[node_id] = 1
            stack.append((node_id, True))
            for child_id in reversed(graph.get(node_id, [])):
                if colors.get(child_id, 0) == 1:
                    raise _AgentHooksValidationError
                if colors.get(child_id, 0) == 0:
                    stack.append((child_id, False))


def _validate_tool_schema(
    schema: dict[str, object],
    *,
    max_bytes: int,
    max_depth: int,
) -> None:
    from jsonschema import Draft202012Validator
    from jsonschema.exceptions import SchemaError

    try:
        _validate_json_value(
            schema,
            max_bytes=max_bytes,
            max_depth=max_depth,
            require_exact_types=True,
        )
        _validate_schema_profile(schema)
        Draft202012Validator.check_schema(schema)
    except (SchemaError, _AgentHooksValidationError):
        raise _AgentHooksValidationError from None


def _validate_tool_post_target(
    target: object,
    state: _ToolPostState,
    context: AgentContext,
) -> None:
    if state.context_invalid:
        raise _AgentHooksContextInvalidError
    if state.unsupported:
        raise _AgentHooksUnsupportedError
    if not isinstance(target, str):
        raise _AgentHooksValidationError
    raw_tool_call = context.get("tool_call")
    if not isinstance(raw_tool_call, dict):
        raise _AgentHooksValidationError
    tool_call = cast(dict[str, object], raw_tool_call)
    if set(tool_call) != {"id", "name", "args"}:
        raise _AgentHooksValidationError
    if (
        tool_call.get("id") != state.invocation_id
        or tool_call.get("name") != state.name
        or tool_call.get("args") != state.arguments
    ):
        raise _AgentHooksValidationError
    raw_tool_result = context.get("tool_result")
    if not isinstance(raw_tool_result, dict):
        raise _AgentHooksValidationError
    tool_result = cast(dict[str, object], raw_tool_result)
    if set(tool_result) != {"value", "is_error", "duration_ms"}:
        raise _AgentHooksValidationError
    duration_ms = tool_result.get("duration_ms")
    if (
        tool_result.get("value") != target
        or tool_result.get("is_error") is not state.is_error
        or isinstance(duration_ms, bool)
        or not isinstance(duration_ms, int | float)
        or not math.isfinite(duration_ms)
        or duration_ms < 0
    ):
        raise _AgentHooksValidationError


def _bounded_blocked_message(reason: str | None) -> str:
    if reason is not None and reason.startswith("host_error:"):
        return "blocked: governance_error"
    return "blocked: policy_denied"


def _validate_json_value(
    value: object,
    *,
    max_bytes: int,
    max_depth: int,
    require_exact_types: bool = False,
) -> None:
    stack: list[tuple[object, int, bool]] = [(value, 1, False)]
    active_container_ids: set[int] = set()
    encoded_size = 0

    def add_size(increment: int) -> None:
        nonlocal encoded_size
        if increment > max_bytes - encoded_size:
            raise _AgentHooksValidationError("JSON value exceeds the configured byte limit")
        encoded_size += increment

    while stack:
        current, depth, exiting = stack.pop()
        if exiting:
            active_container_ids.remove(id(current))
            continue
        if depth > max_depth:
            raise _AgentHooksValidationError("JSON value exceeds the configured depth limit")
        if current is None:
            add_size(4)
            continue
        if (type(current) is str) if require_exact_types else isinstance(current, str):
            add_size(_json_string_size(cast(str, current), max_bytes=max_bytes - encoded_size))
            continue
        if (type(current) is bool) if require_exact_types else isinstance(current, bool):
            add_size(4 if cast(bool, current) else 5)
            continue
        if (type(current) is int) if require_exact_types else isinstance(current, int):
            integer_value = cast(int, current)
            remaining = max_bytes - encoded_size
            if abs(integer_value).bit_length() > max(remaining, 1) * 4:
                raise _AgentHooksValidationError("JSON value exceeds the configured byte limit")
            try:
                integer_text = str(integer_value)
            except (RecursionError, ValueError):
                raise _AgentHooksValidationError("JSON integer is unsupported") from None
            add_size(len(integer_text))
            continue
        if (type(current) is float) if require_exact_types else isinstance(current, float):
            float_value = cast(float, current)
            if not math.isfinite(float_value):
                raise _AgentHooksValidationError("JSON value contains a non-finite number")
            add_size(len(repr(float_value)))
            continue
        is_dict = type(current) is dict if require_exact_types else isinstance(current, dict)
        is_list = type(current) is list if require_exact_types else isinstance(current, list)
        if not is_dict and not is_list:
            raise _AgentHooksValidationError("JSON value contains an unsupported value type")

        container = current
        container_id = id(container)
        if container_id in active_container_ids:
            raise _AgentHooksValidationError("JSON value contains a cycle")
        active_container_ids.add(container_id)
        stack.append((container, depth, True))

        if is_dict:
            current_dict = cast(dict[object, object], current)
            if not all(
                (type(key) is str) if require_exact_types else isinstance(key, str)
                for key in current_dict
            ):
                raise _AgentHooksValidationError("JSON object keys must be strings")
            add_size(2 + max(0, len(current_dict) - 1) + len(current_dict))
            for key in current_dict:
                add_size(
                    _json_string_size(
                        cast(str, key),
                        max_bytes=max_bytes - encoded_size,
                    )
                )
            stack.extend((item, depth + 1, False) for item in current_dict.values())
        else:
            current_list = cast(list[object], current)
            add_size(2 + max(0, len(current_list) - 1))
            stack.extend((item, depth + 1, False) for item in current_list)


def _utf8_size(value: str, *, max_bytes: int) -> int:
    size = 0
    for character in value:
        codepoint = ord(character)
        if codepoint <= 0x7F:
            increment = 1
        elif codepoint <= 0x7FF:
            increment = 2
        elif 0xD800 <= codepoint <= 0xDFFF:
            raise _AgentHooksValidationError("Text contains an invalid Unicode surrogate")
        elif codepoint <= 0xFFFF:
            increment = 3
        else:
            increment = 4
        if increment > max_bytes - size:
            raise _AgentHooksValidationError("Text exceeds the configured byte limit")
        size += increment
    return size


def _json_string_size(value: str, *, max_bytes: int) -> int:
    size = 2
    if size > max_bytes:
        raise _AgentHooksValidationError("JSON value exceeds the configured byte limit")
    for character in value:
        codepoint = ord(character)
        if character in {'"', "\\"} or character in {"\b", "\f", "\n", "\r", "\t"}:
            increment = 2
        elif codepoint < 0x20:
            increment = 6
        elif codepoint <= 0x7F:
            increment = 1
        elif codepoint <= 0x7FF:
            increment = 2
        elif 0xD800 <= codepoint <= 0xDFFF:
            raise _AgentHooksValidationError("JSON value contains an invalid Unicode surrogate")
        elif codepoint <= 0xFFFF:
            increment = 3
        else:
            increment = 4
        if increment > max_bytes - size:
            raise _AgentHooksValidationError("JSON value exceeds the configured byte limit")
        size += increment
    return size


def _is_bounded_identifier(value: object) -> TypeGuard[str]:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        _utf8_size(value, max_bytes=_MAX_IDENTIFIER_BYTES)
        return True
    except _AgentHooksValidationError:
        return False


def _is_nonnegative_int(value: object) -> TypeGuard[int]:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _derive_model_id(model: Model) -> str:
    try:
        configured_model = getattr(model, "model", None)
    except Exception:
        configured_model = None
    if _is_bounded_identifier(configured_model):
        return configured_model
    class_name = type(model).__name__
    if _is_bounded_identifier(class_name):
        return class_name
    bounded_name = class_name.encode("utf-8")[:_MAX_IDENTIFIER_BYTES].decode(
        "utf-8", errors="ignore"
    )
    return bounded_name or "Model"


def _validate_response_input_payload(payload: dict[str, object]) -> None:
    from openai.types.responses import ResponseInputItemParam
    from pydantic import TypeAdapter, ValidationError

    try:
        TypeAdapter(ResponseInputItemParam).validate_python(payload)
    except (TypeError, ValueError, ValidationError):
        raise _AgentHooksUnsupportedError from None


def _extract_plain_text_message(message: object) -> tuple[ResponseOutputMessage, str]:
    from openai.types.responses import ResponseOutputMessage, ResponseOutputText

    if type(message) is not ResponseOutputMessage:
        raise _AgentHooksUnsupportedError
    typed_message = message
    if (
        typed_message.role != "assistant"
        or typed_message.status != "completed"
        or typed_message.phase is not None
        or not _has_only_provider_data_extra(typed_message.model_extra)
        or type(typed_message.content) is not list
        or len(typed_message.content) != 1
    ):
        raise _AgentHooksUnsupportedError
    content_part = typed_message.content[0]
    if type(content_part) is not ResponseOutputText:
        raise _AgentHooksUnsupportedError
    typed_content = content_part
    if (
        not isinstance(typed_content.text, str)
        or type(typed_content.annotations) is not list
        or typed_content.annotations
        or (
            typed_content.logprobs is not None
            and (type(typed_content.logprobs) is not list or typed_content.logprobs)
        )
        or typed_content.model_extra
    ):
        raise _AgentHooksUnsupportedError
    return typed_message, typed_content.text


def _has_only_provider_data_extra(model_extra: object) -> bool:
    if model_extra is None:
        return True
    if type(model_extra) is not dict:
        return False
    keys = tuple(cast(dict[object, object], model_extra))
    return not keys or (len(keys) == 1 and type(keys[0]) is str and keys[0] == "provider_data")


def _validate_function_call_metadata(tool_call: object) -> ResponseFunctionToolCall:
    from openai.types.responses import ResponseFunctionToolCall

    if type(tool_call) is not ResponseFunctionToolCall:
        raise _AgentHooksUnsupportedError
    typed_call = tool_call
    if (
        typed_call.caller is not None
        or typed_call.namespace is not None
        or not _has_only_provider_data_extra(typed_call.model_extra)
        or typed_call.status not in {None, "completed"}
    ):
        raise _AgentHooksUnsupportedError
    return typed_call


def _project_model_pre(
    instructions: object,
    input_items: list[object],
    *,
    tool_names: tuple[str, ...],
) -> tuple[list[dict[str, object]], _ModelPreState]:
    from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage

    from ..util._approvals import parse_function_tool_arguments

    messages: list[dict[str, object]] = []
    entries: list[_ModelPreEntry] = []
    seen_call_ids: set[str] = set()

    if instructions is not None:
        if not isinstance(instructions, str):
            raise _AgentHooksUnsupportedError
        messages.append({"role": "system", "content": instructions})
        entries.append(_ModelPreEntry(kind="instructions", role="system", original=None))

    for input_item in input_items:
        if not isinstance(input_item, dict):
            raise _AgentHooksUnsupportedError
        payload = copy.deepcopy(cast(dict[str, object], input_item))
        item_type = payload.get("type")
        role = payload.get("role")

        if item_type in {None, "message"} and role in {"user", "system", "developer"}:
            content = payload.get("content")
            if not isinstance(content, str):
                raise _AgentHooksUnsupportedError
            _validate_response_input_payload(payload)
            role_value = cast(str, role)
            messages.append({"role": role_value, "content": content})
            entries.append(_ModelPreEntry(kind="message", role=role_value, original=payload))
            continue

        if item_type == "message" and role == "assistant":
            try:
                message, content = _extract_plain_text_message(
                    ResponseOutputMessage.model_validate(payload)
                )
            except (TypeError, ValueError):
                raise _AgentHooksUnsupportedError from None
            messages.append({"role": "assistant", "content": content})
            entries.append(
                _ModelPreEntry(
                    kind="assistant_message",
                    role="assistant",
                    original=message.model_copy(deep=True),
                )
            )
            continue

        if item_type == "function_call":
            try:
                tool_call = _validate_function_call_metadata(
                    ResponseFunctionToolCall.model_validate(payload)
                )
            except (TypeError, ValueError):
                raise _AgentHooksUnsupportedError from None
            call_id = tool_call.call_id
            name = tool_call.name
            arguments = parse_function_tool_arguments(tool_call.arguments)
            if (
                not _is_bounded_identifier(call_id)
                or not _is_bounded_identifier(name)
                or call_id in seen_call_ids
                or name not in tool_names
                or arguments is None
            ):
                raise _AgentHooksUnsupportedError
            seen_call_ids.add(call_id)
            messages.append(
                {
                    "role": "assistant",
                    "content": {
                        "type": "function_call",
                        "call_id": call_id,
                        "name": name,
                        "arguments": arguments,
                    },
                }
            )
            entries.append(
                _ModelPreEntry(
                    kind="function_call",
                    role="assistant",
                    original=tool_call.model_copy(deep=True),
                    call_id=call_id,
                    name=name,
                )
            )
            continue

        if item_type == "function_call_output":
            call_output_id = payload.get("call_id")
            output = payload.get("output")
            if (
                not _is_bounded_identifier(call_output_id)
                or call_output_id not in seen_call_ids
                or not isinstance(output, str)
            ):
                raise _AgentHooksUnsupportedError
            _validate_response_input_payload(payload)
            messages.append(
                {
                    "role": "tool",
                    "content": {
                        "type": "function_call_output",
                        "call_id": call_output_id,
                        "output": output,
                    },
                }
            )
            entries.append(
                _ModelPreEntry(
                    kind="function_call_output",
                    role="tool",
                    original=payload,
                    call_id=call_output_id,
                )
            )
            continue

        raise _AgentHooksUnsupportedError

    return messages, _ModelPreState(entries=tuple(entries))


def _rebuild_model_pre(
    target: object,
    state: _ModelPreState,
) -> tuple[str | None, list[TResponseInputItem]]:
    from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage

    if not isinstance(target, list):
        raise _AgentHooksValidationError
    target_items = cast(list[object], target)
    instructions: str | None = None
    rebuilt_input: list[TResponseInputItem] = []

    for raw_semantic, entry in zip(target_items, state.entries, strict=True):
        if not isinstance(raw_semantic, dict):
            raise _AgentHooksValidationError
        semantic = cast(dict[str, object], raw_semantic)
        content = semantic.get("content")
        if entry.kind == "instructions":
            instructions = cast(str, content)
            continue
        if entry.kind == "message":
            payload = cast(dict[str, object], copy.deepcopy(entry.original))
            payload["content"] = cast(str, content)
            _validate_response_input_payload(payload)
            rebuilt_input.append(cast(TResponseInputItem, payload))
            continue
        if entry.kind == "assistant_message":
            original_message = cast(ResponseOutputMessage, entry.original)
            payload = original_message.model_dump(exclude_unset=True)
            content_payload = original_message.content[0].model_dump(exclude_unset=True)
            content_payload["text"] = cast(str, content)
            payload["content"] = [content_payload]
            rebuilt_message = ResponseOutputMessage.model_validate(payload)
            rebuilt_input.append(
                cast(TResponseInputItem, rebuilt_message.model_dump(exclude_unset=True))
            )
            continue
        if entry.kind == "function_call":
            original_call = cast(ResponseFunctionToolCall, entry.original)
            semantic_content = cast(dict[str, object], content)
            payload = original_call.model_dump(exclude_unset=True)
            payload["arguments"] = json.dumps(
                semantic_content["arguments"],
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            rebuilt_call = ResponseFunctionToolCall.model_validate(payload)
            rebuilt_input.append(
                cast(TResponseInputItem, rebuilt_call.model_dump(exclude_unset=True))
            )
            continue
        payload = cast(dict[str, object], copy.deepcopy(entry.original))
        semantic_content = cast(dict[str, object], content)
        payload["output"] = cast(str, semantic_content["output"])
        _validate_response_input_payload(payload)
        rebuilt_input.append(cast(TResponseInputItem, payload))

    return instructions, rebuilt_input


def _project_model_post(
    response: ModelResponse,
    *,
    detached_usage: Usage,
    tool_names: tuple[str, ...],
    max_tool_calls: int,
    max_context_bytes: int,
    max_context_depth: int,
) -> tuple[dict[str, object], _ModelPostState]:
    from openai.types.responses import (
        ResponseFunctionToolCall,
        ResponseOutputMessage,
        ResponseOutputText,
    )

    if type(response.output) is not list or len(response.output) > max_tool_calls + 1:
        raise _AgentHooksUnsupportedError
    for response_identifier in (response.response_id, response.request_id):
        if response_identifier is not None and not _is_bounded_identifier(response_identifier):
            raise _AgentHooksUnsupportedError

    validated_output: list[ResponseOutputMessage | ResponseFunctionToolCall] = []
    for output_item in response.output:
        if type(output_item) is ResponseOutputMessage:
            message, _ = _extract_plain_text_message(output_item)
            validated_output.append(message)
        elif type(output_item) is ResponseFunctionToolCall:
            validated_output.append(_validate_function_call_metadata(output_item))
        else:
            raise _AgentHooksUnsupportedError
    if sum(type(item) is ResponseFunctionToolCall for item in validated_output) > max_tool_calls:
        raise _AgentHooksUnsupportedError

    remaining_context_bytes = max_context_bytes
    try:
        for output_item in validated_output:
            if type(output_item) is ResponseOutputMessage:
                content_part = cast(ResponseOutputText, output_item.content[0])
                remaining_context_bytes -= _utf8_size(
                    content_part.text,
                    max_bytes=remaining_context_bytes,
                )
            else:
                tool_call = cast(ResponseFunctionToolCall, output_item)
                for semantic_value in (
                    tool_call.arguments,
                    tool_call.call_id,
                    tool_call.name,
                ):
                    remaining_context_bytes -= _utf8_size(
                        semantic_value,
                        max_bytes=remaining_context_bytes,
                    )
    except _AgentHooksValidationError:
        raise _AgentHooksContextInvalidError from None

    entries: list[_ModelPostEntry] = []
    tool_calls: list[dict[str, object]] = []
    content: str | None = None
    message_seen = False
    item_ids: set[str] = set()
    call_ids: set[str] = set()

    for output_item in validated_output:
        item_id: object = output_item.id
        if not _is_bounded_identifier(item_id) or item_id in item_ids:
            raise _AgentHooksUnsupportedError
        item_ids.add(item_id)

        if type(output_item) is ResponseOutputMessage:
            message, output_text = _extract_plain_text_message(output_item)
            if message_seen:
                raise _AgentHooksUnsupportedError
            message_seen = True
            content = output_text
            entries.append(
                _ModelPostEntry(
                    kind="message",
                    item_id=message.id,
                )
            )
            continue

        if type(output_item) is ResponseFunctionToolCall:
            tool_call = _validate_function_call_metadata(output_item)
            call_id = tool_call.call_id
            name = tool_call.name
            arguments = _parse_bounded_json_object(
                tool_call.arguments,
                max_bytes=max_context_bytes,
                max_depth=max_context_depth,
            )
            if (
                not _is_bounded_identifier(call_id)
                or call_id in call_ids
                or not _is_bounded_identifier(name)
                or name not in tool_names
                or arguments is None
            ):
                raise _AgentHooksUnsupportedError
            call_ids.add(call_id)
            tool_calls.append({"id": call_id, "name": name, "args": arguments})
            entries.append(
                _ModelPostEntry(
                    kind="function_call",
                    item_id=item_id,
                    call_id=call_id,
                    name=name,
                    status_completed=tool_call.status == "completed",
                )
            )
            continue

        raise _AgentHooksUnsupportedError

    if len(tool_calls) > max_tool_calls:
        raise _AgentHooksUnsupportedError
    finish_reason = "tool_calls" if tool_calls else "stop"
    target: dict[str, object] = {
        "content": content,
        "tool_calls": tool_calls,
        "finish_reason": finish_reason,
    }
    return target, _ModelPostState(
        usage=detached_usage,
        response_id=response.response_id,
        request_id=response.request_id,
        entries=tuple(entries),
        finish_reason=finish_reason,
    )


def _rebuild_model_post(target: object, state: _ModelPostState) -> ModelResponse:
    from openai.types.responses import (
        ResponseFunctionToolCall,
        ResponseOutputMessage,
        ResponseOutputText,
    )

    from ..items import ModelResponse

    if not isinstance(target, dict):
        raise _AgentHooksValidationError
    semantic = cast(dict[str, object], target)
    transformed_content = semantic.get("content")
    transformed_calls = cast(list[dict[str, object]], semantic.get("tool_calls"))
    call_index = 0
    rebuilt_output: list[TResponseOutputItem] = []

    for entry in state.entries:
        if entry.kind == "message":
            rebuilt_output.append(
                ResponseOutputMessage(
                    id=entry.item_id,
                    content=[
                        ResponseOutputText(
                            annotations=[],
                            text=cast(str, transformed_content),
                            type="output_text",
                        )
                    ],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            )
            continue

        semantic_call = transformed_calls[call_index]
        call_index += 1
        if entry.call_id is None or entry.name is None:
            raise _AgentHooksValidationError
        rebuilt_output.append(
            ResponseFunctionToolCall(
                arguments=json.dumps(
                    semantic_call["args"],
                    allow_nan=False,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                call_id=entry.call_id,
                id=entry.item_id,
                name=entry.name,
                status="completed" if entry.status_completed else None,
                type="function_call",
            )
        )

    return ModelResponse(
        output=rebuilt_output,
        usage=state.usage,
        response_id=state.response_id,
        request_id=state.request_id,
    )


def _validate_verdict(
    verdict: Verdict,
    *,
    max_bytes: int,
    max_depth: int,
    max_collection_items: int,
) -> None:
    if verdict.message is not None or verdict.evidence is not None or verdict.approval is not None:
        raise _AgentHooksValidationError("Free-form verdict metadata is unsupported")
    _validate_opaque_code(verdict.reason)
    if len(verdict.warnings) > max_collection_items:
        raise _AgentHooksValidationError("Verdict has too many warnings")
    for warning in verdict.warnings:
        if warning.message is not None:
            raise _AgentHooksValidationError("Free-form warning messages are unsupported")
        _validate_opaque_code(warning.reason)
    for label in verdict.result_labels:
        _validate_opaque_code(label)
    if verdict.transform is not None:
        _validate_transform_path(verdict.transform.path)
    _validate_json_value(verdict.to_wire(), max_bytes=max_bytes, max_depth=max_depth)


def _validate_opaque_code(value: str | None) -> None:
    if value is None:
        return
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError:
        raise _AgentHooksValidationError from None
    if len(encoded) > _MAX_IDENTIFIER_BYTES or _OPAQUE_CODE_PATTERN.fullmatch(value) is None:
        raise _AgentHooksValidationError


def _validate_transform_path(path: str) -> None:
    try:
        encoded = path.encode("utf-8")
    except UnicodeEncodeError:
        raise _AgentHooksValidationError from None
    if len(encoded) > _MAX_IDENTIFIER_BYTES or _SAFE_TRANSFORM_PATH_PATTERN.fullmatch(path) is None:
        raise _AgentHooksValidationError


def _normalize_host_validator_record(record: InterceptionRecord) -> InterceptionRecord:
    if not record.verdicts:
        return record
    validator_index = record.interceptors_registered - 1
    validator_summary = record.verdicts[-1]
    if validator_summary.index != validator_index or validator_summary.name != _HOST_VALIDATOR_NAME:
        return record

    from agent_hooks import HostError, Verdict

    host_errors = {
        _HOST_TRANSFORM_INVALID: HostError.TRANSFORM_INVALID,
        _HOST_ADAPTER_UNSUPPORTED: HostError.ADAPTER_UNSUPPORTED,
        _HOST_CONTEXT_INVALID: HostError.CONTEXT_INVALID,
    }
    validator_reason = validator_summary.reason
    if validator_reason is None:
        return record
    host_error = host_errors.get(validator_reason)
    if host_error is None:
        return record

    verdicts = (*record.verdicts[:-1], replace(validator_summary, reason=host_error.value))
    return replace(
        record,
        verdict=Verdict.host_error(host_error),
        verdicts=verdicts,
        decided_by=None,
    )


def _sanitize_interception_record(record: InterceptionRecord) -> InterceptionRecord:
    from agent_hooks import Transform, Verdict, Warning

    verdict = record.verdict
    _validate_opaque_code(verdict.reason)
    warnings = tuple(Warning(reason=warning.reason) for warning in verdict.warnings)
    for warning in warnings:
        _validate_opaque_code(warning.reason)
    for label in verdict.result_labels:
        _validate_opaque_code(label)
    transform = None
    if verdict.transform is not None:
        _validate_transform_path(verdict.transform.path)
        transform = Transform(path=verdict.transform.path, value=None)

    sanitized = object.__new__(Verdict)
    object.__setattr__(sanitized, "decision", verdict.decision)
    object.__setattr__(sanitized, "reason", verdict.reason)
    object.__setattr__(sanitized, "message", None)
    object.__setattr__(sanitized, "warnings", warnings)
    object.__setattr__(sanitized, "approval", None)
    object.__setattr__(sanitized, "transform", transform)
    object.__setattr__(sanitized, "evidence", None)
    object.__setattr__(sanitized, "result_labels", verdict.result_labels)
    return replace(record, verdict=sanitized)


def _normalize_interception_record(record: InterceptionRecord) -> InterceptionRecord:
    return _sanitize_interception_record(_normalize_host_validator_record(record))


class _AsyncInterceptorBridge:
    """Bridge the SDK's sync return annotation to the required async callback."""

    __slots__ = (
        "_callback",
        "_max_collection_items",
        "_max_depth",
        "_max_verdict_bytes",
        "_sidecar",
        "_verdict_type",
    )

    def __init__(
        self,
        callback: AsyncInterceptor,
        *,
        verdict_type: type[Verdict],
        max_verdict_bytes: int,
        max_depth: int,
        max_collection_items: int,
        sidecar: _EmissionSidecar,
    ) -> None:
        self._callback = callback
        self._verdict_type = verdict_type
        self._max_verdict_bytes = max_verdict_bytes
        self._max_depth = max_depth
        self._max_collection_items = max_collection_items
        self._sidecar = sidecar

    async def intercept(self, context: AgentContext, /) -> Verdict:
        verdict = await self._callback.intercept(context)
        if not isinstance(verdict, self._verdict_type):
            raise _AgentHooksValidationError("Interceptor returned an unsupported verdict type")
        try:
            _validate_result_labels(
                verdict.result_labels,
                max_items=self._max_collection_items,
            )
        except _AgentHooksValidationError:
            self._sidecar.labels_invalid = True
            from agent_hooks import ALLOW

            return ALLOW
        _validate_verdict(
            verdict,
            max_bytes=self._max_verdict_bytes,
            max_depth=self._max_depth,
            max_collection_items=self._max_collection_items,
        )
        if verdict.decision.permits:
            self._sidecar.add_labels(verdict.result_labels)
        return verdict


class _AuditSink:
    __slots__ = ("_failure", "_sink")

    def __init__(self, sink: _RecordSinkReservation) -> None:
        self._sink: _RecordSinkReservation | None = sink
        self._failure: tuple[int, str] | None = None

    def begin_emission(self) -> None:
        self._failure = None

    def __call__(self, record: InterceptionRecord, /) -> None:
        sink = self._sink
        if sink is None:
            return
        sequence = record.sequence
        try:
            record = _normalize_interception_record(record)
            sink.write(record)
        except Exception as error:
            self._failure = (sequence, type(error).__name__)

    def take_failure(self) -> tuple[int, str] | None:
        failure = self._failure
        self._failure = None
        return failure

    def release(self) -> None:
        self._failure = None
        if self._sink is not None:
            self._sink.release()
        self._sink = None

    @property
    def released(self) -> bool:
        return self._sink is None and self._failure is None


class AgentHooksRunSession:
    """Own one bounded Agent Hooks protocol session for one top-level run."""

    __slots__ = (
        "_audit_sink",
        "_builder",
        "_builder_factory",
        "_closed",
        "_emitter",
        "_max_context_bytes",
        "_max_context_depth",
        "_max_result_labels",
        "_max_tool_calls_per_turn",
        "_max_tool_calls_total",
        "_next_sequence",
        "_opened",
        "_sidecar",
        "_source_labels",
        "_tool_count",
        "_tool_names",
    )

    def __init__(
        self,
        *,
        builder: AgentContextBuilder,
        builder_factory: Callable[[], AgentContextBuilder],
        emitter: InterceptionEmitter,
        audit_sink: _AuditSink,
        limits: AgentHooksLimits,
        tool_names: tuple[str, ...],
        sidecar: _EmissionSidecar,
        max_turns: int,
    ) -> None:
        self._builder: AgentContextBuilder | None = builder
        self._builder_factory: Callable[[], AgentContextBuilder] | None = builder_factory
        self._emitter: InterceptionEmitter | None = emitter
        self._audit_sink = audit_sink
        self._max_context_bytes = limits.max_context_bytes
        self._max_context_depth = limits.max_context_depth
        self._max_result_labels = _MAX_VERDICT_COLLECTION_ITEMS
        self._max_tool_calls_per_turn = limits.max_tool_calls_per_turn
        self._max_tool_calls_total = max_turns * limits.max_tool_calls_per_turn
        self._next_sequence = 0
        self._sidecar = sidecar
        self._source_labels: list[str] = []
        self._tool_count = 0
        self._tool_names = tool_names
        self._opened = False
        self._closed = False

    async def open(self) -> None:
        if self._opened:
            raise RuntimeError("Agent Hooks session has already been opened")
        self._opened = True
        await self._emit_guarded(
            point="agent_startup",
            context=self._build_context(
                point="agent_startup",
                build=lambda builder: builder.agent_startup(
                    tools_registered=list(self._tool_names)
                ),
            ),
        )

    async def emit_input(self, content: str) -> str:
        target = await self._emit_guarded(
            point="input",
            context=self._build_context(
                point="input",
                build=lambda builder: builder.input(content=content),
            ),
        )
        return self._extract_text_target(target, point="input", require_role=True)

    async def emit_output(self, content: object) -> str:
        if not isinstance(content, str):
            raise self._blocked_error(
                point="output",
                reason="host_error:adapter_unsupported",
                sequence=None,
            )
        target = await self._emit_guarded(
            point="output",
            context=self._build_context(
                point="output",
                build=lambda builder: builder.output(content=content),
            ),
        )
        return self._extract_text_target(target, point="output", require_role=False)

    def new_model_call(self, model: Model) -> _ModelCall:
        """Create trusted correlation metadata for one model pre/post pair."""
        return _ModelCall(model_id=_derive_model_id(model), request_id=str(uuid4()))

    async def emit_pre_model(
        self,
        *,
        call: _ModelCall,
        instructions: str | None,
        input_items: list[TResponseInputItem],
    ) -> tuple[str | None, list[TResponseInputItem]]:
        try:
            messages, state = _project_model_pre(
                instructions,
                cast(list[object], input_items),
                tool_names=self._tool_names,
            )
        except _AgentHooksUnsupportedError:
            messages = []
            state = _ModelPreState(entries=(), unsupported=True)
        target = await self._emit_guarded(
            point="pre_model_call",
            context=self._build_context(
                point="pre_model_call",
                build=lambda builder: builder.pre_model_call(
                    model_id=call.model_id,
                    messages=messages,
                    request_id=call.request_id,
                ),
            ),
            validation_state=state,
        )
        return _rebuild_model_pre(target, state)

    async def emit_post_model(
        self,
        *,
        call: _ModelCall,
        response: ModelResponse,
    ) -> ModelResponse:
        detached_usage = self._copy_valid_usage(response.usage)
        usage_supported = True
        try:
            usage = self._project_usage(response)
        except _AgentHooksUnsupportedError:
            usage = None
            usage_supported = False
        try:
            target, state = _project_model_post(
                response,
                detached_usage=detached_usage,
                tool_names=self._tool_names,
                max_tool_calls=self._max_tool_calls_per_turn,
                max_context_bytes=self._max_context_bytes,
                max_context_depth=self._max_context_depth,
            )
        except _AgentHooksContextInvalidError:
            target = cast(
                dict[str, object],
                {"content": None, "tool_calls": [], "finish_reason": "error"},
            )
            state = _ModelPostState(
                usage=detached_usage,
                response_id=None,
                request_id=None,
                entries=(),
                finish_reason="error",
                context_invalid=True,
            )
        except _AgentHooksUnsupportedError:
            target = cast(
                dict[str, object],
                {"content": None, "tool_calls": [], "finish_reason": "stop"},
            )
            state = _ModelPostState(
                usage=detached_usage,
                response_id=None,
                request_id=None,
                entries=(),
                finish_reason="stop",
                unsupported=True,
            )
        if not usage_supported and not state.unsupported:
            state = replace(state, unsupported=True)
        try:
            context = self._build_context(
                point="post_model_call",
                build=lambda builder: builder.post_model_call(
                    model_id=call.model_id,
                    content=target["content"],
                    tool_calls=cast(list[dict[str, object]], target["tool_calls"]),
                    finish_reason=cast(str, target["finish_reason"]),
                    usage=usage,
                    request_id=call.request_id,
                ),
            )
        except Exception as error:
            if not isinstance(error, AgentHooksBlockedError) or error.sequence is not None:
                raise
            record = await self._emit_post_model_context_invalid(call=call)
            raise self._blocked_error(
                point="post_model_call",
                reason="host_error:context_invalid",
                sequence=record.sequence,
            ) from None
        folded_target = await self._emit_guarded(
            point="post_model_call",
            context=context,
            validation_state=state,
        )
        return _rebuild_model_post(folded_target, state)

    async def prepare_model_usage(
        self,
        *,
        call: _ModelCall,
        response: ModelResponse,
    ) -> Usage:
        """Validate and detach untrusted provider usage before run accounting."""
        try:
            if type(response) is not ModelResponse:
                raise _AgentHooksValidationError
            usage = self._copy_valid_usage(response.usage)
            response.usage = usage
        except asyncio.CancelledError:
            raise
        except Exception:
            record = await self._emit_post_model_context_invalid(call=call)
            raise self._blocked_error(
                point="post_model_call",
                reason="host_error:context_invalid",
                sequence=record.sequence,
            ) from None
        return usage

    @staticmethod
    def _copy_valid_usage(usage: object) -> Usage:
        try:
            if type(usage) is not Usage:
                raise _AgentHooksValidationError
            input_details = usage.input_tokens_details
            output_details = usage.output_tokens_details
            entries = usage.request_usage_entries
            if (
                type(input_details) is not InputTokensDetails
                or type(output_details) is not OutputTokensDetails
                or type(entries) is not list
            ):
                raise _AgentHooksValidationError

            def validated_count(value: object) -> int:
                if type(value) is not int or value < 0 or value > 2**63 - 1:
                    raise _AgentHooksValidationError
                return value

            requests = validated_count(usage.requests)
            input_tokens = validated_count(usage.input_tokens)
            output_tokens = validated_count(usage.output_tokens)
            total_tokens = validated_count(usage.total_tokens)
            cached_tokens = validated_count(input_details.cached_tokens)
            cache_write_tokens = validated_count(getattr(input_details, "cache_write_tokens", 0))
            reasoning_tokens = validated_count(output_details.reasoning_tokens)
            if len(entries) > 1:
                raise _AgentHooksValidationError
            detached_entries: list[RequestUsage] = []
            for entry in entries:
                if type(entry) is not RequestUsage:
                    raise _AgentHooksValidationError
                entry_input_details = entry.input_tokens_details
                entry_output_details = entry.output_tokens_details
                if (
                    type(entry_input_details) is not InputTokensDetails
                    or type(entry_output_details) is not OutputTokensDetails
                ):
                    raise _AgentHooksValidationError
                detached_entries.append(
                    RequestUsage(
                        input_tokens=validated_count(entry.input_tokens),
                        output_tokens=validated_count(entry.output_tokens),
                        total_tokens=validated_count(entry.total_tokens),
                        input_tokens_details=_make_input_tokens_details(
                            cached_tokens=validated_count(entry_input_details.cached_tokens),
                            cache_write_tokens=validated_count(
                                getattr(entry_input_details, "cache_write_tokens", 0)
                            ),
                        ),
                        output_tokens_details=OutputTokensDetails(
                            reasoning_tokens=validated_count(entry_output_details.reasoning_tokens)
                        ),
                    )
                )
            return Usage(
                requests=requests,
                input_tokens=input_tokens,
                input_tokens_details=_make_input_tokens_details(
                    cached_tokens=cached_tokens,
                    cache_write_tokens=cache_write_tokens,
                ),
                output_tokens=output_tokens,
                output_tokens_details=OutputTokensDetails(reasoning_tokens=reasoning_tokens),
                total_tokens=total_tokens,
                request_usage_entries=detached_entries,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            raise _AgentHooksValidationError from None

    async def _emit_post_model_context_invalid(
        self,
        *,
        call: _ModelCall,
    ) -> InterceptionRecord:
        state = _ModelPostState(
            usage=Usage(),
            response_id=None,
            request_id=None,
            entries=(),
            finish_reason="error",
            context_invalid=True,
        )
        return await self._emit_unchecked(
            point="post_model_call",
            context=self._build_context(
                point="post_model_call",
                build=lambda builder: builder.post_model_call(
                    model_id=call.model_id,
                    content=None,
                    tool_calls=[],
                    finish_reason="error",
                    request_id=call.request_id,
                ),
            ),
            validation_state=state,
        )

    async def emit_post_model_failure(
        self,
        *,
        call: _ModelCall,
        cancelled: bool,
    ) -> None:
        """Record one payload-free terminal post for a failed dispatched model call."""
        finish_reason = "cancelled" if cancelled else "error"
        state = _ModelPostState(
            usage=Usage(),
            response_id=None,
            request_id=None,
            entries=(),
            finish_reason=finish_reason,
        )
        await self._emit_unchecked(
            point="post_model_call",
            context=self._build_context(
                point="post_model_call",
                build=lambda builder: builder.post_model_call(
                    model_id=call.model_id,
                    content=None,
                    tool_calls=[],
                    finish_reason=finish_reason,
                    request_id=call.request_id,
                ),
            ),
            validation_state=state,
        )

    async def emit_pre_tool(
        self,
        *,
        tool: FunctionTool,
        tool_call: object,
    ) -> ToolPreDecision:
        arguments, state = _project_tool_pre(
            tool=tool,
            tool_call=tool_call,
            max_bytes=self._max_context_bytes,
            max_depth=self._max_context_depth,
        )
        try:
            target = await self._emit_guarded(
                point="pre_tool_call",
                context=self._build_context(
                    point="pre_tool_call",
                    build=lambda builder: builder.pre_tool_call(
                        call_id=state.invocation_id,
                        name=state.name,
                        args=arguments,
                    ),
                ),
                validation_state=state,
            )
        except Exception as error:
            if not isinstance(error, AgentHooksBlockedError):
                raise
            return ToolPreDecision(
                invocation=None,
                blocked_message=_bounded_blocked_message(error.reason),
            )
        transformed_arguments = copy.deepcopy(cast(dict[str, object], target))
        transformed_call = _rebuild_tool_call(tool_call, transformed_arguments)
        return ToolPreDecision(
            invocation=ToolInvocation(
                invocation_id=state.invocation_id,
                name=state.name,
                arguments=transformed_arguments,
                tool_call=transformed_call,
            )
        )

    def prepare_tool_invocation(self, invocation: ToolInvocation) -> object:
        """Rebuild one call from the private arguments approved by pre-tool governance."""
        return _rebuild_tool_call(
            invocation.tool_call,
            copy.deepcopy(invocation.arguments),
        )

    def start_tool_invocation(self) -> float:
        """Count one real invocation and start its monotonic duration clock."""
        if self._tool_count >= self._max_tool_calls_total:
            raise self._blocked_error(
                point="pre_tool_call",
                reason="host_error:adapter_unsupported",
                sequence=None,
            )
        self._tool_count += 1
        return time.monotonic()

    async def emit_post_tool(
        self,
        *,
        invocation: ToolInvocation,
        result: object,
        is_error: bool,
        started_at: float,
    ) -> ToolPostDecision:
        duration_ms = max(0.0, (time.monotonic() - started_at) * 1000.0)
        if isinstance(result, str):
            context_result = result
            state = _ToolPostState(
                invocation_id=invocation.invocation_id,
                name=invocation.name,
                arguments=invocation.arguments,
                is_error=is_error,
            )
        else:
            context_result = ""
            state = _ToolPostState(
                invocation_id=invocation.invocation_id,
                name=invocation.name,
                arguments=invocation.arguments,
                is_error=is_error,
                unsupported=True,
            )
        try:
            context = self._build_context(
                point="post_tool_call",
                build=lambda builder: builder.post_tool_call(
                    call_id=invocation.invocation_id,
                    name=invocation.name,
                    args=invocation.arguments,
                    value=context_result,
                    is_error=is_error,
                    duration_ms=duration_ms,
                ),
            )
        except Exception as error:
            if not isinstance(error, AgentHooksBlockedError):
                raise
            if error.sequence is None:
                record = await self._emit_post_tool_context_invalid(
                    invocation=invocation,
                    duration_ms=duration_ms,
                )
                return ToolPostDecision(
                    result=None,
                    blocked_message=_bounded_blocked_message(record.verdict.reason),
                )
            return ToolPostDecision(
                result=None,
                blocked_message=_bounded_blocked_message(error.reason),
            )
        try:
            target = await self._emit_guarded(
                point="post_tool_call",
                context=context,
                validation_state=state,
            )
        except Exception as error:
            if not isinstance(error, AgentHooksBlockedError):
                raise
            return ToolPostDecision(
                result=None,
                blocked_message=_bounded_blocked_message(error.reason),
            )
        return ToolPostDecision(result=cast(str, target))

    async def _emit_post_tool_context_invalid(
        self,
        *,
        invocation: ToolInvocation,
        duration_ms: float,
    ) -> InterceptionRecord:
        state = _ToolPostState(
            invocation_id=invocation.invocation_id,
            name=invocation.name,
            arguments={},
            is_error=True,
            context_invalid=True,
        )
        return await self._emit_unchecked(
            point="post_tool_call",
            context=self._build_context(
                point="post_tool_call",
                build=lambda builder: builder.post_tool_call(
                    call_id=invocation.invocation_id,
                    name=invocation.name,
                    args={},
                    value="",
                    is_error=True,
                    duration_ms=duration_ms,
                ),
            ),
            validation_state=state,
        )

    async def close(self, reason: str) -> None:
        if not self._opened or self._closed:
            return
        self._closed = True
        await self._emit_unchecked(
            point="agent_shutdown",
            context=self._build_context(
                point="agent_shutdown",
                build=lambda builder: builder.agent_shutdown(reason=reason),
            ),
        )

    def _build_context(
        self,
        *,
        point: str,
        build: Callable[[AgentContextBuilder], AgentContext],
    ) -> AgentContext:
        builder = self._require_builder()
        preview = build(self._require_builder_factory()())
        preview["sequence"] = self._next_sequence
        self._inject_trusted_runtime_context(preview)
        try:
            _validate_json_value(
                preview,
                max_bytes=self._max_context_bytes,
                max_depth=self._max_context_depth,
            )
        except _AgentHooksValidationError:
            raise self._blocked_error(
                point=point,
                reason="host_error:context_invalid",
                sequence=None,
            ) from None
        context = build(builder)
        if context.get("sequence") != self._next_sequence:
            raise RuntimeError("Agent Hooks context builder sequence diverged")
        self._next_sequence += 1
        return context

    def release(self) -> None:
        emitter = self._emitter
        if emitter is not None:
            emitter.take_records()
        self._builder = None
        self._builder_factory = None
        self._emitter = None
        self._sidecar.clear()
        self._source_labels = []
        self._tool_count = 0
        self._next_sequence = 0
        self._tool_names = ()
        self._audit_sink.release()

    @property
    def released(self) -> bool:
        """Whether all adapter-owned runtime references have been released."""
        return (
            self._builder is None
            and self._builder_factory is None
            and self._emitter is None
            and not self._source_labels
            and self._tool_count == 0
            and not self._tool_names
            and self._audit_sink.released
        )

    async def _emit_guarded(
        self,
        *,
        point: str,
        context: AgentContext,
        validation_state: _ValidationState | None = None,
    ) -> object:
        record = await self._emit_unchecked(
            point=point,
            context=context,
            validation_state=validation_state,
        )
        reason = record.verdict.reason
        if not record.proceeds or (reason is not None and reason.startswith("host_error:")):
            raise self._blocked_error(
                point=point,
                reason=reason,
                sequence=record.sequence,
            )
        return context.get("target")

    async def _emit_unchecked(
        self,
        *,
        point: str,
        context: AgentContext,
        validation_state: _ValidationState | None = None,
    ) -> InterceptionRecord:
        self._sidecar.begin(validation_state)
        self._inject_trusted_runtime_context(context)
        context_invalid = False
        try:
            _validate_json_value(
                context,
                max_bytes=self._max_context_bytes,
                max_depth=self._max_context_depth,
            )
        except _AgentHooksValidationError:
            context_invalid = True
        if context_invalid:
            self._sidecar.clear()
            raise self._blocked_error(
                point=point,
                reason="host_error:context_invalid",
                sequence=None,
            ) from None

        emitter = self._require_emitter()
        self._audit_sink.begin_emission()
        try:
            record = await emitter.emit_unchecked(context)
        except BaseException as primary_error:
            audit_error = self._take_audit_error(point=point)
            if audit_error is not None:
                if isinstance(primary_error, asyncio.CancelledError):
                    raise primary_error from audit_error
                raise audit_error from None
            raise
        finally:
            if "record" not in locals():
                self._sidecar.clear()

        try:
            audit_error = self._take_audit_error(point=point)
            if audit_error is not None:
                raise audit_error from None
            normalized_record = _normalize_interception_record(record)
            self._persist_result_labels(normalized_record)
            return normalized_record
        finally:
            self._sidecar.clear()

    def validate_folded_context(self, context: AgentContext) -> None:
        _validate_json_value(
            context,
            max_bytes=self._max_context_bytes,
            max_depth=self._max_context_depth,
        )
        if self._sidecar.labels_invalid:
            raise _AgentHooksValidationError
        combined_labels = [*self._source_labels]
        for label in self._sidecar.labels:
            if label not in combined_labels:
                combined_labels.append(label)
        _validate_result_labels(
            tuple(combined_labels),
            max_items=self._max_result_labels,
        )
        prospective_context = copy.deepcopy(context)
        self._set_source_labels(prospective_context, combined_labels)
        _validate_json_value(
            prospective_context,
            max_bytes=self._max_context_bytes,
            max_depth=self._max_context_depth,
        )
        point = context.get("interception_point")
        target = context.get("target")
        if point == "agent_startup":
            if not isinstance(target, dict):
                raise _AgentHooksValidationError
            startup_target = cast(dict[str, object], target)
            if set(startup_target) != {"tools_registered"}:
                raise _AgentHooksValidationError
            tools_registered = startup_target.get("tools_registered")
            if tools_registered != list(self._tool_names):
                raise _AgentHooksValidationError
            return
        if point == "input":
            self._validate_text_target(target, require_role=True)
            return
        if point == "pre_model_call":
            state = self._sidecar.validation_state
            if not isinstance(state, _ModelPreState):
                raise _AgentHooksUnsupportedError
            _validate_model_pre_target(target, state)
            return
        if point == "post_model_call":
            state = self._sidecar.validation_state
            if not isinstance(state, _ModelPostState):
                raise _AgentHooksUnsupportedError
            _validate_model_post_target(target, state)
            return
        if point == "pre_tool_call":
            state = self._sidecar.validation_state
            if not isinstance(state, _ToolPreState):
                raise _AgentHooksUnsupportedError
            _validate_tool_pre_target(target, state)
            return
        if point == "post_tool_call":
            state = self._sidecar.validation_state
            if not isinstance(state, _ToolPostState):
                raise _AgentHooksUnsupportedError
            _validate_tool_post_target(target, state, context)
            return
        if point == "output":
            self._validate_text_target(target, require_role=False)
            return
        if point == "agent_shutdown":
            if not isinstance(target, dict):
                raise _AgentHooksValidationError
            shutdown_target = cast(dict[str, object], target)
            if set(shutdown_target) != {"reason"}:
                raise _AgentHooksValidationError
            if shutdown_target.get("reason") not in {"completed", "error", "cancelled"}:
                raise _AgentHooksValidationError
            return
        raise _AgentHooksUnsupportedError

    def _inject_trusted_runtime_context(self, context: AgentContext) -> None:
        context["budgets"] = {"tool_call_count": self._tool_count}
        self._set_source_labels(context, self._source_labels)

    @staticmethod
    def _set_source_labels(context: AgentContext, labels: list[str]) -> None:
        if not labels:
            return
        raw_extensions = context.get("extensions")
        extensions: dict[str, object] = (
            dict(cast(dict[str, object], raw_extensions))
            if isinstance(raw_extensions, dict)
            else {}
        )
        raw_openai_agents = extensions.get("openai_agents")
        openai_agents: dict[str, object] = (
            dict(cast(dict[str, object], raw_openai_agents))
            if isinstance(raw_openai_agents, dict)
            else {}
        )
        openai_agents["source_labels"] = list(labels)
        extensions["openai_agents"] = openai_agents
        context["extensions"] = extensions

    def _persist_result_labels(self, record: InterceptionRecord) -> None:
        if not record.verdict.decision.permits:
            return
        for label in record.verdict.result_labels:
            if label not in self._source_labels:
                self._source_labels.append(label)

    @staticmethod
    def _project_usage(response: ModelResponse) -> dict[str, int]:
        prompt_tokens: object = response.usage.input_tokens
        completion_tokens: object = response.usage.output_tokens
        if not _is_nonnegative_int(prompt_tokens) or not _is_nonnegative_int(completion_tokens):
            raise _AgentHooksUnsupportedError
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }

    def _take_audit_error(self, *, point: str) -> Exception | None:
        failure = self._audit_sink.take_failure()
        if failure is None:
            return None
        sequence, sink_error_type = failure
        return create_agent_hooks_audit_error(
            point=point,
            sequence=sequence,
            sink_error_type=sink_error_type,
        )

    def _extract_text_target(
        self,
        target: object,
        *,
        point: str,
        require_role: bool,
    ) -> str:
        try:
            content = self._validate_text_target(target, require_role=require_role)
        except _AgentHooksValidationError:
            raise self._blocked_error(
                point=point,
                reason="host_error:transform_invalid",
                sequence=None,
            ) from None
        return content

    @staticmethod
    def _validate_text_target(target: object, *, require_role: bool) -> str:
        if not isinstance(target, dict):
            raise _AgentHooksValidationError
        text_target = cast(dict[str, object], target)
        expected_keys = {"content", "role"} if require_role else {"content"}
        if set(text_target) != expected_keys or (
            require_role and text_target.get("role") != "user"
        ):
            raise _AgentHooksValidationError
        content = text_target.get("content")
        if not isinstance(content, str):
            raise _AgentHooksValidationError
        return content

    @staticmethod
    def _blocked_error(*, point: str, reason: str | None, sequence: int | None) -> Exception:
        public_reason = (
            reason if reason is not None and reason.startswith("host_error:") else "policy_denied"
        )
        return create_agent_hooks_blocked_error(
            point=point,
            reason=public_reason,
            sequence=sequence,
        )

    def _require_builder(self) -> AgentContextBuilder:
        if self._builder is None:
            raise RuntimeError("Agent Hooks session has been released")
        return self._builder

    def _require_builder_factory(self) -> Callable[[], AgentContextBuilder]:
        if self._builder_factory is None:
            raise RuntimeError("Agent Hooks session has been released")
        return self._builder_factory

    def _require_emitter(self) -> InterceptionEmitter:
        if self._emitter is None:
            raise RuntimeError("Agent Hooks session has been released")
        return self._emitter


class _HostValidatorBridge:
    """Validate the final folded target as the last run-all interceptor."""

    __slots__ = ("_session",)

    def __init__(self, session: AgentHooksRunSession) -> None:
        self._session = session

    async def intercept(self, context: AgentContext, /) -> Verdict:
        from agent_hooks import ALLOW, Decision, Verdict

        try:
            self._session.validate_folded_context(context)
        except _AgentHooksContextInvalidError:
            return Verdict(decision=Decision.DENY, reason=_HOST_CONTEXT_INVALID)
        except _AgentHooksUnsupportedError:
            return Verdict(decision=Decision.DENY, reason=_HOST_ADAPTER_UNSUPPORTED)
        except _AgentHooksValidationError:
            return Verdict(decision=Decision.DENY, reason=_HOST_TRANSFORM_INVALID)
        return ALLOW


_CURRENT_AGENT_HOOKS_SESSION: ContextVar[AgentHooksRunSession | None] = ContextVar(
    "agents_current_agent_hooks_session",
    default=None,
)


def bind_agent_hooks_session(
    session: AgentHooksRunSession | None,
) -> Token[AgentHooksRunSession | None]:
    """Bind a run session, including an explicit ``None`` for disabled nested runs."""
    return _CURRENT_AGENT_HOOKS_SESSION.set(session)


def reset_agent_hooks_session(token: Token[AgentHooksRunSession | None]) -> None:
    """Restore the previous task-local run session."""
    _CURRENT_AGENT_HOOKS_SESSION.reset(token)


def get_current_agent_hooks_session() -> AgentHooksRunSession | None:
    """Return the task-local session inherited by work for the current run."""
    return _CURRENT_AGENT_HOOKS_SESSION.get()


def is_agent_hooks_control_error(error: BaseException) -> bool:
    """Return whether an error must remain free of retained run payloads."""
    return is_host_agent_hooks_error(error) or isinstance(
        error,
        _AgentHooksCancelledError | _AgentHooksKeyboardInterrupt | _AgentHooksSystemExit,
    )


def sanitize_agent_hooks_control_error(error: BaseException) -> BaseException:
    """Return a fresh metadata-only control error without retained object graphs."""
    if isinstance(error, AgentHooksBlockedError) and is_host_agent_hooks_error(error):
        return create_agent_hooks_blocked_error(
            point=error.point,
            reason=error.reason,
            sequence=error.sequence,
        )
    if isinstance(error, AgentHooksAuditError) and is_host_agent_hooks_error(error):
        return create_agent_hooks_audit_error(
            point=error.point,
            sequence=error.sequence,
            sink_error_type=error.sink_error_type,
        )
    if isinstance(error, _AgentHooksSetupError) and is_host_agent_hooks_error(error):
        return create_agent_hooks_setup_error(error.message)
    if isinstance(error, asyncio.CancelledError):
        return _AgentHooksCancelledError()
    if isinstance(error, KeyboardInterrupt):
        return _AgentHooksKeyboardInterrupt()
    if isinstance(error, SystemExit):
        return _AgentHooksSystemExit()
    return create_agent_hooks_execution_error()


def sanitize_agent_hooks_setup_error(error: BaseException) -> BaseException:
    if type(error) is UserError and type(error.message) is str:
        return create_agent_hooks_setup_error(error.message)
    return sanitize_agent_hooks_control_error(error)


def sanitize_agent_hooks_application_error(error: BaseException) -> BaseException:
    """Return a fresh signal for an untrusted model or tool callback failure."""
    if isinstance(error, asyncio.CancelledError):
        return _AgentHooksCancelledError()
    if isinstance(error, KeyboardInterrupt):
        return _AgentHooksKeyboardInterrupt()
    if isinstance(error, SystemExit):
        return _AgentHooksSystemExit()
    return create_agent_hooks_execution_error()


def clear_agent_hooks_error_graph(error: BaseException) -> None:
    pending = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        cause = BaseException.__getattribute__(current, "__cause__")
        context = BaseException.__getattribute__(current, "__context__")
        if isinstance(cause, BaseException):
            pending.append(cause)
        if isinstance(context, BaseException):
            pending.append(context)
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)
        BaseException.__setattr__(current, "__traceback__", None)
        BaseException.__setattr__(current, "__cause__", None)
        BaseException.__setattr__(current, "__context__", None)
        if isinstance(current, AgentsException):
            current.run_data = None


def _derive_max_records(*, max_turns: int, max_tool_calls_per_turn: int) -> int:
    return 4 + max_turns * (2 + 2 * max_tool_calls_per_turn)


def _validate_trusted_context_capacity(
    *,
    config: AgentHooksConfig,
    starting_agent: Agent[TContext],
    max_turns: int,
) -> None:
    session_id = "00000000-0000-0000-0000-000000000000"
    agent: dict[str, object] = {"id": config.agent_id, "framework": _FRAMEWORK}
    if starting_agent.name:
        agent["name"] = starting_agent.name
    session: dict[str, object] = {"id": session_id}
    grouping: dict[str, object] = {}
    if config.session_id is not None:
        grouping["caller_session_id"] = config.session_id
    if config.correlation_id is not None:
        grouping["correlation_id"] = config.correlation_id
    source_labels: list[str] = []
    for index in range(_MAX_VERDICT_COLLECTION_ITEMS):
        prefix = f"label-{index:03d}-"
        source_labels.append(prefix + "x" * (_MAX_LABEL_BYTES - len(prefix)))
    extensions = {
        "openai_agents": {
            **grouping,
            "source_labels": source_labels,
        }
    }

    max_records = _derive_max_records(
        max_turns=max_turns,
        max_tool_calls_per_turn=config.limits.max_tool_calls_per_turn,
    )
    tool_names = [
        get_function_tool_qualified_name(tool) or tool.name for tool in starting_agent.tools
    ]

    def envelope(*, point: str, sequence: int, target: object) -> dict[str, object]:
        context: dict[str, object] = {
            "spec": "agent-hooks/0.1",
            "interception_point": point,
            "timestamp": "2000-01-01T00:00:00.000000Z",
            "sequence": sequence,
            "agent": agent,
            "session": session,
            "trace": {
                "trace_id": "0" * 32,
                "span_id": "0" * 16,
            },
            "target": target,
            "budgets": {"tool_call_count": max_turns * config.limits.max_tool_calls_per_turn},
        }
        context["extensions"] = extensions
        return context

    startup_target: dict[str, object] = {"tools_registered": tool_names}
    startup_context = envelope(
        point="agent_startup",
        sequence=0,
        target=startup_target,
    )
    startup_context["agent_init"] = startup_target
    shutdown_target: dict[str, object] = {"reason": "cancelled"}
    shutdown_context = envelope(
        point="agent_shutdown",
        sequence=max_records - 1,
        target=shutdown_target,
    )
    shutdown_context["summary"] = shutdown_target

    model_response: dict[str, object] = {
        "content": None,
        "tool_calls": [],
        "finish_reason": "error",
    }
    model_post_context = envelope(
        point="post_model_call",
        sequence=max_records - 2,
        target=model_response,
    )
    model_post_context["model"] = {"id": "m" * _MAX_IDENTIFIER_BYTES}
    model_post_context["response"] = model_response
    model_post_context["request_id"] = "00000000-0000-0000-0000-000000000000"

    trusted_contexts = [startup_context, shutdown_context, model_post_context]
    if tool_names:
        tool_call: dict[str, object] = {
            "id": "00000000-0000-0000-0000-000000000000",
            "name": max(tool_names, key=lambda name: len(name.encode("utf-8"))),
            "args": {},
        }
        tool_result: dict[str, object] = {
            "value": "",
            "is_error": True,
            "duration_ms": 0.0,
        }
        tool_post_context = envelope(
            point="post_tool_call",
            sequence=max_records - 2,
            target="",
        )
        tool_post_context["tool_call"] = tool_call
        tool_post_context["tool_result"] = tool_result
        trusted_contexts.append(tool_post_context)

    invalid_capacity = False
    try:
        for trusted_context in trusted_contexts:
            _validate_json_value(
                trusted_context,
                max_bytes=config.limits.max_context_bytes,
                max_depth=config.limits.max_context_depth,
            )
    except _AgentHooksValidationError:
        invalid_capacity = True
    if invalid_capacity:
        raise UserError(
            "Agent Hooks context limits cannot encode the trusted lifecycle envelopes"
        ) from None


def validate_agent_hooks_admission(
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
    """Reject unsupported governed run shapes before protocol state or side effects."""
    if type(starting_agent) is not Agent:
        raise UserError("Agent Hooks requires an exact plain Agent instance")
    if not isinstance(input, str):
        raise UserError("Agent Hooks requires string input and does not support RunState or items")
    if isinstance(max_turns, bool) or not isinstance(max_turns, int) or max_turns <= 0:
        raise UserError("Agent Hooks requires max_turns to be a positive finite integer")
    if session is not None:
        raise UserError("Agent Hooks does not support Session; use a fresh run")
    if previous_response_id is not None or auto_previous_response_id or conversation_id is not None:
        raise UserError("Agent Hooks does not support server-managed conversations")
    if error_handlers is not None:
        raise UserError("Agent Hooks does not support run error handlers")
    if starting_agent.handoffs:
        raise UserError("Agent Hooks does not support handoffs")
    if starting_agent.mcp_servers:
        raise UserError("Agent Hooks does not support MCP servers")
    if starting_agent.prompt is not None:
        raise UserError("Agent Hooks does not support prompts")
    if callable(starting_agent.instructions):
        raise UserError("Agent Hooks does not support dynamic instructions")
    if starting_agent.output_type not in (None, str):
        raise UserError("Agent Hooks supports only plain string output")
    if starting_agent.tool_use_behavior != "run_llm_again":
        raise UserError("Agent Hooks does not support tool-as-final output behavior")

    if run_config.handoff_input_filter is not None:
        raise UserError("Agent Hooks does not support handoff input filters")
    if run_config.nest_handoff_history or run_config.handoff_history_mapper is not None:
        raise UserError("Agent Hooks does not support handoff history configuration")
    if run_config.session_input_callback is not None or run_config.session_settings is not None:
        raise UserError("Agent Hooks does not support session callbacks or settings")
    if run_config.call_model_input_filter is not None:
        raise UserError("Agent Hooks does not support call-model input filters")
    if run_config.sandbox is not None:
        raise UserError("Agent Hooks does not support sandbox runs")
    if run_config.tool_error_formatter is not None:
        raise UserError("Agent Hooks does not support tool error formatters")
    if run_config.tool_not_found_behavior != "raise_error":
        raise UserError("Agent Hooks requires tool_not_found_behavior='raise_error'")

    from .turn_preparation import get_model_settings

    model_settings = get_model_settings(starting_agent, run_config)
    retry_settings = model_settings.retry
    if retry_settings is None or retry_settings.max_retries != 0:
        raise UserError("Agent Hooks requires effective model settings with retry.max_retries=0")

    if starting_agent.tools:
        if model_settings.parallel_tool_calls is not False:
            raise UserError(
                "Agent Hooks requires effective parallel_tool_calls=False when tools are configured"
            )
        tool_execution = run_config.tool_execution
        if tool_execution is None or tool_execution.max_function_tool_concurrency != 1:
            raise UserError("Agent Hooks requires tool_execution.max_function_tool_concurrency=1")
        if tool_execution.pre_approval_tool_input_guardrails:
            raise UserError("Agent Hooks does not support pre-approval tool input guardrails")

    config = run_config.agent_hooks
    if config is None:
        raise RuntimeError("Agent Hooks admission requires configured Agent Hooks")

    admitted_tool_names: set[str] = set()
    for tool in starting_agent.tools:
        if not isinstance(tool, FunctionTool):
            raise UserError("Agent Hooks supports only static FunctionTool entries")
        if not is_async_function_tool(tool):
            raise UserError("Agent Hooks requires asynchronous FunctionTool handlers")
        try:
            _validate_tool_schema(
                cast(dict[str, object], tool.params_json_schema),
                max_bytes=config.limits.max_context_bytes,
                max_depth=config.limits.max_context_depth,
            )
        except _AgentHooksValidationError:
            raise UserError(
                "Agent Hooks requires every FunctionTool parameter schema to be valid"
            ) from None
        if tool.is_enabled is not True:
            raise UserError("Agent Hooks requires every FunctionTool to be statically enabled")
        if tool.needs_approval is not False:
            raise UserError("Agent Hooks does not support native tool approval")
        if tool.defer_loading:
            raise UserError("Agent Hooks does not support deferred FunctionTool loading")
        if get_explicit_function_tool_namespace(tool) is not None:
            raise UserError("Agent Hooks does not support namespaced FunctionTool entries")
        if tool.allowed_callers is not None:
            raise UserError("Agent Hooks does not support FunctionTool caller restrictions")
        if tool.output_json_schema is not None:
            raise UserError("Agent Hooks supports only string FunctionTool output")
        if tool.custom_data_extractor is not None:
            raise UserError("Agent Hooks does not support FunctionTool custom data extraction")
        tool_name = get_function_tool_qualified_name(tool) or tool.name
        if not _is_bounded_identifier(tool_name) or tool_name in admitted_tool_names:
            raise UserError("Agent Hooks requires unique bounded FunctionTool names")
        admitted_tool_names.add(tool_name)
        origin = get_function_tool_origin(tool)
        if origin is not None and origin.type is not ToolOriginType.FUNCTION:
            raise UserError(
                "Agent Hooks does not support MCP or agent-as-tool FunctionTool entries"
            )

    _validate_trusted_context_capacity(
        config=config,
        starting_agent=starting_agent,
        max_turns=max_turns,
    )


def create_agent_hooks_session(
    *,
    config: AgentHooksConfig,
    starting_agent: Agent[TContext],
    max_turns: int,
) -> AgentHooksRunSession:
    """Lazily create one fixed-profile SDK session after admission succeeds."""
    from agent_hooks import (
        JCS_SHA256,
        AgentContextBuilder,
        CompositionConfig,
        InterceptionEmitter,
        Interceptor,
        Verdict,
    )

    session_id = str(uuid4())
    grouping: dict[str, object] = {}
    if config.session_id is not None:
        grouping["caller_session_id"] = config.session_id
    if config.correlation_id is not None:
        grouping["correlation_id"] = config.correlation_id
    l2: dict[str, object] = {
        "trace": {
            "trace_id": uuid4().hex,
            "span_id": uuid4().hex[:16],
        }
    }
    if grouping:
        l2["extensions"] = {"openai_agents": grouping}

    def create_builder() -> AgentContextBuilder:
        return AgentContextBuilder(
            agent_id=config.agent_id,
            framework=_FRAMEWORK,
            session_id=session_id,
            agent_name=starting_agent.name,
        ).with_l2(**l2)

    builder = create_builder()

    emitter = InterceptionEmitter(
        mode=config.mode,
        resolver=None,
        timeout=config.interceptor_timeout_seconds,
        composition=CompositionConfig.run_all(),
        identity_provider=JCS_SHA256,
    )
    max_records = _derive_max_records(
        max_turns=max_turns,
        max_tool_calls_per_turn=config.limits.max_tool_calls_per_turn,
    )
    emitter.set_max_records(max_records)

    sidecar = _EmissionSidecar()
    interceptor_count = len(config.interceptors)
    max_verdict_bytes = max(1, config.limits.max_verdict_bytes // interceptor_count)
    max_collection_items = max(1, _MAX_VERDICT_COLLECTION_ITEMS // interceptor_count)
    for index, interceptor in enumerate(config.interceptors):
        bridge = _AsyncInterceptorBridge(
            interceptor,
            verdict_type=Verdict,
            max_verdict_bytes=max_verdict_bytes,
            max_depth=config.limits.max_context_depth,
            max_collection_items=max_collection_items,
            sidecar=sidecar,
        )
        # The upstream protocol annotates only synchronous returns although its emitter awaits
        # awaitables. This cast is confined to that external typing mismatch.
        emitter.register(cast(Interceptor, bridge), name=f"interceptor-{index}")

    tool_names = tuple(
        get_function_tool_qualified_name(tool) or tool.name for tool in starting_agent.tools
    )
    try:
        reservation = config.record_sink.reserve(max_records)
    except BufferError:
        raise UserError(
            f"Agent Hooks RecordSink capacity must reserve {max_records} records for this run"
        ) from None

    audit_sink = _AuditSink(reservation)
    try:
        emitter.set_record_sink(audit_sink)
        session = AgentHooksRunSession(
            builder=builder,
            builder_factory=create_builder,
            emitter=emitter,
            audit_sink=audit_sink,
            limits=config.limits,
            tool_names=tool_names,
            sidecar=sidecar,
            max_turns=max_turns,
        )
        emitter.register(
            cast(Interceptor, _HostValidatorBridge(session)),
            name=_HOST_VALIDATOR_NAME,
        )
        return session
    except BaseException:
        audit_sink.release()
        raise
