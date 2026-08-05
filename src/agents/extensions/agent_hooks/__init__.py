"""Optional Agent Hooks configuration and callback contracts."""

from __future__ import annotations

import copy
import inspect
import math
from collections.abc import Awaitable, Iterator, Sequence
from dataclasses import dataclass, field
from threading import Lock
from typing import TYPE_CHECKING, Protocol, overload, runtime_checkable

try:
    from agent_hooks import AgentContext, EnforcementMode, InterceptionRecord, Verdict
except ImportError as error:
    raise ImportError(
        "agents.extensions.agent_hooks requires the 'agent-hooks' extra. "
        "Install it with: pip install 'openai-agents[agent-hooks]'"
    ) from error

from ..._config_coercion import coerce_dataclass_config
from ...run_internal.agent_hooks_errors import (
    AgentHooksAuditError,
    AgentHooksBlockedError,
    AgentHooksExecutionError,
)

__all__ = [
    "AgentContext",
    "AgentHooksAuditError",
    "AgentHooksBlockedError",
    "AgentHooksConfig",
    "AgentHooksExecutionError",
    "AgentHooksLimits",
    "AsyncInterceptor",
    "EnforcementMode",
    "InterceptionRecord",
    "RecordSink",
    "Verdict",
]

_MAX_IDENTIFIER_BYTES = 256
_MAX_CONTEXT_DEPTH = 128


def _clone_record(record: InterceptionRecord) -> InterceptionRecord:
    return copy.deepcopy(record) if isinstance(record, InterceptionRecord) else record


@runtime_checkable
class AsyncInterceptor(Protocol):
    """An Agent Hooks interceptor that always returns its verdict asynchronously."""

    def intercept(self, context: AgentContext, /) -> Awaitable[Verdict]: ...


class RecordSink(Sequence[InterceptionRecord]):
    """A bounded in-memory sink for payload-free interception records."""

    __slots__ = ("_lock", "_max_records", "_records", "_reserved")

    def __init__(self, *, max_records: int) -> None:
        if isinstance(max_records, bool) or not isinstance(max_records, int) or max_records <= 0:
            raise ValueError("RecordSink.max_records must be a positive integer")
        self._lock = Lock()
        self._max_records = max_records
        self._records: list[InterceptionRecord] = []
        self._reserved = 0

    @property
    def max_records(self) -> int:
        """Return the maximum number of retained records."""
        return self._max_records

    def write(self, record: InterceptionRecord, /) -> None:
        """Retain one record or fail without evicting earlier audit history."""
        with self._lock:
            if len(self._records) + self._reserved >= self._max_records:
                raise BufferError("Agent Hooks record sink is full")
            self._records.append(_clone_record(record))

    def reserve(self, max_records: int, /) -> _RecordSinkReservation:
        """Atomically reserve finite capacity for one governed run."""
        if isinstance(max_records, bool) or not isinstance(max_records, int) or max_records <= 0:
            raise ValueError("RecordSink reservation must be a positive integer")
        with self._lock:
            if max_records > self._max_records - len(self._records) - self._reserved:
                raise BufferError("Agent Hooks RecordSink capacity is insufficient for the run")
            self._reserved += max_records
        return _RecordSinkReservation(self, max_records)

    def _write_reserved(
        self,
        reservation: _RecordSinkReservation,
        record: InterceptionRecord,
    ) -> None:
        with self._lock:
            if reservation._released or reservation._remaining <= 0:
                raise BufferError("Agent Hooks RecordSink reservation is exhausted")
            self._records.append(_clone_record(record))
            reservation._remaining -= 1
            self._reserved -= 1

    def _release_reservation(self, reservation: _RecordSinkReservation) -> None:
        with self._lock:
            if reservation._released:
                return
            self._reserved -= reservation._remaining
            reservation._remaining = 0
            reservation._released = True

    def _reservation_released(self, reservation: _RecordSinkReservation) -> bool:
        with self._lock:
            return reservation._released

    def snapshot(self) -> tuple[InterceptionRecord, ...]:
        """Return an immutable view of retained records."""
        with self._lock:
            return tuple(_clone_record(record) for record in self._records)

    def drain(self) -> tuple[InterceptionRecord, ...]:
        """Remove and return all retained records."""
        with self._lock:
            records = tuple(_clone_record(record) for record in self._records)
            self._records.clear()
            return records

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)

    @overload
    def __getitem__(self, index: int) -> InterceptionRecord: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[InterceptionRecord, ...]: ...

    def __getitem__(
        self,
        index: int | slice,
    ) -> InterceptionRecord | tuple[InterceptionRecord, ...]:
        with self._lock:
            if isinstance(index, slice):
                return tuple(_clone_record(record) for record in self._records[index])
            return _clone_record(self._records[index])

    def __iter__(self) -> Iterator[InterceptionRecord]:
        return iter(self.snapshot())


class _RecordSinkReservation:
    """Own one run's atomically reserved record capacity."""

    __slots__ = ("_released", "_remaining", "_sink")

    def __init__(self, sink: RecordSink, max_records: int) -> None:
        self._sink = sink
        self._remaining = max_records
        self._released = False

    def write(self, record: InterceptionRecord, /) -> None:
        self._sink._write_reserved(self, record)

    def release(self) -> None:
        self._sink._release_reservation(self)

    @property
    def released(self) -> bool:
        return self._sink._reservation_released(self)


def _validate_positive_limit(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"agent_hooks.limits.{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class AgentHooksLimits:
    """Finite resource limits for one Agent Hooks run."""

    max_context_bytes: int = 5 * 1024 * 1024
    max_context_depth: int = 128
    max_interceptors: int = 8
    max_tool_calls_per_turn: int = 32
    max_verdict_bytes: int = 16 * 1024

    def __post_init__(self) -> None:
        _validate_positive_limit(self.max_context_bytes, "max_context_bytes")
        _validate_positive_limit(self.max_context_depth, "max_context_depth")
        _validate_positive_limit(self.max_interceptors, "max_interceptors")
        _validate_positive_limit(self.max_tool_calls_per_turn, "max_tool_calls_per_turn")
        _validate_positive_limit(self.max_verdict_bytes, "max_verdict_bytes")
        if self.max_context_depth > _MAX_CONTEXT_DEPTH:
            raise ValueError(
                f"agent_hooks.limits.max_context_depth must not exceed {_MAX_CONTEXT_DEPTH}"
            )


def _validate_identifier(value: object, name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"agent_hooks.{name} must be a string")
    if not value.strip():
        raise ValueError(f"agent_hooks.{name} must not be empty")
    try:
        encoded_value = value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"agent_hooks.{name} must contain valid UTF-8 text") from error
    if len(encoded_value) > _MAX_IDENTIFIER_BYTES:
        raise ValueError(f"agent_hooks.{name} must not exceed {_MAX_IDENTIFIER_BYTES} UTF-8 bytes")


def _is_async_callable(callback: object) -> bool:
    return inspect.iscoroutinefunction(callback) or inspect.iscoroutinefunction(
        inspect.getattr_static(type(callback), "__call__", None)
    )


@dataclass(frozen=True, slots=True)
class AgentHooksConfig:
    """Trusted Agent Hooks configuration for one agent run."""

    agent_id: str
    interceptors: tuple[AsyncInterceptor, ...]
    record_sink: RecordSink
    session_id: str | None = None
    correlation_id: str | None = None
    mode: EnforcementMode = EnforcementMode.ENFORCE
    interceptor_timeout_seconds: float = 5.0
    limits: AgentHooksLimits = field(default_factory=AgentHooksLimits)

    if TYPE_CHECKING:

        def __init__(
            self,
            agent_id: str,
            interceptors: tuple[AsyncInterceptor, ...],
            record_sink: RecordSink,
            session_id: str | None = None,
            correlation_id: str | None = None,
            mode: EnforcementMode = EnforcementMode.ENFORCE,
            interceptor_timeout_seconds: float = 5.0,
            limits: AgentHooksLimits | dict[str, object] = ...,
        ) -> None: ...

    def __post_init__(self) -> None:
        normalized_limits = coerce_dataclass_config(
            self.limits,
            AgentHooksLimits,
            parameter_name="agent_hooks.limits",
        )
        object.__setattr__(self, "limits", normalized_limits)

        _validate_identifier(self.agent_id, "agent_id")
        if self.session_id is not None:
            _validate_identifier(self.session_id, "session_id")
        if self.correlation_id is not None:
            _validate_identifier(self.correlation_id, "correlation_id")

        if not isinstance(self.interceptors, tuple):
            raise TypeError("agent_hooks.interceptors must be a tuple")
        if not self.interceptors:
            raise ValueError("agent_hooks.interceptors must contain at least one interceptor")
        if len(self.interceptors) > normalized_limits.max_interceptors:
            raise ValueError(
                "agent_hooks.interceptors must not exceed agent_hooks.limits.max_interceptors"
            )
        for index, interceptor in enumerate(self.interceptors):
            intercept = getattr(interceptor, "intercept", None)
            if not callable(intercept):
                raise TypeError(f"agent_hooks.interceptors[{index}].intercept must be callable")
            if not _is_async_callable(intercept):
                raise TypeError(f"agent_hooks.interceptors[{index}].intercept must be async")

        if type(self.record_sink) is not RecordSink:
            raise TypeError("agent_hooks.record_sink must be an exact RecordSink instance")

        timeout = self.interceptor_timeout_seconds
        if isinstance(timeout, bool) or not isinstance(timeout, int | float):
            raise TypeError("agent_hooks.interceptor_timeout_seconds must be a number")
        normalized_timeout = float(timeout)
        if not math.isfinite(normalized_timeout) or normalized_timeout <= 0:
            raise ValueError("agent_hooks.interceptor_timeout_seconds must be finite and positive")
        object.__setattr__(self, "interceptor_timeout_seconds", normalized_timeout)

        if not isinstance(self.mode, EnforcementMode):
            raise TypeError("agent_hooks.mode must be an EnforcementMode")
