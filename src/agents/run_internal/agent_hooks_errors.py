"""Dependency-free control errors for the optional Agent Hooks integration."""

from __future__ import annotations

from ..exceptions import AgentsException, UserError

_HOST_ERROR_PROVENANCE = object()


class AgentHooksBlockedError(AgentsException):
    """Report a payload-free Agent Hooks denial or invalid transform."""

    __slots__ = ("_host_provenance", "point", "reason", "sequence")

    def __init__(self, *, point: str, reason: str | None, sequence: int | None) -> None:
        self._host_provenance: object | None = None
        self.point = point
        self.reason = reason
        self.sequence = sequence
        super().__init__(f"Agent Hooks blocked the {point} action")


class AgentHooksAuditError(AgentsException):
    """Report payload-free audit delivery failure metadata."""

    __slots__ = ("_host_provenance", "point", "sequence", "sink_error_type")

    def __init__(self, *, point: str, sequence: int | None, sink_error_type: str) -> None:
        self._host_provenance: object | None = None
        self.point = point
        self.sequence = sequence
        self.sink_error_type = sink_error_type
        super().__init__(f"Agent Hooks audit delivery failed for the {point} action")


class AgentHooksExecutionError(AgentsException):
    """Report a governed execution failure without retaining untrusted payloads."""

    __slots__ = ("_host_provenance",)

    def __init__(self) -> None:
        self._host_provenance: object | None = None
        super().__init__("Agent Hooks governed execution failed")


class _AgentHooksSetupError(UserError):
    __slots__ = ("_host_provenance",)

    def __init__(self, message: str) -> None:
        self._host_provenance: object | None = None
        super().__init__(message)


def create_agent_hooks_blocked_error(
    *,
    point: str,
    reason: str | None,
    sequence: int | None,
) -> AgentHooksBlockedError:
    error = AgentHooksBlockedError(point=point, reason=reason, sequence=sequence)
    error._host_provenance = _HOST_ERROR_PROVENANCE
    return error


def create_agent_hooks_audit_error(
    *,
    point: str,
    sequence: int | None,
    sink_error_type: str,
) -> AgentHooksAuditError:
    error = AgentHooksAuditError(
        point=point,
        sequence=sequence,
        sink_error_type=sink_error_type,
    )
    error._host_provenance = _HOST_ERROR_PROVENANCE
    return error


def create_agent_hooks_execution_error() -> AgentHooksExecutionError:
    error = AgentHooksExecutionError()
    error._host_provenance = _HOST_ERROR_PROVENANCE
    return error


def create_agent_hooks_setup_error(message: str) -> _AgentHooksSetupError:
    error = _AgentHooksSetupError(message)
    error._host_provenance = _HOST_ERROR_PROVENANCE
    return error


def is_host_agent_hooks_error(error: BaseException) -> bool:
    if not isinstance(
        error,
        AgentHooksAuditError
        | AgentHooksBlockedError
        | AgentHooksExecutionError
        | _AgentHooksSetupError,
    ):
        return False
    try:
        provenance = BaseException.__getattribute__(error, "_host_provenance")
    except BaseException:
        return False
    return provenance is _HOST_ERROR_PROVENANCE
