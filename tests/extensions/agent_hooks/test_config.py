from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from threading import Barrier
from typing import Any, cast

import pytest
from agent_hooks import ALLOW, AgentContext, Verdict

from agents import RunConfig
from agents.extensions.agent_hooks import (
    AgentHooksConfig,
    AgentHooksLimits,
    AsyncInterceptor,
    EnforcementMode,
    RecordSink,
)


class AllowInterceptor:
    async def intercept(self, context: AgentContext, /) -> Verdict:
        return ALLOW


record_sink = RecordSink(max_records=100)


class SyncInterceptor:
    def intercept(self, context: AgentContext, /) -> Verdict:
        return ALLOW


def test_agent_hooks_config_defaults() -> None:
    interceptor = AllowInterceptor()

    config = AgentHooksConfig(
        agent_id="support-agent-v1",
        interceptors=(interceptor,),
        record_sink=record_sink,
    )

    assert config.agent_id == "support-agent-v1"
    assert config.interceptors == (interceptor,)
    assert config.record_sink is record_sink
    assert config.session_id is None
    assert config.correlation_id is None
    assert config.mode is EnforcementMode.ENFORCE
    assert config.interceptor_timeout_seconds == 5.0
    assert config.limits == AgentHooksLimits()

    with pytest.raises(FrozenInstanceError):
        config.agent_id = "replacement"  # type: ignore[misc]


@pytest.mark.parametrize("agent_id", ["", "   ", "a" * 257])
def test_agent_hooks_config_rejects_invalid_agent_id(agent_id: str) -> None:
    with pytest.raises(ValueError, match=r"agent_hooks\.agent_id"):
        AgentHooksConfig(
            agent_id=agent_id,
            interceptors=(AllowInterceptor(),),
            record_sink=record_sink,
        )


@pytest.mark.parametrize("timeout", [0.0, -1.0, float("inf"), float("nan")])
def test_agent_hooks_config_rejects_nonpositive_or_nonfinite_timeout(timeout: float) -> None:
    with pytest.raises(
        ValueError,
        match="agent_hooks.interceptor_timeout_seconds must be finite and positive",
    ):
        AgentHooksConfig(
            agent_id="support-agent-v1",
            interceptors=(AllowInterceptor(),),
            record_sink=record_sink,
            interceptor_timeout_seconds=timeout,
        )


def test_agent_hooks_limits_reject_nonpositive_values() -> None:
    with pytest.raises(
        ValueError,
        match="agent_hooks.limits.max_context_bytes must be a positive integer",
    ):
        AgentHooksLimits(max_context_bytes=0)


def test_agent_hooks_limits_reject_excessive_context_depth() -> None:
    with pytest.raises(ValueError, match="max_context_depth must not exceed 128"):
        AgentHooksLimits(max_context_depth=129)


def test_agent_hooks_config_enforces_interceptor_count_limit() -> None:
    interceptor = AllowInterceptor()

    with pytest.raises(ValueError, match="agent_hooks.limits.max_interceptors"):
        AgentHooksConfig(
            agent_id="support-agent-v1",
            interceptors=(interceptor, interceptor),
            record_sink=record_sink,
            limits=AgentHooksLimits(max_interceptors=1),
        )


def test_agent_hooks_config_rejects_synchronous_interceptor() -> None:
    with pytest.raises(
        TypeError,
        match=r"agent_hooks\.interceptors\[0\]\.intercept must be async",
    ):
        AgentHooksConfig(
            agent_id="support-agent-v1",
            interceptors=(cast(AsyncInterceptor, SyncInterceptor()),),
            record_sink=record_sink,
        )


def test_agent_hooks_config_rejects_noncallable_interceptor() -> None:
    with pytest.raises(
        TypeError,
        match=r"agent_hooks\.interceptors\[0\]\.intercept must be callable",
    ):
        AgentHooksConfig(
            agent_id="support-agent-v1",
            interceptors=(cast(AsyncInterceptor, object()),),
            record_sink=record_sink,
        )


def test_agent_hooks_config_rejects_record_sink_subclass() -> None:
    class UnsafeRecordSink(RecordSink):
        pass

    with pytest.raises(TypeError, match="exact RecordSink instance"):
        AgentHooksConfig(
            agent_id="support-agent-v1",
            interceptors=(AllowInterceptor(),),
            record_sink=UnsafeRecordSink(max_records=100),
        )


def test_agent_hooks_config_rejects_invalid_record_sink() -> None:
    with pytest.raises(TypeError, match="exact RecordSink instance"):
        AgentHooksConfig(
            agent_id="support-agent-v1",
            interceptors=(AllowInterceptor(),),
            record_sink=cast(RecordSink, object()),
        )


def test_record_sink_is_bounded_and_drainable() -> None:
    sink = RecordSink(max_records=1)
    record = cast(Any, object())
    sink.write(record)

    with pytest.raises(BufferError, match="record sink is full"):
        sink.write(record)

    assert sink.snapshot() == (record,)
    assert sink.drain() == (record,)
    assert sink.snapshot() == ()


def test_record_sink_concurrent_writes_preserve_capacity() -> None:
    sink = RecordSink(max_records=3)
    start = Barrier(9)

    def write(record: object) -> bool:
        start.wait()
        try:
            sink.write(cast(Any, record))
        except BufferError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(write, index) for index in range(8)]
        start.wait()
        accepted = [future.result() for future in futures]

    assert sum(accepted) == 3
    assert len(sink.snapshot()) == 3


def test_record_sink_concurrent_drain_does_not_lose_accepted_writes() -> None:
    sink = RecordSink(max_records=32)
    start = Barrier(9)

    def write(record: int) -> int:
        start.wait()
        sink.write(cast(Any, record))
        return record

    def drain() -> tuple[int, ...]:
        start.wait()
        return cast(tuple[int, ...], sink.drain())

    with ThreadPoolExecutor(max_workers=9) as executor:
        write_futures = [executor.submit(write, index) for index in range(8)]
        drain_future = executor.submit(drain)
        accepted = [future.result() for future in write_futures]
        drained = drain_future.result()

    retained = cast(tuple[int, ...], sink.snapshot())
    assert sorted((*drained, *retained)) == sorted(accepted)


@pytest.mark.parametrize("max_records", [0, -1, True])
def test_record_sink_rejects_invalid_capacity(max_records: object) -> None:
    with pytest.raises(ValueError, match="RecordSink.max_records"):
        RecordSink(max_records=cast(Any, max_records))


def test_run_config_preserves_typed_agent_hooks_config() -> None:
    agent_hooks = AgentHooksConfig(
        agent_id="support-agent-v1",
        interceptors=(AllowInterceptor(),),
        record_sink=record_sink,
    )

    config = RunConfig(agent_hooks=agent_hooks)

    assert config.agent_hooks is agent_hooks


def test_run_config_coerces_equivalent_agent_hooks_dictionary() -> None:
    interceptor = AllowInterceptor()
    agent_hooks: dict[str, object] = {
        "agent_id": "support-agent-v1",
        "interceptors": (interceptor,),
        "record_sink": record_sink,
        "session_id": "session-1",
        "correlation_id": "correlation-1",
        "limits": {"max_interceptors": 2},
    }

    config = RunConfig(agent_hooks=agent_hooks)

    assert isinstance(config.agent_hooks, AgentHooksConfig)
    assert config.agent_hooks.agent_id == "support-agent-v1"
    assert config.agent_hooks.interceptors == (interceptor,)
    assert config.agent_hooks.session_id == "session-1"
    assert config.agent_hooks.correlation_id == "correlation-1"
    assert config.agent_hooks.limits.max_interceptors == 2


def test_run_config_rejects_unknown_agent_hooks_dictionary_fields() -> None:
    with pytest.raises(TypeError, match="Unknown run_config.agent_hooks settings: unknown"):
        RunConfig(
            agent_hooks={
                "agent_id": "support-agent-v1",
                "interceptors": (AllowInterceptor(),),
                "record_sink": record_sink,
                "unknown": True,
            }
        )
