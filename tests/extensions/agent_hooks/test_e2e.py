from __future__ import annotations

from typing import cast

import httpx
import pytest
from agent_hooks import (
    ALLOW,
    AgentContext,
    Decision,
    InterceptionPoint,
    Verdict,
)
from openai import AsyncOpenAI

from agents import (
    Agent,
    ModelSettings,
    OpenAIChatCompletionsModel,
    RunConfig,
    Runner,
    ToolExecutionConfig,
    function_tool,
)
from agents.extensions.agent_hooks import AgentHooksConfig, RecordSink
from agents.retry import ModelRetrySettings

_OLLAMA_BASE_URL = "http://localhost:11434/v1"


class _DenySensitiveTool:
    async def intercept(self, context: AgentContext, /) -> Verdict:
        if context.get("interception_point") == InterceptionPoint.PRE_TOOL_CALL.value:
            tool_call = context.get("tool_call")
            if isinstance(tool_call, dict) and tool_call.get("name") == "record_sensitive_action":
                return Verdict(decision=Decision.DENY, reason="e2e:tool_not_permitted")
        return ALLOW


async def _require_local_qwen() -> str:
    try:
        async with httpx.AsyncClient(timeout=3.0, trust_env=False) as client:
            response = await client.get(f"{_OLLAMA_BASE_URL}/models")
            response.raise_for_status()
            payload: object = response.json()
    except (httpx.HTTPError, ValueError):
        pytest.skip("Ollama OpenAI-compatible endpoint is unavailable")

    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        pytest.skip("Ollama returned an invalid model inventory")
    model_ids: list[str] = []
    for item in payload["data"]:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        if isinstance(model_id, str):
            model_ids.append(model_id)
    qwen_models = [model_id for model_id in model_ids if "qwen" in model_id.lower()]
    if not qwen_models:
        pytest.skip("Ollama has no locally installed Qwen model")
    return sorted(qwen_models, key=lambda model_id: ("qwen2.5" not in model_id.lower(), model_id))[
        0
    ]


@pytest.mark.e2e
@pytest.mark.serial
@pytest.mark.allow_call_model_methods
async def test_local_qwen_denied_tool_has_no_side_effect_and_complete_records() -> None:
    model_id = await _require_local_qwen()
    side_effects: list[str] = []
    records = RecordSink(max_records=256)

    @function_tool(name_override="record_sensitive_action")
    async def record_sensitive_action(value: str) -> str:
        """Record a test-only sensitive action."""
        side_effects.append(value)
        return "recorded"

    client = AsyncOpenAI(
        base_url=_OLLAMA_BASE_URL,
        api_key="ollama-local-placeholder",
        max_retries=0,
        timeout=300.0,
    )
    try:
        model = OpenAIChatCompletionsModel(model=model_id, openai_client=client)
        agent: Agent[None] = Agent(
            name="local-qwen-governed-agent",
            instructions=(
                "Call record_sensitive_action once with value 'denied-e2e-probe'. "
                "After receiving the tool result, answer briefly without calling another tool."
            ),
            model=model,
            tools=[record_sensitive_action],
        )
        result = await Runner.run(
            agent,
            "Perform the requested test action and then report whether it was permitted.",
            max_turns=3,
            run_config=RunConfig(
                agent_hooks=AgentHooksConfig(
                    agent_id="local-qwen-agent-hooks-e2e",
                    interceptors=(_DenySensitiveTool(),),
                    record_sink=records,
                ),
                trace_include_sensitive_data=False,
                model_settings=ModelSettings(
                    temperature=0,
                    tool_choice="required",
                    parallel_tool_calls=False,
                    max_tokens=128,
                    store=False,
                    retry=ModelRetrySettings(max_retries=0),
                ),
                tool_execution=ToolExecutionConfig(max_function_tool_concurrency=1),
            ),
        )
    finally:
        await client.close()

    output = cast(str, result.final_output)
    assert output.strip()
    assert side_effects == []

    points = [record.interception_point for record in records]
    assert points[:2] == [InterceptionPoint.AGENT_STARTUP, InterceptionPoint.INPUT]
    assert points[-2:] == [InterceptionPoint.OUTPUT, InterceptionPoint.AGENT_SHUTDOWN]
    assert points.count(InterceptionPoint.PRE_TOOL_CALL) >= 1
    assert InterceptionPoint.POST_TOOL_CALL not in points
    assert points.count(InterceptionPoint.PRE_MODEL_CALL) == points.count(
        InterceptionPoint.POST_MODEL_CALL
    )
    assert points.count(InterceptionPoint.AGENT_SHUTDOWN) == 1
    assert [record.sequence for record in records] == list(range(len(records)))
    assert len({record.session_id for record in records}) == 1
    assert all(record.identity_provider == "jcs-sha256" for record in records)
    assert all(record.composition.profile.value == "sequential/run_all" for record in records)
