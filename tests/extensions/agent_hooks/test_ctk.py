from __future__ import annotations

import pytest
from agent_hooks.ctk import load_vectors, run_vector

from tests.extensions.agent_hooks._ctk_harness import (
    OpenAIAgentsCTKHarness,
    project_vector_to_fixed_surface,
)

EXECUTED_VECTOR_IDS = (
    "AH-CTK-001",
    "AH-CTK-002",
    "AH-CTK-003",
    "AH-CTK-010",
    "AH-CTK-011",
    "AH-CTK-012",
    "AH-CTK-021",
    "AH-CTK-022",
    "AH-CTK-040",
    "AH-CTK-060",
    "AH-CTK-070",
    "AH-CTK-071",
    "AH-CTK-074",
    "AH-CTK-081",
    "AH-CTK-097",
    "AH-CTK-100",
    "AH-CTK-102",
    "AH-CTK-104",
)

SKIPPED_VECTOR_CATEGORIES = {
    "approval resolver unsupported": {
        "AH-CTK-030",
        "AH-CTK-031",
        "AH-CTK-032",
        "AH-CTK-072",
        "AH-CTK-073",
        "AH-CTK-080",
        "AH-CTK-082",
        "AH-CTK-083",
        "AH-CTK-086",
        "AH-CTK-088",
        "AH-CTK-098",
        "AH-CTK-099",
        "AH-CTK-103",
        "AH-CTK-105",
    },
    "composition profile outside fixed sequential/run_all": {
        "AH-CTK-084",
        "AH-CTK-085",
        "AH-CTK-087",
        "AH-CTK-094",
    },
    "identity provider outside fixed jcs-sha256": {"AH-CTK-089", "AH-CTK-096"},
    "numeric JSON capability not declared": {"AH-CTK-090", "AH-CTK-091", "AH-CTK-095"},
    "non-string tool result outside adapter projection": {"AH-CTK-020"},
    "free-form verdict metadata outside adapter projection": {
        "AH-CTK-050",
        "AH-CTK-092",
        "AH-CTK-093",
    },
    "zero external interceptors rejected by public config": {"AH-CTK-061", "AH-CTK-101"},
}


def test_ctk_vector_inventory_is_explicit_and_exhaustive() -> None:
    official_ids = {item["id"] for item in load_vectors()}
    skipped_ids = set().union(*SKIPPED_VECTOR_CATEGORIES.values())

    assert set(EXECUTED_VECTOR_IDS).isdisjoint(skipped_ids)
    assert set(EXECUTED_VECTOR_IDS) | skipped_ids == official_ids


@pytest.mark.parametrize("vector_id", EXECUTED_VECTOR_IDS)
async def test_applicable_ctk_vector_uses_production_runner(vector_id: str) -> None:
    vector = next(item for item in load_vectors() if item["id"] == vector_id)

    result = await run_vector(
        OpenAIAgentsCTKHarness(),
        project_vector_to_fixed_surface(vector),
    )

    assert result.status == "pass", {"detail": result.detail, "failures": result.failures}
