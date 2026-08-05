from __future__ import annotations

import subprocess
import sys
from textwrap import dedent


def _run_isolated_python(source: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", dedent(source)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_import_agents_does_not_load_agent_hooks_when_dependency_is_blocked() -> None:
    result = _run_isolated_python(
        """
        import importlib.abc
        import sys

        class BlockAgentHooks(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                if fullname == "agent_hooks" or fullname.startswith("agent_hooks."):
                    raise ModuleNotFoundError("blocked agent_hooks dependency")
                return None

        sys.meta_path.insert(0, BlockAgentHooks())

        import agents

        config = agents.RunConfig()

        assert config.agent_hooks is None
        assert "agent_hooks" not in sys.modules
        assert "agents.extensions.agent_hooks" not in sys.modules
        """
    )

    assert result.returncode == 0, result.stderr


def test_disabled_run_failure_does_not_import_agent_hooks_dependency() -> None:
    result = _run_isolated_python(
        """
        import importlib.abc
        import sys

        class BlockAgentHooks(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                if fullname == "agent_hooks" or fullname.startswith("agent_hooks."):
                    raise ModuleNotFoundError("blocked agent_hooks dependency")
                return None

        sys.meta_path.insert(0, BlockAgentHooks())

        import agents

        try:
            agents.Runner.run_sync(
                agents.Agent(name="disabled-agent"),
                "hello",
                run_config={
                    "tool_execution": {"max_function_tool_concurrency": 0},
                },
            )
        except ValueError as error:
            assert "max_function_tool_concurrency must be at least 1" in str(error)
        else:
            raise AssertionError("expected invalid disabled-run configuration to fail")

        assert "agent_hooks" not in sys.modules
        assert "agents.extensions.agent_hooks" not in sys.modules
        """
    )

    assert result.returncode == 0, result.stderr


def test_agent_hooks_extension_import_error_points_to_optional_extra() -> None:
    result = _run_isolated_python(
        """
        import importlib.abc
        import sys

        class BlockAgentHooks(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                if fullname == "agent_hooks" or fullname.startswith("agent_hooks."):
                    raise ModuleNotFoundError("blocked agent_hooks dependency")
                return None

        sys.meta_path.insert(0, BlockAgentHooks())

        try:
            import agents.extensions.agent_hooks
        except ImportError as error:
            message = str(error)
            assert "requires the 'agent-hooks' extra" in message
            assert "pip install 'openai-agents[agent-hooks]'" in message
            assert isinstance(error.__cause__, ImportError)
        else:
            raise AssertionError("expected the blocked optional dependency import to fail")
        """
    )

    assert result.returncode == 0, result.stderr
