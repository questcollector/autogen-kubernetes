import pytest
from autogen_kubernetes.code_executors import PodCommandLineCodeExecutor
from autogen_kubernetes_mcp._executor import make_executor, run_code
from conftest import kubernetes_enabled, state_kubernetes_enabled


@pytest.mark.skipif(not state_kubernetes_enabled, reason="kubernetes not accessible")
@pytest.mark.asyncio
async def test_vanilla_executor() -> None:
    async with make_executor({}) as executor:
        assert isinstance(executor, PodCommandLineCodeExecutor)
        result = await run_code(executor, 'print("Hello")')
        assert "Hello" in result
