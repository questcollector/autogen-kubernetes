import socket

import pytest
from autogen_core import CancellationToken
from autogen_kubernetes.code_executors import (
    PodCommandLineCodeExecutor,
    PodJupyterCodeExecutor,
)
from autogen_kubernetes_mcp._executor import make_executor, run_code
from conftest import kubernetes_enabled, state_kubernetes_enabled


def can_resolve_svc_fqdn() -> bool:
    try:
        socket.gethostbyname("kubernetes.default")
        return True
    except socket.error:
        return False


@pytest.mark.skipif(not state_kubernetes_enabled, reason="kubernetes not accessible")
@pytest.mark.asyncio
async def test_vanilla_commandline_executor() -> None:
    instances = await make_executor({"type": "commandline"})
    executor = instances[0]
    assert isinstance(executor, PodCommandLineCodeExecutor)
    result = await run_code(executor, 'print("Hello")', CancellationToken())
    assert "Hello" in result
    for instance in instances:
        await instance.stop()


@pytest.mark.skipif(not state_kubernetes_enabled or not can_resolve_svc_fqdn(), reason="kubernetes not accessible")
@pytest.mark.asyncio
async def test_vanilla_jupyter_executor() -> None:
    instances = await make_executor({"type": "jupyter"})
    executor = instances[0]
    assert isinstance(executor, PodJupyterCodeExecutor)
    result = await run_code(executor, 'print("Hello")', CancellationToken())
    assert "Hello" in result
    for instance in instances:
        await instance.stop()
