import asyncio
import inspect
import logging
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx as hx
import pytest
from autogen_agentchat.agents import CodeExecutorAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_core.code_executor import (
    Alias,
    CodeBlock,
    FunctionWithRequirements,
    ImportFromModule,
    with_requirements,
)
from autogen_kubernetes.code_executors import PodCommandLineCodeExecutor
from conftest import kubernetes_enabled, state_kubernetes_enabled

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@pytest.mark.skipif(not state_kubernetes_enabled, reason="kubernetes not accessible")
@pytest.mark.asyncio
async def test_pod_spec_yaml_file() -> None:
    yaml_file_path = Path(os.path.dirname(__file__)) / Path("spec-files/test-pod.yaml")
    async with PodCommandLineCodeExecutor(pod_spec=str(yaml_file_path)) as executor:
        code_result = await executor.execute_code_blocks(
            code_blocks=[
                CodeBlock(language="python", code="print('Hello, World!')"),
            ],
            cancellation_token=CancellationToken(),
        )
        logger.info(code_result)
        assert code_result.exit_code == 0
        assert "Hello, World!" in code_result.output


@pytest.mark.skipif(not state_kubernetes_enabled, reason="kubernetes not accessible")
@pytest.mark.asyncio
async def test_pod_spec_yaml_str(pod_yaml_str: Callable[[str], str]) -> None:
    async with PodCommandLineCodeExecutor(pod_spec=pod_yaml_str("test-pod-yaml-str")) as executor:
        code_result = await executor.execute_code_blocks(
            code_blocks=[
                CodeBlock(language="python", code="print('Hello, World!')"),
            ],
            cancellation_token=CancellationToken(),
        )
        assert code_result.exit_code == 0
        assert "Hello, World!" in code_result.output


@pytest.mark.skipif(not state_kubernetes_enabled, reason="kubernetes not accessible")
@pytest.mark.asyncio
async def test_pod_spec_v1_pod(v1_pod: Callable[[str, str], Any]) -> None:
    v1_pod_spec = v1_pod("test-pod-spec-v1-pod", "autogen-executor")
    async with PodCommandLineCodeExecutor(pod_spec=v1_pod_spec) as executor:
        code_result = await executor.execute_code_blocks(
            code_blocks=[
                CodeBlock(language="python", code="print('Hello, World!')"),
            ],
            cancellation_token=CancellationToken(),
        )
        assert code_result.exit_code == 0
        assert "Hello, World!" in code_result.output


@pytest.mark.skipif(not state_kubernetes_enabled, reason="kubernetes not accessible")
@pytest.mark.asyncio
async def test_pod_spec_dict(pod_spec: Callable[[str], dict[str, Any]]) -> None:
    pod_spec_dict = pod_spec("test-pod-spec-dict")
    async with PodCommandLineCodeExecutor(pod_spec=pod_spec_dict) as executor:
        code_result = await executor.execute_code_blocks(
            code_blocks=[
                CodeBlock(language="python", code="print('Hello, World!')"),
            ],
            cancellation_token=CancellationToken(),
        )
        assert code_result.exit_code == 0
        assert "Hello, World!" in code_result.output


@pytest.mark.skipif(not state_kubernetes_enabled, reason="kubernetes not accessible")
@pytest.mark.asyncio
async def test_pod_default(generated_pod_name_regex: str) -> None:
    async with PodCommandLineCodeExecutor() as executor:
        code_result = await executor.execute_code_blocks(
            code_blocks=[
                CodeBlock(language="python", code="print('Hello, World!')"),
            ],
            cancellation_token=CancellationToken(),
        )
        assert re.fullmatch(generated_pod_name_regex, executor._pod_name) is not None
        assert code_result.exit_code == 0
        assert "Hello, World!" in code_result.output


@pytest.mark.skipif(not state_kubernetes_enabled, reason="kubernetes not accessible")
@pytest.mark.asyncio
async def test_timeout_error() -> None:
    with pytest.raises(ValueError):
        PodCommandLineCodeExecutor(image="python:3-slim", timeout=0)


@pytest.mark.skipif(not state_kubernetes_enabled, reason="kubernetes not accessible")
@pytest.mark.asyncio
async def test_relative_path_error() -> None:
    with pytest.raises(ValueError):
        PodCommandLineCodeExecutor(image="python:3-slim", workspace_path="./workspace")


@pytest.mark.skipif(not state_kubernetes_enabled, reason="kubernetes not accessible")
@pytest.mark.asyncio
async def test_volume_yaml_file(generated_pod_name_regex: str) -> None:
    yaml_file_path = Path(os.path.dirname(__file__)) / Path("spec-files/test-volume.yaml")
    async with PodCommandLineCodeExecutor(volume=str(yaml_file_path)) as executor:
        code_result = await executor.execute_code_blocks(
            code_blocks=[
                CodeBlock(language="python", code="print('Hello, World!')"),
            ],
            cancellation_token=CancellationToken(),
        )
        assert re.fullmatch(generated_pod_name_regex, executor._pod_name) is not None
        assert code_result.exit_code == 0
        assert "Hello, World!" in code_result.output


@pytest.mark.skipif(not state_kubernetes_enabled, reason="kubernetes not accessible")
@pytest.mark.asyncio
async def test_volume_yaml_str(generated_pod_name_regex: str) -> None:
    yaml_file_path = Path(os.path.dirname(__file__)) / Path("spec-files/test-volume.yaml")
    async with PodCommandLineCodeExecutor(volume=yaml_file_path.read_text()) as executor:
        code_result = await executor.execute_code_blocks(
            code_blocks=[
                CodeBlock(language="python", code="print('Hello, World!')"),
            ],
            cancellation_token=CancellationToken(),
        )
        assert re.fullmatch(generated_pod_name_regex, executor._pod_name) is not None
        assert code_result.exit_code == 0
        assert "Hello, World!" in code_result.output


@pytest.mark.skipif(not state_kubernetes_enabled, reason="kubernetes not accessible")
@pytest.mark.asyncio
async def test_volume_dict(volume_dict: Callable[[str], dict[str, Any]], generated_pod_name_regex: str) -> None:
    volume = volume_dict("test")
    async with PodCommandLineCodeExecutor(volume=volume) as executor:
        code_result = await executor.execute_code_blocks(
            code_blocks=[
                CodeBlock(language="python", code="print('Hello, World!')"),
            ],
            cancellation_token=CancellationToken(),
        )
        assert re.fullmatch(generated_pod_name_regex, executor._pod_name) is not None
        assert code_result.exit_code == 0
        assert "Hello, World!" in code_result.output


@pytest.mark.skipif(not state_kubernetes_enabled, reason="kubernetes not accessible")
@pytest.mark.asyncio
async def test_volume_V1Volume(v1_volume: Callable[[str], Any], generated_pod_name_regex: str) -> None:
    v1_volume_obj = v1_volume("test")
    async with PodCommandLineCodeExecutor(volume=v1_volume_obj) as executor:
        code_result = await executor.execute_code_blocks(
            code_blocks=[
                CodeBlock(language="python", code="print('Hello, World!')"),
            ],
            cancellation_token=CancellationToken(),
        )
        assert re.fullmatch(generated_pod_name_regex, executor._pod_name) is not None
        assert code_result.exit_code == 0
        assert "Hello, World!" in code_result.output


@pytest.mark.skipif(not state_kubernetes_enabled, reason="kubernetes not accessible")
@pytest.mark.asyncio
async def test_pod_name_error() -> None:
    invalid_pod_name = "!%#$SDAbsdbawpup230bkvbl;kouqw98q;asdnlkjdvnblafjdg8WU0 2Q3Y5OUDAJFNBJADBVNA;DWERY"
    with pytest.raises(ValueError):
        PodCommandLineCodeExecutor(pod_name=invalid_pod_name)


@pytest.mark.skipif(not state_kubernetes_enabled, reason="kubernetes not accessible")
@pytest.mark.asyncio
async def test_func_modules(generated_pod_name_regex: str) -> None:
    test_function = FunctionWithRequirements.from_str(
        (f"{inspect.getsource(kubernetes_enabled)}" "\nkubernetes_enabled()"),
        ["kubernetes"],
        [
            ImportFromModule(module="kubernetes.client", imports=("CoreV1Api",)),
            ImportFromModule(module="kubernetes.config", imports=("load_config",)),
        ],
    )
    async with PodCommandLineCodeExecutor(functions=[test_function]) as executor:
        code_result = await executor.execute_code_blocks(
            code_blocks=[
                CodeBlock(language="python", code="print('Hello, World!')"),
            ],
            cancellation_token=CancellationToken(),
        )
        assert re.fullmatch(generated_pod_name_regex, executor._pod_name) is not None
        assert code_result.exit_code == 0
        assert "Hello, World!" in code_result.output


@with_requirements(python_packages=["httpx"], global_imports=[Alias(name="httpx", alias="hx"), ImportFromModule(module="typing", imports=("Any",)),],)  # fmt: skip
def load_data() -> Any:
    """
    fetch cat fact api

    Returns:
        Any: status for request to cat fact api
    """
    client = hx.Client()
    if client.base_url is None:
        pass
    return "success"


@pytest.mark.skipif(not state_kubernetes_enabled, reason="kubernetes not accessible")
@pytest.mark.asyncio
async def test_func_modules_with_requirements(generated_pod_name_regex: str) -> None:
    async with PodCommandLineCodeExecutor(functions=[load_data]) as executor:
        code = f"from {executor._functions_module} import load_data;print(load_data())"
        code_result = await executor.execute_code_blocks(
            code_blocks=[
                CodeBlock(language="python", code=code),
            ],
            cancellation_token=CancellationToken(),
        )
        assert re.fullmatch(generated_pod_name_regex, executor._pod_name) is not None
        assert code_result.exit_code == 0
        assert "success" in code_result.output


@pytest.mark.skipif(not state_kubernetes_enabled, reason="kubernetes not accessible")
@pytest.mark.asyncio
async def test_with_code_executor_agent() -> None:
    async with PodCommandLineCodeExecutor() as executor:
        code_executor_agent = CodeExecutorAgent("code_executor", code_executor=executor)
        task = TextMessage(
            content="""Here is some code
```python
print('Hello world!')
```
""",
            source="user",
        )
        response = await code_executor_agent.on_messages([task], CancellationToken())
        assert "Hello world!" in response.chat_message.to_model_text()


@pytest.mark.skipif(not state_kubernetes_enabled, reason="kubernetes not accessible")
@pytest.mark.asyncio
async def test_load_component() -> None:
    executor = PodCommandLineCodeExecutor()
    dumped_config = executor.dump_component()
    loaded_executor = PodCommandLineCodeExecutor.load_component(dumped_config)

    assert isinstance(executor, PodCommandLineCodeExecutor)
    assert executor._pod_name == loaded_executor._pod_name


@pytest.mark.skipif(not state_kubernetes_enabled, reason="kubernetes not accessible")
@pytest.mark.asyncio
async def test_pod_exec_cancellation(generated_pod_name_regex: str) -> None:
    cancellation_token = CancellationToken()
    async with PodCommandLineCodeExecutor() as executor:
        coro = executor.execute_code_blocks(
            code_blocks=[
                CodeBlock(
                    language="python",
                    code="""import time
time.sleep(10)
print('Hello, World!')""",
                ),
            ],
            cancellation_token=cancellation_token,
        )
        await asyncio.sleep(5)
        cancellation_token.cancel()
        code_result = await coro

        assert re.fullmatch(generated_pod_name_regex, executor._pod_name) is not None
        assert code_result.exit_code == 1 and "cancelled" in code_result.output
