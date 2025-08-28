import logging
import os
import re
import socket
from pathlib import Path

import pytest
from autogen_core import CancellationToken
from autogen_core.code_executor import CodeBlock
from autogen_kubernetes.code_executors import PodJupyterCodeExecutor, PodJupyterServer
from conftest import kubernetes_enabled, state_kubernetes_enabled

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def can_resolve_svc_fqdn() -> bool:
    try:
        socket.gethostbyname("kubernetes.default")
        return True
    except socket.error:
        return False


@pytest.mark.skipif(
    not state_kubernetes_enabled or not can_resolve_svc_fqdn(),
    reason="kubernetes not accessible",
)
@pytest.mark.asyncio
async def test_pod_default(generated_jupyter_pod_regex: str) -> None:
    async with PodJupyterServer() as jupyter_server:
        async with PodJupyterCodeExecutor(jupyter_server) as executor:
            code_result = await executor.execute_code_blocks(
                code_blocks=[
                    CodeBlock(language="python", code="print('Hello, World!')"),
                ],
                cancellation_token=CancellationToken(),
            )
            assert re.fullmatch(generated_jupyter_pod_regex, jupyter_server._pod_name) is not None
            assert code_result.exit_code == 0
            assert "Hello, World!" in code_result.output


@pytest.mark.skipif(
    not state_kubernetes_enabled or not can_resolve_svc_fqdn(),
    reason="kubernetes not accessible",
)
@pytest.mark.asyncio
async def test_pod_yaml_file() -> None:
    yaml_file_path = Path(os.path.dirname(__file__)) / Path("spec-files/test-jupyter.yaml")

    async with PodJupyterServer(
        pod_spec=yaml_file_path,
        secret_spec=yaml_file_path,
        service_spec=yaml_file_path,
    ) as jupyter_server:
        async with PodJupyterCodeExecutor(jupyter_server) as executor:
            code_result = await executor.execute_code_blocks(
                code_blocks=[
                    CodeBlock(language="python", code="print('Hello, World!')"),
                ],
                cancellation_token=CancellationToken(),
            )
            assert jupyter_server.connection_info.host == "test-jupyter.default.svc.cluster.local"
            assert jupyter_server.connection_info.port == 8888
            assert code_result.exit_code == 0
            assert "Hello, World!" in code_result.output
