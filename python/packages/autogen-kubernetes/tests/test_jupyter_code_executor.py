import logging
import re
import socket

import pytest
from autogen_core import CancellationToken
from autogen_core.code_executor import CodeBlock
from autogen_kubernetes.code_executors import PodJupyterCodeExecutor, PodJupyterServer
from conftest import kubernetes_enabled, state_kubernetes_enabled

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@pytest.mark.skipif(not state_kubernetes_enabled, reason="kubernetes not accessible")
@pytest.mark.asyncio
async def test_pod_default(generated_jupyter_pod_regex: str) -> None:
    async with PodJupyterServer() as jupyter_server:
        connection_info = jupyter_server.connection_info
        try:
            socket.gethostbyname(connection_info.host)
        except socket.error:
            pytest.skip("cannot access jupyter server")
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
