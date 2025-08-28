import asyncio
import base64
import json
import os
import tempfile
import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import List, Optional, Union

import httpx
from autogen_core import CancellationToken, Component
from autogen_core.code_executor import CodeBlock, CodeExecutor, CodeResult
from pydantic import BaseModel, ConfigDict
from typing_extensions import Self

from ._jupyter_server import (
    JupyterKernelClient,
    PodJupyterClient,
    PodJupyterConnectionInfo,
    PodJupyterServer,
)
from ._utils import silence_pip


# Source below based from: https://github.com/microsoft/autogen/blob/main/python/packages/autogen-ext/src/autogen_ext/code_executors/docker_jupyter/_docker_jupyter.py
# Credit to original authors
# Original code Licensed under the MIT License.
# See the License file for the full license text.
@dataclass
class PodJupyterCodeResult(CodeResult):
    """(Experimental) A code result class for IPython code executor."""

    output_files: list[Path]


class PodJupyterCodeExecutorConfig(BaseModel):
    """Configuration for JupyterCodeExecutor"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    jupyter_server: Union[PodJupyterServer, PodJupyterConnectionInfo]
    kernel_name: str = "python3"
    timeout: int = 60
    output_dir: Optional[Union[Path, str]] = None


class PodJupyterCodeExecutor(CodeExecutor, Component[PodJupyterCodeExecutorConfig]):
    """(Experimental) A code executor class that executes code statefully using
    a Jupyter server supplied to this class.

    Each execution is stateful and can access variables created from previous
    executions in the same session.

    Args:
        jupyter_server (Union[JupyterConnectable, JupyterConnectionInfo]): The Jupyter server to use.
        kernel_name (str): The kernel name to use. Make sure it is installed.
            By default, it is "python3".
        timeout (int): The timeout for code execution, by default 60.
        output_dir (Path): The directory to save output files, by default None.

    Example of using it directly:

    .. code-block:: python

        import asyncio
        from autogen_core import CancellationToken
        from autogen_core.code_executor import CodeBlock
        from autogen_kubernetes.code_executors import PodJupyterCodeExecutor, PodJupyterServer


        async def main() -> None:
            async with PodJupyterServer() as jupyter_server:
                async with PodJupyterCodeExecutor(jupyter_server=jupyter_server) as executor:
                    code_blocks = [CodeBlock(code="print('hello world!')", language="python")]
                    code_result = await executor.execute_code_blocks(code_blocks, cancellation_token=CancellationToken())
                    print(code_result)


        asyncio.run(main())

    Example of using it with your own jupyter image:

    .. code-block:: python

        import asyncio
        from autogen_core import CancellationToken
        from autogen_core.code_executor import CodeBlock
        from autogen_kubernetes.code_executors import PodJupyterCodeExecutor, PodJupyterServer


        async def main() -> None:
            async with PodJupyterServer(image="your_custom_images_name", port=8888) as jupyter_server:
                async with PodJupyterCodeExecutor(jupyter_server=jupyter_server) as executor:
                    code_blocks = [CodeBlock(code="print('hello world!')", language="python")]
                    code_result = await executor.execute_code_blocks(code_blocks, cancellation_token=CancellationToken())
                    print(code_result)


        asyncio.run(main())

    Example of using it with :class:`~autogen_ext.tools.code_execution.PythonCodeExecutionTool`:

    .. code-block:: python

        import asyncio
        from autogen_agentchat.agents import AssistantAgent
        from autogen_kubernetes.code_executors import PodJupyterCodeExecutor, PodJupyterServer
        from autogen_ext.models.openai import OpenAIChatCompletionClient
        from autogen_ext.tools.code_execution import PythonCodeExecutionTool


        async def main() -> None:
            async with PodJupyterServer() as jupyter_server:
                async with PodJupyterCodeExecutor(jupyter_server=jupyter_server) as executor:
                    tool = PythonCodeExecutionTool(executor)
                    model_client = OpenAIChatCompletionClient(model="gpt-4o")
                    agent = AssistantAgent("assistant", model_client=model_client, tools=[tool])
                    result = await agent.run(task="What is the 10th Fibonacci number? Use Python to calculate it.")
                    print(result)


        asyncio.run(main())

    Example of using it inside a :class:`~autogen_agentchat.agents._code_executor_agent.CodeExecutorAgent`:

    .. code-block:: python

        import asyncio
        from autogen_agentchat.agents import CodeExecutorAgent
        from autogen_agentchat.messages import TextMessage
        from autogen_kubernetes.code_executors import PodJupyterCodeExecutor, PodJupyterServer
        from autogen_core import CancellationToken


        async def main() -> None:
            async with PodJupyterServer() as jupyter_server:
                async with PodJupyterCodeExecutor(jupyter_server=jupyter_server) as executor:
                    code_executor_agent = CodeExecutorAgent("code_executor", code_executor=executor)
                    task = TextMessage(
                        content='''Here is some code
                ```python
                print('Hello world')
                ```
                ''',
                        source="user",
                    )
                    response = await code_executor_agent.on_messages([task], CancellationToken())
                    print(response.chat_message)


        asyncio.run(main())

    """

    component_config_schema = PodJupyterCodeExecutorConfig
    component_provider_override = "autogen_kubernetes.code_executors.PodJupyterCodeExecutor"

    def __init__(
        self,
        jupyter_server: Union[PodJupyterServer, PodJupyterConnectionInfo],
        kernel_name: str = "python3",
        timeout: int = 60,
        output_dir: Path | None = None,
    ):
        if timeout < 1:
            raise ValueError("Timeout must be greater than or equal to 1.")

        if isinstance(jupyter_server, PodJupyterServer):
            self._connection_info = jupyter_server.connection_info
        elif isinstance(jupyter_server, PodJupyterConnectionInfo):
            self._connection_info = jupyter_server
        else:
            raise ValueError("jupyter_server must be a PodJupyterServer or PodJupyterConnectionInfo.")

        self._output_dir = output_dir
        if not self._output_dir:
            with tempfile.TemporaryDirectory() as temp_dir:
                self._output_dir = Path(temp_dir)
                self._output_dir.mkdir(exist_ok=True)

        self._jupyter_client = PodJupyterClient(self._connection_info)

        self._kernel_name = kernel_name
        self._timeout = timeout
        self._async_jupyter_kernel_client: Optional[JupyterKernelClient] = None
        self._kernel_id: Optional[str] = None

    async def _ensure_async_kernel_client(self) -> JupyterKernelClient:
        """Ensure that an async kernel client exists and return it."""
        if self._kernel_id is None:
            await self.start()
            assert self._kernel_id is not None
        if self._async_jupyter_kernel_client is None:
            self._async_jupyter_kernel_client = await self._jupyter_client.get_kernel_client(self._kernel_id)
        return self._async_jupyter_kernel_client

    async def execute_code_blocks(
        self, code_blocks: List[CodeBlock], cancellation_token: CancellationToken
    ) -> PodJupyterCodeResult:
        """(Experimental) Execute a list of code blocks and return the result.

        This method executes a list of code blocks as cells in the Jupyter kernel.
        See: https://jupyter-client.readthedocs.io/en/stable/messaging.html
        for the message protocol.

        Args:
            code_blocks (List[CodeBlock]): A list of code blocks to execute.

        Returns:
            PodJupyterCodeResult: The result of the code execution.
        """
        kernel_client = await self._ensure_async_kernel_client()
        # Wait for kernel to be ready using async client
        is_ready = await kernel_client.wait_for_ready(timeout_seconds=self._timeout)
        if not is_ready:
            return PodJupyterCodeResult(exit_code=1, output="ERROR: Kernel not ready", output_files=[])

        outputs: List[str] = []
        output_files: List[Path] = []
        for code_block in code_blocks:
            code = silence_pip(code_block.code, code_block.language)
            # Execute code using async client
            exec_task = asyncio.create_task(kernel_client.execute(code, timeout_seconds=self._timeout))
            cancellation_token.link_future(exec_task)
            result = await exec_task
            if result.is_ok:
                outputs.append(result.output)
                for data in result.data_items:
                    if data.mime_type == "image/png":
                        path = self._save_image(data.data)
                        outputs.append(path)
                        output_files.append(Path(path))
                    elif data.mime_type == "text/html":
                        path = self._save_html(data.data)
                        outputs.append(path)
                        output_files.append(Path(path))
                    else:
                        outputs.append(json.dumps(data.data))
            else:
                existing_output = "\n".join([str(output) for output in outputs])
                return PodJupyterCodeResult(
                    exit_code=1,
                    output=existing_output + "\nERROR: " + result.output,
                    output_files=output_files,
                )
        return PodJupyterCodeResult(
            exit_code=0,
            output="\n".join([str(output) for output in outputs]),
            output_files=output_files,
        )

    async def restart(self) -> None:
        """(Experimental) Restart a new session."""
        # Use async client to restart kernel
        if self._kernel_id is not None:
            await self._jupyter_client.restart_kernel(self._kernel_id)
        # Reset the clients to force recreation
        if self._async_jupyter_kernel_client is not None:
            await self._async_jupyter_kernel_client.stop()
            self._async_jupyter_kernel_client = None

    async def _wait_for_server(self) -> None:
        while True:
            try:
                response = await self._jupyter_client.list_kernel_specs()
                assert "kernelspecs" in response
                break
            except (
                httpx.ConnectError,
                httpx.HTTPStatusError,
                AssertionError,
                json.decoder.JSONDecodeError,
            ):
                await asyncio.sleep(1)

    async def start(self) -> None:
        """(Experimental) Start a new session."""
        try:
            await asyncio.wait_for(self._wait_for_server(), timeout=self._timeout)
            available_kernels = await self._jupyter_client.list_kernel_specs()
            if self._kernel_name not in available_kernels["kernelspecs"]:
                raise ValueError(f"Kernel {self._kernel_name} is not installed.")
            self._kernel_id = await self._jupyter_client.start_kernel(self._kernel_name)
        except asyncio.TimeoutError:
            warnings.warn(
                f"jupyter server not accessible after connection tries for {self._timeout} seconds. close context.",
                stacklevel=2,
            )
            raise

    def _save_image(self, image_data_base64: str) -> str:
        """Save image data to a file."""
        image_data = base64.b64decode(image_data_base64)
        filename = f"{uuid.uuid4().hex}.png"
        path = os.path.join(str(self._output_dir), filename)
        with open(path, "wb") as f:
            f.write(image_data)
        return os.path.abspath(path)

    def _save_html(self, html_data: str) -> str:
        """Save html data to a file."""
        filename = f"{uuid.uuid4().hex}.html"
        path = os.path.join(str(self._output_dir), filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(html_data)
        return os.path.abspath(path)

    async def stop(self) -> None:
        """Stop the kernel."""
        if self._kernel_id is not None:
            await self._jupyter_client.delete_kernel(self._kernel_id)
        if self._async_jupyter_kernel_client is not None:
            await self._async_jupyter_kernel_client.stop()
            self._async_jupyter_kernel_client = None
        await self._jupyter_client.close()

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.stop()

    def _to_config(self) -> PodJupyterCodeExecutorConfig:
        """(Experimental) Convert the component to a config object"""

        return PodJupyterCodeExecutorConfig(
            jupyter_server=self._connection_info,
            kernel_name=self._kernel_name,
            timeout=self._timeout,
            output_dir=self._output_dir,
        )

    @classmethod
    def _from_config(cls, config: PodJupyterCodeExecutorConfig) -> Self:
        """(Experimental) Create a component from a config object"""

        return cls(
            jupyter_server=config.jupyter_server,
            kernel_name=config.kernel_name,
            timeout=config.timeout,
            output_dir=(config.output_dir if not isinstance(config.output_dir, str) else Path(config.output_dir)),
        )
