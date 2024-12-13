# File based from: https://github.com/microsoft/autogen/blob/main/python/packages/autogen-ext/src/autogen_ext/code_executors/_docker_code_executor.py
# Credit to original authors
# Original code Licensed under the MIT License.
# Modifications made by kiyoung you(questcollector)
# See the License file for the full license text.

from __future__ import annotations

import asyncio
import logging
import re
import shlex
import sys
import time
import uuid
import warnings
from collections.abc import Coroutine, Sequence
from hashlib import sha256
from pathlib import Path
from types import TracebackType
from typing import Any, Callable, ClassVar, Dict, List, Optional, ParamSpec, Type, Union

import kubernetes.client
import kubernetes.config
import yaml
from autogen_core import CancellationToken
from autogen_core.code_executor import (
    CodeBlock,
    CodeExecutor,
)
from httpx import HTTPStatusError

from ._utils import (
    POD_NAME_PATTERN,
    CommandLineCodeResult,
    FunctionWithRequirements,
    FunctionWithRequirementsStr,
    StreamChannel,
    build_python_functions_file,
    clean_none_value,
    create_pod,
    delete_pod,
    get_file_name_from_content,
    get_pod_logs,
    lang_to_cmd,
    pod_exec_stream,
    silence_pip,
    wait_for_ready,
)

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


A = ParamSpec("A")


# TODO autogen compatibility
class PodCommandLineCodeExecutor(CodeExecutor):
    """Executes code through a command line environment in a container on Kubernetes Pod.

    The executor first saves each code block in a file in the working
    directory, and then executes the code file in the container.
    The executor executes the code blocks in the order they are received.
    Currently, the executor only supports Python and shell scripts.
    For Python code, use the language "python" for the code block.
    For shell scripts, use the language "bash", "shell", or "sh" for the code
    block.

    Args:
        image (_type_, optional): container image to use for code execution.
            Defaults to "python:3-slim".
        pod_name (Optional[str], optional): Name of the kubernetes pod
            which is created. If None, will autogenerate a name. Defaults to None.
        timeout (int, optional): The timeout for code execution. Defaults to 60.
        workspace_path (Union[Path, str], optional): The workspace directory for code executor container.
            Generated code script files will be stored in this directory.
            Not supports relative path(use absolute path). Defaults to Path("/workspace")
        namespace (str, optional): Name of the namespace of kubernetes cluster on which code executor pod will be created.
            Defaults to "default".
        volume (Union[dict, str, Path, kubernetes.client.models.V1Volume, None], optional): The volume for the code execution pod.
            Supports the formats of a dictionary, a YAML string, a YAML file path, and kubernetes `V1Volume` model format.
            Must conform to the kubernetes `V1Volume` model format.
            Must have appropriate access mode(such as ReadWriteMany, ReadWriteOnce, ReadWriteOncePod, in case of PersistentVolumeClaim)
            If None, no volume attached to code executor pod. Defaults to None.
        pod_spec (Union[dict, str, Path, kubernetes.client.models.V1Pod, None], optional): pod specification for code executor.
            Must contain a container which name is "autogen-executor" for execution for codes.
            Supports the formats of a dictionary, a YAML string, a YAML file path, and a kubernetes V1Pod model.
            Must conform to the kubernetes `V1Pod` model format.
            If None, will use above parameters to create code executor pod. Defaults to None.
        kube_config_file (Union[Path, str, None], optional): kubernetes configuration file(kubeconfig) path.
            If None, will use KUBECONFIG environment variables or service account token(incluster config).
            Using service account token, service account must have at least those namespaced permissions below.
            [
              {
                "resource": "pods", "verb": ["get", "create", "delete"]
              },
              {
                "resource": "pods/status", "verb": ["get"]
              },
              {
                "resource": "pods/exec", "verb": ["create"]
              },
              {
                "resource": "pods/log", "verb": ["get"]
              }
            ]
        auto_remove (bool, optional): If true, will automatically remove the
            container when remove is called, when the context manager exits or when
            the Python process exits with atext. Defaults to True.
        functions (List[Union[FunctionWithRequirements[Any, A], Callable[..., Any]]]): A list of functions that are available to the code executor. Default is an empty list.
        functions_module (str, optional): The name of the module that will be created to store the functions. Defaults to "functions".
    """

    SUPPORTED_LANGUAGES: ClassVar[List[str]] = [
        "bash",
        "shell",
        "sh",
        "pwsh",
        "powershell",
        "ps1",
        "python",
    ]

    LANGUAGE_FILE_EXTENSION: ClassVar[Dict[str, str]] = {
        "python": "py",
        "bash": "sh",
        "shell": "sh",
        "sh": "sh",
        "pwsh": "ps1",
        "powershell": "ps1",
        "ps1": "ps1",
    }

    FUNCTION_PROMPT_TEMPLATE: ClassVar[
        str
    ] = """You have access to the following user defined functions. They can be accessed from the module called `$module_name` by their function names.

For example, if there was a function called `foo` you could import it by writing `from $module_name import foo`

$functions"""

    def __init__(
        self,
        image: str = "python:3-slim",
        pod_name: Optional[str] = None,
        *,
        timeout: int = 60,
        workspace_path: Union[Path, str] = Path("/workspace"),
        namespace: str = "default",
        volume: Union[dict[str, Any], str, Path, Type[kubernetes.client.models.V1Volume], None] = None,
        pod_spec: Union[dict[str, Any], str, Path, Type[kubernetes.client.models.V1Pod], None] = None,
        kube_config_file: Union[Path, str, None] = None,
        auto_remove: bool = True,
        functions: Sequence[
            Union[
                FunctionWithRequirements[Any, A],
                Callable[..., Any],
                FunctionWithRequirementsStr,
            ]
        ] = [],
        functions_module: str = "functions",
    ):
        if timeout < 1:
            raise ValueError("Timeout must be greater than or equal to 1.")

        ## kubeconfig
        if isinstance(kube_config_file, str):
            kube_config_file = Path(kube_config_file)

        if kube_config_file is None:  ## configuration from default kubeconfig or incluster
            kubernetes.config.load_config()  # type: ignore
        else:
            kubernetes.config.load_config(config_file=kube_config_file)  # type: ignore
        self._kube_config = kubernetes.client.Configuration.get_default_copy()  # type: ignore

        ## workspace
        if isinstance(workspace_path, str):  ## path string to Path
            workspace_path = Path(workspace_path)

        if not workspace_path.is_absolute():  ## validate workspace_path
            raise ValueError("Not supports workspace path as relative path")

        self._workspace_path: Path = workspace_path

        ## volume
        volume_json = None
        if isinstance(volume, str):
            try:
                if Path(volume).exists():  # YAML file path string
                    volume = Path(volume)
            except OSError:
                pass

        if isinstance(volume, Path):  # YAML file to dict
            volume_json = yaml.safe_load(volume.read_text(encoding="utf-8"))
        elif isinstance(volume, str):  # YAML string to dict
            volume_json = yaml.safe_load(volume)
        elif isinstance(volume, kubernetes.client.models.V1Volume):
            volume_json = clean_none_value(dict(volume.to_dict()))
        elif isinstance(volume, dict):
            volume_json = volume

        self._volume = volume_json

        ## pod_name
        if pod_name is None:
            self._pod_name = f"autogen-code-exec-{uuid.uuid4()}"
        else:
            self._pod_name = pod_name

        if not re.fullmatch(POD_NAME_PATTERN, self._pod_name):
            raise ValueError(
                "Pod name validation failed: pod name must start and end with lower alphanumeric characters, "
                "and contain only lowercase alphanumeric or dash('-') characters."
            )

        self._container_name = "autogen-executor"
        self._namespace = namespace
        self._timeout = timeout
        self._auto_remove = auto_remove
        self._image = image

        ## pod_spec
        if isinstance(pod_spec, str):  # YAML file path string
            try:
                if Path(pod_spec).exists():
                    pod_spec = Path(pod_spec)
            except OSError:
                pass

        pod_spec_dict = {}
        if isinstance(pod_spec, Path):  # YAML file to dict
            pod_spec_dict = yaml.safe_load(pod_spec.read_text(encoding="utf-8"))
        elif isinstance(pod_spec, str):  # YAML string to dict
            pod_spec_dict = yaml.safe_load(pod_spec)
        elif isinstance(pod_spec, kubernetes.client.models.V1Pod):
            pod_spec_dict = clean_none_value(dict(pod_spec.to_dict()))
        elif isinstance(pod_spec, dict):
            pod_spec_dict = pod_spec

        if pod_spec is None:  ## create pod_spec from other parameters
            self._pod = self._define_pod_spec()
        else:
            self._pod = pod_spec_dict

        self._validate_pod()

        if not functions_module.isidentifier():
            raise ValueError("Module name must be a valid Python identifier")

        self._functions_module = functions_module
        self._functions = functions
        # Setup could take some time so we intentionally wait for the first code block to do it.
        if len(functions) > 0:
            self._setup_functions_complete = False
        else:
            self._setup_functions_complete = True

        self._running = False

    @property
    def timeout(self) -> int:
        """(Experimental) The timeout for code execution."""
        return self._timeout

    @property
    def workspace_path(self) -> Path:
        """(Experimental) The working directory for the code execution."""
        return self._workspace_path

    def _validate_pod(self) -> None:
        try:
            # pod_name, namespace
            self._pod_name = self._pod["metadata"]["name"]
            self._namespace = self._pod["metadata"]["namespace"]

            # container
            executor_container_flag = True
            for container in self._pod["spec"]["containers"]:
                if container["name"] == "autogen-executor":
                    executor_container_flag = False
            if executor_container_flag:
                raise ValueError("No container which name is autogen-executor")

        except KeyError as e:
            raise ValueError(e) from e
        except ValueError as e:
            raise e from e

    def _define_pod_spec(self) -> Any:
        pod = kubernetes.client.models.V1Pod()
        metadata = kubernetes.client.models.V1ObjectMeta(name=self._pod_name, namespace=self._namespace)

        executor_container = kubernetes.client.models.V1Container(
            args=["-c", "while true;do sleep 5; done"],
            command=["/bin/sh"],
            name=self._container_name,
            image=self._image,
        )
        pod_spec = kubernetes.client.models.V1PodSpec(restart_policy="Never", containers=[executor_container])

        pod.metadata = metadata
        pod.spec = pod_spec

        pod_manifest: Dict[str, Any] = clean_none_value(dict(pod.to_dict()))

        if self._volume is not None:
            # add volume
            volume: Dict[str, Any] = self._volume
            pod_manifest["spec"]["volumes"] = [volume]

            # add volume mounts to executor container
            pod_manifest["spec"]["containers"][0]["volumeMounts"] = [
                {"mountPath": str(self.workspace_path), "name": volume["name"]}
            ]

        return pod_manifest

    async def _setup_functions(self, cancellation_token: CancellationToken) -> None:  # type: ignore
        func_file_content = build_python_functions_file(self._functions)
        func_file = self.workspace_path / f"{self._functions_module}.py"

        write_func_file_command = (
            f'if [ ! -d "{self.workspace_path}" ]; then\n  mkdir {self.workspace_path}\nfi\n'
            f"cat <<EOF >{func_file}\n{func_file_content}\nEOF\nchmod +x {func_file}"
        )
        write_func_file_stderr_msg: List[str] = []
        write_func_file_stdout_msg: List[str] = []
        write_func_file_exit_code = 0
        async for channel, msg, exit_code in pod_exec_stream(
            self._kube_config,
            self._pod["metadata"]["name"],
            self._pod["metadata"]["namespace"],
            ["sh", "-c", write_func_file_command],
            self._container_name,
        ):
            if channel == StreamChannel.STDOUT_CHANNEL:
                write_func_file_stdout_msg.append(msg)
            elif channel == StreamChannel.STDERR_CHANNEL:
                write_func_file_stderr_msg.append(msg)
            if exit_code is not None:
                write_func_file_exit_code = exit_code

        if write_func_file_exit_code != 0:
            write_func_file_stdout = "".join(write_func_file_stdout_msg)
            write_func_file_stderr = "".join(write_func_file_stderr_msg)
            raise ValueError(f"write function module file failed. \n{write_func_file_stdout}\n{write_func_file_stderr}")

        # Collect requirements
        lists_of_packages = [
            x.python_packages
            for x in self._functions
            if isinstance(x, (FunctionWithRequirements, FunctionWithRequirementsStr))
        ]
        flattened_packages = [item for sublist in lists_of_packages for item in sublist]
        required_packages = list(set(flattened_packages))
        if len(required_packages) > 0:
            logging.info("Ensuring packages are installed in executor.")

            packages = shlex.join(required_packages)

            install_packages_stderr_msg: List[str] = []
            install_packages_stdout_msg: List[str] = []
            install_packages_exit_code = 0
            async for channel, msg, exit_code in pod_exec_stream(
                self._kube_config,
                self._pod["metadata"]["name"],
                self._pod["metadata"]["namespace"],
                ["sh", "-c", f"python -m pip install {packages}"],
                self._container_name,
            ):
                if channel == StreamChannel.STDOUT_CHANNEL:
                    install_packages_stdout_msg.append(msg)
                elif channel == StreamChannel.STDERR_CHANNEL:
                    install_packages_stderr_msg.append(msg)
                if exit_code is not None:
                    install_packages_exit_code = exit_code

            if install_packages_exit_code != 0:
                install_packages_stdout = "".join(install_packages_stdout_msg)
                install_packages_stderr = "".join(install_packages_stderr_msg)
                raise ValueError(f"Pip install failed. \n{install_packages_stdout}\n{install_packages_stderr}")

        # Attempt to load the function file to check for syntax errors, imports etc.
        # TODO use exec
        exec_stderr_msg: List[str] = []
        exec_stdout_msg: List[str] = []
        exec_exit_code = 0
        async for channel, msg, exit_code in pod_exec_stream(
            self._kube_config,
            self._pod["metadata"]["name"],
            self._pod["metadata"]["namespace"],
            ["sh", "-c", "python", str(func_file)],
            self._container_name,
        ):
            if channel == StreamChannel.STDOUT_CHANNEL:
                exec_stdout_msg.append(msg)
            elif channel == StreamChannel.STDERR_CHANNEL:
                exec_stderr_msg.append(msg)
            if exit_code is not None:
                exec_exit_code = exit_code

        if exec_exit_code != 0:
            exec_stdout = "".join(exec_stdout_msg)
            exec_stderr = "".join(exec_stderr_msg)
            raise ValueError(f"Functions failed to load: \n{exec_stdout}\n{exec_stderr}")

        self._setup_functions_complete = True

    async def _execute_code_dont_check_setup(
        self, code_blocks: List[CodeBlock], cancellation_token: CancellationToken
    ) -> CommandLineCodeResult:  # type: ignore
        if self._pod is None or not self._running:
            raise ValueError("Container is not running. Must first be started with either start or a context manager.")

        if len(code_blocks) == 0:
            raise ValueError("No code blocks to execute.")

        outputs: List[str] = []
        files: List[Path] = []
        last_exit_code = 0
        for code_block in code_blocks:
            lang = code_block.language.lower()
            code = silence_pip(code_block.code, lang)

            # Check if there is a filename comment
            try:
                filename = get_file_name_from_content(code, self._workspace_path)
            except ValueError:
                outputs.append("Filename is not in the workspace")
                last_exit_code = 1
                break

            if not filename:
                extension = self.LANGUAGE_FILE_EXTENSION[lang]
                filename = f"tmp_code_{sha256(code.encode()).hexdigest()}.{extension}"

            code_path = self._workspace_path / filename
            mkdir_and_write_code = (
                f'if [ ! -d "{self.workspace_path}" ]; then\n  mkdir {self.workspace_path}\nfi\n'
                f"cat <<EOF >{code_path}\n{code}\nEOF\nchmod +x {code_path}"
            )

            code_write_command = ["sh", "-c", mkdir_and_write_code]
            stderr_msg: List[str] = []
            stdout_msg: List[str] = []
            last_exit_code = 0
            async for channel, msg, exit_code in pod_exec_stream(
                self._kube_config,
                self._pod["metadata"]["name"],
                self._pod["metadata"]["namespace"],
                code_write_command,
                self._container_name,
            ):
                if channel == StreamChannel.STDOUT_CHANNEL:
                    stdout_msg.append(msg)
                elif channel == StreamChannel.STDERR_CHANNEL:
                    stderr_msg.append(msg)
                if exit_code is not None:
                    last_exit_code = exit_code

            outputs.extend(stderr_msg)
            if last_exit_code != 0:
                break

            files.append(code_path)
            command = ["timeout", str(self.timeout), lang_to_cmd(lang), str(code_path)]

            stderr_msg = []
            stdout_msg = []
            async for channel, msg, exit_code in pod_exec_stream(
                self._kube_config,
                self._pod["metadata"]["name"],
                self._pod["metadata"]["namespace"],
                command,
                self._container_name,
            ):
                if channel == StreamChannel.STDOUT_CHANNEL:
                    stdout_msg.append(msg)
                elif channel == StreamChannel.STDERR_CHANNEL:
                    stderr_msg.append(msg)
                if exit_code is not None:
                    last_exit_code = exit_code

            if last_exit_code == 124:
                stdout_msg.append("\n Timeout")
            outputs.extend(stdout_msg)
            outputs.extend(stderr_msg)

            if last_exit_code != 0:
                break

        code_file = str(files[0]) if files else None
        return CommandLineCodeResult(exit_code=last_exit_code, output="".join(outputs), code_file=code_file)

    async def execute_code_blocks(
        self, code_blocks: List[CodeBlock], cancellation_token: CancellationToken
    ) -> CommandLineCodeResult:
        """(Experimental) Execute the code blocks and return the result.

        Args:
            code_blocks (List[CodeBlock]): The code blocks to execute.

        Returns:
            CommandlineCodeResult: The result of the code execution."""

        def raise_not_implemented() -> None:
            raise NotImplementedError("Cancellation is not yet supported for PodCommandLineCodeExecutor")

        cancellation_token.add_callback(lambda: raise_not_implemented())

        if not self._setup_functions_complete:
            await self._setup_functions(cancellation_token)

        return await self._execute_code_dont_check_setup(code_blocks, cancellation_token)

    async def remove(self) -> None:
        await delete_pod(
            self._kube_config,
            self._pod["metadata"]["name"],
            self._pod["metadata"]["namespace"],
        )
        self._running = False

    async def start(self) -> None:
        import asyncio_atexit

        self._pod = await create_pod(self._kube_config, self._pod)

        self._pod = await wait_for_ready(
            self._kube_config,
            self._pod["metadata"]["name"],
            self._pod["metadata"]["namespace"],
            self._timeout,
        )

        async def cleanup() -> None:
            await self.remove()
            asyncio_atexit.unregister(cleanup)  # type: ignore

        if self._auto_remove:
            asyncio_atexit.register(cleanup)  # type: ignore

        # Check if the container is running
        if self._pod["status"]["phase"] != "Running":
            logs_str = await get_pod_logs(
                self._kube_config,
                self._pod["metadata"]["name"],
                self._pod["metadata"]["namespace"],
                self._container_name,
            )
            raise ValueError(f"Failed to start container from image {self._image}. Logs: {logs_str}")

        self._running = True

    async def restart(self) -> None:
        warnings.warn(
            "Restarting Pod command line code executor is not supported. No action is taken.",
            stacklevel=2,
        )

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> Optional[bool]:
        await self.remove()
        return None
