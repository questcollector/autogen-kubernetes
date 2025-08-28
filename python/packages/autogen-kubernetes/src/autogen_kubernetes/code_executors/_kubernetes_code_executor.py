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
import uuid
import warnings
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path
from types import TracebackType
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    ParamSpec,
    Tuple,
    Type,
    Union,
)

import asyncio_atexit
import kubernetes.client
import kubernetes.config
import yaml
from autogen_core import CancellationToken, Component
from autogen_core.code_executor import (
    CodeBlock,
    CodeExecutor,
    FunctionWithRequirements,
    FunctionWithRequirementsStr,
)
from autogen_core.code_executor._func_with_reqs import (
    build_python_functions_file,
)
from httpx import HTTPStatusError
from kubernetes.client.models import (
    V1Container,
    V1ObjectMeta,
    V1Pod,
    V1PodSpec,
    V1Volume,
)
from pydantic import BaseModel

from ._utils import (
    POD_NAME_PATTERN,
    CommandLineCodeResult,
    StreamChannel,
    create_namespaced_corev1_resource,
    delete_namespaced_corev1_resource,
    get_file_name_from_content,
    get_pod_logs,
    lang_to_cmd,
    pod_exec_stream,
    sanitize_for_serialization,
    silence_pip,
    wait_for_ready,
)

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


A = ParamSpec("A")

DEFAULT_COMMAND = ["/bin/sh", "-c", "while true;do sleep 5; done"]


class PodCommandLineCodeExecutorConfig(BaseModel):
    """Configuration for PodCommandLineCodeExecutor"""

    image: str = "python:3-slim"
    pod_name: Optional[str] = None
    command: Optional[list[str]] = None
    args: Optional[list[str]] = None
    timeout: int = 60
    workspace_path: Union[Path, str] = "/workspace"
    namespace: str = "default"
    volume: Union[dict[str, Any], str, Path, Type[V1Volume], None] = None
    pod_spec: Union[dict[str, Any], str, Path, Type[V1Pod], None] = None
    kube_config_file: Union[Path, str, None] = None
    auto_remove: bool = True
    functions_module: str = "functions"


class PodCommandLineCodeExecutor(CodeExecutor, Component[PodCommandLineCodeExecutorConfig]):
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
        command (Optional[list[str]], optional): container command. Defaults to DEFAULT_COMMAND.
        args (Optional[list[str]], optional): container argument. Defaults to None.
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
        pod_spec (Union[dict, str, Path, kubernetes.client.models.V1Pod, None], optional): Custom pod specification for code executor.
            Must contain a container which name is "autogen-executor" for execution for codes.
            Supports the formats of a dictionary, a YAML string, a YAML file path, and a kubernetes V1Pod model.
            Must conform to the kubernetes `V1Pod` model format.
            If None, will use above parameters to create code executor pod. Defaults to None.
        kube_config_file (Union[Path, str, None], optional): kubernetes configuration file(kubeconfig) path.
            If None, will use `KUBECONFIG` environment variables or service account token(incluster config).
            Service account must have at least those namespaced permissions below.
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

    component_config_schema = PodCommandLineCodeExecutorConfig
    component_provider_override = "autogen_kubernetes.code_executors.PodCommandLineCodeExecutor"

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

    def __init__(
        self,
        *,
        image: Optional[str] = None,
        pod_name: Optional[str] = None,
        command: Optional[list[str]] = None,
        args: Optional[list[str]] = None,
        timeout: int = 60,
        workspace_path: Union[Path, str] = Path("/workspace"),
        namespace: str = "default",
        volume: Union[dict[str, Any], str, Path, Type[V1Volume], None] = None,
        pod_spec: Union[dict[str, Any], str, Path, Type[V1Pod], None] = None,
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
        self._kube_config_file = kube_config_file

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
        elif isinstance(volume, V1Volume):
            volume_json = sanitize_for_serialization(volume)
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
        self._image = image or "python:3-slim"
        self._command = command or DEFAULT_COMMAND
        self._args = args

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
        elif isinstance(pod_spec, V1Pod):
            pod_spec_dict = sanitize_for_serialization(pod_spec)
        elif isinstance(pod_spec, dict):
            pod_spec_dict = pod_spec

        if pod_spec is None:  ## create pod_spec from other parameters
            self._pod = self._define_pod_spec()
        else:
            ## merge default pod spec
            pod_spec_dict = self._merge_with_default_pod_spec(pod_spec_dict)
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

    def _merge_with_default_pod_spec(self, pod_spec_dict: dict[str, Any]) -> dict[str, Any]:
        clean_pod_spec_dict: dict[str, Any] = sanitize_for_serialization(pod_spec_dict)
        clean_pod_spec_dict["kind"] = "Pod"
        default_pod_spec = self._define_pod_spec()
        if "metadata" not in clean_pod_spec_dict:
            clean_pod_spec_dict |= {"metadata": default_pod_spec["metadata"]}
        if "name" not in clean_pod_spec_dict["metadata"]:
            clean_pod_spec_dict["metadata"] |= {"name": default_pod_spec["metadata"]["name"]}
        if "namespace" not in clean_pod_spec_dict["metadata"]:
            clean_pod_spec_dict["metadata"] |= {"namespace": default_pod_spec["metadata"]["namespace"]}

        if "spec" not in clean_pod_spec_dict:
            clean_pod_spec_dict |= {"spec": default_pod_spec["spec"]}

        if "automountServiceAccountToken" not in clean_pod_spec_dict["spec"]:
            clean_pod_spec_dict["spec"] |= {
                "automountServiceAccountToken": default_pod_spec["spec"]["automountServiceAccountToken"]
            }
        if "containers" not in clean_pod_spec_dict["spec"]:
            clean_pod_spec_dict["spec"] |= {"containers": default_pod_spec["spec"]["containers"]}
        if "restartPolicy" not in clean_pod_spec_dict["spec"]:
            clean_pod_spec_dict["spec"] |= {"restartPolicy": default_pod_spec["spec"]["restartPolicy"]}
        has_executor_pod = False
        for container in clean_pod_spec_dict["spec"]["containers"]:
            if container["name"] == "autogen-executor":
                has_executor_pod = True
        if not has_executor_pod:
            clean_pod_spec_dict["spec"]["containers"].append(default_pod_spec["spec"]["containers"][0])

        if "volumes" in default_pod_spec["spec"]:
            if "volumes" not in clean_pod_spec_dict["spec"]:
                clean_pod_spec_dict["spec"] |= {"volumes": []}
            for default_volume in default_pod_spec["spec"]["volumes"]:
                has_volume = False
                for volume in clean_pod_spec_dict["spec"]["volumes"]:
                    if volume["name"] == default_volume["name"]:
                        has_volume = True
                if not has_volume:
                    clean_pod_spec_dict["spec"]["volumes"].append(default_volume)

        return clean_pod_spec_dict

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
        pod = V1Pod(kind="Pod")
        metadata = V1ObjectMeta(name=self._pod_name, namespace=self._namespace)

        executor_container = V1Container(
            args=self._args,
            command=self._command,
            name=self._container_name,
            image=self._image,
            working_dir=str(self._workspace_path),
        )
        pod_spec = V1PodSpec(restart_policy="Never", containers=[executor_container])
        pod_spec.automount_service_account_token = False

        pod.metadata = metadata
        pod.spec = pod_spec

        pod_manifest: Dict[str, Any] = sanitize_for_serialization(pod)  # type: ignore

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

    async def _execute_command(self, command: List[str]) -> Tuple[List[str], int]:
        stderr_msg: List[str] = []
        stdout_msg: List[str] = []
        outputs: List[str] = []
        last_exit_code = 0
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

        return outputs, last_exit_code

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
            code_write_exec = asyncio.create_task(self._execute_command(code_write_command))
            cancellation_token.link_future(code_write_exec)

            try:
                std_outputs, exit_code = await code_write_exec
                outputs.extend(std_outputs)
                last_exit_code = exit_code
                if last_exit_code != 0:
                    break
            except asyncio.CancelledError:
                return CommandLineCodeResult(exit_code=1, output="Code execution was cancelled.", code_file=None)

            files.append(code_path)
            command = ["timeout", str(self.timeout), lang_to_cmd(lang), str(code_path)]
            command_exec = asyncio.create_task(self._execute_command(command))
            cancellation_token.link_future(command_exec)

            try:
                std_outputs, exit_code = await command_exec
                outputs.extend(std_outputs)
                last_exit_code = exit_code
                if last_exit_code != 0:
                    break
            except asyncio.CancelledError:
                return CommandLineCodeResult(exit_code=1, output="Code execution was cancelled.", code_file=None)

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

        if not self._setup_functions_complete:
            await self._setup_functions(cancellation_token)

        return await self._execute_code_dont_check_setup(code_blocks, cancellation_token)

    async def remove(self) -> None:
        try:
            await delete_namespaced_corev1_resource(
                self._kube_config,
                self._pod,
            )
        except HTTPStatusError:
            pass
        finally:
            self._running = False

    async def stop(self) -> None:
        await self.remove()

    async def start(self) -> None:
        self._pod = await create_namespaced_corev1_resource(self._kube_config, self._pod)

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

    def _to_config(self) -> PodCommandLineCodeExecutorConfig:
        """(Experimental) Convert the component to a config object"""
        if self._functions:
            logging.info("Functions will not be included in serialized configuration")

        return PodCommandLineCodeExecutorConfig(
            image=self._image,
            pod_name=self._pod_name,
            command=self._command,
            args=self._args,
            timeout=self._timeout,
            workspace_path=self._workspace_path,
            namespace=self._namespace,
            volume=self._volume,
            pod_spec=self._pod,
            kube_config_file=self._kube_config_file,
            auto_remove=self._auto_remove,
            functions_module=self._functions_module,
        )

    @classmethod
    def _from_config(cls, config: PodCommandLineCodeExecutorConfig) -> Self:
        """(Experimental) Create a component from a config object"""

        return cls(
            image=config.image,
            pod_name=config.pod_name,
            command=config.command,
            args=config.args,
            timeout=config.timeout,
            workspace_path=config.workspace_path,
            namespace=config.namespace,
            volume=config.volume,
            pod_spec=config.pod_spec,
            kube_config_file=config.kube_config_file,
            auto_remove=config.auto_remove,
            functions_module=config.functions_module,
            functions=[],
        )
