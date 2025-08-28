import asyncio
import base64
import datetime
import json
import re
import secrets
import time
import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Type,
    Union,
    cast,
    runtime_checkable,
)
from urllib.parse import urlparse

import asyncio_atexit
import httpx
import kubernetes.client
import kubernetes.config
import yaml
from autogen_core import Component, ComponentBase
from httpx import HTTPStatusError
from kubernetes.client.models import (
    V1Container,
    V1ContainerPort,
    V1EnvFromSource,
    V1EnvVar,
    V1EnvVarSource,
    V1ObjectMeta,
    V1Pod,
    V1PodSpec,
    V1Secret,
    V1SecretEnvSource,
    V1Service,
    V1ServicePort,
    V1ServiceSpec,
    V1Volume,
    V1VolumeMount,
)
from pydantic import BaseModel, SecretStr
from typing_extensions import Self
from websockets.asyncio.client import ClientConnection, connect

from ._utils import (
    create_namespaced_corev1_resource,
    delete_namespaced_corev1_resource,
    get_pod_logs,
    sanitize_for_serialization,
    wait_for_ready,
)


class PodJupyterConnectionInfo(BaseModel):
    host: str
    port: Optional[int] = None
    token: Optional[SecretStr] = None


class PodJupyterClient:
    def __init__(self, connection_info: PodJupyterConnectionInfo):
        self._connection_info = connection_info
        self._http_client: httpx.AsyncClient = httpx.AsyncClient(headers=self._get_headers())

    def _get_headers(self) -> Dict[str, str]:
        if self._connection_info.token is None:
            return {}
        return {"Authorization": f"token {self._connection_info.token.get_secret_value()}"}

    def _get_api_base_url(self) -> str:
        port = f":{self._connection_info.port}" if self._connection_info.port else ""
        if urlparse(self._connection_info.host).scheme:
            api_server_url = self._connection_info.host
        else:
            api_server_url = f"http://{self._connection_info.host}"
        return f"{api_server_url}{port}"

    def _get_ws_base_url(self) -> str:
        api_server_url = self._get_api_base_url().replace("http://", "ws://").replace("https://", "wss://")
        return api_server_url

    async def list_kernel_specs(self) -> Dict[str, Dict[str, str]]:
        response = await self._http_client.get(
            f"{self._get_api_base_url()}/api/kernelspecs",
        )
        response.raise_for_status()
        return cast(Dict[str, Dict[str, str]], response.json())

    async def list_kernels(self) -> List[Dict[str, str]]:
        response = await self._http_client.get(f"{self._get_api_base_url()}/api/kernels")
        response.raise_for_status()
        return cast(List[Dict[str, str]], response.json())

    async def start_kernel(self, kernel_spec_name: str) -> str:
        """Start a new kernel asynchronously.

        Args:
            kernel_spec_name (str): Name of the kernel spec to start

        Returns:
            str: ID of the started kernel
        """
        response = await self._http_client.post(
            f"{self._get_api_base_url()}/api/kernels",
            json={"name": kernel_spec_name},
        )
        response.raise_for_status()
        data = response.json()
        return cast(str, data["id"])

    async def delete_kernel(self, kernel_id: str) -> None:
        response = await self._http_client.delete(
            f"{self._get_api_base_url()}/api/kernels/{kernel_id}",
        )
        response.raise_for_status()

    async def restart_kernel(self, kernel_id: str) -> None:
        response = await self._http_client.post(
            f"{self._get_api_base_url()}/api/kernels/{kernel_id}/restart",
        )
        response.raise_for_status()

    async def get_kernel_client(self, kernel_id: str) -> "JupyterKernelClient":
        ws_path = f"/api/kernels/{kernel_id}/channels"
        # Using websockets library for async websocket connections
        ws = await connect(
            uri=self._get_ws_base_url() + ws_path,
            additional_headers=self._get_headers(),
        )
        return JupyterKernelClient(ws)

    async def close(self) -> None:
        """Close the async session"""
        await self._http_client.aclose()


DEFAULT_COMMAND = [
    "/bin/bash",
    "-o",
    "pipefail",
    "-c",
    """
    mamba install --yes jupyter_kernel_gateway ipykernel && \
        mamba clean --all -f -y && \
        fix-permissions "${CONDA_DIR}" && \
        fix-permissions "/home/${NB_USER}"; \
    python -m jupyter kernelgateway --KernelGatewayApp.ip=0.0.0.0 \
        --JupyterApp.answer_yes=true \
        --JupyterWebsocketPersonality.list_kernels=true;
    """,
]
DEFAULT_PORT = 8888


class PodJupyterServerConfig(BaseModel):
    """Configuration for PodJupyterServer"""

    image: Optional[str] = None
    pod_name: Optional[str] = None
    command: Optional[list[str]] = None
    args: Optional[list[str]] = None
    timeout: int = 60
    workspace_path: Union[Path, str] = Path("/workspace")
    namespace: str = "default"
    volume: Union[dict[str, Any], str, Path, Type[V1Volume], None] = None
    pod_spec: Union[dict[str, Any], str, Path, Type[V1Pod], None] = None
    auto_remove: bool = True
    port: int = DEFAULT_PORT
    service_spec: Union[dict[str, Any], str, Path, Type[V1Service], None] = None
    token: Optional[SecretStr] = None
    secret_spec: Union[dict[str, Any], str, Path, Type[V1Secret], SecretStr, None] = None
    """when secret spec is provided as SecretStr, It should be a json format"""
    kube_config_file: Union[Path, str, None] = None


class PodJupyterServer(Component[PodJupyterServerConfig], ComponentBase[BaseModel]):
    class GenerateToken:
        pass

    component_type = "jupyter_server"
    component_version = 1
    component_description = "jupyter server component as kubernetes pod"
    component_config_schema = PodJupyterServerConfig
    component_provider_override = "autogen_kubernetes.code_executors.PodJupyterServer"

    def __init__(
        self,
        *,
        image: Optional[str] = None,
        pod_name: Optional[str] = None,
        command: Optional[list[str]] = None,
        args: Optional[list[str]] = None,
        timeout: int = 60,
        workspace_path: Union[Path, str] = "/workspace",
        namespace: str = "default",
        volume: Union[dict[str, Any], str, Path, Type[V1Volume], None] = None,
        pod_spec: Union[dict[str, Any], str, Path, Type[V1Pod], None] = None,
        auto_remove: bool = True,
        port: int = DEFAULT_PORT,
        service_spec: Union[dict[str, Any], str, Path, Type[V1Service], None] = None,
        token: Optional[Union[str, GenerateToken]] = None,
        secret_spec: Union[dict[str, Any], str, Path, Type[V1Secret], None] = None,
        kube_config_file: Union[Path, str, None] = None,
    ):
        """Start a Jupyter kernel gateway server in a Kubernetes container.

        Args:
            image (Optional[str], optional): jupyter server docker image. Defaults to None and converted to quay.io/jupyter/docker-stacks-foundation.
            pod_name (Optional[str], optional): jupyter server pod name. Defaults to None and will be converted to auto generated name.
            command (Optional[list[str]], optional): jupyter server container command. Defaults to DEFAULT_COMMAND.
            args (Optional[list[str]], optional): jupyter server container argument. Defaults to None.
            timeout (int, optional): wait time in secondes for the pod to be running phase. Defaults to 60.
            workspace_path (str, optional): default workspace directory. Defaults to "/workspace".
            namespace (str, optional): namespace for jupyter server pod. Defaults to "default".
            volume (Union[dict[str, Any], str, Path, Type[V1Volume], None], optional): The volume for the jupyter server pod.
                Supports the formats of a dictionary, a YAML/json string, a YAML file path, and kubernetes `V1Volume` model format.
                Must conform to the kubernetes `V1Volume` model format.
                Must have appropriate access mode(such as ReadWriteMany, ReadWriteOnce, ReadWriteOncePod, in case of PersistentVolumeClaim)
                If None, no volume attached to code executor pod. Defaults to None.
            pod_spec (Union[dict[str, Any], str, Path, Type[V1Pod], None], optional): Custom pod specification for jupyter server.
                Must contain a container which name is "autogen-executor" for execution for codes.
                Supports the formats of a dictionary, a YAML/json string, a YAML file path, and a kubernetes V1Pod model.
                Must conform to the kubernetes `V1Pod` model format.
                It has priority than above parameters.
                If None, will use above parameters to create code executor pod. Defaults to None.
            auto_remove (bool, optional): If true, will automatically remove the
                container when remove is called, when the context manager exits or when
                the Python process exits with atext. Defaults to True.
            port (int, optional): port number for jupyter server. Defaults to DEFAULT_PORT.
            service_spec (Union[dict[str, Any], str, Path, Type[V1Service], None], optional): Custom service specificatin for jupyter server.
                Supports the formats of a dictionary, a YAML/json string, a YAML file path, and kubernetes `V1Service` model format.
                Must conform to the kubernetes `V1Service` model format.
                The first port should be the name `jupyter` otherwise PodJupyterClient cannot use the port.
                It has priority than port parameter.
                Defaults to None.
            token (Optional[Union[str, GenerateToken]], optional): token string. Defaults to None and will be converted to auto generated token.
            secret_spec (Union[dict[str, Any], str, Path, Type[V1Secret], None], optional): Custom secret specificatin for jupyter server.
                Supports the formats of a dictionary, a YAML/json string, a YAML file path, and kubernetes `V1Secret` model format.
                Must conform to the kubernetes `V1Secret` model format.
                It is possible to add data `KG_AUTH_TOKEN` to use custom authentication token for jupyter kernel gateway server.
                It has priority than token parameter.
                Defaults to None.
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
                    "resource": "pods/log", "verb": ["get"]
                },
                {
                    "resource": "services", "verb": ["get", "create", "delete"]
                },
                {
                    "resource": "secrets", "verb": ["get", "create", "delete"]
                },
                ]
        """
        # Generate container name if not provided
        self._pod_name = pod_name or f"autogen-jupyter-{uuid.uuid4()}"
        self._secret_name = self._pod_name + "-secret"

        if not re.fullmatch(r"^[a-z0-9](?:[a-z0-9-]{0,54})[a-z0-9]?$", self._pod_name):
            raise ValueError(
                "Pod name validation failed: pod name must start and end with lower alphanumeric characters with length 56, "
                "and contain only lowercase alphanumeric or dash('-') characters."
            )

        # Determine and prepare Docker image
        self._image = image or "quay.io/jupyter/docker-stacks-foundation"

        # Set up authentication token
        if token is None:
            token = PodJupyterServer.GenerateToken()
        self._token = secrets.token_hex(32) if isinstance(token, PodJupyterServer.GenerateToken) else token

        ## kubeconfig
        if isinstance(kube_config_file, str):
            kube_config_file = Path(kube_config_file)

        if kube_config_file is None:  ## configuration from default kubeconfig or incluster
            kubernetes.config.load_config()  # type: ignore
        else:
            kubernetes.config.load_config(config_file=kube_config_file)  # type: ignore
        self._kube_config = kubernetes.client.Configuration.get_default_copy()  # type: ignore
        self._kube_config_file = kube_config_file

        self._port = port
        self._container_name = "autogen-executor"
        self._namespace = namespace
        self._timeout = timeout
        self._auto_remove = auto_remove
        self._command = command or DEFAULT_COMMAND
        self._args = args
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

        ## set default resources
        self._secret: Optional[dict[str, Any]] = None
        self._define_resources_spec()

        if pod_spec is not None:
            self._pod = self._read_from_resource_spec("Pod", pod_spec)
        if secret_spec is not None:
            self._secret = self._read_from_resource_spec("Secret", secret_spec)
            if "KG_AUTH_TOKEN" in self._secret["data"]:
                self._token = base64.b64decode(self._secret["data"]["KG_AUTH_TOKEN"]).decode("utf-8")
            elif "KG_AUTH_TOKEN" in self._secret["stringData"]:
                self._token = self._secret["stringData"]["KG_AUTH_TOKEN"]
        if service_spec is not None:
            self._service = self._read_from_resource_spec("Service", service_spec)
            has_jupyter_port = False
            for p in self._service["spec"]["ports"]:
                if p["name"] == "jupyter":
                    self._port = p["port"]
                    has_jupyter_port = True
                    break
            if not has_jupyter_port:
                raise ValueError(
                    "custom service should have a port named `jupyter`. add a port named `jupyter` in service_spec"
                )

        self._running = False

    def _read_from_resource_spec(self, kind: str, spec: Any) -> dict[str, Any]:
        types = {"Pod": V1Pod, "Secret": V1Secret, "Service": V1Service}

        if isinstance(spec, str):  # YAML file path string
            try:
                if Path(spec).exists():
                    spec = Path(spec)
            except OSError:
                pass

        resource_dict = {}
        if isinstance(spec, Path):  # YAML file to dict
            for obj in yaml.safe_load_all(spec.read_text(encoding="utf-8")):
                if obj["kind"] == kind:
                    resource_dict = obj
            if not resource_dict:
                raise ValueError(f"spec file {str(spec)} has no manifest for kind {kind}")
        elif isinstance(spec, str):  # YAML string to dict
            resource_dict = yaml.safe_load(spec)
        elif isinstance(spec, types[kind]):
            resource_dict = sanitize_for_serialization(spec)  # type: ignore
        elif isinstance(spec, dict):
            resource_dict = spec

        return resource_dict

    def _define_resources_spec(self) -> None:
        label = {"autogen-kubernetes/jupyter-server": self._pod_name}
        pod = V1Pod(
            kind="Pod",
            metadata=V1ObjectMeta(
                name=self._pod_name,
                namespace=self._namespace,
                labels=label,
            ),
        )

        executor_container = V1Container(
            args=self._args,
            command=self._command,
            name=self._container_name,
            image=self._image,
            env=[
                V1EnvVar(name="KG_PORT", value=str(self._port)),
            ],
            ports=[V1ContainerPort(container_port=self._port, name="jupyter")],
            working_dir=str(self._workspace_path),
        )
        if self._token:
            executor_container.env_from = [
                V1EnvFromSource(secret_ref=V1SecretEnvSource(name=self._secret_name)),
            ]

            secret = V1Secret(
                kind="Secret",
                metadata=V1ObjectMeta(name=self._secret_name, namespace=self._namespace, labels=label),
                string_data={"KG_AUTH_TOKEN": self._token},
            )
            self._secret = sanitize_for_serialization(secret)

        pod_spec = V1PodSpec(
            automount_service_account_token=False,
            containers=[executor_container],
        )

        if self._volume:
            executor_container.volume_mounts = [
                V1VolumeMount(mount_path=str(self._workspace_path), name=self._volume["name"])
            ]
            pod_spec.volumes = [V1Volume(**self._volume)]

        pod.spec = pod_spec
        self._pod = sanitize_for_serialization(pod)

        service = V1Service(
            kind="Service",
            metadata=V1ObjectMeta(
                name=self._pod_name,
                namespace=self._namespace,
                labels=label,
            ),
            spec=V1ServiceSpec(
                type="ClusterIP",
                selector=label,
                ports=[
                    V1ServicePort(
                        name="jupyter",
                        protocol="TCP",
                        port=self._port,
                        target_port="jupyter",
                    )
                ],
            ),
        )
        self._service = sanitize_for_serialization(service)

    @property
    def connection_info(self) -> PodJupyterConnectionInfo:
        return PodJupyterConnectionInfo(
            host=f"{self._service['metadata']['name']}.{self._service['metadata']['namespace']}.svc.cluster.local",
            port=self._port,
            token=SecretStr(self._token) if self._token else None,
        )

    async def remove(self, spec: Optional[dict[str, Any]]) -> None:
        if spec is None:
            return
        try:
            await delete_namespaced_corev1_resource(self._kube_config, spec)
        except HTTPStatusError:
            pass

    async def stop(self) -> None:
        await asyncio.gather(*[self.remove(spec) for spec in (self._pod, self._secret, self._service)])
        self._running = False

    async def start(self) -> None:
        for spec in [self._secret, self._pod, self._service]:
            if spec is None:
                continue
            await create_namespaced_corev1_resource(self._kube_config, spec)

        self._pod = await wait_for_ready(
            self._kube_config,
            self._pod["metadata"]["name"],
            self._pod["metadata"]["namespace"],
            self._timeout,
        )

        async def cleanup() -> None:
            await self.stop()
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

    async def get_client(self) -> PodJupyterClient:
        return PodJupyterClient(self.connection_info)

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        await self.stop()

    def _to_config(self) -> PodJupyterServerConfig:
        """(Experimental) Convert the component to a config object"""

        return PodJupyterServerConfig(
            image=self._image,
            pod_name=self._pod_name,
            timeout=self._timeout,
            workspace_path=self._workspace_path,
            namespace=self._namespace,
            volume=self._volume,
            pod_spec=self._pod,
            kube_config_file=self._kube_config_file,
            auto_remove=self._auto_remove,
            port=self._port,
            command=self._command,
            service_spec=self._service,
            token=SecretStr(self._token),
            secret_spec=SecretStr(str(self._secret)),
        )

    @classmethod
    def _from_config(cls, config: PodJupyterServerConfig) -> Self:
        """(Experimental) Create a component from a config object"""

        return cls(
            image=config.image,
            pod_name=config.pod_name,
            command=config.command,
            timeout=config.timeout,
            workspace_path=config.workspace_path,
            namespace=config.namespace,
            volume=config.volume,
            pod_spec=config.pod_spec,
            auto_remove=config.auto_remove,
            port=config.port,
            service_spec=config.service_spec,
            token=(config.token.get_secret_value() if config.token is not None else config.token),
            secret_spec=(
                config.secret_spec
                if not isinstance(config.secret_spec, SecretStr)
                else json.loads(config.secret_spec.get_secret_value())
            ),
            kube_config_file=config.kube_config_file,
        )


# Source below based from: https://github.com/microsoft/autogen/blob/main/python/packages/autogen-ext/src/autogen_ext/code_executors/docker_jupyter/_jupyter_server.py
# Credit to original authors
# Original code Licensed under the MIT License.
# See the License file for the full license text.


@dataclass
class DataItem:
    mime_type: str
    data: str


@dataclass
class ExecutionResult:
    is_ok: bool
    output: str
    data_items: List[DataItem]


class JupyterKernelClient:
    """An asynchronous client for communicating with a Jupyter kernel."""

    def __init__(self, websocket: ClientConnection) -> None:
        self._session_id = uuid.uuid4().hex
        self._websocket = websocket

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        await self.stop()

    async def stop(self) -> None:
        await self._websocket.close()

    async def _send_message(self, *, content: Dict[str, Any], channel: str, message_type: str) -> str:
        timestamp = datetime.datetime.now().isoformat()
        message_id = uuid.uuid4().hex
        message = {
            "header": {
                "username": "autogen",
                "version": "5.0",
                "session": self._session_id,
                "msg_id": message_id,
                "msg_type": message_type,
                "date": timestamp,
            },
            "parent_header": {},
            "channel": channel,
            "content": content,
            "metadata": {},
            "buffers": {},
        }
        await self._websocket.send(json.dumps(message))
        return message_id

    async def _receive_message(self, timeout_seconds: Optional[float]) -> Optional[Dict[str, Any]]:
        try:
            if timeout_seconds is not None:
                data = await asyncio.wait_for(self._websocket.recv(), timeout=timeout_seconds)
            else:
                data = await self._websocket.recv()
            if isinstance(data, bytes):
                return cast(Dict[str, Any], json.loads(data.decode("utf-8")))
            return cast(Dict[str, Any], json.loads(data))
        except asyncio.TimeoutError:
            return None

    async def wait_for_ready(self, timeout_seconds: Optional[float] = None) -> bool:
        message_id = await self._send_message(content={}, channel="shell", message_type="kernel_info_request")
        while True:
            message = await self._receive_message(timeout_seconds)
            # This means we timed out with no new messages.
            if message is None:
                return False
            if (
                message.get("parent_header", {}).get("msg_id") == message_id
                and message["msg_type"] == "kernel_info_reply"
            ):
                return True

    async def execute(self, code: str, timeout_seconds: Optional[float] = None) -> ExecutionResult:
        message_id = await self._send_message(
            content={
                "code": code,
                "silent": False,
                "store_history": True,
                "user_expressions": {},
                "allow_stdin": False,
                "stop_on_error": True,
            },
            channel="shell",
            message_type="execute_request",
        )

        text_output: List[str] = []
        data_output: List[DataItem] = []
        while True:
            message = await self._receive_message(timeout_seconds)
            if message is None:
                return ExecutionResult(
                    is_ok=False,
                    output="ERROR: Timeout waiting for output from code block.",
                    data_items=[],
                )

            # Ignore messages that are not for this execution.
            if message.get("parent_header", {}).get("msg_id") != message_id:
                continue

            msg_type = message["msg_type"]
            content = message["content"]
            if msg_type in ["execute_result", "display_data"]:
                for data_type, data in content["data"].items():
                    if data_type == "text/plain":
                        text_output.append(data)
                    elif data_type.startswith("image/") or data_type == "text/html":
                        data_output.append(DataItem(mime_type=data_type, data=data))
                    else:
                        text_output.append(json.dumps(data))
            elif msg_type == "stream":
                text_output.append(content["text"])
            elif msg_type == "error":
                # Output is an error.
                return ExecutionResult(
                    is_ok=False,
                    output=f"ERROR: {content['ename']}: {content['evalue']}\n{content['traceback']}",
                    data_items=[],
                )
            if msg_type == "status" and content["execution_state"] == "idle":
                break
        return ExecutionResult(
            is_ok=True,
            output="\n".join([str(output) for output in text_output]),
            data_items=data_output,
        )
