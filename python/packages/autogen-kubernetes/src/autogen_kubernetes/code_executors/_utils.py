from __future__ import annotations

import asyncio
import datetime
import functools
import inspect
import json
import logging
import os
import re
import ssl
import urllib.parse
from dataclasses import dataclass, field
from enum import IntEnum
from importlib.abc import SourceLoader
from importlib.util import module_from_spec, spec_from_loader
from pathlib import Path
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Generic,
    Optional,
    Sequence,
    Set,
    Type,
    TypeVar,
    Union,
)

import httpx
import kubernetes.client
import yaml
from autogen_core.code_executor import CodeResult
from typing_extensions import ParamSpec
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed
from websockets.typing import Subprotocol

T = TypeVar("T")
P = ParamSpec("P")


class StreamChannel(IntEnum):
    STDIN_CHANNEL = 0
    STDOUT_CHANNEL = 1
    STDERR_CHANNEL = 2
    ERROR_CHANNEL = 3
    RESIZE_CHANNEL = 4


class HttpStatusCode(IntEnum):
    OK = 200
    CREATED = 201
    ACCEPTED = 202


POD_NAME_PATTERN = r"^[a-z0-9](?:[a-z0-9-]{0,61})[a-z0-9]?$"


# Source below based from: https://github.com/kubernetes-client/python/blob/master/kubernetes/client/api_client.py
# Credit to original authors
# Original code Licensed under the Apache-2.0 license
# See the License file for the full license text.
def sanitize_for_serialization(obj: Any) -> Any:
    if obj is None:
        return None
    elif isinstance(obj, (float, bool, bytes, str, int)):
        return obj
    elif isinstance(obj, list):
        return [sanitize_for_serialization(sub_obj) for sub_obj in obj]
    elif isinstance(obj, tuple):
        return tuple(sanitize_for_serialization(sub_obj) for sub_obj in obj)
    elif isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat

    if isinstance(obj, dict):
        obj_dict = obj
    else:
        obj_dict = {
            obj.attribute_map[attr]: getattr(obj, attr)
            for attr, _ in obj.openapi_types.items()
            if getattr(obj, attr) is not None
        }

    return {key: sanitize_for_serialization(val) for key, val in obj_dict.items()}


async def wait_for_ready(
    kube_config: Any,
    pod_name: str,
    namespace: str,
    timeout: int = 60,
    stop_time: float = 0.1,
) -> Any:
    elapsed_time = 0.0
    while True:
        pod = await _read_pod_status(kube_config, pod_name, namespace)
        if pod["status"]["phase"] == "Running" or elapsed_time >= timeout:
            break
        await asyncio.sleep(stop_time)
        elapsed_time += stop_time
    if pod["status"]["phase"] != "Running":
        raise ValueError("Pod failed to start")
    else:
        return pod


def _create_ssl_context_and_headers(
    kube_config: Any,
) -> tuple[ssl.SSLContext, dict[str, str]]:
    ca_cert = kube_config.ssl_ca_cert
    ssl_context = ssl.create_default_context(cafile=ca_cert)
    headers = {}

    if "authorization" in kube_config.api_key:  ## Bearer token
        headers.update({"Authorization": kube_config.api_key["authorization"]})
    else:  ## ssl cafile and keyfile
        ssl_context.load_cert_chain(certfile=kube_config.cert_file, keyfile=kube_config.key_file)
    return ssl_context, headers


websocket_subprotocol_kubernetes_api = os.environ.get("WEBSOCKET_SUBPROTOCOL_KUBERNETES_API", "v4.channel.k8s.io")


async def pod_exec_stream(
    kube_config: Any,
    pod_name: str,
    namespace: str,
    command: list[str],
    container_name: str,
) -> AsyncGenerator[tuple[int, str, Optional[int]], None]:
    api_server_url = kube_config.host

    ssl_context, headers = _create_ssl_context_and_headers(kube_config)
    additional_headers = [(key, value) for key, value in headers.items()]

    # websocket
    url = f"{api_server_url}/api/v1/namespaces/{namespace}/pods/{pod_name}/exec"
    params = {
        "container": container_name,
        "stdin": "false",
        "stdout": "true",
        "stderr": "true",
        "tty": "false",
    }
    if not command:
        raise ValueError("command must not be empty list")
    command_query_string = "&".join(f"command={urllib.parse.quote_plus(cmd)}" for cmd in command)
    query_string = "&".join(f"{key}={urllib.parse.quote_plus(value)}" for key, value in params.items() if value)
    websocket_url = f"{url}?{command_query_string}&{query_string}".replace("https://", "wss://")
    subprotocols = [Subprotocol(websocket_subprotocol_kubernetes_api)]

    async with connect(
        websocket_url,
        ssl=ssl_context,
        additional_headers=additional_headers,
        subprotocols=subprotocols,
    ) as ws:
        try:
            while True:
                message = await ws.recv(decode=False)
                if len(message) < 2:
                    continue
                channel = int(message[0])
                msg_content = message[1:]
                if isinstance(msg_content, bytes):
                    content = msg_content.decode("utf-8")
                else:
                    content = msg_content

                returncode = None
                if channel == StreamChannel.ERROR_CHANNEL:  # ERROR_CHANNEL
                    error_info = yaml.safe_load(content)
                    if error_info["status"] == "Success":
                        returncode = 0
                    else:
                        returncode = int(error_info["details"]["causes"][0]["message"])

                yield (channel, content, returncode)

        except ConnectionClosed as e:
            logging.info(f"Websocket connection closed: {e}")
        except asyncio.CancelledError:
            payload = json.dumps({"signal": "SIGKILL"}).encode("utf-8")
            frame = b"\x03" + payload
            await ws.send(frame)

            logging.info("send SIGKILL")
            raise


async def _read_pod_status(kube_config: Any, pod_name: str, namespace: str) -> Any:
    api_server_url = kube_config.host
    ssl_context, headers = _create_ssl_context_and_headers(kube_config)
    headers.update({"Accept": "application/json"})

    url = f"{api_server_url}/api/v1/namespaces/{namespace}/pods/{pod_name}/status"

    async with httpx.AsyncClient(verify=ssl_context) as httpx_client:
        response = await httpx_client.get(url, headers=headers)
        if response.status_code == HttpStatusCode.OK:
            return response.json()
        else:
            logging.info("Failed to get pod status: {response.status_code}")
            response.raise_for_status()


async def get_pod_logs(
    kube_config: Any,
    pod_name: str,
    namespace: str,
    container_name: str,
) -> Any:
    api_server_url = kube_config.host
    ssl_context, headers = _create_ssl_context_and_headers(kube_config)
    headers.update({"Accept": "application/json", "Content-Type": "application/json"})

    url = f"{api_server_url}/api/v1/namespaces/{namespace}/pods/{pod_name}/log?container={container_name}"

    async with httpx.AsyncClient(verify=ssl_context) as httpx_client:
        response = await httpx_client.get(url, headers=headers)
        if response.status_code == HttpStatusCode.OK:
            return response.text
        else:
            logging.info("Failed to get pod status: {response.status_code}")
            response.raise_for_status()


async def create_namespaced_corev1_resource(
    kube_config: Any, resource_spec: dict[str, Any], dry_run: bool = False
) -> Any:
    api_server_url = kube_config.host
    ssl_context, headers = _create_ssl_context_and_headers(kube_config)
    headers.update(
        {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
    )

    namespace = resource_spec["metadata"]["namespace"]
    kind = str(resource_spec["kind"]).lower() + "s"
    url = f"{api_server_url}/api/v1/namespaces/{namespace}/{kind}"
    if dry_run:
        url += "?dryRun=All"

    async with httpx.AsyncClient(verify=ssl_context) as httpx_client:
        response = await httpx_client.post(url, headers=headers, json=resource_spec)
        if response.status_code == HttpStatusCode.CREATED:
            return response.json()
        else:
            logging.info("Failed to create pod: {response.status_code}")
            response.raise_for_status()


async def delete_namespaced_corev1_resource(kube_config: Any, resource: dict[str, Any]) -> Any:
    api_server_url = kube_config.host
    ssl_context, headers = _create_ssl_context_and_headers(kube_config)
    headers.update({"Accept": "application/json"})

    kind = resource["kind"]
    name = resource["metadata"]["name"]
    namespace = resource["metadata"]["namespace"]

    url = f"{api_server_url}/api/v1/namespaces/{namespace}/{kind.lower() + 's'}/{name}"

    async with httpx.AsyncClient(verify=ssl_context) as httpx_client:
        response = await httpx_client.delete(url, headers=headers)
        if response.status_code in [HttpStatusCode.OK, HttpStatusCode.ACCEPTED]:
            return response.json()
        else:
            logging.info("Failed to get pod status: {response.status_code}")
            response.raise_for_status()


# Source below based from: https://github.com/microsoft/autogen/blob/main/python/packages/autogen-ext/src/autogen_ext/code_executors/_common.py
# Credit to original authors
# Original code Licensed under the MIT License.
# See the License file for the full license text.


@dataclass
class CommandLineCodeResult(CodeResult):
    """A code result class for command line code executor."""

    code_file: Optional[str]


# Raises ValueError if the file is not in the workspace
def get_file_name_from_content(code: str, workspace_path: Path) -> Optional[str]:
    first_line = code.split("\n")[0]
    # TODO - support other languages
    if first_line.startswith("# filename:"):
        filename = first_line.split(":")[1].strip()

        # Handle relative paths in the filename
        path = Path(filename)
        if not path.is_absolute():
            path = workspace_path / path
        path = path.resolve()
        # Throws an error if the file is not in the workspace
        relative = path.relative_to(workspace_path.resolve())
        return str(relative)

    return None


PYTHON_VARIANTS = ["python", "Python", "py"]


def lang_to_cmd(lang: str) -> str:
    if lang in PYTHON_VARIANTS:
        return "python"
    if lang.startswith("python") or lang in ["bash", "sh"]:
        return lang
    if lang in ["shell"]:
        return "sh"
    else:
        raise ValueError(f"Unsupported language: {lang}")


def silence_pip(code: str, lang: str) -> str:
    """Apply -qqq flag to pip install commands."""
    if lang == "python":
        regex = r"^! ?pip install"
    elif lang in ["bash", "shell", "sh", "pwsh", "powershell", "ps1"]:
        regex = r"^pip install"
    else:
        return code

    # Find lines that start with pip install and make sure "-qqq" flag is added.
    lines = code.split("\n")
    for i, line in enumerate(lines):
        # use regex to find lines that start with pip install.
        match = re.search(regex, line)
        if match is not None:
            if "-qqq" not in line:
                lines[i] = line.replace(match.group(0), match.group(0) + " -qqq")
    return "\n".join(lines)
