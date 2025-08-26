import textwrap
from typing import Any, Callable

import pytest
from kubernetes import config
from kubernetes.client import CoreV1Api
from kubernetes.client.configuration import Configuration
from kubernetes.client.models import (
    V1Container,
    V1EmptyDirVolumeSource,
    V1ObjectMeta,
    V1Pod,
    V1PodSpec,
    V1Volume,
)


def kubernetes_enabled() -> bool:
    try:
        config.load_config()  # type: ignore
        api_client = CoreV1Api()
        api_client.list_namespace()  # type: ignore
        return True
    except Exception:
        return False


state_kubernetes_enabled = kubernetes_enabled()


@pytest.fixture
def kubeconfig() -> Any:
    config.load_config()  # type: ignore
    return Configuration.get_default_copy()  # type: ignore


@pytest.fixture
def pod_spec() -> Callable[[str], dict[str, Any]]:
    def default_pod_spec(name: str = "test") -> dict[str, Any]:
        return {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "annotations": {"test": "true"},
                "name": name,
                "namespace": "default",
            },
            "spec": {
                "containers": [
                    {
                        "image": "python:3-slim",
                        "name": "autogen-executor",
                        "args": [
                            "/bin/sh",
                            "-c",
                            "echo 'test container'; while true;do sleep 5; done",
                        ],
                    },
                ],
            },
        }

    return default_pod_spec


@pytest.fixture
def pod_yaml_str() -> Callable[[str], str]:
    def default_pod_yaml(name: str = "test") -> str:
        return (
            "apiVersion: v1\n"
            "kind: Pod\n"
            "metadata:\n"
            "  annotations:\n"
            "    test: 'true'\n"
            f"  name: {name}\n"
            "  namespace: default\n"
            "spec:\n"
            "  containers:\n"
            "  - args:\n"
            "    - sh\n"
            "    - -c\n"
            "    - while true;do sleep 5; done\n"
            "    image: python:3-slim\n"
            "    name: autogen-executor\n"
        )

    return default_pod_yaml


@pytest.fixture
def v1_pod() -> Callable[[str, str], Any]:
    def default_v1_pod(name: str = "test", container_name: str = "autogen-executor") -> Any:
        return V1Pod(
            metadata=V1ObjectMeta(name=name, namespace="default"),
            spec=V1PodSpec(
                restart_policy="Never",
                containers=[
                    V1Container(
                        args=[
                            "-c",
                            "echo 'test container'; while true;do sleep 5; done",
                        ],
                        command=["/bin/sh"],
                        name=container_name,
                        image="python:3-slim",
                    )
                ],
            ),
        )

    return default_v1_pod


@pytest.fixture
def volume_dict() -> Callable[[str], dict[str, Any]]:
    def default_volume_dict(name: str = "test") -> dict[str, Any]:
        return {"name": name, "emptyDir": {"medium": "Memory", "sizeLimit": "5Mi"}}

    return default_volume_dict


@pytest.fixture
def v1_volume() -> Callable[[str], Any]:
    def default_v1_volume(name: str = "test") -> Any:
        return V1Volume(
            name=name,
            empty_dir=V1EmptyDirVolumeSource(medium="Memory", size_limit="5Mi"),
        )

    return default_v1_volume


@pytest.fixture
def generated_pod_name_regex() -> str:
    return r"^autogen-code-exec-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-4[0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$"


@pytest.fixture
def generated_jupyter_pod_regex() -> str:
    return r"^autogen-jupyter-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-4[0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$"
