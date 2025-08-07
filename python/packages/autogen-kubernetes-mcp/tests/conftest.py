from typing import Any

import pytest
from kubernetes import config
from kubernetes.client import CoreV1Api
from kubernetes.client.configuration import Configuration


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
