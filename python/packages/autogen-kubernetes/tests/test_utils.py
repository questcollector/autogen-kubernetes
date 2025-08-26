from typing import Any, Callable

import pytest
from autogen_kubernetes.code_executors._utils import (
    StreamChannel,
    create_namespaced_corev1_resource,
    delete_namespaced_corev1_resource,
    get_pod_logs,
    pod_exec_stream,
    sanitize_for_serialization,
    wait_for_ready,
)
from conftest import state_kubernetes_enabled
from kubernetes.client.models import (
    V1PersistentVolumeClaimVolumeSource,
    V1Volume,
)


def test_clean_none_value() -> None:
    volume = V1Volume(
        name="test",
        persistent_volume_claim=V1PersistentVolumeClaimVolumeSource(claim_name="test-pvc", read_only=False),
    )
    volume_dict = sanitize_for_serialization(volume)  # type: ignore
    assert "awsElasticBlockStore" not in volume_dict
    assert volume_dict["name"] == "test"
    assert volume_dict["persistentVolumeClaim"] == {
        "claimName": "test-pvc",
        "readOnly": False,
    }


@pytest.mark.skipif(not state_kubernetes_enabled, reason="kubernetes not accessible")
@pytest.mark.asyncio
async def test_create_pod(kubeconfig: Any, pod_spec: Callable[[str], dict[str, Any]]) -> None:
    create_pod_spec = pod_spec("test-create")
    pod = await create_namespaced_corev1_resource(kubeconfig, create_pod_spec, dry_run=True)

    assert pod["metadata"]["name"] == "test-create"
    assert len(pod["spec"]["containers"]) == 1


@pytest.mark.skipif(not state_kubernetes_enabled, reason="kubernetes not accessible")
@pytest.mark.asyncio
async def test_delete_pod(kubeconfig: Any, pod_spec: Callable[[str], dict[str, Any]]) -> None:
    delete_pod_spec = pod_spec("test-delete")
    pod = await create_namespaced_corev1_resource(kubeconfig, delete_pod_spec)
    deleted_pod = await delete_namespaced_corev1_resource(
        kubeconfig,
        pod,
    )

    assert deleted_pod["metadata"]["name"] == "test-delete"


@pytest.mark.skipif(not state_kubernetes_enabled, reason="kubernetes not accessible")
@pytest.mark.asyncio
async def test_get_pod_log(kubeconfig: Any, pod_spec: Callable[[str], dict[str, Any]]) -> None:
    get_log_pod_spec = pod_spec("test-get-log")
    pod = await create_namespaced_corev1_resource(kubeconfig, get_log_pod_spec)
    pod = await wait_for_ready(
        kubeconfig,
        pod_name=pod["metadata"]["name"],
        namespace=pod["metadata"]["namespace"],
    )
    log = await get_pod_logs(
        kubeconfig,
        pod_name=pod["metadata"]["name"],
        namespace=pod["metadata"]["namespace"],
        container_name="autogen-executor",
    )
    await delete_namespaced_corev1_resource(
        kubeconfig,
        pod,
    )

    assert "test container" in log


@pytest.mark.skipif(not state_kubernetes_enabled, reason="kubernetes not accessible")
@pytest.mark.asyncio
async def test_exec_pod(kubeconfig: Any, pod_spec: Callable[[str], dict[str, Any]]) -> None:
    exec_pod_spec = pod_spec("test-exec")
    pod = await create_namespaced_corev1_resource(kubeconfig, exec_pod_spec)
    pod = await wait_for_ready(
        kubeconfig,
        pod_name=pod["metadata"]["name"],
        namespace=pod["metadata"]["namespace"],
    )
    command = [
        "sh",
        "-c",
        "python3 - << EOF\nprint('stdout')\nimport sys\nprint('stderr', file=sys.stderr)\nEOF",
    ]
    last_exit_code = 0
    async for channel, msg, exit_code in pod_exec_stream(
        kubeconfig,
        pod_name=pod["metadata"]["name"],
        namespace=pod["metadata"]["namespace"],
        command=command,
        container_name="autogen-executor",
    ):
        if channel == StreamChannel.STDOUT_CHANNEL:
            assert "stdout" in msg
        elif channel == StreamChannel.STDERR_CHANNEL:
            assert "stderr" in msg
        if exit_code is not None:
            last_exit_code = exit_code

    assert last_exit_code == 0

    await delete_namespaced_corev1_resource(kubeconfig, pod)


@pytest.mark.skipif(not state_kubernetes_enabled, reason="kubernetes not accessible")
@pytest.mark.asyncio
async def test_exec_pod_timeout(kubeconfig: Any, pod_spec: Callable[[str], dict[str, Any]]) -> None:
    exec_pod_spec = pod_spec("test-exec-timeout")
    pod = await create_namespaced_corev1_resource(kubeconfig, exec_pod_spec)
    pod = await wait_for_ready(
        kubeconfig,
        pod_name=pod["metadata"]["name"],
        namespace=pod["metadata"]["namespace"],
    )
    command = ["sh", "-c", "timeout 3 sleep 5"]
    last_exit_code = 0
    async for _, _, exit_code in pod_exec_stream(
        kubeconfig,
        pod_name=pod["metadata"]["name"],
        namespace=pod["metadata"]["namespace"],
        command=command,
        container_name="autogen-executor",
    ):
        if exit_code is not None:
            last_exit_code = exit_code

    assert last_exit_code == 124

    await delete_namespaced_corev1_resource(kubeconfig, pod)


@pytest.mark.skipif(not state_kubernetes_enabled, reason="kubernetes not accessible")
@pytest.mark.asyncio
async def test_exec_pod_error(kubeconfig: Any, pod_spec: Callable[[str], dict[str, Any]]) -> None:
    exec_pod_spec = pod_spec("test-exec-error")
    pod = await create_namespaced_corev1_resource(kubeconfig, exec_pod_spec)
    pod = await wait_for_ready(
        kubeconfig,
        pod_name=pod["metadata"]["name"],
        namespace=pod["metadata"]["namespace"],
    )
    command = ["sh", "-c", "python3 - << EOF\nraise ValueError('test error')\nEOF"]
    last_exit_code = 0
    async for _, _, exit_code in pod_exec_stream(
        kubeconfig,
        pod_name=pod["metadata"]["name"],
        namespace=pod["metadata"]["namespace"],
        command=command,
        container_name="autogen-executor",
    ):
        if exit_code is not None:
            last_exit_code = exit_code

    assert last_exit_code != 0

    await delete_namespaced_corev1_resource(kubeconfig, pod)
