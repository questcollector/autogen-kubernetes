# autogen-kubernetes

This is autogen(>=0.4) kubernetes extension which provides code executor on kuberentes pod

We plan to add "autogen" features needs Kubernetes features.

## Usage

### PodCommandLineCodeExecutor

Like DockerCommandLineCodeExecutor, this code executor runs codes on a container in a kubernetes pod

Unlike DockerCommandLineCodeExecutor, PodCommandLineCodeExecutor is not support container restart feature.

```python
from autogen_core import CancellationToken
from autogen_core.code_executor import CodeBlock
from autogen_kubernetes.code_executors import PodCommandLineCodeExecutor

async with PodCommandLineCodeExecutor() as executor:
    code_result = await executor.execute_code_blocks(
        code_blocks=[
            CodeBlock(language="python", code="print('Hello, World!')"),
        ],
        cancellation_token=CancellationToken(),
    )
    print(code_result)
```
```
CommandLineCodeResult(exit_code=0, output='Hello, World!\n', code_file='/workspace/tmp_code_07da107bb575cc4e02b0e1d6d99cc204.py')
```

in default options, pod will be created like

```python
{
  "kind": "Pod",
  "apiVersion": "v1",
  "metadata": {
    # name is like autogen-code-exec-{uuid4}
    "name": "autogen-code-exec-a2826c87-9b8d-46ec-be36-5fffc5d8f899",
    # created on default namespace
    "namespace": "default",
    ...
  },
  "spec": {
    "containers": [
      {
        # container named autogen-executor
        "name": "autogen-executor",
        # default image python:3-slim
        "image": "python:3-slim",
        # the container is kept alive by running keep-alive loop: /bin/sh -c while true;do sleep 5; done
        "command": [
          "/bin/sh"
        ],
        "args": [
          "-c",
          "while true;do sleep 5; done"
        ],
        ...
    ],
    ...
  }
}
```

There are arguments that change pod specification of PodCommandLineCodeExecutor.

There are several arguments to change pod settings

|argument|datatype|describe|
|--|--|--|
|image|str|Container image|
|pod_name|str|Pod name|
|namespace|str|Pod namespace|
|workspace_path|str, Path|Path in container where the LLM generated code script will be saved|
|volume|dict, str, Path, V1Volume, None|Volume to be mounted on pods|
|pod_spec|dict, str, Path, V1Pod, None|Pod specification for command line code executor|

Other parameters image, pod_name, namespace and volume are ignored when pod_spec parameter is provided.

#### volume parameter usage

"volume" parameter is to mount pre-existing volume on code executor container

The volume will be mounted in the container's workspace path provided as the executor's parameter "workspace_path".

"volume" parameter can be provided like:

- kubernetes.client.models.V1Volume

```python
from autogen_kubernetes.code_executors import PodCommandLineCodeExecutor
from kubernetes.client.models import V1Volume

volume = V1Volume(
    name="test-volume",
    empty_dir=V1EmptyDirVolumeSource(medium="Memory", size_limit="5Mi"),
)

executor = PodCommandLineCodeExecutor(volume=volume)
...

```

- dictionary, must conform kubernetes volume specification.
  reference: [
    https://github.com/kubernetes-client/python/blob/master/kubernetes/docs/V1Volume.md, 
    https://kubernetes.io/docs/concepts/storage/volumes/
  ]

```python
from autogen_kubernetes.code_executors import PodCommandLineCodeExecutor

volume = {"name": "test-volume", "emptyDir": {"medium": "Memory", "sizeLimit": "5Mi"}}

executor = PodCommandLineCodeExecutor(volume=volume)
...
```

- string which is expresses yaml or json format

```python
from autogen_kubernetes.code_executors import PodCommandLineCodeExecutor

volume = """
name: "test-volume"
emptyDir:
  medium: "Memory"
  sizeLimit: "5Mi"
"""

executor = PodCommandLineCodeExecutor(volume=volume)
...
```

- file path of yaml or json format
```python
from autogen_kubernetes.code_executors import PodCommandLineCodeExecutor

volume = "./test/test-volume.yaml"

executor = PodCommandLineCodeExecutor(volume=volume)
...
```

#### pod_spec parameter usage

To create more complex pod specification of PodCommandLineCodeExecutor, use pod_spec parameter.

Like volume parameter, pod_spec parameter can be provided like:

- kubernetes.client.models.V1Volume

```python
from autogen_kubernetes.code_executors import PodCommandLineCodeExecutor

pod_spec = V1Pod(
    metadata=V1ObjectMeta(name="test-pod", namespace="default"),
    spec=V1PodSpec(
        restart_policy="Never",
        containers=[
            V1Container(
                args=[
                    "-c",
                    "echo 'test container'; while true;do sleep 5; done",
                ],
                command=["/bin/sh"],
                name="autogen-executor", # container named "autogen-executor" must be included
                image="python:3-slim",
            )
        ],
    ),
)

executor = PodCommandLineCodeExecutor(pod_spec=pod_spec)
...
```

- dictionary, must conform kubernetes pod specification.
  reference: [
    https://github.com/kubernetes-client/python/blob/master/kubernetes/docs/V1Pod.md, 
    https://kubernetes.io/docs/concepts/workloads/pods/
  ]

```python
from autogen_kubernetes.code_executors import PodCommandLineCodeExecutor

pod = {
    "apiVersion": "v1",
    "kind": "Pod",
    "metadata": {
        "annotations": {"test": "true"},
        "name": "test-pod",
        "namespace": "default",
    },
    "spec": {
        "containers": [
            {
                "image": "python:3-slim",
                "name": "autogen-executor", # container named "autogen-executor" must be included
                "args": [
                    "/bin/sh",
                    "-c",
                    "echo 'test container'; while true;do sleep 5; done",
                ],
            },
        ],
    },
}

executor = PodCommandLineCodeExecutor(pod_spec=pod_spec)
...
```

- string which is expresses yaml or json format

```python
from autogen_kubernetes.code_executors import PodCommandLineCodeExecutor

pod = (
    "apiVersion: v1\n"
    "kind: Pod\n"
    "metadata:\n"
    "  annotations:\n"
    "    test: 'true'\n"
    "  name: test-pod\n"
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

executor = PodCommandLineCodeExecutor(pod_spec=pod_spec)
...
```

- file path of yaml or json format

```python
from autogen_kubernetes.code_executors import PodCommandLineCodeExecutor

pod = "./test/test-pod.yaml"

executor = PodCommandLineCodeExecutor(pod_spec=pod_spec)
...
```

#### Kubeconfig usage

By default, PodCommandLineCodeExecutor uses default kubeconfig file to communicate with kubernetes API server.

Default kubeconfig file path is provded by Environment variable "KUBECONFIG"

To use other kubeconfig file, provide it's path on "KUBECONFIG" environment variable.

https://github.com/kubernetes-client/python/blob/master/kubernetes/base/config/kube_config.py#L48
```python
...
KUBE_CONFIG_DEFAULT_LOCATION = os.environ.get('KUBECONFIG', '~/.kube/config')
...
```

If the PodCommandLineExecutor is initialized on kubernetes object, incluster config(serviceAccount tokens) is used.

To use incluster config, make sure to have sufficient permissions.

below is minimum permission:

|resource|verb|
|--|--|
|pods|get, create, delete|
|pods/status|get|
|pods/exec|create|
|pods/log|get|


For example, create serviceAccount and bind role with sufficient permissions.

creatae ServiceAccount
```sh
kubectl create serviceaccount autogen-executor-sa
```

create clusterRole/Role(namespaced role) with sufficient permissions
```sh
kubectl create clusterrole autogen-executor-role \
  --resource=pods --verb=get,create,delete \
  --resource=pods/exec --verb=create \
  --resource=pods/status,pods/log --verb=get
```

bind clusterRole/role with ServiceAccount
```sh
kubectl create rolebinding autogen-executor-rolebinding \
  --clusterrole autogen-executor-role --serviceaccount default:autogen-executor-sa
```

Then, PodCommandLineCodeExecutor will work alright where the pod uses the serviceAccount created before.

create pod uses serviceAccount
```sh
kubectl run autogen-executor --image python:3 \
  --overrides='{"spec": {"serviceAccount": "autogen-executor-sa"}}' \
  -- sh -c 'pip install autogen-kubernetes && sleep infinity'
```

execute the pod
```sh
kubectl exec autogen-executor -it -- python
```

execute PodCommandLineCodeExecutor
```python
from autogen_core import CancellationToken
from autogen_core.code_executor import CodeBlock
from autogen_kubernetes.code_executors import PodCommandLineCodeExecutor

async with PodCommandLineCodeExecutor() as executor:
    code_result = await executor.execute_code_blocks(
        code_blocks=[
            CodeBlock(language="python", code="print('Hello, World!')"),
        ],
        cancellation_token=CancellationToken(),
    )
    print(code_result)
```
```
CommandLineCodeResult(exit_code=0, output='Hello, World!\n', code_file='/workspace/tmp_code_07da107bb575cc4e02b0e1d6d99cc204.py')
```

#### Function module usage

To make code executor pod to have pre-installed packages, provide "functions" parameter.

```python
import pandas as pd
from autogen_core import CancellationToken
from autogen_core.code_executor import CodeBlock
from autogen_kubernetes.code_executors import PodCommandLineCodeExecutor, Alias, with_requirements

@with_requirements(python_packages=["pandas"], global_imports=[Alias(name="pandas", alias="pd")])
def load_data() -> pd.DataFrame:
    """
    Load pandas sample dataframe

    Returns:
        pd.DataFrame: sample Dataframe with columns name(str), age(int)
    """
    data = {
        "name": ["Sam", "Brown"],
        "age": [37, 57],
    }
    return pd.DataFrame(data)

async with PodCommandLineCodeExecutor(functions=[load_data]) as executor:
    code = f"from {executor._functions_module} import load_data\nprint(load_data())"
    code_result = await executor.execute_code_blocks(
        code_blocks=[
            CodeBlock(language="python", code=f"from "),
        ],
        cancellation_token=CancellationToken(),
    )
    print(code_result)

```

```
CommandLineCodeResult(exit_code=0, output='    name  age\n0    Sam   37\n1  Brown   57\n', code_file='/workspace/tmp_code_bd92ac3930fbd4f6f627885646227d5ff54753166555b98206cc01fdc023a7ef.py')
```

Another example with FunctionWithRequirements and ImportFromModule.

```python
import inspect
from autogen_core import CancellationToken
from autogen_core.code_executor import CodeBlock
from autogen_kubernetes.code_executors import PodCommandLineCodeExecutor, ImportFromModule, FunctionWithRequirements

## this section will be written on function module file(workspace_path/functions.py)
from kubernetes.client import CoreV1Api
from kubernetes.config import load_config

def kubernetes_enabled() -> bool:
    try:
        load_config()  # type: ignore
        api_client = CoreV1Api()
        api_client.list_namespace()
        return True
    except Exception:
        return False

##

test_function = FunctionWithRequirements.from_str(
    inspect.getsource(kubernetes_enabled),
    ["kubernetes"],
    [
        ImportFromModule(module="kubernetes.client", imports=("CoreV1Api",)),
        ImportFromModule(module="kubernetes.config", imports=("load_config",)),
    ],
)
async with PodCommandLineCodeExecutor(functions=[test_function]) as executor:
    code = f"from {executor._functions_module} import kubernetes_enabled\nprint(kubernetes_enabled())"
    code_result = await executor.execute_code_blocks(
        code_blocks=[
            CodeBlock(language="python", code=code),
        ],
        cancellation_token=CancellationToken(),
    )
    print(code_result)
```
```
CommandLineCodeResult(exit_code=0, output='kube_config_path not provided and default location (~/.kube/config) does not exist. Using inCluster Config. This might not work.\nTrue\n', code_file='/workspace/tmp_code_c61a3c1e421357bd54041ad195e242d8205b86a3a4a0778b8e2684bc373aac22.py')
```

### PodJupyterServer

- Creates a Jupyter Server Pod using `jupyter-kernel-gateway`, along with the required token secret and service resources.
- Similar to `PodCommandLineCodeExecutor`, the `service_spec`, and `secret_spec` can be customized in multiple formats: Python dictionary, YAML/json string, YAML/json file path, or Kubernetes client model.
- When used with default arguments, it runs on the `quay.io/jupyter/docker-stacks-foundation` image with `jupyter-kernel-gateway` and `ipykernal` installed, and executes via `jupyter-kernel-gateway`. For efficiency in production, building a custom image is recommended. Below is the sample Dockerfile

```Dockerfile
FROM quay.io/jupyter/docker-stacks-foundation

RUN mamba install --yes jupyter_kernel_gateway ipykernel && \
    mamba clean --all -f -y && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"
CMD python -m jupyter kernelgateway --KernelGatewayApp.ip=0.0.0.0 \
        --JupyterApp.answer_yes=true \
        --JupyterWebsocketPersonality.list_kernels=true
```

### PodJupyterCodeExecutor

- A `CodeExecutor` that leverages PodJupyterServer and the jupyter-kernel-gateway server to executor code statefully and retrieve results
- When used together with PodJupyterServer,note that the server generates PodJupyterConnectionInfo based on the service FQDN. This means it cannot be directly used in a non-incluster environment.
  - After creating the PodJupyterServer, you must construct and provide PodJupyterConnectionInfo so that PodJupyterCodeExecutor can access it properly.

Using with PodJupyterServer
```python
async with PodJupyeterServer() as jupyter_server:
    async with PodJupyterCodeExecutor(jupyter_server) as executor:
        code_result = await executor.execute_code_blocks(
            code_blocks=[
                CodeBlock(language="python", code="print('Hello, World!')"),
            ],
            cancellation_token=CancellationToken(),
        )
        print(code_result)
```

Using custom PodJupyterConnectionInfo
```python
async with PodJupyeterServer() as jupyter_server:
    # connection info for created jupyter server pod
    connection_info = PodJupyetrConnectionInfo(
        host="https://jupyter-server/access/path",
        port="443",
        token=SecretStr("token-string")
    )
    async with PodJupyterCodeExecutor(connection_info) as executor:
        code_result = await executor.execute_code_blocks(
            code_blocks=[
                CodeBlock(language="python", code="print('Hello, World!')"),
            ],
            cancellation_token=CancellationToken(),
        )
        print(code_result)
```

- Even without using PodJupyterServer, you can configure a custom Jupyter server that meets the required conditions and provide PodJupyterConnectionInfo for PodJupyterCodeExecutor to work with

## Contribute

This project's structure conforms microsoft/autogen python package

```
├── LICENSE
└── python
    ├── run_task_in_pkgs_if_exist.py
    ├── pyproject.toml
    ├── shared_tasks.toml
    ├── uv.lock
    └── packages
        └── autogen-kubernetes
            ├── tests
            │   ├── test_utils.py
            │   ├── conftest.py
            │   ├── test_kubernetees_code_executor.py
            │   ├── test-pod.yaml
            │   └── test-volume.yaml
            ├── LICENSE-CODE
            ├── pyproject.toml
            ├── README.md
            └── src
                └── autogen_kubernetes
                    ├── py.typed
                    └── code_executors
                        ├── _utils.py
                        └── _kubernetes_code_executor.py
```

### Install uv

Install uv according to your environment.

https://docs.astral.sh/uv/getting-started/installation/

### Sync project

```sh
cd autogen-kubernetes/python
uv venv --python ">=3.10"
source .venv/bin/activate
uv sync --locked --all-extras
```

### Common tasks

- Format: `poe format`
- Lint: `poe lint`
- Test: `poe test`
- Mypy: `poe mypy`
- Check all: `poe check`

## Licensing

This project is licensed under the MIT License.

### Code Modification

This project includes code from the microsoft/autogen project (licensed under the MIT License), 

with modifications made by kiyoung you(questcollector), See the [LICENSE-CODE](python/packages/autogen-kubernetes/LICENSE-CODE) file for details


### Third-Party Dependencies

This project uses the following third-party dependencies:

1. **kubernetes**
    License: Apache License, Version 2.0
    Source: https://github.com/kubernetes-client/python
2. **httpx**
    License: BSD 3-Clause "New" or "Revised"
    Source: https://github.com/encode/httpx
3. **websockets**
    License: BSD 3-Clause "New" or "Revised"
    Source: https://github.com/python-websockets/websockets
4. **PyYAML**
    License: MIT License
    Source: https://github.com/yaml/pyyaml

For details, see the LICENCE-THIRD-PARTY file.