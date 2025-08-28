from typing import Any

from autogen_core import CancellationToken, ComponentModel
from autogen_core.code_executor import CodeBlock, CodeExecutor
from autogen_kubernetes.code_executors import (
    PodCommandLineCodeExecutor,
    PodCommandLineCodeExecutorConfig,
    PodJupyterCodeExecutor,
    PodJupyterCodeExecutorConfig,
    PodJupyterServer,
    PodJupyterServerConfig,
)


async def make_executor(args: dict[str, Any]) -> list[Any]:
    closable_instances = []
    if args["type"] == "jupyter":
        jupyter_server_config = PodJupyterServerConfig(**args)
        jupyter_server = PodJupyterServer.load_component(
            ComponentModel(
                provider="autogen_kubernetes.code_executors.PodJupyterServer",
                component_type=PodJupyterServer.component_type,
                version=PodJupyterServer.component_version,
                component_version=PodJupyterServer.component_version,
                description=PodJupyterServer.component_description,
                label=PodJupyterServer.__name__,
                config=jupyter_server_config.model_dump(),
            )
        )
        await jupyter_server.start()
        jupyter_executor_config = PodJupyterCodeExecutorConfig(jupyter_server=jupyter_server, **args)
        jupyter_executor = PodJupyterCodeExecutor.load_component(
            ComponentModel(
                provider="autogen_kubernetes.code_executors.PodJupyterCodeExecutor",
                component_type=PodJupyterCodeExecutor.component_type,
                version=PodJupyterCodeExecutor.component_version,
                component_version=PodJupyterCodeExecutor.component_version,
                description=PodJupyterCodeExecutor.component_description,
                label=PodJupyterCodeExecutor.__name__,
                config=jupyter_executor_config.model_dump(),
            )
        )
        await jupyter_executor.start()
        closable_instances.extend([jupyter_executor, jupyter_server])
    else:
        pod_commandline_executor_config = PodCommandLineCodeExecutorConfig(**args)
        component_model = ComponentModel(
            provider="autogen_kubernetes.code_executors.PodCommandLineCodeExecutor",
            component_type=PodCommandLineCodeExecutor.component_type,
            version=PodCommandLineCodeExecutor.component_version,
            component_version=PodCommandLineCodeExecutor.component_version,
            description=PodCommandLineCodeExecutor.component_description,
            label=PodCommandLineCodeExecutor.__name__,
            config=pod_commandline_executor_config.model_dump(exclude_none=True),
        )
        cmd_executor = PodCommandLineCodeExecutor.load_component(component_model)
        await cmd_executor.start()
        closable_instances.append(cmd_executor)
    return closable_instances


async def run_code(
    executor: CodeExecutor,
    code: str,
    cancellation_token: CancellationToken,
) -> str:
    code_result = await executor.execute_code_blocks(
        code_blocks=[
            CodeBlock(language="python", code=code),
        ],
        cancellation_token=cancellation_token,
    )
    return code_result.output
