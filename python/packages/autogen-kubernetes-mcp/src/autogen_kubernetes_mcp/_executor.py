from typing import Any

from autogen_core import CancellationToken, ComponentModel
from autogen_core.code_executor import CodeBlock
from autogen_kubernetes.code_executors import (
    PodCommandLineCodeExecutor,
    PodCommandLineCodeExecutorConfig,
)


def make_executor(args: dict[str, Any]) -> PodCommandLineCodeExecutor:
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
    return PodCommandLineCodeExecutor.load_component(component_model)


async def run_code(executor: PodCommandLineCodeExecutor, code: str) -> str:
    code_result = await executor.execute_code_blocks(
        code_blocks=[
            CodeBlock(language="python", code=code),
        ],
        cancellation_token=CancellationToken(),
    )
    return code_result.output
