import argparse

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from ._executor import make_executor, run_code

parser = argparse.ArgumentParser(description="autogen-kubernetes arguments")
parser.add_argument("--kubeconfig", dest="kube_config_file", default=None)
parser.add_argument("--image", default="python:3-slim")
parser.add_argument("--pod-name", default=None)
parser.add_argument("--timeout", type=int, default=60)
parser.add_argument("--workspace_path", default="/workspace")
parser.add_argument("-n", "--namespace", default="default")
parser.add_argument("--volume", default=None)
parser.add_argument("--pod-spec", default=None)

args = parser.parse_args()

mcp = FastMCP(
    name="python",
    instructions=r"""
Use this tool to execute Python code in your chain of thought. The code will not be shown to the user. This tool should be used for internal reasoning, but not for code that is intended to be visible to the user (e.g. when creating plots, tables, or files).
When you send a message containing python code to python, it will be executed in a stateless docker container, and the stdout of that process will be returned to you.
""".strip(),
)


@mcp.tool(
    name="python",
    title="Execute Python code",
    description="""
Use this tool to execute Python code in your chain of thought. The code will not be shown to the user. This tool should be used for internal reasoning, but not for code that is intended to be visible to the user (e.g. when creating plots, tables, or files).
When you send a message containing python code to python, it will be executed in a stateless docker container, and the stdout of that process will be returned to you.
    """,
)
async def python(code: str) -> str:
    async with make_executor(vars(args)) as executor:
        result: str = await run_code(executor, code)
        return result


if __name__ == "__main__":
    """MCP autogen-kubernetes - code interpreter deployed on kubernetes workload"""
    mcp.run(transport="sse")
