import argparse
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from autogen_kubernetes_mcp._executor import make_executor, run_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="autogen-kubernetes arguments")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--kubeconfig", dest="kube_config_file", default=None)
    parser.add_argument("--image", default="python:3-slim")
    parser.add_argument("--pod-name", default=None)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--workspace-path", default="/workspace")
    parser.add_argument("-n", "--namespace", default="default")
    parser.add_argument("--volume", default=None)
    parser.add_argument("--pod-spec", default=None)

    return parser


def build_server(args: dict[str, Any]) -> FastMCP:
    mcp = FastMCP(
        name="python",
        instructions=r"""
    Use this tool to execute Python code in your chain of thought. The code will not be shown to the user. This tool should be used for internal reasoning, but not for code that is intended to be visible to the user (e.g. when creating plots, tables, or files).
    When you send a message containing python code to python, it will be executed in a stateless docker container, and the stdout of that process will be returned to you.
    """.strip(),
        host=args["host"],
        port=args["port"],
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
        async with make_executor(args) as executor:
            result: str = await run_code(executor, code)
            return result

    return mcp


def main() -> None:
    """MCP autogen-kubernetes - code interpreter deployed on kubernetes workload"""
    args = build_parser().parse_args()
    mcp = build_server(vars(args))
    mcp.run(transport="sse")
