import argparse
import asyncio
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, TypedDict

from autogen_core import CancellationToken
from mcp.server.fastmcp import Context, FastMCP
from mcp.types import ToolAnnotations

from autogen_kubernetes_mcp._executor import make_executor, run_code

STATEFUL_DESCRIPTION = r"""Use this tool to execute Python code in your chain of thought. The code will not be shown to the user. This tool should be used for internal reasoning, but not for code that is intended to be visible to the user (e.g. when creating plots, tables, or files).
When you send a message containing Python code to python, it will be executed in a stateful Jupyter notebook environment. python will respond with the output of the execution or time out after 120.0 seconds. The drive at '/mnt/data' can be used to save and persist user files. Internet access for this session is UNKNOWN. Depends on the cluster"""
STATELESS_DESCRIPTION = r"""Use this tool to execute Python code in your chain of thought. The code will not be shown to the user. This tool should be used for internal reasoning, but not for code that is intended to be visible to the user (e.g. when creating plots, tables, or files).
When you send a message containing python code to python, it will be executed in a stateless docker container, and the stdout of that process will be returned to you."""

sessions: dict[str, list[Any]] = {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="autogen-kubernetes-mcp arguments")
    parser.add_argument(
        dest="type",
        choices=["commandline", "jupyter"],
        help="choose a code executor type (commandline, jupyter)",
    )
    parser.add_argument("--host", default="0.0.0.0", help="MCP server host")
    parser.add_argument("--port", type=int, default=8000, help="MCP server port")
    parser.add_argument("--kubeconfig", dest="kube_config_file", default=None)
    parser.add_argument("--image", default=None, help="image for code execution pod")
    parser.add_argument("--pod-name", default=None)
    parser.add_argument("--timeout", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--workspace-path", default="/workspace")
    parser.add_argument("-n", "--namespace", default="default")
    parser.add_argument("--volume", default=None)
    parser.add_argument("--pod-spec", default=None)
    parser.add_argument("--command", nargs="+", help="jupyter server pod commands", default=None)
    parser.add_argument("--args", nargs="+", help="jupyter server pod arguments", default=None)

    return parser


class State(TypedDict):
    sid: str


def build_server(args: dict[str, Any]) -> FastMCP:
    tool_description = STATELESS_DESCRIPTION
    if args["type"] == "jupyter":
        tool_description = STATEFUL_DESCRIPTION
        if "timeout" not in args:
            args["timeout"] = 120

    @asynccontextmanager
    async def session_lifespan(app: FastMCP) -> AsyncIterator[State]:
        sid = str(uuid.uuid4())
        executor = await make_executor(args)
        sessions[sid] = executor

        yield {"sid": sid}

        for instance in executor:
            await instance.stop()
        sessions.pop(sid, None)

    mcp = FastMCP(
        name="python",
        instructions=tool_description.strip(),
        host=args["host"],
        port=args["port"],
        lifespan=session_lifespan,
    )

    @mcp.tool(
        name="python",
        title="Execute Python code",
        description=tool_description,
    )
    async def python(code: str, ctx: Context) -> str:  # type: ignore
        sid = ctx.request_context.lifespan_context["sid"]
        executor = sessions[sid][0]
        cancellation_token = CancellationToken()
        try:
            result: str = await run_code(executor, code, cancellation_token)
            return result
        except asyncio.CancelledError:
            cancellation_token.cancel()
            return ""

    return mcp


def main() -> None:
    """MCP autogen-kubernetes - code interpreter deployed on kubernetes workload"""
    args = build_parser().parse_args()
    mcp = build_server(vars(args))
    mcp.run(transport="sse")
