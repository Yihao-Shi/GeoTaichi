"""GeoTaichi MCP server entry point."""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any, Optional

from . import __version__
from .core.mcp_resources import register_resources
from .tools import configure_task_manager, register_tools


try:
    from fastmcp import FastMCP as _FastMCP
except ImportError:
    _FastMCP = None


SERVER_INSTRUCTIONS = (
    "GeoTaichi model-building and execution server for MPM, DEM, MPDEM/CFDEM, IGA, and IGAMPM. "
    "Browse a known capability path and query only when the path is unknown. Inspect generated models before running. "
    "Run every model in an isolated task process because Taichi initialization, dimension, precision, and backend are "
    "process-global. Pass task_id to geotaichi_execute_code for cooperative execution in the task's shared __main__ "
    "namespace at a complete-step checkpoint. Read geotaichi://index before loading detailed resources. "
    "SolverJob and arbitrary-code tools execute trusted local Python and require an explicit trusted profile. "
    "Validate physical observables separately from successful execution."
)


def create_server(tool_profile: str = "trusted", auth: Any = None) -> Any:
    if _FastMCP is None:
        raise RuntimeError(
            "FastMCP is not installed. Install the optional server dependency with: pip install -e '.[mcp]'"
        )
    server = _FastMCP("GeoTaichi MCP Server", instructions=SERVER_INSTRUCTIONS, auth=auth)
    register_tools(server, profile=tool_profile)
    register_resources(server)
    return server


mcp: Optional[Any] = create_server() if _FastMCP is not None else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="geotaichi-mcp",
        description=(
            "GeoTaichi documentation, model inspection, persistent tasks, "
            "and cooperative live execution over MCP"
        ),
    )
    parser.add_argument("--version", "-v", action="version", version="geotaichi-mcp %s" % __version__)
    parser.add_argument("--transport", choices=("stdio", "http", "sse"), default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--repo-root", help="GeoTaichi checkout; otherwise auto-detected")
    parser.add_argument("--workspace", help="Persistent task directory; defaults to .geotaichi-mcp/tasks")
    parser.add_argument("--log-level", choices=("debug", "info", "warning", "error"), default="warning")
    parser.add_argument(
        "--tool-profile",
        choices=("auto", "safe", "trusted"),
        default="auto",
        help="auto uses trusted tools for stdio and safe tools for HTTP/SSE",
    )
    parser.add_argument(
        "--auth-token-env",
        default="GEOTAICHI_MCP_AUTH_TOKEN",
        help="environment variable containing the HTTP/SSE bearer token",
    )
    parser.add_argument(
        "--allow-unauthenticated-http",
        action="store_true",
        help="allow loopback HTTP/SSE without a bearer token; trusted local development only",
    )
    return parser.parse_args()


def _http_auth(args: argparse.Namespace) -> Any:
    if args.transport not in {"http", "sse"}:
        return None
    token = os.environ.get(args.auth_token_env, "")
    if not token:
        if args.allow_unauthenticated_http:
            return None
        raise SystemExit(
            "%s must contain a bearer token for HTTP/SSE, or pass "
            "--allow-unauthenticated-http for trusted loopback development" % args.auth_token_env
        )
    try:
        from fastmcp.server.auth.providers.jwt import StaticTokenVerifier
    except ImportError as exc:
        raise SystemExit("installed FastMCP does not provide StaticTokenVerifier: %s" % exc) from exc
    return StaticTokenVerifier(
        tokens={token: {"client_id": "geotaichi-local", "scopes": ["geotaichi:access"]}},
        required_scopes=["geotaichi:access"],
    )


def main() -> None:
    args = parse_args()
    if args.port < 1 or args.port > 65535:
        raise SystemExit("--port must be between 1 and 65535")
    if args.repo_root:
        os.environ["GEOTAICHI_REPO_ROOT"] = os.path.abspath(os.path.expanduser(args.repo_root))
    if args.workspace:
        os.environ["GEOTAICHI_MCP_WORKSPACE"] = os.path.abspath(os.path.expanduser(args.workspace))

    level = getattr(logging, args.log_level.upper())
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
    try:
        profile = args.tool_profile
        if profile == "auto":
            profile = "trusted" if args.transport == "stdio" else "safe"
        server = create_server(tool_profile=profile, auth=_http_auth(args))
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    configure_task_manager()
    run_kwargs = {"transport": args.transport, "show_banner": False}
    if args.transport in {"http", "sse"}:
        run_kwargs.update({"host": args.host, "port": args.port})
    try:
        server.run(**run_kwargs)
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
