"""Launch the Deka web API with uvicorn.

Usage::

    uv run deka-web
    # or
    python -m src.web_api
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from dotenv import load_dotenv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.web_api",
        description="Deka web API server.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable uvicorn auto-reload (dev only)",
    )
    args = parser.parse_args(argv)

    # Pass an explicit path when DEKA_ENV_FILE is set (production
    # container builds bind-mount /app/.env and set the env var). In
    # dev, fall back to dotenv's auto-discovery which walks up the
    # call stack — that breaks in bytecode-only images because the
    # frames' co_filename points at deleted .py files, so find_dotenv
    # walks off the top of the stack with AttributeError.
    env_file = os.environ.get("DEKA_ENV_FILE")
    if env_file:
        load_dotenv(env_file, override=False)
    else:
        load_dotenv(override=False)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
    )

    # Import here so logging is configured before FastAPI/uvicorn set up their own.
    import uvicorn

    uvicorn.run(
        "src.web_api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
