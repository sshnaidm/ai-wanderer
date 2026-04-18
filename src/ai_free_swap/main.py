from __future__ import annotations

import argparse
import logging
import sys

import uvicorn

from .config import load_config
from .server import create_app

# Ensure provider modules are imported so they register themselves
import ai_free_swap.providers  # noqa: F401


def main():
    parser = argparse.ArgumentParser(
        prog="ai-free-swap",
        description="OpenAI-compatible proxy routing through free-tier AI providers",
    )
    parser.add_argument("--config", "-c", default="config.yaml", help="Path to config file")
    parser.add_argument("--host", default=None, help="Override server host")
    parser.add_argument("--port", type=int, default=None, help="Override server port")
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Log level",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        sys.exit(1)

    host = args.host or config.server.host
    port = args.port or config.server.port

    app = create_app(config)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
