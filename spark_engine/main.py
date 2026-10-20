"""CLI entry: 'turbine serve <model> [flags]' parses args, builds the engine, starts uvicorn."""

import argparse
from collections.abc import Sequence
from typing import Any, get_args

from spark_engine.config import EngineConfig, log_resolved
from spark_engine.log import configure_logging, get_logger

log = get_logger("main")


def build_parser() -> argparse.ArgumentParser:
    """`turbine serve <model>` with one --kebab-case flag per EngineConfig field."""
    parser = argparse.ArgumentParser(prog="turbine")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="serve a model over the OpenAI-compatible API")
    serve.add_argument("model", help="HF model id or local path")
    for name, field in EngineConfig.model_fields.items():
        if name == "model":
            continue
        kwargs: dict[str, Any] = {"default": field.default, "help": f"(default: {field.default})"}
        if field.annotation in (int, float, str):
            kwargs["type"] = field.annotation
        else:
            kwargs["choices"] = get_args(field.annotation)
        serve.add_argument(f"--{name.replace('_', '-')}", dest=name, **kwargs)
    return parser


def config_from_args(argv: Sequence[str] | None) -> EngineConfig:
    """Parse argv into an EngineConfig, configure logging, log `config.resolved`."""
    args = vars(build_parser().parse_args(argv))
    args.pop("command")
    config = EngineConfig(**args)
    configure_logging(config.log_level, config.log_file)
    log_resolved(config)
    return config


def serve(config: EngineConfig) -> None:
    raise NotImplementedError("serve is wired in Task 15")


def cli(argv: Sequence[str] | None = None) -> None:
    serve(config_from_args(argv))


if __name__ == "__main__":
    cli()
