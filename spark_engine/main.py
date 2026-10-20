"""CLI entry: 'turbine serve <model> [flags]' parses args, builds the engine, starts uvicorn."""

import argparse
from collections.abc import Sequence
from typing import Any, Literal, get_args, get_origin

import uvicorn
from pydantic import BaseModel

from spark_engine.api.server import create_app
from spark_engine.config import EngineConfig, log_resolved
from spark_engine.engine import build_engine
from spark_engine.engine_client import EngineClient
from spark_engine.log import configure_logging, get_logger

log = get_logger("main")


def _add_field_flags(parser: argparse.ArgumentParser, model_cls: type[BaseModel]) -> None:
    """One --kebab-case flag per field (except `model`); unsupported field types raise TypeError."""
    for name, field in model_cls.model_fields.items():
        if name == "model":
            continue
        annotation = field.annotation
        if field.is_required():
            raise TypeError(
                f"unsupported EngineConfig field type: {name}: {annotation} (no default)"
            )
        kwargs: dict[str, Any] = {"default": field.default, "help": f"(default: {field.default})"}
        if annotation in (int, float, str):
            kwargs["type"] = annotation
        elif get_origin(annotation) is Literal and all(
            isinstance(v, str) for v in get_args(annotation)
        ):
            kwargs["choices"] = get_args(annotation)
        else:
            raise TypeError(f"unsupported EngineConfig field type: {name}: {annotation}")
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name, **kwargs)


def build_parser() -> argparse.ArgumentParser:
    """`turbine serve <model>` with one --kebab-case flag per EngineConfig field."""
    parser = argparse.ArgumentParser(prog="turbine")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="serve a model over the OpenAI-compatible API")
    serve.add_argument("model", help="HF model id or local path")
    _add_field_flags(serve, EngineConfig)
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
    """Build engine + tokenizer, wrap the engine in its client, run the API under uvicorn.

    The engine thread starts and stops in the app's lifespan; logging is already configured
    by `config_from_args`, so uvicorn's own log config is disabled.
    """
    engine, tokenizer = build_engine(config)
    client = EngineClient(engine, config.max_queue)
    app = create_app(client, tokenizer, config.model, config.max_model_len)
    log.info("server.started", host=config.host, port=config.port, model=config.model)
    uvicorn.run(app, host=config.host, port=config.port, log_config=None)


def cli(argv: Sequence[str] | None = None) -> None:
    serve(config_from_args(argv))


if __name__ == "__main__":
    cli()
