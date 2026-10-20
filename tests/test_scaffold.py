"""Smoke test: every module in the package imports (catches syntax errors in stubs and blocks)."""

import importlib
import pkgutil

import spark_engine


def test_every_module_imports() -> None:
    for info in pkgutil.walk_packages(spark_engine.__path__, prefix="spark_engine."):
        importlib.import_module(info.name)
