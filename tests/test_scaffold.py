"""Smoke test: every module in the package imports (catches syntax errors in stubs and blocks)."""

import importlib
import pkgutil

import turbine_serve


def test_every_module_imports() -> None:
    for info in pkgutil.walk_packages(turbine_serve.__path__, prefix="turbine_serve."):
        importlib.import_module(info.name)
