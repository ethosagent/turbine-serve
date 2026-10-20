"""Protocol conformance tests.

Each block task registers its concrete class by appending ONE line to CONFORMANCE:

    CONFORMANCE.append((ConcreteClass, ProtocolName, lambda: ConcreteClass(...)))

Put the line right below the CONFORMANCE definition. Cases are generated lazily in
pytest_generate_tests, so the list is read at collection time and ids are the class names.

Entry shape: (concrete class, Protocol it must satisfy, zero-arg factory building an instance).
The test asserts isinstance(factory(), Protocol) (Protocols are @runtime_checkable).
With no entries the test is skipped, not errored.
"""

from collections.abc import Callable

import pytest
from spark_engine.interfaces import (
    AttentionBackend,
    Detokenizer,
    KVCacheManager,
    Model,
    ModelRunner,
    Sampler,
    Scheduler,
    Tokenizer,
)
from spark_engine.kv_cache import PagedKVCache
from spark_engine.types import AttentionMetadata

CONFORMANCE: list[tuple[type, type, Callable[[], object]]] = []
CONFORMANCE.append(
    (PagedKVCache, KVCacheManager, lambda: PagedKVCache(1, 1, 1, 16, 0, "cpu", num_blocks=4))
)


def test_concrete_classes_satisfy_protocols(
    cls: type, protocol: type, factory: Callable[[], object]
) -> None:
    obj = factory()
    assert isinstance(obj, cls)
    assert isinstance(obj, protocol)


def test_protocols_importable_and_runtime_checkable() -> None:
    protocols = [
        Tokenizer,
        Detokenizer,
        KVCacheManager,
        Scheduler,
        AttentionBackend,
        Model,
        ModelRunner,
        Sampler,
    ]
    for p in protocols:
        assert getattr(p, "_is_runtime_protocol", False), p
        assert not isinstance(object(), p)


def test_scheduler_has_fail() -> None:
    assert hasattr(Scheduler, "fail")


def test_attention_metadata_is_plain_base_with_none_defaults() -> None:
    meta = AttentionMetadata()
    assert (meta.q_start, meta.q_len, meta.ctx_len, meta.block_tables) == (None, None, None, None)


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if metafunc.function is not test_concrete_classes_satisfy_protocols:
        return
    metafunc.parametrize(
        ("cls", "protocol", "factory"),
        [pytest.param(c, p, f, id=f"{c.__name__}-{p.__name__}") for c, p, f in CONFORMANCE],
    )


def test_case_count_matches_conformance(request: pytest.FixtureRequest) -> None:
    items = [
        i
        for i in request.session.items
        if getattr(i, "originalname", None) == "test_concrete_classes_satisfy_protocols"
    ]
    # an empty list yields one skipped NOTSET placeholder case
    assert len(items) == (len(CONFORMANCE) or 1)
