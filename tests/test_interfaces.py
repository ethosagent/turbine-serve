"""Protocol conformance tests.

Each block task registers its concrete class by appending ONE line to CONFORMANCE:

    CONFORMANCE.append((ConcreteClass, ProtocolName, lambda: ConcreteClass(...)))

Entry shape: (concrete class, Protocol it must satisfy, zero-arg factory building an instance).
The test asserts isinstance(factory(), Protocol) (Protocols are @runtime_checkable).
With no entries the test is skipped, not errored.
"""

from collections.abc import Callable

import pytest
from spark_engine import interfaces
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
from spark_engine.types import AttentionMetadata

CONFORMANCE: list[tuple[type, type, Callable[[], object]]] = []

_PLACEHOLDER = pytest.param(object, object, object, marks=pytest.mark.skip(reason="no entries yet"))


@pytest.mark.parametrize(
    ("cls", "protocol", "factory"),
    CONFORMANCE or [_PLACEHOLDER],
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
    assert interfaces.Scheduler is Scheduler


def test_scheduler_has_fail() -> None:
    assert hasattr(Scheduler, "fail")


def test_attention_metadata_is_plain_base_with_none_defaults() -> None:
    meta = AttentionMetadata()
    assert (meta.q_start, meta.q_len, meta.ctx_len, meta.block_tables) == (None, None, None, None)
