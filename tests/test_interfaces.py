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
import torch
from transformers import Qwen3Config
from turbine_serve.attention.sdpa import SDPAAttention
from turbine_serve.detokenizer import IncrementalDetokenizer
from turbine_serve.engine import Engine
from turbine_serve.interfaces import (
    AttentionBackend,
    Detokenizer,
    EngineLike,
    KVCacheManager,
    Model,
    ModelRunner,
    Sampler,
    Scheduler,
    Tokenizer,
)
from turbine_serve.kv_cache import PagedKVCache
from turbine_serve.model_runner import GPUModelRunner
from turbine_serve.models.qwen3 import Qwen3Model
from turbine_serve.sampler import TorchSampler
from turbine_serve.scheduler import FCFSScheduler
from turbine_serve.tokenizer import HFTokenizer
from turbine_serve.types import AttentionMetadata

from tests.fakes import FakeRunner, make_engine

CONFORMANCE: list[tuple[type, type, Callable[[], object]]] = []
CONFORMANCE.append((Engine, EngineLike, lambda: make_engine()[0]))
CONFORMANCE.append(
    (PagedKVCache, KVCacheManager, lambda: PagedKVCache(1, 1, 1, 16, 0, "cpu", num_blocks=4))
)
CONFORMANCE.append(
    (
        FCFSScheduler,
        Scheduler,
        lambda: FCFSScheduler(PagedKVCache(1, 1, 1, 16, 0, "cpu", num_blocks=4), 8, (7,)),
    )
)
CONFORMANCE.append((SDPAAttention, AttentionBackend, lambda: SDPAAttention(4, 2, 8)))
CONFORMANCE.append((TorchSampler, Sampler, TorchSampler))
CONFORMANCE.append((FakeRunner, ModelRunner, FakeRunner))
CONFORMANCE.append((HFTokenizer, Tokenizer, lambda: HFTokenizer("Qwen/Qwen3-4B", 8192)))
CONFORMANCE.append(
    (
        IncrementalDetokenizer,
        Detokenizer,
        lambda: HFTokenizer("Qwen/Qwen3-4B", 8192).new_detokenizer(),
    )
)


def _tiny_cfg() -> Qwen3Config:
    return Qwen3Config(
        num_hidden_layers=1,
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        vocab_size=16,
        rope_parameters={"rope_type": "default", "rope_theta": 1e6},
    )


def _tiny_model() -> Qwen3Model:
    return Qwen3Model(_tiny_cfg(), "cpu", torch.float32)


CONFORMANCE.append((Qwen3Model, Model, _tiny_model))
CONFORMANCE.append(
    (
        GPUModelRunner,
        ModelRunner,
        lambda: GPUModelRunner(
            _tiny_model(),
            PagedKVCache(1, 1, 4, 16, 0, "cpu", num_blocks=4),
            SDPAAttention(2, 1, 4),
            "cpu",
        ),
    )
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
