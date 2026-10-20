"""Model registry: HF model_type -> Model class."""

from spark_engine.models.qwen3 import Qwen3Model

MODEL_CLASSES: dict[str, type[Qwen3Model]] = {"qwen3": Qwen3Model}


def get_model_cls(model_type: str) -> type[Qwen3Model]:
    try:
        return MODEL_CLASSES[model_type]
    except KeyError:
        raise ValueError(
            f"unsupported model_type {model_type!r}; known: {sorted(MODEL_CLASSES)}"
        ) from None
