"""Strict loader for the oracle's small, reviewable fixture specification."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TensorSpec:
    tensor_id: int
    role: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class FixtureSpec:
    raw: dict[str, Any]
    tensors: tuple[TensorSpec, ...]

    @property
    def model(self) -> dict[str, int]:
        return self.raw["model"]

    @property
    def tokenizer(self) -> dict[str, Any]:
        return self.raw["tokenizer"]

    @property
    def numeric(self) -> dict[str, Any]:
        return self.raw["numeric"]

    @property
    def recipe(self) -> dict[str, Any]:
        return self.raw["tensor_recipe"]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"invalid fixture spec: {message}")


def load_fixture_spec(path: str | Path) -> FixtureSpec:
    """Load and validate the deliberately closed tiny-model contract."""

    spec_path = Path(path)
    raw = json.loads(spec_path.read_text(encoding="utf-8"))
    _require(raw.get("fixture_version") == 1, "fixture_version must be 1")

    artifact = raw.get("artifact")
    _require(isinstance(artifact, dict), "artifact must be an object")
    _require(
        artifact
        == {
            "artifact_id": "sha256:e49321cefc980ab59cd449341edfc624ccfc4b0b703cd175184e056d296d9ed3",
            "object_digest": "sha256:6b2b8a1bbb2854084b1e1fe1e5787a9cfdb021b397e774fc7dbef79ac9d24bf6",
            "object_length": 7904,
            "page_table_digest": "sha256:29383b56a150f9e5705f3666ca7707f21bc3fbd663ffd7249f4ecb938da6a62d",
            "page_table_length": 96,
        },
        "artifact identity differs from tiny-v1",
    )

    model = raw.get("model")
    _require(isinstance(model, dict), "model must be an object")
    expected_model = {
        "context_length": 16,
        "expert_hidden_size": 12,
        "hidden_size": 8,
        "num_experts": 4,
        "num_heads": 2,
        "num_layers": 1,
        "top_k": 2,
        "vocab_size": 32,
    }
    _require(model == expected_model, "model dimensions differ from tiny-v1")

    numeric = raw.get("numeric")
    _require(isinstance(numeric, dict), "numeric must be an object")
    _require(numeric.get("dtype") == "float32", "only float32 is supported")
    _require(
        numeric.get("rms_norm_epsilon_power_of_two") == -12,
        "RMS epsilon must be 2^-12",
    )
    _require(
        numeric.get("attention_scale_power_of_two") == -1,
        "attention scale must be 2^-1",
    )

    recipe = raw.get("tensor_recipe")
    _require(isinstance(recipe, dict), "tensor_recipe must be an object")
    _require(recipe.get("multiplier_tensor") == 37, "tensor multiplier must be 37")
    _require(recipe.get("multiplier_index") == 17, "index multiplier must be 17")
    _require(recipe.get("modulus") == 29, "recipe modulus must be 29")
    _require(recipe.get("center") == 14, "recipe center must be 14")
    _require(
        recipe.get("ordinary_denominator_power_of_two") == 5,
        "ordinary tensor denominator must be 2^5",
    )
    _require(
        recipe.get("normalization_denominator_power_of_two") == 6,
        "normalization tensor denominator must be 2^6",
    )
    _require(recipe.get("normalization_base") == 1, "normalization base must be 1")
    _require(
        recipe.get("normalization_tensor_ids") == [1, 6, 20],
        "normalization tensor IDs must be [1, 6, 20]",
    )

    expected_tensors: list[tuple[str, tuple[int, ...]]] = [
        ("token_embedding", (32, 8)),
        ("layers.0.attn_norm", (8,)),
        ("layers.0.attn_q", (8, 8)),
        ("layers.0.attn_k", (8, 8)),
        ("layers.0.attn_v", (8, 8)),
        ("layers.0.attn_out", (8, 8)),
        ("layers.0.ffn_norm", (8,)),
        ("layers.0.router", (4, 8)),
    ]
    for expert_id in range(4):
        prefix = f"layers.0.experts.{expert_id}"
        expected_tensors.extend(
            [
                (f"{prefix}.gate", (12, 8)),
                (f"{prefix}.up", (12, 8)),
                (f"{prefix}.down", (8, 12)),
            ]
        )
    expected_tensors.extend([("final_norm", (8,)), ("lm_head", (32, 8))])

    tensors_raw = raw.get("tensors")
    _require(isinstance(tensors_raw, list), "tensors must be an array")
    tensors: list[TensorSpec] = []
    for expected_id, item in enumerate(tensors_raw):
        _require(isinstance(item, dict), f"tensor {expected_id} must be an object")
        _require(item.get("id") == expected_id, "tensor IDs must be contiguous")
        role = item.get("role")
        shape = item.get("shape")
        _require(isinstance(role, str) and role, f"tensor {expected_id} has no role")
        _require(
            isinstance(shape, list)
            and shape
            and all(isinstance(dim, int) and dim > 0 for dim in shape),
            f"tensor {expected_id} has an invalid shape",
        )
        tensors.append(TensorSpec(expected_id, role, tuple(shape)))
    _require(len(tensors) == 22, "tiny-v1 requires exactly 22 tensors")
    _require(len({tensor.role for tensor in tensors}) == 22, "tensor roles must be unique")
    actual_tensors = [(tensor.role, tensor.shape) for tensor in tensors]
    _require(actual_tensors == expected_tensors, "tensor role or shape table differs from tiny-v1")

    tokenizer = raw.get("tokenizer")
    expected_tokens = [
        "<eos>",
        "<bos>",
        *list("abcdefghijklmnopqrstuvwxyz"),
        " ",
        ".",
        ",",
        "?",
    ]
    _require(isinstance(tokenizer, dict), "tokenizer must be an object")
    _require(tokenizer.get("tokens") == expected_tokens, "token table differs from tiny-v1")
    _require(tokenizer.get("encode_prepends_bos") is True, "encoder must prepend BOS")
    _require(tokenizer.get("eos_token_id") == 0, "EOS ID must be 0")
    _require(tokenizer.get("bos_token_id") == 1, "BOS ID must be 1")

    generation = raw.get("generation_fixture")
    _require(
        generation == {
            "max_new_tokens": 4,
            "prompt": "moe",
            "strategy": "greedy",
        },
        "generation fixture must be greedy 'moe' with four new tokens",
    )
    _require(
        raw.get("comparison") == {"atol": 0.00001, "rtol": 0.0001},
        "comparison tolerance must be atol=1e-5 and rtol=1e-4",
    )
    return FixtureSpec(raw=raw, tensors=tuple(tensors))
