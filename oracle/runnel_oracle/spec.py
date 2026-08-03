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
    storage_dtype: str


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

    @property
    def fixture_version(self) -> int:
        return self.raw["fixture_version"]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"invalid fixture spec: {message}")


def load_fixture_spec(path: str | Path) -> FixtureSpec:
    """Load and validate the deliberately closed tiny-model contract."""

    spec_path = Path(path)
    raw = json.loads(spec_path.read_text(encoding="utf-8"))
    fixture_version = raw.get("fixture_version")
    _require(
        isinstance(fixture_version, int)
        and not isinstance(fixture_version, bool)
        and fixture_version in {1, 2, 3},
        "fixture_version must be 1, 2, or 3",
    )

    expected_identity = f"runnel-tiny-causal-moe-v{fixture_version}"
    _require(raw.get("identity") == expected_identity, "fixture identity is not canonical")

    artifact = raw.get("artifact")
    _require(isinstance(artifact, dict), "artifact must be an object")
    expected_artifacts = {
        1: {
            "artifact_id": "sha256:e49321cefc980ab59cd449341edfc624ccfc4b0b703cd175184e056d296d9ed3",
            "object_digest": "sha256:6b2b8a1bbb2854084b1e1fe1e5787a9cfdb021b397e774fc7dbef79ac9d24bf6",
            "object_length": 7904,
            "page_table_digest": "sha256:29383b56a150f9e5705f3666ca7707f21bc3fbd663ffd7249f4ecb938da6a62d",
            "page_table_length": 96,
        },
        2: {
            "artifact_id": "sha256:606baa0c1082b369632b5dd000d30dc51ae20321b2c032ef3395aaa0bfd7c76c",
            "object_digest": "sha256:275f985b05a85d4f85d78fc290c10c9d39e9169c46449f4d3a3ed6513a8965ab",
            "object_length": 5600,
            "page_table_digest": "sha256:7d660764b861f97afbc800efb361bdd59ac38fdea5fc424c62f9fb6a30b2896c",
            "page_table_length": 96,
        },
        3: {
            "artifact_id": "sha256:382856e13f688b5176ad1e5f06c26bcd85bcaeb719a10a60e9adc9ff387c945c",
            "object_digest": "sha256:275f985b05a85d4f85d78fc290c10c9d39e9169c46449f4d3a3ed6513a8965ab",
            "object_length": 5600,
            "page_table_digest": "sha256:7d660764b861f97afbc800efb361bdd59ac38fdea5fc424c62f9fb6a30b2896c",
            "page_table_length": 96,
        },
    }
    _require(
        artifact == expected_artifacts[fixture_version],
        f"artifact identity differs from tiny-v{fixture_version}",
    )

    model = raw.get("model")
    _require(isinstance(model, dict), "model must be an object")
    expected_model = {
        "context_length": 1024 if fixture_version == 3 else 16,
        "expert_hidden_size": 12,
        "hidden_size": 8,
        "num_experts": 4,
        "num_heads": 2,
        "num_layers": 1,
        "top_k": 2,
        "vocab_size": 32,
    }
    _require(
        model == expected_model,
        f"model dimensions differ from tiny-v{fixture_version}",
    )

    numeric = raw.get("numeric")
    _require(isinstance(numeric, dict), "numeric must be an object")
    _require(numeric.get("dtype") == "float32", "only float32 is supported")
    if fixture_version == 1:
        _require(
            "expert_storage_dtype" not in numeric
            and "expert_storage_conversion" not in numeric,
            "tiny-v1 must not declare compact expert storage",
        )
    else:
        _require(
            numeric.get("expert_storage_dtype") == "bfloat16",
            f"tiny-v{fixture_version} expert storage dtype must be bfloat16",
        )
        _require(
            numeric.get("expert_storage_conversion")
            == "float32-to-bfloat16-rne-to-float32",
            f"tiny-v{fixture_version} expert conversion must be the frozen BF16 round trip",
        )
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
        expected_keys = {"id", "role", "shape"}
        if fixture_version in {2, 3}:
            expected_keys.add("dtype")
        _require(
            set(item) == expected_keys,
            f"tensor {expected_id} keys differ from tiny-v{fixture_version}",
        )
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
        storage_dtype = item.get("dtype", "f32-le")
        expected_dtype = (
            "bf16-le"
            if fixture_version in {2, 3} and 8 <= expected_id <= 19
            else "f32-le"
        )
        _require(
            storage_dtype == expected_dtype,
            f"tensor {expected_id} storage dtype differs from tiny-v{fixture_version}",
        )
        tensors.append(TensorSpec(expected_id, role, tuple(shape), storage_dtype))
    _require(
        len(tensors) == 22,
        f"tiny-v{fixture_version} requires exactly 22 tensors",
    )
    _require(len({tensor.role for tensor in tensors}) == 22, "tensor roles must be unique")
    actual_tensors = [(tensor.role, tensor.shape) for tensor in tensors]
    _require(
        actual_tensors == expected_tensors,
        f"tensor role or shape table differs from tiny-v{fixture_version}",
    )

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
    _require(
        tokenizer.get("tokens") == expected_tokens,
        f"token table differs from tiny-v{fixture_version}",
    )
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
