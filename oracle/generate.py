"""Generate or verify the committed deterministic oracle vectors."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from .runnel_oracle import TinyMoEOracle, TinyTokenizer, formula_tensor, load_fixture_spec


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = REPOSITORY_ROOT / "fixtures" / "tiny" / "spec.json"


def _rounded(values: torch.Tensor) -> list[Any]:
    data = values.detach().cpu().tolist()

    def visit(value: Any) -> Any:
        if isinstance(value, list):
            return [visit(item) for item in value]
        if isinstance(value, float):
            return round(value, 9)
        return value

    return visit(data)


def greedy_generate(
    model: TinyMoEOracle, input_ids: list[int], max_new_tokens: int
) -> tuple[list[int], list[int], list[dict[str, Any]], str]:
    """Run full-prefix greedy decisions with the runtime's admission contract."""

    if not input_ids:
        raise ValueError("input IDs must not be empty")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must not be negative")
    required_positions = len(input_ids) + max(max_new_tokens - 1, 0)
    if required_positions > model.context_length:
        raise ValueError("generation would exceed the context length")

    full_ids = list(input_ids)
    generated_ids: list[int] = []
    logit_steps: list[dict[str, Any]] = []
    stop_reason = "max_new_tokens"
    for step in range(max_new_tokens):
        output = model(full_ids)
        final_logits = output.logits[-1]
        if not bool(torch.all(torch.isfinite(final_logits))):
            raise ValueError("oracle produced non-finite logits")
        # torch.argmax returns the first maximum, which is the lowest token ID.
        next_token_id = int(torch.argmax(final_logits).item())
        logit_steps.append(
            {
                "input_ids": list(full_ids),
                "logits": _rounded(final_logits),
                "next_token_id": next_token_id,
                "step": step,
            }
        )
        generated_ids.append(next_token_id)
        full_ids.append(next_token_id)
        if next_token_id == 0:
            stop_reason = "eos"
            break
    return full_ids, generated_ids, logit_steps, stop_reason


def build_vectors(spec_path: Path) -> dict[str, dict[str, Any]]:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    spec = load_fixture_spec(spec_path)
    model = TinyMoEOracle(spec)
    tokenizer = TinyTokenizer(spec)
    generation = spec.raw["generation_fixture"]
    prompt = generation["prompt"]
    input_ids = tokenizer.encode(prompt)
    full_ids, generated_ids, logit_steps, stop_reason = greedy_generate(
        model, input_ids, generation["max_new_tokens"]
    )

    final_output = model(full_ids)
    positions = []
    for position, token_id in enumerate(full_ids):
        positions.append(
            {
                "position": position,
                "router_scores": _rounded(final_output.routes.router_scores[position]),
                "selected_experts": _rounded(final_output.routes.selected_experts[position]),
                "selected_weights": _rounded(final_output.routes.selected_weights[position]),
                "source": "input" if position < len(input_ids) else "generated",
                "token_id": token_id,
            }
        )

    identity = spec.raw["identity"]
    comparison = spec.raw["comparison"]
    routes = {
        "fixture": identity,
        "generated_ids": generated_ids,
        "input_ids": input_ids,
        "positions": positions,
        "prompt": prompt,
    }
    logits = {
        "comparison": comparison,
        "fixture": identity,
        "positions": [
            {
                "logits": _rounded(final_output.logits[position]),
                "position": position,
                "token_id": token_id,
            }
            for position, token_id in enumerate(full_ids)
        ],
        "prompt": prompt,
        "steps": logit_steps,
    }
    tokens = {
        "fixture": identity,
        "full_ids": full_ids,
        "generated_ids": generated_ids,
        "generated_text": tokenizer.decode(generated_ids),
        "input_ids": input_ids,
        "max_new_tokens": generation["max_new_tokens"],
        "prompt": prompt,
        "stop_reason": stop_reason,
    }
    metadata = {
        "artifact": spec.raw["artifact"],
        "comparison": comparison,
        "fixture": identity,
        "oracle": {
            "framework": "PyTorch",
            "pytorch_version": torch.__version__,
            "torch_dtype": "float32",
            "torch_threads": torch.get_num_threads(),
        },
        "tokenizer_vector": {
            "encoded": tokenizer.encode("abcdefghijklmnopqrstuvwxyz .,?"),
            "text": "abcdefghijklmnopqrstuvwxyz .,?",
        },
    }
    if spec.fixture_version == 2:
        bf16_tensors = [tensor for tensor in spec.tensors if tensor.storage_dtype == "bf16-le"]
        f32_tensors = [tensor for tensor in spec.tensors if tensor.storage_dtype == "f32-le"]
        changed_elements = 0
        for tensor in bf16_tensors:
            source = formula_tensor(spec, tensor)
            changed_elements += int(
                torch.count_nonzero(source != model.weights[tensor.role]).item()
            )
        bf16_elements = sum(math.prod(tensor.shape) for tensor in bf16_tensors)
        f32_elements = sum(math.prod(tensor.shape) for tensor in f32_tensors)
        metadata["adapter"] = {
            "id": "runnel.tiny-causal-moe",
            "version": 2,
        }
        metadata["oracle"]["expert_storage_dtype"] = "bfloat16"
        metadata["oracle"]["expert_storage_round_trip"] = (
            "float32-to-bfloat16-rne-to-float32"
        )
        metadata["storage"] = {
            "bf16_expert_bytes": bf16_elements * 2,
            "bf16_expert_elements": bf16_elements,
            "bf16_expert_tensor_count": len(bf16_tensors),
            "f32_nonexpert_bytes": f32_elements * 4,
            "f32_nonexpert_elements": f32_elements,
            "f32_nonexpert_tensor_count": len(f32_tensors),
            "round_trip_changed_elements": changed_elements,
            "rounding": "round-to-nearest-ties-to-even",
            "widening": "exact-bf16-to-float32",
        }
    return {
        "golden_logits.json": logits,
        "golden_metadata.json": metadata,
        "golden_routes.json": routes,
        "golden_tokens.json": tokens,
    }


def _assert_close(actual: Any, expected: Any, path: str, atol: float, rtol: float) -> None:
    if isinstance(expected, float):
        if not isinstance(actual, (int, float)):
            raise AssertionError(f"{path}: expected a number, got {type(actual).__name__}")
        if not math.isfinite(float(actual)) or not math.isfinite(expected):
            raise AssertionError(f"{path}: non-finite value is not comparable")
        difference = abs(float(actual) - expected)
        if difference > atol + rtol * abs(expected):
            raise AssertionError(
                f"{path}: {actual!r} != {expected!r} (difference {difference})"
            )
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise AssertionError(f"{path}: list shape differs")
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected, strict=True)):
            _assert_close(actual_item, expected_item, f"{path}[{index}]", atol, rtol)
        return
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise AssertionError(f"{path}: object keys differ")
        for key in expected:
            _assert_close(actual[key], expected[key], f"{path}.{key}", atol, rtol)
        return
    if actual != expected:
        raise AssertionError(f"{path}: {actual!r} != {expected!r}")


def write_vectors(spec_path: Path) -> None:
    for filename, payload in build_vectors(spec_path).items():
        destination = spec_path.parent / filename
        destination.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {destination.relative_to(REPOSITORY_ROOT)}")


def check_vectors(spec_path: Path) -> None:
    generated = build_vectors(spec_path)
    spec = load_fixture_spec(spec_path)
    atol = float(spec.raw["comparison"]["atol"])
    rtol = float(spec.raw["comparison"]["rtol"])
    for filename, actual in generated.items():
        source = spec_path.parent / filename
        expected = json.loads(source.read_text(encoding="utf-8"))
        _assert_close(actual, expected, filename, atol, rtol)
        print(f"verified {source.relative_to(REPOSITORY_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true", help="rewrite the golden JSON files")
    action.add_argument("--check", action="store_true", help="verify committed golden JSON files")
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC, help="fixture specification")
    arguments = parser.parse_args()
    if arguments.write:
        write_vectors(arguments.spec.resolve())
    else:
        check_vectors(arguments.spec.resolve())


if __name__ == "__main__":
    main()
