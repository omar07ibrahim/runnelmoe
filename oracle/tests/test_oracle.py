from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from oracle.generate import (
    _assert_close,
    build_vectors,
    check_vectors,
    greedy_generate,
)
from oracle.runnel_oracle import (
    TinyMoEOracle,
    TinyTokenizer,
    bf16_storage_round_trip,
    formula_tensor,
    load_fixture_spec,
    stable_top_k,
)


ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = ROOT / "fixtures" / "tiny" / "spec.json"
V2_SPEC_PATH = ROOT / "fixtures" / "tiny-v2" / "spec.json"
V3_SPEC_PATH = ROOT / "fixtures" / "tiny-v3" / "spec.json"


class OracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)
        cls.spec = load_fixture_spec(SPEC_PATH)
        cls.tokenizer = TinyTokenizer(cls.spec)
        cls.model = TinyMoEOracle(cls.spec)
        cls.v2_spec = load_fixture_spec(V2_SPEC_PATH)
        cls.v2_model = TinyMoEOracle(cls.v2_spec)
        cls.v3_spec = load_fixture_spec(V3_SPEC_PATH)
        cls.v3_model = TinyMoEOracle(cls.v3_spec)

    def test_tokenizer_contract(self) -> None:
        text = "abcdefghijklmnopqrstuvwxyz .,?"
        self.assertEqual(self.tokenizer.encode(text), list(range(1, 32)))
        self.assertEqual(self.tokenizer.decode([*range(1, 32), 0]), text)
        self.assertEqual(self.tokenizer.encode(""), [1])
        self.assertEqual(self.tokenizer.decode([2, 0, 3]), "a")
        with self.assertRaisesRegex(ValueError, "position 0"):
            self.tokenizer.encode("A")

    def test_exact_tensor_recipe(self) -> None:
        embedding = self.model.weights["token_embedding"].reshape(-1)
        expected_numerators = [((37 + 17 * (index + 1)) % 29) - 14 for index in range(8)]
        self.assertEqual(embedding[:8].tolist(), [value / 32 for value in expected_numerators])
        norm = self.model.weights["layers.0.attn_norm"].reshape(-1)
        expected_norm = [
            1 + (((74 + 17 * (index + 1)) % 29) - 14) / 64 for index in range(8)
        ]
        self.assertEqual(norm.tolist(), expected_norm)

    def test_bf16_storage_round_trip_uses_ties_to_even(self) -> None:
        half_ulp = 2.0**-8
        source = torch.tensor(
            [
                1.0 + half_ulp,
                1.0 + 3.0 * half_ulp,
                -1.0 - half_ulp,
                -1.0 - 3.0 * half_ulp,
            ],
            dtype=torch.float32,
        )
        expected = torch.tensor(
            [1.0, 1.0 + 2.0**-6, -1.0, -1.0 - 2.0**-6],
            dtype=torch.float32,
        )
        self.assertTrue(torch.equal(bf16_storage_round_trip(source), expected))
        with self.assertRaisesRegex(ValueError, "float32"):
            bf16_storage_round_trip(source.to(torch.float64))
        with self.assertRaisesRegex(ValueError, "finite"):
            bf16_storage_round_trip(torch.tensor([float("inf")], dtype=torch.float32))

    def test_v2_round_trips_only_expert_matrices_through_bf16(self) -> None:
        bf16_tensors = [
            tensor for tensor in self.v2_spec.tensors if tensor.storage_dtype == "bf16-le"
        ]
        f32_tensors = [
            tensor for tensor in self.v2_spec.tensors if tensor.storage_dtype == "f32-le"
        ]
        self.assertEqual([tensor.tensor_id for tensor in bf16_tensors], list(range(8, 20)))
        self.assertEqual(
            sum(math.prod(tensor.shape) for tensor in bf16_tensors),
            1_152,
        )
        self.assertEqual(len(f32_tensors), 10)
        self.assertTrue(
            all(weight.dtype == torch.float32 for weight in self.v2_model.weights.values())
        )

        changed_elements = 0
        for tensor in bf16_tensors:
            source = formula_tensor(self.v2_spec, tensor)
            expected = bf16_storage_round_trip(source)
            actual = self.v2_model.weights[tensor.role]
            self.assertEqual(actual.dtype, torch.float32)
            self.assertTrue(torch.equal(actual, expected))
            changed_elements += int(torch.count_nonzero(source != actual).item())
        self.assertEqual(changed_elements, 0)

        for tensor in f32_tensors:
            self.assertTrue(
                torch.equal(
                    self.v2_model.weights[tensor.role],
                    formula_tensor(self.v2_spec, tensor),
                )
            )

        metadata = build_vectors(V2_SPEC_PATH)["golden_metadata.json"]
        self.assertEqual(metadata["adapter"]["version"], 2)
        self.assertEqual(metadata["artifact"]["object_length"], 5_600)
        self.assertEqual(metadata["storage"]["bf16_expert_bytes"], 2_304)
        self.assertEqual(metadata["storage"]["f32_nonexpert_bytes"], 3_296)
        self.assertEqual(metadata["storage"]["round_trip_changed_elements"], 0)

    def test_compact_versions_execute_twelve_conversions_and_v1_executes_none(
        self,
    ) -> None:
        target = "oracle.runnel_oracle.model.bf16_storage_round_trip"
        with mock.patch(target, wraps=bf16_storage_round_trip) as conversion:
            TinyMoEOracle(self.spec)
            self.assertEqual(conversion.call_count, 0)
        with mock.patch(target, wraps=bf16_storage_round_trip) as conversion:
            TinyMoEOracle(self.v2_spec)
            self.assertEqual(conversion.call_count, 12)
        with mock.patch(target, wraps=bf16_storage_round_trip) as conversion:
            TinyMoEOracle(self.v3_spec)
            self.assertEqual(conversion.call_count, 12)

    def test_v3_retains_v2_numerics_with_a_larger_context_contract(self) -> None:
        self.assertEqual(self.v2_model.context_length, 16)
        self.assertEqual(self.v3_model.context_length, 1_024)
        self.assertEqual(
            [tensor.storage_dtype for tensor in self.v3_spec.tensors],
            [tensor.storage_dtype for tensor in self.v2_spec.tensors],
        )
        for role, version_two_weight in self.v2_model.weights.items():
            self.assertTrue(torch.equal(self.v3_model.weights[role], version_two_weight))

        vectors = build_vectors(V3_SPEC_PATH)
        metadata = vectors["golden_metadata.json"]
        self.assertEqual(metadata["adapter"]["version"], 3)
        self.assertEqual(metadata["artifact"]["object_length"], 5_600)
        self.assertEqual(
            vectors["golden_tokens.json"]["fixture"],
            "runnel-tiny-causal-moe-v3",
        )
        check_vectors(V3_SPEC_PATH)

    def test_stable_top_k_breaks_ties_by_expert_id(self) -> None:
        scores = torch.tensor([[2.0, 3.0, 3.0, 1.0], [-0.0, 0.0, 0.0, -0.0]])
        self.assertEqual(stable_top_k(scores, 2).tolist(), [[1, 2], [0, 1]])
        with self.assertRaisesRegex(ValueError, "finite"):
            stable_top_k(torch.tensor([[0.0, float("nan")]]), 1)

    def test_forward_shape_and_route_weight(self) -> None:
        output = self.model(self.tokenizer.encode("moe"))
        self.assertEqual(tuple(output.logits.shape), (4, 32))
        self.assertEqual(tuple(output.routes.selected_experts.shape), (4, 2))
        torch.testing.assert_close(
            output.routes.selected_weights.sum(dim=-1), torch.ones(4), atol=1e-6, rtol=0
        )

    def test_generation_is_repeatable(self) -> None:
        self.assertEqual(build_vectors(SPEC_PATH), build_vectors(SPEC_PATH))
        self.assertEqual(build_vectors(V2_SPEC_PATH), build_vectors(V2_SPEC_PATH))
        self.assertEqual(build_vectors(V3_SPEC_PATH), build_vectors(V3_SPEC_PATH))
        with self.assertRaisesRegex(AssertionError, "non-finite"):
            _assert_close(float("nan"), 0.0, "test", 1e-5, 1e-4)

    def test_fixture_loader_accepts_only_frozen_versions_and_dtypes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            invalid_version = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
            invalid_version["fixture_version"] = 4
            path = Path(temporary) / "version.json"
            path.write_text(json.dumps(invalid_version), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be 1, 2, or 3"):
                load_fixture_spec(path)

            invalid_dtype = json.loads(V2_SPEC_PATH.read_text(encoding="utf-8"))
            invalid_dtype["tensors"][8]["dtype"] = "f32-le"
            path = Path(temporary) / "dtype.json"
            path.write_text(json.dumps(invalid_dtype), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "storage dtype"):
                load_fixture_spec(path)

            invalid_context = json.loads(V3_SPEC_PATH.read_text(encoding="utf-8"))
            invalid_context["model"]["context_length"] = 16
            path = Path(temporary) / "context.json"
            path.write_text(json.dumps(invalid_context), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "model dimensions"):
                load_fixture_spec(path)

    def test_context_admission_matches_runtime(self) -> None:
        context = self.model.context_length
        self.assertEqual(len(greedy_generate(self.model, [1] * context, 1)[1]), 1)
        with self.assertRaisesRegex(ValueError, "context"):
            greedy_generate(self.model, [1] * context, 2)
        self.assertEqual(len(greedy_generate(self.model, [1] * (context - 1), 2)[1]), 2)
        with self.assertRaisesRegex(ValueError, "context"):
            greedy_generate(self.model, [1] * (context - 1), 3)
        self.assertEqual(greedy_generate(self.model, [1] * context, 0)[1], [])

    def test_causal_prefix_is_independent_of_suffix(self) -> None:
        left = self.model([1, 2, 3])
        right = self.model([1, 2, 4])
        torch.testing.assert_close(left.logits[:2], right.logits[:2])
        self.assertTrue(torch.equal(left.routes.selected_experts[:2], right.routes.selected_experts[:2]))


if __name__ == "__main__":
    unittest.main()
