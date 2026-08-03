from __future__ import annotations

from pathlib import Path
import unittest

import torch

from oracle.generate import _assert_close, build_vectors, greedy_generate
from oracle.runnel_oracle import TinyMoEOracle, TinyTokenizer, load_fixture_spec, stable_top_k


ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = ROOT / "fixtures" / "tiny" / "spec.json"


class OracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)
        cls.spec = load_fixture_spec(SPEC_PATH)
        cls.tokenizer = TinyTokenizer(cls.spec)
        cls.model = TinyMoEOracle(cls.spec)

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
        with self.assertRaisesRegex(AssertionError, "non-finite"):
            _assert_close(float("nan"), 0.0, "test", 1e-5, 1e-4)

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
