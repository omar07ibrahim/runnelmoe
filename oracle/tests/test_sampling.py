from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

from oracle import sampling


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
VECTOR_PATH = REPOSITORY_ROOT / "fixtures" / "scheduler" / "sampling-v1.json"
EXPECTED_FILE_SHA256 = "5db404a505c5560654c04a77435485e176fe5c2bf868d3aaf4ad1b543ebdda82"
EXPECTED_VECTOR_ID = "sha256:76a258c07a32e1a2dc61ad5d4d59827cfca469372e25b052bcf1f0bde38d6ccb"


class SamplingOracleTests(unittest.TestCase):
    def test_splitmix64_has_frozen_transitions_and_units(self) -> None:
        expected = {
            0x0000_0000_0000_0000: (
                0x9E37_79B9_7F4A_7C15,
                0xE220_A839_7B1D_CDAF,
                "0x1.c4415072f63b9p-1",
            ),
            0x0000_0000_0000_0001: (
                0x9E37_79B9_7F4A_7C16,
                0x910A_2DEC_8902_5CC1,
                "0x1.22145bd91204bp-1",
            ),
            0x0123_4567_89AB_CDEF: (
                0x9F5A_BF21_08F6_4A04,
                0x157A_3807_A48F_AA9D,
                "0x1.57a3807a48fa8p-4",
            ),
            0xFFFF_FFFF_FFFF_FFFF: (
                0x9E37_79B9_7F4A_7C14,
                0xE4D9_7177_1B65_2C20,
                "0x1.c9b2e2ee36ca5p-1",
            ),
        }

        for state, (next_state, word, unit_hex) in expected.items():
            with self.subTest(state=state):
                preview = sampling.splitmix64_preview(state)
                self.assertEqual(preview.next_state, next_state)
                self.assertEqual(preview.word, word)
                self.assertEqual(preview.unit.hex(), unit_hex)

    def test_binary32_inputs_widen_exactly(self) -> None:
        self.assertEqual(sampling.f32_from_bits(0x0000_0001).hex(), "0x1.0000000000000p-149")
        self.assertEqual(sampling.f32_from_bits(0x8000_0001).hex(), "-0x1.0000000000000p-149")
        self.assertEqual(sampling.f32_from_bits(0x3F00_0001).hex(), "0x1.0000020000000p-1")
        self.assertTrue(math.isnan(sampling.f32_from_bits(0x7FC0_0000)))

    def test_categorical_strict_greater_boundaries(self) -> None:
        candidates = ((7, 1.0), (3, 1.0))
        boundaries = (
            (0.0, 7),
            (math.nextafter(0.0, 1.0), 7),
            (0.5, 3),
            (math.nextafter(1.0, 0.0), 3),
        )
        for unit, expected_token in boundaries:
            with self.subTest(unit=unit.hex()):
                result = sampling.categorical_select(candidates, unit)
                self.assertEqual(result.token_id, expected_token)
                self.assertEqual(
                    tuple(value.hex() for value in result.cumulative),
                    ("0x1.0000000000000p-1", "0x1.0000000000000p+0"),
                )
                self.assertFalse(result.used_fallback)

        fallback = sampling.categorical_select(
            ((7, 8.0), (3, 9.0), (11, 9.0), (5, 9.0)),
            math.nextafter(1.0, 0.0),
        )
        self.assertEqual(fallback.token_id, 5)
        self.assertEqual(fallback.cumulative[-1].hex(), "0x1.fffffffffffffp-1")
        self.assertTrue(fallback.used_fallback)

    def test_sampling_cases_freeze_order_retention_and_tokens(self) -> None:
        document = sampling.build_vectors()
        expected = {
            "stable-logit-ties": (0, [1, 2, 0, 3], [1, 2, 0, 3]),
            "top-k-one": (1, [1], [1]),
            "top-p-half-equality": (0, [0, 1], [0]),
            "top-p-half-neighbor-below": (0, [0, 1], [0]),
            "top-p-half-neighbor-above": (1, [0, 1], [0, 1]),
            "smallest-positive-temperature": (1, [0, 1, 2], [0, 1, 2]),
            "underflowed-nonmaximum-weight": (0, [0, 1, 2], [0, 1]),
        }
        self.assertEqual(len(document["sample_cases"]), len(expected))
        for case in document["sample_cases"]:
            with self.subTest(case=case["id"]):
                chosen, top_k_tokens, retained = expected[case["id"]]
                self.assertEqual(case["output"]["chosen_token_id"], chosen)
                self.assertEqual(case["output"]["top_k_token_ids"], top_k_tokens)
                self.assertEqual(case["output"]["retained_token_ids"], retained)

        underflow = document["sample_cases"][-1]["output"]
        self.assertEqual(underflow["softmax_weights_f64_hex"][-1], "0x0.0p+0")

    def test_invalid_vectors_cover_nonfinite_logits_and_configuration(self) -> None:
        document = sampling.build_vectors()
        case_ids = {case["id"] for case in document["invalid_cases"]}
        self.assertEqual(
            case_ids,
            {
                "empty-logits",
                "logit-qnan",
                "logit-positive-infinity",
                "logit-negative-infinity",
                "temperature-zero",
                "temperature-negative",
                "temperature-qnan",
                "temperature-infinity",
                "top-k-zero",
                "top-k-above-vocab",
                "top-p-zero",
                "top-p-above-one",
                "top-p-qnan",
                "top-p-infinity",
            },
        )
        self.assertTrue(
            all(case["expected_error"] == "invalid_request" for case in document["invalid_cases"])
        )

    def test_categorical_rejects_nonfinite_and_degenerate_inputs(self) -> None:
        invalid_calls = (
            lambda: sampling.categorical_select((), 0.0),
            lambda: sampling.categorical_select(((0, 0.0),), 0.0),
            lambda: sampling.categorical_select(((0, math.nan),), 0.0),
            lambda: sampling.categorical_select(((0, 1.0),), math.nan),
            lambda: sampling.categorical_select(((0, 1.0),), 1.0),
            lambda: sampling.categorical_select(((0, 1.0), (0, 2.0)), 0.0),
        )
        for call in invalid_calls:
            with self.subTest(call=call):
                with self.assertRaises(sampling.SamplingError) as raised:
                    call()
                self.assertEqual(raised.exception.code, "invalid_request")

    def test_committed_vectors_equal_regeneration_and_frozen_hashes(self) -> None:
        raw = VECTOR_PATH.read_bytes()
        document = sampling.parse_document_bytes(raw)
        self.assertEqual(raw, sampling.generated_bytes())
        self.assertEqual(hashlib.sha256(raw).hexdigest(), EXPECTED_FILE_SHA256)
        self.assertEqual(document["vector_id"], EXPECTED_VECTOR_ID)
        self.assertEqual(sampling.check_path(VECTOR_PATH), EXPECTED_FILE_SHA256)

    def test_stale_identity_and_noncanonical_perturbations_fail(self) -> None:
        document = sampling.build_vectors()
        document["sample_cases"][0]["output"]["chosen_token_id"] = 99
        with self.assertRaises(sampling.SamplingError):
            sampling.parse_document_bytes(sampling.canonical_bytes(document))

        raw = sampling.generated_bytes()
        with self.assertRaises(sampling.SamplingError):
            sampling.parse_document_bytes(raw.replace(b"\n", b" \n", 1))

    def test_self_consistent_alternate_vectors_still_fail_frozen_check(self) -> None:
        alternate = copy.deepcopy(sampling.build_vectors())
        case = alternate["rng_cases"][0]
        replacement = sampling.splitmix64_preview(2)
        case.update(
            {
                "state_before_hex": "0x0000000000000002",
                "state_after_hex": f"0x{replacement.next_state:016x}",
                "unit_f64_hex": replacement.unit.hex(),
                "word_hex": f"0x{replacement.word:016x}",
            }
        )
        alternate["vector_id"] = sampling.vector_identity(alternate)
        payload = sampling.canonical_bytes(alternate)
        sampling.parse_document_bytes(payload)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sampling-v1.json"
            path.write_bytes(payload)
            with self.assertRaises(sampling.SamplingError):
                sampling.check_path(path)

    def test_cli_write_and_check_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sampling-v1.json"
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                self.assertEqual(sampling.main(["--write", "--path", str(path)]), 0)
                self.assertEqual(sampling.main(["--check", "--path", str(path)]), 0)
            self.assertEqual(stderr.getvalue(), "")
            self.assertIn(f"sha256:{EXPECTED_FILE_SHA256}", stdout.getvalue())
            self.assertEqual(path.read_bytes(), sampling.generated_bytes())


if __name__ == "__main__":
    unittest.main()
