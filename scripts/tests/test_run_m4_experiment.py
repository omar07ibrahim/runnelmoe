from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "run_m4_experiment.py"
SPEC = importlib.util.spec_from_file_location("run_m4_experiment", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
evidence = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = evidence
SPEC.loader.exec_module(evidence)


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def case_rows() -> list[dict]:
    return [
        {
            "schema": evidence.CASE_SCHEMA,
            "case_id": case.case_id,
            "rows": case.rows,
            "columns": case.columns,
            "calls_per_worker": case.calls_per_worker,
            "matrix_bytes": case.matrix_bytes,
            "input_bytes": case.input_bytes,
            "output_bytes": case.output_bytes,
            "element_products_per_worker": case.element_products_per_worker,
            "role": case.role,
            "weight_sha256": evidence.EXPECTED_CASE_DIGESTS[case.case_id][0],
            "input_sha256": evidence.EXPECTED_CASE_DIGESTS[case.case_id][1],
        }
        for case in evidence.CASES
    ]


def correctness_rows(failed_check: str | None = None) -> list[dict]:
    rows = []
    for check_id, kind, case_id, backend in evidence.expected_correctness_checks():
        status = "failed" if check_id == failed_check else "ok"
        if kind == "kernel":
            metrics = {
                "components": evidence.CASE_BY_ID[case_id].rows,
                "max_abs_error": 0.0,
                "max_error_to_bound_ratio": 0.0,
                "worst_index": 0,
                "worst_abs_error": 0.0,
                "worst_error_bound": 1e-7,
                "cross_scalar_max_abs": 0.0,
                "cross_scalar_max_rel": 0.0,
                "cross_scalar_max_ulp": 0,
            }
            if status == "failed":
                metrics.update(
                    {
                        "max_abs_error": 2e-7,
                        "max_error_to_bound_ratio": 2.0,
                        "worst_abs_error": 2e-7,
                        "worst_error_bound": 1e-7,
                    }
                )
        else:
            contract = evidence.MODEL_CONTRACTS[check_id]
            selected = None
            if check_id == "tiny-v2-scalar":
                selected = "scalar"
            elif check_id == "tiny-v2-avx2":
                selected = "avx2"
            error = {
                "max_abs": 0.0,
                "max_rel": 0.0,
                "max_tolerance_ratio": 0.0,
                "worst_position": 0,
                "worst_index": 0,
            }
            metrics = {
                "adapter_version": contract["adapter_version"],
                "representation": contract["representation"],
                "requested_backend": contract["requested_backend"],
                "selected_backend": selected,
                "fixture": copy.deepcopy(contract["fixture"]),
                "goldens": copy.deepcopy(contract["goldens"]),
                "prompt": "moe",
                "input_ids": [1, 14, 16, 6],
                "full_ids": [1, 14, 16, 6, 15, 11, 20, 9],
                "positions": list(range(8)),
                "repetitions": 2,
                "tokens_exact": True,
                "expert_ids_exact": True,
                "deterministic": True,
                "tolerance": {"atol": 1e-5, "rtol": 1e-4},
                "logits_error": copy.deepcopy(error),
                "router_score_error": copy.deepcopy(error),
                "route_weight_error": copy.deepcopy(error),
            }
            if status == "failed":
                metrics["tokens_exact"] = False
        failure = None
        if status == "failed":
            failure = (
                "numerical_bound_exceeded"
                if kind == "kernel"
                else "exactness_mismatch"
            )
        rows.append(
            {
                "schema": evidence.CORRECTNESS_SCHEMA,
                "check_id": check_id,
                "kind": kind,
                "case_id": case_id,
                "backend": backend,
                "status": status,
                "failure": failure,
                "metrics": metrics,
            }
        )
    return rows


def independently_recompute_case_digests(case) -> tuple[str, str]:
    """Implement ADR 0006's fixture stream/RNE rules without harness helpers."""

    case_id = case.case_id.encode("ascii")
    encoded_length = struct.pack("<H", len(case_id))

    bf16_words = []
    f32_bytes = []
    for digest_byte in range(256):
        signed = digest_byte - 128
        bits = struct.unpack("<I", struct.pack("<f", signed / 128.0))[0]
        rounded = bits + 0x7FFF + ((bits >> 16) & 1)
        bf16_words.append((rounded >> 16) & 0xFFFF)
        f32_bytes.append(struct.pack("<f", signed / 64.0))

    low = bytes(word & 0xFF for word in bf16_words)
    high = bytes(word >> 8 for word in bf16_words)
    weight = hashlib.sha256()
    weight_prefix = b"runnel-m4-weight-v1\0" + encoded_length + case_id
    remaining = case.rows * case.columns
    counter = 0
    encoded = bytearray(64)
    while remaining:
        source = hashlib.sha256(weight_prefix + struct.pack("<Q", counter)).digest()
        encoded[0::2] = source.translate(low)
        encoded[1::2] = source.translate(high)
        taken = min(remaining, len(source))
        weight.update(encoded[: taken * 2])
        remaining -= taken
        counter += 1

    f32_tables = [bytes(value[offset] for value in f32_bytes) for offset in range(4)]
    input_digest = hashlib.sha256()
    input_prefix = b"runnel-m4-input-v1\0" + encoded_length + case_id
    remaining = case.columns
    counter = 0
    encoded_input = bytearray(128)
    while remaining:
        source = hashlib.sha256(input_prefix + struct.pack("<Q", counter)).digest()
        for offset, table in enumerate(f32_tables):
            encoded_input[offset::4] = source.translate(table)
        if counter == 0:
            encoded_input[:4] = struct.pack("<f", 1.0)
        taken = min(remaining, len(source))
        input_digest.update(encoded_input[: taken * 4])
        remaining -= taken
        counter += 1

    return "sha256:" + weight.hexdigest(), "sha256:" + input_digest.hexdigest()


def worker_rows(cell, cpu_base: int = 2) -> list[dict]:
    return [
        {
            "worker_index": index,
            "cpu": cpu_base + index,
            "cpu_before": cpu_base + index,
            "cpu_after": cpu_base + index,
            "affinity": [cpu_base + index],
            "mxcsr_before": 0x1F80,
            "mxcsr_after": 0x1F80,
        }
        for index in range(cell.workers)
    ]


def successful_observation(
    cell,
    child_sequence: int,
    pair_sequence: int,
    pair_order: str,
    variant_sequence: int,
    variant: str,
) -> dict:
    case = evidence.CASE_BY_ID[cell.case_id]
    baseline = 1_000_000 + pair_sequence * 1_000 + child_sequence * 10
    elapsed = baseline if variant == "baseline" else baseline * 9 // 10
    return {
        "schema": evidence.OBSERVATION_SCHEMA,
        "cell_id": cell.cell_id,
        "case_id": cell.case_id,
        "child_sequence": child_sequence,
        "pair_sequence": pair_sequence,
        "pair_order": pair_order,
        "variant_sequence": variant_sequence,
        "variant": variant,
        "implementation": cell.implementation(variant),
        "status": "ok",
        "failure": None,
        "elapsed_ns": elapsed,
        "expected_calls": case.calls_per_worker * cell.workers,
        "executed_calls": case.calls_per_worker * cell.workers,
        "sink_sha256": "sha256:" + digest(f"{cell.cell_id}-{variant}-sink"),
        "output_sha256": "sha256:" + digest(
            f"{cell.cell_id}-{variant}-output"
        ),
        "addresses_mod_64": {"weights": 0, "input": 0, "output": 0},
        "workers": worker_rows(cell),
        "resource_usage": {name: 0 for name in evidence.RESOURCE_KEYS},
    }


def observation_rows() -> list[dict]:
    rows = []
    for child_sequence, cell_id in enumerate(evidence.cell_order()):
        cell = evidence.CELL_BY_ID[cell_id]
        for pair_sequence, order in enumerate(evidence.measured_pair_orders(cell_id)):
            for variant_sequence, variant in enumerate(evidence._variant_order(order)):
                rows.append(
                    successful_observation(
                        cell,
                        child_sequence,
                        pair_sequence,
                        order,
                        variant_sequence,
                        variant,
                    )
                )
    return rows


def warmup_rows(cell, child_sequence: int) -> list[dict]:
    return [
        {
            "schema": evidence.WARMUP_SCHEMA,
            "cell_id": cell.cell_id,
            "case_id": cell.case_id,
            "child_sequence": child_sequence,
            "pair_sequence": pair_sequence,
            "order": order,
            "baseline_status": "ok",
            "candidate_status": "ok",
        }
        for pair_sequence, order in enumerate(evidence.WARMUP_ORDERS)
    ]


def mark_failed(row: dict, status: str = "kernel_error") -> None:
    row.update(
        {
            "status": status,
            "failure": "injected row failure",
            "elapsed_ns": None,
            "executed_calls": 0,
            "sink_sha256": None,
            "output_sha256": None,
            "addresses_mod_64": None,
            "workers": [],
            "resource_usage": None,
        }
    )


class JsonContractTests(unittest.TestCase):
    def test_duplicate_keys_nonfinite_and_blank_jsonl_are_rejected(self) -> None:
        with self.assertRaisesRegex(evidence.EvidenceError, "duplicate JSON key"):
            evidence.parse_json_bytes(b'{"a":1,"a":2}', "duplicate")
        with self.assertRaisesRegex(evidence.EvidenceError, "non-finite"):
            evidence.parse_json_bytes(b'{"a":NaN}', "nan")
        with self.assertRaisesRegex(evidence.EvidenceError, "blank JSONL"):
            evidence.parse_jsonl_bytes(b"{}\n\n", "blank")
        with self.assertRaisesRegex(evidence.EvidenceError, "end with LF"):
            evidence.parse_jsonl_bytes(b"{}", "unterminated")
        with self.assertRaisesRegex(evidence.EvidenceError, "invalid JSON"):
            evidence.parse_json_bytes(b'{"n":' + b"9" * 5_000 + b"}", "huge integer")

    def test_case_schema_rejects_unknown_fields_and_boolean_integer(self) -> None:
        rows = case_rows()
        rows[0]["unknown"] = 1
        with self.assertRaisesRegex(evidence.EvidenceError, "unknown=.*unknown"):
            evidence.validate_cases(rows)
        rows = case_rows()
        rows[0]["rows"] = True
        with self.assertRaisesRegex(evidence.EvidenceError, "unsigned integer"):
            evidence.validate_cases(rows)

    def test_case_digests_are_independently_recomputed_from_adr(self) -> None:
        for case in evidence.CASES:
            with self.subTest(case=case.case_id):
                self.assertEqual(
                    independently_recompute_case_digests(case),
                    evidence.EXPECTED_CASE_DIGESTS[case.case_id],
                )
        rows = case_rows()
        rows[0]["weight_sha256"] = "sha256:" + "0" * 64
        with self.assertRaisesRegex(evidence.EvidenceError, "frozen case"):
            evidence.validate_cases(rows)

    def test_correctness_join_rejects_missing_duplicate_and_wrong_case(self) -> None:
        rows = correctness_rows()
        with self.assertRaisesRegex(evidence.EvidenceError, "expected 16"):
            evidence.validate_correctness(rows[:-1])
        duplicated = copy.deepcopy(rows)
        duplicated[1] = copy.deepcopy(duplicated[0])
        with self.assertRaisesRegex(evidence.EvidenceError, "check_id"):
            evidence.validate_correctness(duplicated)
        wrong_case = copy.deepcopy(rows)
        wrong_case[0]["case_id"] = evidence.CASES[1].case_id
        with self.assertRaisesRegex(evidence.EvidenceError, "case_id"):
            evidence.validate_correctness(wrong_case)

    def test_componentwise_correctness_proofs_and_golden_bindings_are_enforced(self) -> None:
        rows = correctness_rows()
        kernel = rows[0]["metrics"]
        kernel["max_abs_error"] = 2e-7
        kernel["max_error_to_bound_ratio"] = 2.0
        kernel["worst_abs_error"] = 2e-7
        kernel["worst_error_bound"] = 1e-7
        with self.assertRaisesRegex(evidence.EvidenceError, "componentwise f64 bound"):
            evidence.validate_correctness(rows)

        rows = correctness_rows()
        rows[-1]["metrics"]["logits_error"]["max_tolerance_ratio"] = 1.0001
        with self.assertRaisesRegex(evidence.EvidenceError, "componentwise oracle parity"):
            evidence.validate_correctness(rows)

        rows = correctness_rows()
        rows[-1]["metrics"]["goldens"]["logits_sha256"] = "0" * 64
        with self.assertRaisesRegex(evidence.EvidenceError, "wrong oracle evidence"):
            evidence.validate_correctness(rows)

    def test_historical_oracle_hash_cohort_is_exact_and_not_mixable(self) -> None:
        legacy = correctness_rows()
        for row in legacy[-3:]:
            check_id = row["check_id"]
            row["metrics"]["goldens"] = copy.deepcopy(
                evidence.M4_20260803_MODEL_CONTRACTS[check_id]["goldens"]
            )
        evidence.validate_correctness(
            legacy,
            model_contracts=evidence.M4_20260803_MODEL_CONTRACTS,
        )
        self.assertIs(
            evidence._model_contracts_for_commit(evidence.M4_20260803_GIT_COMMIT),
            evidence.M4_20260803_MODEL_CONTRACTS,
        )
        self.assertIs(
            evidence._model_contracts_for_commit("a" * 40),
            evidence.MODEL_CONTRACTS,
        )

        mixed = copy.deepcopy(legacy)
        check_id = mixed[-1]["check_id"]
        mixed[-1]["metrics"]["goldens"] = copy.deepcopy(
            evidence.MODEL_CONTRACTS[check_id]["goldens"]
        )
        with self.assertRaisesRegex(evidence.EvidenceError, "wrong oracle evidence"):
            evidence.validate_correctness(
                mixed,
                model_contracts=evidence.M4_20260803_MODEL_CONTRACTS,
            )

    def test_typed_avx2_unsupported_row_is_retained_without_false_proofs(self) -> None:
        rows = correctness_rows()
        row = rows[-1]
        row["status"] = "unsupported"
        row["failure"] = "backend_unavailable"
        row["metrics"]["selected_backend"] = None
        for name in (
            "tokens_exact",
            "expert_ids_exact",
            "deterministic",
        ):
            row["metrics"][name] = False
        row["metrics"]["logits_error"] = None
        row["metrics"]["router_score_error"] = None
        row["metrics"]["route_weight_error"] = None
        validated = evidence.validate_correctness(rows)
        self.assertEqual(validated[-1]["status"], "unsupported")
        self.assertFalse(evidence.correctness_passed(validated))

    def test_failed_checks_retain_negative_proofs_or_typed_execution_failure(self) -> None:
        numerical = correctness_rows(failed_check="tail-257x513-avx2")
        validated = evidence.validate_correctness(numerical)
        self.assertEqual(validated[1]["failure"], "numerical_bound_exceeded")
        self.assertEqual(validated[1]["metrics"]["max_error_to_bound_ratio"], 2.0)

        execution = correctness_rows()
        execution[1]["status"] = "failed"
        execution[1]["failure"] = "kernel_execution_error"
        for name in evidence.KERNEL_METRIC_KEYS[1:]:
            execution[1]["metrics"][name] = None
        evidence.validate_correctness(execution)

        independent_candidate = correctness_rows()
        independent_candidate[0]["status"] = "failed"
        independent_candidate[0]["failure"] = "kernel_execution_error"
        for name in evidence.KERNEL_METRIC_KEYS[1:]:
            independent_candidate[0]["metrics"][name] = None
        for name in (
            "cross_scalar_max_abs",
            "cross_scalar_max_rel",
            "cross_scalar_max_ulp",
        ):
            independent_candidate[1]["metrics"][name] = None
        validated = evidence.validate_correctness(independent_candidate)
        self.assertEqual(validated[1]["status"], "ok")
        self.assertEqual(validated[1]["metrics"]["max_error_to_bound_ratio"], 0.0)

        tolerance = correctness_rows()
        tolerance[-1]["status"] = "failed"
        tolerance[-1]["failure"] = "tolerance_exceeded"
        tolerance[-1]["metrics"]["logits_error"]["max_tolerance_ratio"] = 1.25
        evidence.validate_correctness(tolerance)

        model_execution = correctness_rows()
        model_execution[-1]["status"] = "failed"
        model_execution[-1]["failure"] = "model_execution_error"
        model_execution[-1]["metrics"]["selected_backend"] = None
        for name in (
            "tokens_exact",
            "expert_ids_exact",
            "deterministic",
        ):
            model_execution[-1]["metrics"][name] = False
        model_execution[-1]["metrics"]["logits_error"] = None
        model_execution[-1]["metrics"]["router_score_error"] = None
        model_execution[-1]["metrics"]["route_weight_error"] = None
        evidence.validate_correctness(model_execution)


class DeterministicScheduleTests(unittest.TestCase):
    def test_preregistered_cell_order_vector(self) -> None:
        self.assertEqual(
            evidence.cell_order(),
            [
                "avx-two-worker-stream-expand",
                "avx-natural-llc",
                "staged-natural-stream-contract",
                "avx-natural-l2",
                "avx-two-worker-stream-contract",
                "avx-natural-stream-contract",
                "staged-natural-llc",
                "avx-offset-stream-expand",
                "avx-offset-stream-contract",
                "avx-offset-tail",
                "staged-natural-stream-expand",
                "avx-natural-tail",
                "avx-natural-stream-expand",
            ],
        )

    def test_preregistered_pair_order_vector_and_balance(self) -> None:
        self.assertEqual(
            evidence.measured_pair_orders("avx-natural-tail"),
            [
                "baseline-candidate",
                "candidate-baseline",
                "baseline-candidate",
                "candidate-baseline",
                "candidate-baseline",
                "baseline-candidate",
                "candidate-baseline",
                "baseline-candidate",
                "candidate-baseline",
                "baseline-candidate",
                "baseline-candidate",
                "baseline-candidate",
                "baseline-candidate",
                "candidate-baseline",
                "baseline-candidate",
                "candidate-baseline",
                "baseline-candidate",
                "candidate-baseline",
                "baseline-candidate",
                "candidate-baseline",
                "baseline-candidate",
                "candidate-baseline",
                "candidate-baseline",
                "baseline-candidate",
                "candidate-baseline",
                "candidate-baseline",
                "baseline-candidate",
                "candidate-baseline",
                "baseline-candidate",
                "candidate-baseline",
            ],
        )
        for cell in evidence.CELLS:
            orders = evidence.measured_pair_orders(cell.cell_id)
            self.assertEqual(orders.count("baseline-candidate"), 15)
            self.assertEqual(orders.count("candidate-baseline"), 15)
        self.assertEqual(
            evidence.WARMUP_ORDERS,
            (
                "baseline-candidate",
                "candidate-baseline",
                "baseline-candidate",
                "candidate-baseline",
                "baseline-candidate",
            ),
        )

    def test_requests_freeze_cpus_warmups_and_pair_orders(self) -> None:
        cell = evidence.CELL_BY_ID["avx-two-worker-stream-expand"]
        request = evidence.build_cell_request(cell, 0, [2, 4])
        self.assertEqual(request["cpus"], [2, 4])
        self.assertEqual(request["warmup_orders"], list(evidence.WARMUP_ORDERS))
        self.assertEqual(len(request["measured_orders"]), 30)
        with self.assertRaisesRegex(evidence.EvidenceError, "distinct CPUs"):
            evidence.build_cell_request(cell, 0, [2, 2])


class DatasetAndAntiElisionTests(unittest.TestCase):
    def test_exact_780_row_grid_validates(self) -> None:
        rows = evidence.validate_dataset(observation_rows())
        self.assertEqual(len(rows), 780)
        self.assertEqual({row["cell_id"] for row in rows}, set(evidence.CELL_BY_ID))

    def test_row_deletion_and_duplication_cannot_be_hidden(self) -> None:
        rows = observation_rows()
        with self.assertRaisesRegex(evidence.EvidenceError, "expected 780"):
            evidence.validate_dataset(rows[:-1])
        rows[-1] = copy.deepcopy(rows[-2])
        with self.assertRaisesRegex(evidence.EvidenceError, "cell_id|variant"):
            evidence.validate_dataset(rows)

    def test_case_join_and_pair_address_join_are_enforced(self) -> None:
        rows = observation_rows()
        rows[0]["case_id"] = "tail-257x513"
        with self.assertRaisesRegex(evidence.EvidenceError, "case_id"):
            evidence.validate_dataset(rows)
        rows = observation_rows()
        rows[1]["addresses_mod_64"]["output"] = 4
        with self.assertRaisesRegex(evidence.EvidenceError, "different buffers"):
            evidence.validate_dataset(rows)

    def test_cpu_custody_and_repeated_batch_proofs_are_enforced(self) -> None:
        rows = observation_rows()
        for name in ("cpu", "cpu_before", "cpu_after"):
            rows[1]["workers"][0][name] = 999
        rows[1]["workers"][0]["affinity"] = [999]
        with self.assertRaisesRegex(evidence.EvidenceError, "different worker CPUs"):
            evidence.validate_dataset(rows)

        rows = observation_rows()
        with self.assertRaisesRegex(evidence.EvidenceError, "unrequested CPU"):
            evidence.validate_dataset(rows, benchmark_cpus=([0], [0, 1]))

        rows = observation_rows()
        same_variant = next(
            row
            for row in rows[2:]
            if row["cell_id"] == rows[0]["cell_id"]
            and row["variant"] == rows[0]["variant"]
        )
        same_variant["output_sha256"] = "sha256:" + digest("changed-output")
        with self.assertRaisesRegex(evidence.EvidenceError, "repeated identical batches"):
            evidence.validate_dataset(rows)

    def test_successful_row_with_missing_timed_call_is_rejected(self) -> None:
        rows = observation_rows()
        rows[0]["executed_calls"] -= 1
        with self.assertRaisesRegex(evidence.EvidenceError, "omitted a timed call"):
            evidence.validate_dataset(rows)

    def test_mxcsr_and_cpu_residency_are_enforced(self) -> None:
        rows = observation_rows()
        rows[0]["workers"][0]["mxcsr_after"] |= 1 << 15
        with self.assertRaisesRegex(evidence.EvidenceError, "MXCSR changed"):
            evidence.validate_dataset(rows)
        rows = observation_rows()
        rows[0]["workers"][0]["cpu_after"] += 1
        with self.assertRaisesRegex(evidence.EvidenceError, "residency mismatch"):
            evidence.validate_dataset(rows)

    def test_failed_row_retains_independent_partial_probe_evidence(self) -> None:
        rows = observation_rows()
        self.assertEqual(evidence.CELL_BY_ID[rows[0]["cell_id"]].workers, 2)
        row = rows[0]
        row["status"] = "kernel_error"
        row["failure"] = "injected resource and residency probe failure"
        row["workers"] = [row["workers"][1]]
        row["workers"][0]["cpu_after"] = 999
        row["resource_usage"] = None
        validated = evidence.validate_dataset(rows)
        self.assertIsNotNone(validated[0]["addresses_mod_64"])
        self.assertEqual(validated[0]["workers"][0]["worker_index"], 1)
        self.assertIsNone(validated[0]["resource_usage"])


class StatisticsTests(unittest.TestCase):
    def test_descriptive_statistics_use_sample_deviation_and_nearest_rank_p95(self) -> None:
        summary = evidence.descriptive_statistics(list(range(1, 31)))
        self.assertEqual(summary["n"], 30)
        self.assertEqual(summary["p50"], 15.5)
        self.assertEqual(summary["p95"], 29.0)
        self.assertAlmostEqual(summary["sample_standard_deviation"], statistics_stdev_1_to_30())
        with self.assertRaisesRegex(evidence.EvidenceError, "not booleans"):
            evidence.descriptive_statistics([1, True])

    def test_bootstrap_vector_is_deterministic_and_uses_frozen_endpoints(self) -> None:
        values = [0.80 + index * 0.01 for index in range(30)]
        first = evidence.bootstrap_median_interval(values, cell_id="avx-natural-tail")
        second = evidence.bootstrap_median_interval(values, cell_id="avx-natural-tail")
        self.assertEqual(first, second)
        self.assertEqual(first["low_index"], 249)
        self.assertEqual(first["high_index"], 9_749)
        self.assertEqual(first["resamples"], 10_000)
        self.assertEqual(
            first["stream_prefix_sha256"],
            "sha256:858026bb14ca91df20754aadc21fb7f09416677c732bd9056e7e2011c6d92968",
        )
        self.assertEqual(first["low"], 0.895)
        self.assertEqual(first["high"], 0.995)

    def test_incomplete_cell_reports_status_only_and_no_interval(self) -> None:
        rows = observation_rows()
        failed_cell = rows[0]["cell_id"]
        mark_failed(rows[0])
        summary = evidence.build_summary(
            rows,
            correctness_rows(),
            cases_digest="sha256:" + digest("cases"),
            correctness_digest="sha256:" + digest("correctness"),
            observations_digest="sha256:" + digest("observations"),
            bootstrap_resamples=25,
        )
        cell = next(value for value in summary["cells"] if value["cell_id"] == failed_cell)
        self.assertFalse(cell["complete"])
        self.assertEqual(cell["complete_pairs"], 0)
        self.assertIsNone(cell["paired_candidate_over_baseline"])
        self.assertIsNone(cell["exploratory_lower_elapsed_time"])

    def test_failed_correctness_invalidates_every_timing_cell(self) -> None:
        summary = evidence.build_summary(
            observation_rows(),
            correctness_rows(failed_check="tail-257x513-avx2"),
            cases_digest="sha256:" + digest("cases"),
            correctness_digest="sha256:" + digest("correctness"),
            observations_digest="sha256:" + digest("observations"),
            bootstrap_resamples=25,
        )
        self.assertFalse(summary["correctness_passed"])
        self.assertTrue(all(not cell["complete"] for cell in summary["cells"]))
        self.assertFalse(summary["general_claim_rule"]["eligible"])
        self.assertIsNone(summary["general_claim_rule"]["satisfied"])


def statistics_stdev_1_to_30() -> float:
    mean = 15.5
    return math.sqrt(sum((value - mean) ** 2 for value in range(1, 31)) / 29)


class CaptureContractTests(unittest.TestCase):
    def test_jsonl_child_boundary_accepts_valid_cross_language_float_spelling(self) -> None:
        result = evidence.BoundedProcessResult(
            return_code=0,
            stdout=b'{"z":1e-05,"a":2}\n',
            stderr=b"",
            timed_out=False,
            stdout_exceeded=False,
            stderr_exceeded=False,
            launch_error=None,
        )
        with mock.patch.object(evidence, "run_bounded_process", return_value=result):
            rows = evidence._run_jsonl_command(
                ("unused",), expected_rows=1, context="cross-language JSON"
            )
        self.assertEqual(rows, [{"z": 1e-5, "a": 2}])

    def test_every_phase_is_bounded_by_one_absolute_capture_deadline(self) -> None:
        with mock.patch.object(evidence.time, "monotonic", return_value=100.0):
            self.assertEqual(
                evidence._remaining_timeout(105.0, 30.0, "test phase"),
                5.0,
            )
            with self.assertRaisesRegex(evidence.EvidenceError, "deadline expired"):
                evidence._remaining_timeout(100.0, 30.0, "test phase")

    def test_recorded_native_argv_is_executable_build_script_custody(self) -> None:
        source = (ROOT / "crates" / "runnel-kernels" / "build.rs").read_text()
        evidence._verify_native_build_argv_contract(source)
        with self.assertRaisesRegex(evidence.EvidenceError, "recorded native argv differs"):
            evidence._verify_native_build_argv_contract(
                source.replace('.arg("-O3")', '.arg("-O2")', 1)
            )

    def test_timed_out_cell_retains_every_valid_streamed_observation_prefix(self) -> None:
        child_sequence = 1
        cell = evidence.CELL_BY_ID[evidence.cell_order()[child_sequence]]
        self.assertEqual(cell.workers, 1)
        measured = [
            row for row in observation_rows() if row["cell_id"] == cell.cell_id
        ]
        prefix = [*warmup_rows(cell, child_sequence), *measured[:7]]
        result = evidence.BoundedProcessResult(
            return_code=-9,
            stdout=evidence._jsonl_bytes(prefix) + b'{"partial":',
            stderr=b"",
            timed_out=True,
            stdout_exceeded=False,
            stderr_exceeded=False,
            launch_error=None,
        )
        with mock.patch.object(evidence, "run_bounded_process", return_value=result):
            rows = evidence._run_cell(
                Path("/unused/kernel"), cell, child_sequence, [2]
            )
        self.assertEqual(len(rows), 60)
        self.assertEqual(rows[:7], measured[:7])
        self.assertTrue(all(row["status"] == "timeout" for row in rows[7:]))
        self.assertTrue(all(row["elapsed_ns"] is None for row in rows[7:]))

    def test_repository_and_tmpfs_reserves_are_independent(self) -> None:
        build_root = Path("/dev/shm")
        tmpfs_required = evidence.MIN_FREE_BYTES_AFTER_CAPTURE + evidence.MAX_DIRECTORY_BYTES
        repository_required = evidence.MIN_EVIDENCE_FILESYSTEM_FREE_BYTES

        with mock.patch.object(
            evidence,
            "_filesystem_available",
            side_effect=[tmpfs_required, repository_required],
        ):
            evidence._require_capture_reserves(build_root)
        with mock.patch.object(
            evidence,
            "_filesystem_available",
            side_effect=[tmpfs_required, repository_required - 1],
        ):
            with self.assertRaisesRegex(evidence.EvidenceError, "32-MiB"):
                evidence._require_capture_reserves(build_root)
        with mock.patch.object(
            evidence,
            "_filesystem_available",
            return_value=tmpfs_required - 1,
        ):
            with self.assertRaisesRegex(evidence.EvidenceError, "two GiB of tmpfs"):
                evidence._require_capture_reserves(build_root)

    def test_capture_derives_thirteen_kernel_correctness_rows(self) -> None:
        with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
            temporary = Path(directory)
            private_root = temporary / "private-build"
            private_root.mkdir()
            private = evidence.PrivateBuild(
                root=private_root,
                kernel_binary=temporary / "kernel",
                kernel_binary_sha256="d" * 64,
                model_binary=temporary / "model",
                model_binary_sha256="e" * 64,
                cargo_version="cargo test",
                rustc_version="rustc test",
                cc_path="/usr/bin/cc",
                cc_sha256="1" * 64,
                cc_version="cc test",
                ar_path="/usr/bin/ar",
                ar_sha256="2" * 64,
                ar_version="ar test",
                build_root_available_bytes=evidence.MIN_FREE_BYTES_AFTER_CAPTURE,
            )
            calls = []

            def run_jsonl(command, *, expected_rows, context, deadline=None):
                calls.append((tuple(command), expected_rows, context))
                if context == "kernel cases":
                    return case_rows()
                raise RuntimeError("stop after kernel correctness contract")

            with (
                mock.patch.object(evidence, "_require_clean_commit", return_value="c" * 64),
                mock.patch.object(evidence, "_resolve_new_output", return_value=temporary / "out"),
                mock.patch.object(
                    evidence,
                    "_resolve_build_root",
                    return_value=(temporary, evidence.MIN_FREE_BYTES_AFTER_CAPTURE),
                ),
                mock.patch.object(evidence, "_require_capture_reserves"),
                mock.patch.object(
                    evidence,
                    "_rusage_snapshot",
                    return_value={name: 0 for name in evidence.RESOURCE_KEYS},
                ),
                mock.patch.object(evidence, "_build_release_binaries", return_value=private),
                mock.patch.object(evidence, "_run_jsonl_command", side_effect=run_jsonl),
            ):
                with self.assertRaisesRegex(RuntimeError, "kernel correctness contract"):
                    evidence.capture(str(temporary), "unused", "a" * 40)
            self.assertEqual(calls[1][1:], (13, "kernel correctness"))


class CustodyAndRegenerationTests(unittest.TestCase):
    def build_artifacts(self) -> dict[str, bytes]:
        cases = evidence.validate_cases(case_rows())
        correctness = evidence.validate_correctness(correctness_rows())
        observations = evidence.validate_dataset(observation_rows())
        cases_bytes = evidence._jsonl_bytes(cases)
        correctness_bytes = evidence._jsonl_bytes(correctness)
        observations_bytes = evidence._jsonl_bytes(observations)
        summary = evidence.build_summary(
            observations,
            correctness,
            cases_digest=evidence._artifact_digest(cases_bytes),
            correctness_digest=evidence._artifact_digest(correctness_bytes),
            observations_digest=evidence._artifact_digest(observations_bytes),
        )
        snapshot = {name: 0 for name in evidence.RESOURCE_KEYS}
        topology = [
            {
                "cpu": 2,
                "physical_package_id": 0,
                "core_id": 0,
                "thread_siblings": [2],
            },
            {
                "cpu": 3,
                "physical_package_id": 0,
                "core_id": 1,
                "thread_siblings": [3],
            },
        ]
        with (
            mock.patch.object(os, "sched_getaffinity", return_value={2, 3}),
            mock.patch.object(evidence, "_cpu_topology", return_value=topology),
        ):
            environment = evidence.build_environment(
                captured_at="2026-08-03T12:34:56Z",
                harness_sha256="c" * 64,
                kernel_binary_sha256="d" * 64,
                model_binary_sha256="e" * 64,
                cargo_version="cargo 1.97.1",
                rustc_version="rustc 1.97.1",
                cc_path="/usr/bin/cc",
                cc_sha256="1" * 64,
                cc_version="cc test",
                ar_path="/usr/bin/ar",
                ar_sha256="2" * 64,
                ar_version="ar test",
                benchmark_cpus=([2], [2, 3]),
                build_root_available_bytes=evidence.MIN_FREE_BYTES_AFTER_CAPTURE + 1,
                build_root_remaining_bytes=evidence.MIN_FREE_BYTES_AFTER_CAPTURE + 1,
                process_before=snapshot,
                process_after=snapshot,
            )
        artifacts = {
            "environment.json": evidence._json_file_bytes(environment),
            "cases.jsonl": cases_bytes,
            "correctness.jsonl": correctness_bytes,
            "observations.jsonl": observations_bytes,
            "summary.json": evidence._json_file_bytes(summary),
            **evidence.render_figures(summary),
        }
        experiment = evidence.build_experiment(
            captured_at="2026-08-03T12:34:56Z",
            commit="a" * 40,
            kernel_binary_sha256="d" * 64,
            model_binary_sha256="e" * 64,
            artifacts={name: evidence._artifact_digest(data) for name, data in artifacts.items()},
        )
        artifacts["experiment.json"] = evidence._json_file_bytes(experiment)
        return artifacts

    def write_tree(self, root: Path, artifacts: dict[str, bytes]) -> None:
        (root / "figures").mkdir()
        for name, data in artifacts.items():
            path = root / name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(data)

    def test_complete_tree_byte_regenerates(self) -> None:
        artifacts = self.build_artifacts()
        with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
            root = Path(directory)
            self.write_tree(root, artifacts)
            with mock.patch.object(evidence, "_verify_commit_harness_sha256"):
                result = evidence.verify_directory(root)
            self.assertEqual(result["observation_rows"], 780)
            self.assertEqual(result["complete_cells"], 13)

    def test_tampered_summary_and_unknown_file_are_rejected(self) -> None:
        artifacts = self.build_artifacts()
        with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
            root = Path(directory)
            self.write_tree(root, artifacts)
            summary = json.loads((root / "summary.json").read_text())
            summary["correctness_passed"] = False
            (root / "summary.json").write_bytes(evidence._json_file_bytes(summary))
            with mock.patch.object(evidence, "_verify_commit_harness_sha256"):
                with self.assertRaisesRegex(evidence.EvidenceError, "digest differs"):
                    evidence.verify_directory(root)
        with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
            root = Path(directory)
            self.write_tree(root, artifacts)
            (root / "unknown.txt").write_text("unexpected")
            with self.assertRaisesRegex(evidence.EvidenceError, "unknown entry"):
                evidence._read_evidence_files(root)

    def test_historical_harness_hash_is_checked(self) -> None:
        blob = b"historical M4 harness\n"
        recorded = "sha256:" + hashlib.sha256(blob).hexdigest()
        with mock.patch.object(evidence, "_git_blob", return_value=blob):
            evidence._verify_commit_harness_sha256("a" * 40, recorded)
            with self.assertRaisesRegex(evidence.EvidenceError, "differs"):
                evidence._verify_commit_harness_sha256("a" * 40, "sha256:" + "0" * 64)

    def test_experiment_schema_rejects_nested_mutation(self) -> None:
        artifacts = self.build_artifacts()
        experiment = json.loads(artifacts["experiment.json"])
        experiment["schedule_contract"]["measured_rows"] = 779
        with self.assertRaisesRegex(evidence.EvidenceError, "frozen M4 contract"):
            evidence._validate_experiment(experiment)


if __name__ == "__main__":
    unittest.main()
