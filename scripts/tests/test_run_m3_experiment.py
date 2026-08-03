from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import run_m3_experiment as evidence


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def generated_trace_bytes(
    family: str = "stationary_zipf",
    replicate: int = 0,
    measured_steps: int = 1,
) -> bytes:
    pages = []
    for page_id in range(384):
        expert, ordinal = divmod(page_id, 3)
        pages.append(
            {
                "id": page_id,
                "logical_bytes": evidence.PAGE_BYTES,
                "charge_bytes": evidence.PAGE_BYTES,
                "class": {
                    "kind": "expert",
                    "layer": 0,
                    "expert": expert,
                    "ordinal": ordinal,
                },
            }
        )
    events = []
    sequence = 0
    for step in range(measured_steps):
        first = (step * 2) % 128
        second = (first + 1) % 128
        for expert in (first, second):
            for ordinal in range(3):
                events.append(
                    {
                        "kind": "demand",
                        "sequence": sequence,
                        "request": 0,
                        "step": step,
                        "page": expert * 3 + ordinal,
                    }
                )
                sequence += 1
    header = {
        "kind": "header",
        "schema": "runnel.cache-trace/1",
        "trace_id": evidence._expected_trace_id(family, replicate, measured_steps),
        "page_count": 384,
        "event_count": len(events),
        "charge_quantum": evidence.PAGE_BYTES,
        "prefetch_model": "instant-between-events-v1",
    }
    return b"".join(
        json.dumps(row, separators=(",", ":"), ensure_ascii=True).encode("ascii")
        + b"\n"
        for row in (header, *pages, *events)
    )


def generated_trace(
    family: str = "stationary_zipf",
    replicate: int = 0,
    measured_steps: int = 1,
) -> evidence.GeneratedTraceEvidence:
    return evidence.validate_generated_trace(
        generated_trace_bytes(family, replicate, measured_steps),
        family=family,
        replicate=replicate,
        measured_steps=measured_steps,
    )


def policy_spec(policy: str, capacity: int) -> dict:
    if policy in {"lru", "belady"}:
        return {"name": policy}
    if policy == "slru":
        return {"name": policy, "protected_fraction_ppm": 750_000}
    if policy == "tiny-lfu":
        return {
            "name": policy,
            "config": {
                "sketch_depth": 4,
                "sketch_width": 2_048,
                "sample_accesses": capacity // evidence.PAGE_BYTES * 10,
            },
        }
    return {
        "name": policy,
        "config": {
            "protected_fraction_ppm": 750_000,
            "minimum_score_ppm": 100_000,
            "max_experts_per_signal": 2,
            "max_pages_per_signal": 6,
            "max_prefetch_bytes_per_signal": 393_216,
        },
    }


def metrics(*, accesses: int, misses: int) -> dict[str, int]:
    page = evidence.PAGE_BYTES
    hits = accesses - misses
    return {
        "demand_accesses": accesses,
        "demand_logical_bytes": accesses * page,
        "ordinary_demand_hits": hits,
        "ordinary_demand_hit_bytes": hits * page,
        "useful_prefetch_hits": 0,
        "useful_prefetch_hit_bytes": 0,
        "demand_misses": misses,
        "demand_miss_bytes": misses * page,
        "demand_load_bytes": misses * page,
        "prefetch_offered": 0,
        "prefetch_offered_bytes": 0,
        "prefetch_admitted": 0,
        "prefetch_load_bytes": 0,
        "prefetch_useful": 0,
        "prefetch_useful_bytes": 0,
        "prefetch_wasted": 0,
        "prefetch_wasted_bytes": 0,
        "prefetch_redundant": 0,
        "prefetch_redundant_bytes": 0,
        "prefetch_dropped": 0,
        "prefetch_dropped_bytes": 0,
        "admissions": misses,
        "bypasses": 0,
        "evictions": 0,
        "evicted_charge_bytes": 0,
        "final_resident_charge_bytes": misses * page,
        "peak_resident_charge_bytes": misses * page,
        "policy_metadata_bytes": 0,
        "policy_metadata_limit_bytes": 0,
        "total_physical_load_bytes": misses * page,
    }


def router_prefetch_metrics(*, accesses: int) -> dict[str, int]:
    page = evidence.PAGE_BYTES
    value = metrics(accesses=accesses, misses=3)
    value.update(
        {
            "ordinary_demand_hits": accesses - 4,
            "ordinary_demand_hit_bytes": (accesses - 4) * page,
            "useful_prefetch_hits": 1,
            "useful_prefetch_hit_bytes": page,
            "prefetch_offered": 3,
            "prefetch_offered_bytes": 3 * page,
            "prefetch_admitted": 2,
            "prefetch_load_bytes": 2 * page,
            "prefetch_useful": 1,
            "prefetch_useful_bytes": page,
            "prefetch_wasted": 1,
            "prefetch_wasted_bytes": page,
            "prefetch_redundant": 1,
            "prefetch_redundant_bytes": page,
            "admissions": 5,
            "final_resident_charge_bytes": 5 * page,
            "peak_resident_charge_bytes": 5 * page,
            "total_physical_load_bytes": 5 * page,
        }
    )
    return value


def matrix_output(
    family: str = "stationary_zipf",
    replicate: int = 0,
    measured_steps: int = 1,
) -> dict:
    generated = generated_trace(family, replicate, measured_steps)
    trace_hash = generated.trace_sha256
    trace_id = evidence._expected_trace_id(family, replicate, measured_steps)
    results = []
    for capacity in evidence.CAPACITIES:
        for policy in evidence.POLICIES:
            miss_count = 3 if policy == "belady" else 4
            results.append(
                {
                    "schema": evidence.RESULT_SCHEMA,
                    "trace_id": trace_id,
                    "trace_sha256": trace_hash,
                    "policy": policy,
                    "policy_spec": policy_spec(policy, capacity),
                    "capacity_bytes": capacity,
                    "oracle_optimal": policy == "belady",
                    "metrics": (
                        router_prefetch_metrics(accesses=measured_steps * 6)
                        if policy == "router-prefetch"
                        else metrics(
                            accesses=measured_steps * 6,
                            misses=miss_count,
                        )
                    ),
                    "decision_sha256": digest(
                        f"decision:{family}:{replicate}:{capacity}:{policy}"
                    ),
                }
            )
    return {
        "schema": evidence.MATRIX_SCHEMA,
        "family": family,
        "replicate": replicate,
        "measured_steps": measured_steps,
        "seed_sha256": generated.seed_sha256,
        "full_route_sha256": digest(f"full:{family}:{replicate}"),
        "measured_route_sha256": generated.measured_route_sha256,
        "trace_sha256": trace_hash,
        "results": results,
    }


def normalized_matrix() -> tuple[dict, list[dict]]:
    return evidence.validate_matrix(
        matrix_output(),
        expected_family="stationary_zipf",
        expected_replicate=0,
        expected_measured_steps=1,
        generated_trace=generated_trace(),
    )


class JsonValidationTests(unittest.TestCase):
    def test_duplicate_keys_are_rejected_at_any_depth(self) -> None:
        with self.assertRaisesRegex(evidence.EvidenceError, "duplicate JSON key"):
            evidence.parse_json_bytes(b'{"outer":{"x":1,"x":2}}', "fixture")

    def test_nonfinite_numbers_are_rejected(self) -> None:
        with self.assertRaisesRegex(evidence.EvidenceError, "non-finite"):
            evidence.parse_json_bytes(b'{"x":NaN}', "fixture")

    def test_bool_is_not_an_integer(self) -> None:
        value = matrix_output()
        value["replicate"] = False
        with self.assertRaisesRegex(evidence.EvidenceError, "unsigned integer"):
            evidence.validate_matrix(
                value,
                expected_family="stationary_zipf",
                expected_replicate=0,
                expected_measured_steps=1,
                generated_trace=generated_trace(),
            )

    def test_unknown_matrix_field_is_rejected(self) -> None:
        value = matrix_output()
        value["surprise"] = 1
        with self.assertRaisesRegex(evidence.EvidenceError, "unknown=.*surprise"):
            evidence.validate_matrix(
                value,
                expected_family="stationary_zipf",
                expected_replicate=0,
                expected_measured_steps=1,
                generated_trace=generated_trace(),
            )

    def test_jsonl_requires_lf_and_rejects_blank_records(self) -> None:
        with self.assertRaisesRegex(evidence.EvidenceError, "end with LF"):
            evidence.parse_jsonl_bytes(b'{"x":1}', "rows")
        with self.assertRaisesRegex(evidence.EvidenceError, "blank"):
            evidence.parse_jsonl_bytes(b'{"x":1}\n\n', "rows")


class GeneratedTraceTests(unittest.TestCase):
    def test_seed_and_measured_route_are_independently_reconstructed(self) -> None:
        generated = generated_trace(measured_steps=2)
        self.assertEqual(
            generated.seed_sha256,
            evidence._expected_seed_sha256("stationary_zipf", 0),
        )
        self.assertEqual(generated.page_count, 384)
        self.assertEqual(generated.demand_events, 12)
        self.assertEqual(generated.router_signal_events, 0)

    def test_malformed_demand_grouping_is_rejected(self) -> None:
        lines = generated_trace_bytes().splitlines()
        demand = json.loads(lines[385])
        demand["page"] = 1
        lines[385] = json.dumps(demand, separators=(",", ":")).encode("ascii")
        with self.assertRaisesRegex(evidence.EvidenceError, "top-2 route|page grouping"):
            evidence.validate_generated_trace(
                b"\n".join(lines) + b"\n",
                family="stationary_zipf",
                replicate=0,
                measured_steps=1,
            )

    def test_matrix_seed_tamper_is_rejected(self) -> None:
        value = matrix_output()
        value["seed_sha256"] = digest("wrong-seed")
        with self.assertRaisesRegex(evidence.EvidenceError, "independent seed"):
            evidence.validate_matrix(
                value,
                expected_family="stationary_zipf",
                expected_replicate=0,
                expected_measured_steps=1,
                generated_trace=generated_trace(),
            )


class MatrixContractTests(unittest.TestCase):
    def test_small_synthetic_matrix_normalizes_to_one_trace_and_18_rows(self) -> None:
        trace, observations = normalized_matrix()
        self.assertEqual(trace["result_count"], 18)
        self.assertEqual(len(observations), 18)
        self.assertEqual(
            {(row["capacity_bytes"], row["policy"]) for row in observations},
            {
                (capacity, policy)
                for capacity in evidence.CAPACITIES
                for policy in evidence.POLICIES
            },
        )

    def test_duplicate_result_is_rejected(self) -> None:
        value = matrix_output()
        value["results"][1] = copy.deepcopy(value["results"][0])
        with self.assertRaisesRegex(evidence.EvidenceError, "duplicate|order"):
            evidence.validate_matrix(
                value,
                expected_family="stationary_zipf",
                expected_replicate=0,
                expected_measured_steps=1,
                generated_trace=generated_trace(),
            )

    def test_missing_result_is_rejected(self) -> None:
        value = matrix_output()
        value["results"].pop()
        with self.assertRaisesRegex(evidence.EvidenceError, "expected 18"):
            evidence.validate_matrix(
                value,
                expected_family="stationary_zipf",
                expected_replicate=0,
                expected_measured_steps=1,
                generated_trace=generated_trace(),
            )

    def test_belady_must_not_exceed_an_online_policy(self) -> None:
        value = matrix_output()
        value["results"][5]["metrics"] = metrics(accesses=6, misses=5)
        with self.assertRaisesRegex(evidence.EvidenceError, "beats exact belady"):
            evidence.validate_matrix(
                value,
                expected_family="stationary_zipf",
                expected_replicate=0,
                expected_measured_steps=1,
                generated_trace=generated_trace(),
            )

    def test_metric_accounting_tamper_is_rejected(self) -> None:
        value = matrix_output()
        value["results"][0]["metrics"]["total_physical_load_bytes"] += 1
        with self.assertRaisesRegex(evidence.EvidenceError, "physical-load byte"):
            evidence.validate_matrix(
                value,
                expected_family="stationary_zipf",
                expected_replicate=0,
                expected_measured_steps=1,
                generated_trace=generated_trace(),
            )

    def test_trace_hash_tamper_is_rejected_by_dataset_join(self) -> None:
        trace, observations = normalized_matrix()
        observations[0]["trace_sha256"] = digest("tampered")
        with self.assertRaisesRegex(evidence.EvidenceError, "differs from trace ledger"):
            evidence.validate_dataset(
                [trace], observations, measured_steps=1, require_full_matrix=False
            )

    def test_duplicate_observation_is_rejected(self) -> None:
        trace, observations = normalized_matrix()
        with self.assertRaisesRegex(evidence.EvidenceError, "duplicate observation"):
            evidence.validate_dataset(
                [trace],
                observations + [copy.deepcopy(observations[0])],
                measured_steps=1,
                require_full_matrix=False,
            )

    def test_duplicate_measured_route_is_rejected_even_with_distinct_trace_hashes(self) -> None:
        first_trace, first_observations = normalized_matrix()
        second_generated = generated_trace("scan_pollution", 0, 1)
        second_trace, second_observations = evidence.validate_matrix(
            matrix_output("scan_pollution", 0, 1),
            expected_family="scan_pollution",
            expected_replicate=0,
            expected_measured_steps=1,
            generated_trace=second_generated,
        )
        self.assertNotEqual(first_trace["trace_sha256"], second_trace["trace_sha256"])
        self.assertEqual(
            first_trace["measured_route_sha256"],
            second_trace["measured_route_sha256"],
        )
        with self.assertRaisesRegex(evidence.EvidenceError, "measured route digest reused"):
            evidence.validate_dataset(
                [first_trace, second_trace],
                first_observations + second_observations,
                measured_steps=1,
                require_full_matrix=False,
            )


class ProvenanceAndBoundedReadTests(unittest.TestCase):
    @staticmethod
    def _empty_evidence_tree(root: Path) -> None:
        (root / "figures").mkdir()
        for name in evidence.EXPECTED_FILES:
            path = root / name
            if "/" not in name:
                path.write_bytes(b"")
            else:
                path.write_bytes(b"")

    def test_clean_commit_rejects_untracked_input(self) -> None:
        def git_output(arguments: tuple[str, ...]) -> str:
            if arguments[0] == "rev-parse":
                return "a" * 40
            if arguments[0] == "status":
                return "?? untracked.rs"
            raise AssertionError(arguments)

        with mock.patch.object(evidence, "_git_output", side_effect=git_output):
            with self.assertRaisesRegex(evidence.EvidenceError, "clean worktree"):
                evidence._require_clean_commit("a" * 40)

    def test_clean_commit_rejects_harness_blob_mismatch(self) -> None:
        def git_output(arguments: tuple[str, ...]) -> str:
            if arguments[0] == "rev-parse":
                return "a" * 40
            if arguments[0] == "status":
                return ""
            if arguments[0] == "ls-files":
                return "scripts/run_m3_experiment.py"
            raise AssertionError(arguments)

        with (
            mock.patch.object(evidence, "_git_output", side_effect=git_output),
            mock.patch.object(evidence, "_commit_harness_sha256", return_value="b" * 64),
            mock.patch.object(evidence, "sha256_file", return_value="c" * 64),
        ):
            with self.assertRaisesRegex(evidence.EvidenceError, "commit blob"):
                evidence._require_clean_commit("a" * 40)

    def test_git_metadata_output_is_hard_bounded(self) -> None:
        overflow = evidence.BoundedProcessResult(
            return_code=None,
            stdout=b"x" * (evidence.MAX_GIT_OUTPUT_BYTES + 1),
            stderr=b"",
            timed_out=False,
            stdout_exceeded=True,
            stderr_exceeded=False,
            launch_error=None,
        )
        with mock.patch.object(evidence, "run_bounded_process", return_value=overflow):
            with self.assertRaisesRegex(evidence.EvidenceError, "bounded output"):
                evidence._git_output(("status", "--porcelain=v1"))

    def test_directory_enumeration_rejects_the_first_unknown_entry(self) -> None:
        class Entry:
            def __init__(self, name: str) -> None:
                self.name = name

        class Listing:
            def __init__(self) -> None:
                self.entries = iter((Entry("allowed"), Entry("unknown"), Entry("unread")))
                self.yielded = 0

            def __enter__(self) -> "Listing":
                return self

            def __exit__(self, *_: object) -> None:
                return None

            def __iter__(self) -> "Listing":
                return self

            def __next__(self) -> Entry:
                entry = next(self.entries)
                self.yielded += 1
                return entry

        listing = Listing()
        with mock.patch.object(evidence.os, "scandir", return_value=listing):
            with self.assertRaisesRegex(evidence.EvidenceError, "unknown entry"):
                evidence._bounded_directory_entries(123, {"allowed"}, "test directory")
        self.assertEqual(listing.yielded, 2)

    def test_oversized_sparse_file_is_rejected_before_read(self) -> None:
        with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
            root = Path(directory)
            self._empty_evidence_tree(root)
            with (root / "observations.jsonl").open("wb") as output:
                output.truncate(evidence.MAX_DIRECTORY_BYTES + 1)
            with mock.patch.object(evidence.os, "read") as read:
                with self.assertRaisesRegex(evidence.EvidenceError, "hard cap"):
                    evidence._read_evidence_files(root)
                read.assert_not_called()

    def test_retained_descriptor_reader_accepts_exact_regular_file_set(self) -> None:
        with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
            root = Path(directory)
            self._empty_evidence_tree(root)
            files = evidence._read_evidence_files(root)
            self.assertEqual(set(files), evidence.EXPECTED_FILES)
            self.assertTrue(all(value == b"" for value in files.values()))

    def test_top_level_symlink_is_rejected_without_resolution(self) -> None:
        with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
            parent = Path(directory)
            root = parent / "evidence"
            root.mkdir()
            self._empty_evidence_tree(root)
            link = parent / "link"
            os.symlink(root, link)
            with self.assertRaisesRegex(evidence.EvidenceError, "safely read"):
                evidence._read_evidence_files(link)

    def test_build_root_requires_tmpfs_and_rejects_symlink(self) -> None:
        resolved, available = evidence._resolve_build_root("/dev/shm")
        self.assertEqual(resolved, Path("/dev/shm"))
        self.assertGreaterEqual(available, evidence.MIN_BUILD_ROOT_FREE_BYTES)
        with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
            link = Path(directory) / "link"
            os.symlink("/dev/shm", link)
            with self.assertRaisesRegex(evidence.EvidenceError, "non-symlink"):
                evidence._resolve_build_root(str(link))

    def test_archival_harness_hash_is_checked_against_commit_blob(self) -> None:
        blob = b"historical schema-v2 harness\n"
        recorded = "sha256:" + hashlib.sha256(blob).hexdigest()
        with mock.patch.object(evidence, "_git_blob", return_value=blob):
            evidence._verify_commit_harness_sha256("a" * 40, recorded)
            with self.assertRaisesRegex(evidence.EvidenceError, "commit blob"):
                evidence._verify_commit_harness_sha256(
                    "a" * 40, "sha256:" + "0" * 64
                )

    def test_environment_rejects_recorded_build_root_below_free_space_floor(self) -> None:
        environment = evidence.build_environment(
            binary_sha256="b" * 64,
            harness_sha256="c" * 64,
            captured_at="2026-08-03T12:34:56Z",
            cargo_version="cargo 1.97.1",
            rustc_version="rustc 1.97.1",
            build_root_available_bytes=evidence.MIN_BUILD_ROOT_FREE_BYTES - 1,
        )
        with self.assertRaisesRegex(evidence.EvidenceError, "at least one GiB"):
            evidence._validate_environment(environment)


class ManifestContractTests(unittest.TestCase):
    def experiment(self) -> dict:
        artifacts = {
            name: "sha256:" + digest(name)
            for name in evidence.EXPECTED_FILES
            if name != "experiment.json"
        }
        return evidence.build_experiment(
            captured_at="2026-08-03T12:34:56Z",
            commit="a" * 40,
            binary_sha256="b" * 64,
            measured_steps=evidence.DEFAULT_MEASURED_STEPS,
            artifacts=artifacts,
        )

    def test_closed_experiment_manifest_round_trips(self) -> None:
        experiment = self.experiment()
        self.assertIs(evidence._validate_experiment(experiment), experiment)
        self.assertEqual(len(experiment["commands"]["matrix"]), 8)
        self.assertEqual(experiment["matrix_contract"]["observation_rows"], 3_240)
        self.assertFalse(experiment["analysis_contract"]["omnibus_conclusion_permitted"])
        self.assertEqual(experiment["trace_contract"], evidence.TRACE_CONTRACT)
        self.assertEqual(
            experiment["build"]["timeout_seconds"], evidence.BUILD_TIMEOUT_SECONDS
        )
        self.assertNotIn(b"/dev/shm", evidence.canonical_json(experiment["build"]))

    def test_manifest_rejects_bool_and_unknown_nested_field(self) -> None:
        experiment = self.experiment()
        experiment["tracked_worktree_clean"] = 1
        with self.assertRaisesRegex(evidence.EvidenceError, "expected boolean"):
            evidence._validate_experiment(experiment)
        experiment = self.experiment()
        experiment["matrix_contract"]["extra"] = False
        with self.assertRaisesRegex(evidence.EvidenceError, "unknown=.*extra"):
            evidence._validate_experiment(experiment)


class StatisticsAndFigureTests(unittest.TestCase):
    def test_descriptive_statistics_reject_boolean_and_reports_sample_stdev(self) -> None:
        with self.assertRaisesRegex(evidence.EvidenceError, "not booleans"):
            evidence.descriptive_statistics([1, True])
        summary = evidence.descriptive_statistics([1, 2, 3])
        self.assertEqual(summary["n"], 3)
        self.assertEqual(summary["median"], 2.0)
        self.assertAlmostEqual(summary["sample_stdev"], 1.0)

    def test_bootstrap_is_sha_seeded_and_deterministic(self) -> None:
        first = evidence.bootstrap_median_interval(
            [0.8, 0.9, 1.1, 1.2], label="cell-a", resamples=250
        )
        second = evidence.bootstrap_median_interval(
            [0.8, 0.9, 1.1, 1.2], label="cell-a", resamples=250
        )
        other = evidence.bootstrap_median_interval(
            [0.8, 0.9, 1.1, 1.2], label="cell-b", resamples=250
        )
        self.assertEqual(first, second)
        self.assertNotEqual(first["seed_sha256"], other["seed_sha256"])
        self.assertEqual(first["resamples"], 250)
        self.assertLessEqual(first["low"], first["high"])

    def test_summary_roles_are_exploratory_and_exclude_baseline_and_oracle(self) -> None:
        _, observations = normalized_matrix()
        summary = evidence.build_summary(
            observations,
            traces_digest="sha256:" + digest("traces"),
            observations_digest="sha256:" + digest("observations"),
            bootstrap_resamples=50,
        )
        lru = next(
            cell
            for cell in summary["cells"]
            if cell["capacity_bytes"] == evidence.CAPACITIES[0]
            and cell["policy"] == "lru"
        )
        belady = next(
            cell
            for cell in summary["cells"]
            if cell["capacity_bytes"] == evidence.CAPACITIES[0]
            and cell["policy"] == "belady"
        )
        self.assertEqual(lru["comparison_role"], "baseline")
        self.assertEqual(
            lru["paired_vs_lru"]["interval_position_vs_one"], "not_applicable"
        )
        self.assertEqual(belady["comparison_role"], "oracle")
        self.assertEqual(
            belady["paired_vs_lru"]["interval_position_vs_one"], "not_applicable"
        )
        self.assertEqual(summary["methodology"]["scope"], "exploratory-per-cell")
        self.assertFalse(summary["methodology"]["omnibus_conclusion_permitted"])

    def test_svg_generation_is_deterministic_accessible_and_raw_sourced(self) -> None:
        _, observations = normalized_matrix()
        summary = evidence.build_summary(
            observations,
            traces_digest="sha256:" + digest("traces"),
            observations_digest="sha256:" + digest("observations"),
            bootstrap_resamples=50,
        )
        with mock.patch.object(
            evidence,
            "bootstrap_median_interval",
            side_effect=AssertionError("figures must not bootstrap"),
        ):
            first = evidence.render_figures(summary)
            second = evidence.render_figures(summary)
        self.assertEqual(first, second)
        self.assertEqual(set(first), {
            "figures/optimal-gap.svg",
            "figures/paired-change.svg",
            "figures/prefetch-accounting.svg",
        })
        for svg in first.values():
            text = svg.decode("utf-8")
            self.assertIn('role="img"', text)
            self.assertIn("<title", text)
            self.assertIn("<desc", text)
            self.assertIn("Source: summary.json", text)
            self.assertIn("host timing", text)
        paired = first["figures/paired-change.svg"].decode("utf-8")
        for policy in evidence.POLICIES[1:5]:
            self.assertIn(evidence.POLICY_LABELS[policy], paired)
        self.assertIn("2 MiB", paired)
        self.assertIn("4 MiB", paired)
        self.assertIn("8 MiB", paired)

    def test_canonical_json_refuses_nan(self) -> None:
        with self.assertRaises(ValueError):
            evidence.canonical_json({"value": math.nan})


if __name__ == "__main__":
    unittest.main()
