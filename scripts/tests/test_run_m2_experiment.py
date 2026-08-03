from __future__ import annotations

import copy
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from scripts import run_m2_experiment as evidence


MULTI_OBJECT_DIGEST = (
    "sha256:9bb8fcc8e9d6f6ca3512bcb6daf6f89"
    "b6f0134874c9803f8782b100b39c854ae"
)


def demo_trace_events() -> list[dict]:
    expected = (
        ("miss", 0, 65_536),
        ("load_started", 0, 65_536),
        ("admitted", 0, 65_536),
        ("miss", 1, 65_536),
        ("evicted", 0, 65_536),
        ("load_started", 1, 65_536),
        ("admitted", 1, 65_536),
        ("miss", 0, 65_536),
        ("evicted", 1, 65_536),
        ("load_started", 0, 65_536),
        ("admitted", 0, 65_536),
        ("miss", 2, 17),
        ("evicted", 0, 65_536),
        ("load_started", 2, 17),
        ("admitted", 2, 17),
        ("hit", 2, 17),
    )
    return [
        {
            "sequence": sequence,
            "outcome": outcome,
            "reason": "demand",
            "object_digest": MULTI_OBJECT_DIGEST,
            "page_size": 65_536,
            "page_index": page_index,
            "logical_bytes": logical_bytes,
        }
        for sequence, (outcome, page_index, logical_bytes) in enumerate(expected)
    ]


def demo_output() -> dict:
    return {
        "schema_version": 1,
        "fixtures": {
            "tiny": {
                "artifact_id": (
                    "sha256:e49321cefc980ab59cd449341edfc624c"
                    "cfc4b0b703cd175184e056d296d9ed3"
                ),
                "object_digest": "sha256:" + "2" * 64,
                "object_length": 7_904,
                "page_table_digest": "sha256:" + "3" * 64,
                "page_table_length": 80,
            },
            "multi_page": {
                "artifact_id": (
                    "sha256:15feda585327bf8e25c124692de1b2e1"
                    "01315185e427b811a07b8eaca54e1630"
                ),
                "object_digest": MULTI_OBJECT_DIGEST,
                "object_length": 131_089,
                "page_table_digest": "sha256:" + "6" * 64,
                "page_table_length": 144,
            },
        },
        "prompt": "moe",
        "max_new_tokens": 4,
        "generated_ids": [15, 11, 20, 9],
        "generated_text": "njsh",
        "parity": True,
        "generation_cache": {
            "completed_generated_tokens": 4,
            "physical_read_bytes": 7_904,
            "bytes_per_generated_token": {
                "numerator_bytes": 7_904,
                "denominator_tokens": 4,
            },
        },
        "trace": {
            "access": "demand",
            "page_indices": [0, 1, 0, 2, 2],
            "page_lengths": [65_536, 65_536, 17],
            "event_count": 16,
            "events": demo_trace_events(),
            "outcomes": {
                "hit": 1,
                "miss": 4,
                "load_started": 4,
                "load_coalesced": 0,
                "late_prefetch": 0,
                "prefetch_coalesced": 0,
                "admitted": 4,
                "evicted": 3,
                "retired": 0,
                "load_failed": 0,
                "cancelled": 0,
                "prefetch_useful": 0,
                "prefetch_wasted": 0,
                "prefetch_redundant": 0,
                "prefetch_dropped": 0,
            },
        },
        "cache_capacity_bytes": 65_536,
        "metrics": {
            "demand_bytes": 196_642,
            "physical_read_bytes": 196_625,
            "hits": 1,
            "misses": 4,
            "admissions": 4,
            "evictions": 3,
            "coalesced_demands": 0,
            "prefetch": {
                "bytes": 0,
                "coalesced": 0,
                "late": 0,
                "useful": 0,
                "wasted": 0,
                "redundant": 0,
                "dropped": 0,
            },
            "wait_nanoseconds": 100,
            "io_nanoseconds": 200,
            "accounted": {
                "active_loads": 0,
                "page_pool_bytes": 64,
                "inflight_bytes": 0,
                "resident_bytes": 64,
                "retiring_bytes": 0,
                "leases": 0,
            },
            "observed_rss": {"resident_bytes": 1, "peak_bytes": 2},
            "trace_events_dropped": 0,
        },
    }


class PathValidationTests(unittest.TestCase):
    def test_accepts_portable_experiment_id_and_relative_command_path(self) -> None:
        self.assertEqual(
            evidence.validate_experiment_id("m2-data-plane-20260803"),
            "m2-data-plane-20260803",
        )
        self.assertEqual(
            evidence.validate_repository_relative_path("target/release/runnel").as_posix(),
            "target/release/runnel",
        )

    def test_rejects_absolute_traversal_and_multicomponent_ids(self) -> None:
        invalid_ids = (
            "/tmp/evidence",
            "../evidence",
            "m2/evidence",
            "m2\\evidence",
            "C:\\evidence",
            "m2..evidence",
            "UPPERCASE",
            "-leading",
            "trailing-",
        )
        for value in invalid_ids:
            with self.subTest(value=value), self.assertRaises(evidence.EvidenceError):
                evidence.validate_experiment_id(value)
        invalid_paths = (
            "/target/release/runnel",
            "../target/release/runnel",
            "target/../../runnel",
            "target//release/runnel",
            "./target/release/runnel",
            "target/./release/runnel",
            "target\\release\\runnel",
            "C:\\target\\runnel.exe",
        )
        for value in invalid_paths:
            with self.subTest(value=value), self.assertRaises(evidence.EvidenceError):
                evidence.validate_repository_relative_path(value)

    def test_effective_invocation_includes_every_parameter(self) -> None:
        arguments = evidence.parse_arguments(
            [
                "m2-exp",
                "--warmups",
                "4",
                "--repetitions",
                "31",
                "--timeout-seconds",
                "12.5",
                "--bootstrap-seed",
                "9",
                "--bootstrap-resamples",
                "2000",
            ]
        )
        invocation = evidence.effective_harness_invocation(arguments)
        self.assertEqual(invocation["argv"][2], "m2-exp")
        self.assertIn("--warmups 4", invocation["display"])
        self.assertIn("--repetitions 31", invocation["display"])
        self.assertIn("--timeout-seconds 12.5", invocation["display"])
        self.assertIn("--bootstrap-seed 9", invocation["display"])
        self.assertIn("--bootstrap-resamples 2000", invocation["display"])

    def test_release_build_is_bounded_offline_and_records_only_allowlisted_env(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "runnel"
            binary.write_bytes(b"binary")
            binary.chmod(0o700)
            disk = SimpleNamespace(
                f_bavail=evidence.MIN_BUILD_FREE_BYTES,
                f_frsize=1,
            )
            completed = evidence.BoundedProcessResult(
                return_code=0,
                stdout=b"",
                stderr=b"",
                timed_out=False,
                output_exceeded=False,
                launch_error=False,
            )
            with (
                mock.patch.object(evidence.os, "statvfs", return_value=disk),
                mock.patch.object(
                    evidence, "run_bounded_process", return_value=completed
                ) as run,
            ):
                result = evidence.build_release_binary(binary)
        self.assertEqual(result["command_argv"], list(evidence.BUILD_COMMAND))
        self.assertEqual(result["environment"], dict(sorted(evidence.BUILD_ENVIRONMENT.items())))
        invocation = run.call_args
        self.assertIn("--locked", invocation.args[0])
        self.assertIn("--offline", invocation.args[0])
        self.assertNotIn("AWS_SECRET_ACCESS_KEY", invocation.kwargs["environment"])

    def test_release_build_refuses_low_disk_before_launch(self) -> None:
        disk = SimpleNamespace(
            f_bavail=evidence.MIN_BUILD_FREE_BYTES - 1,
            f_frsize=1,
        )
        with (
            mock.patch.object(evidence.os, "statvfs", return_value=disk),
            mock.patch.object(evidence, "run_bounded_process") as run,
            self.assertRaises(evidence.EvidenceError),
        ):
            evidence.build_release_binary(Path("missing"))
        run.assert_not_called()

    def test_streaming_capture_kills_an_over_limit_producer(self) -> None:
        result = evidence.run_bounded_process(
            (
                sys.executable,
                "-c",
                "import os; os.write(1, b'x' * 4096)",
            ),
            cwd=evidence.ROOT,
            environment=evidence.CONTROLLED_ENVIRONMENT,
            timeout_seconds=5,
            maximum_capture_bytes=1_024,
        )
        self.assertTrue(result.output_exceeded)
        self.assertLessEqual(len(result.stdout), 1_025)


class ProjectionTests(unittest.TestCase):
    def test_projection_excludes_only_timing_and_observed_rss(self) -> None:
        first = demo_output()
        second = copy.deepcopy(first)
        second["metrics"]["wait_nanoseconds"] = 999
        second["metrics"]["io_nanoseconds"] = 888
        second["metrics"]["observed_rss"] = {
            "resident_bytes": 999_999,
            "peak_bytes": 1_000_000,
        }
        self.assertEqual(
            evidence.project_correctness(first), evidence.project_correctness(second)
        )
        second["metrics"]["physical_read_bytes"] += 1
        self.assertNotEqual(
            evidence.project_correctness(first), evidence.project_correctness(second)
        )

    def test_projection_requires_every_volatile_observation(self) -> None:
        output = demo_output()
        del output["metrics"]["observed_rss"]["peak_bytes"]
        with self.assertRaises(evidence.EvidenceError):
            evidence.project_correctness(output)

    def test_fixed_projection_gate_accepts_golden_and_rejects_false_parity(self) -> None:
        projection = evidence.project_correctness(demo_output())
        evidence.validate_correctness_projection(projection)
        projection["parity"] = False
        with self.assertRaises(evidence.EvidenceError):
            evidence.validate_correctness_projection(projection)

    def test_fixture_and_trace_digests_are_stable_and_domain_separated(self) -> None:
        projection = evidence.project_correctness(demo_output())
        first = evidence.fixture_and_trace_records(projection)
        second = evidence.fixture_and_trace_records(copy.deepcopy(projection))
        self.assertEqual(first, second)
        self.assertNotEqual(first["fixture"]["digest"], first["trace"]["digest"])

    def test_trace_gate_rejects_wrong_event_order_or_identity(self) -> None:
        projection = evidence.project_correctness(demo_output())
        projection["trace"]["events"][4]["page_index"] = 2
        with self.assertRaises(evidence.EvidenceError):
            evidence.validate_correctness_projection(projection)


class StatisticsTests(unittest.TestCase):
    def test_contract_statistics_use_sample_dispersion_and_type_7_percentile(self) -> None:
        summary = evidence.summarize_samples(
            [1, 2, 3, 4], bootstrap_seed=7, bootstrap_resamples=500
        )
        assert summary is not None
        self.assertEqual(summary["sample_count"], 4)
        self.assertEqual(summary["median"], 2.5)
        self.assertEqual(summary["mean"], 2.5)
        self.assertEqual(summary["p50"], 2.5)
        self.assertAlmostEqual(summary["p95"], 3.85)
        self.assertTrue(
            math.isclose(summary["sample_standard_deviation"], math.sqrt(5 / 3))
        )

    def test_bootstrap_interval_is_seeded_and_deterministic(self) -> None:
        first = evidence.bootstrap_median_interval(
            [2, 3, 5, 8, 13], seed=1234, resamples=1_000
        )
        second = evidence.bootstrap_median_interval(
            [2, 3, 5, 8, 13], seed=1234, resamples=1_000
        )
        self.assertEqual(first, second)
        assert first is not None
        self.assertLessEqual(first["lower"], 5)
        self.assertGreaterEqual(first["upper"], 5)

    def test_bootstrap_calculates_exactly_one_median_per_resample(self) -> None:
        with mock.patch.object(
            evidence.statistics, "median", wraps=evidence.statistics.median
        ) as median:
            evidence.bootstrap_median_interval([1, 2, 3], seed=8, resamples=37)
        self.assertEqual(median.call_count, 37)

    def test_empty_success_population_has_no_statistics(self) -> None:
        self.assertIsNone(
            evidence.summarize_samples(
                [], bootstrap_seed=1, bootstrap_resamples=1_000
            )
        )

    def test_summary_preserves_failures_and_uses_only_successful_timings(self) -> None:
        rows = [
            {
                "status": "ok",
                "wall_time_nanoseconds": 10,
                "correctness_projection_matches_gate": True,
                "child_page_faults": {"major": 1, "minor": 2},
                "demo": demo_output(),
            },
            {
                "status": "timeout",
                "wall_time_nanoseconds": 100,
                "correctness_projection_matches_gate": False,
                "child_page_faults": {"major": 0, "minor": 0},
            },
        ]
        summary = evidence.build_summary(
            experiment_id="m2-test",
            rows=rows,
            gate_projection_digest="sha256:" + "0" * 64,
            observations_digest="1" * 64,
            binary_unchanged=True,
            bootstrap_seed=1,
            bootstrap_resamples=100,
        )
        self.assertEqual(summary["status_counts"], {"ok": 1, "timeout": 1})
        self.assertEqual(summary["wall_time_nanoseconds"]["sample_count"], 1)
        self.assertEqual(
            summary["cache_and_rss_distributions"]["cache_wait_nanoseconds"][
                "statistics"
            ]["median"],
            100.0,
        )
        self.assertEqual(
            summary["correctness"]["projection_match_fraction"],
            {"numerator": 1, "denominator": 2, "value": 0.5},
        )
        self.assertEqual(
            summary["correctness"]["exact_generation_bytes_per_token"][
                "bytes_per_generated_token"
            ],
            1_976.0,
        )
        self.assertEqual(summary["correctness"]["outcome"], "fail")

    def test_summary_cannot_pass_with_missing_observability(self) -> None:
        output = demo_output()
        del output["metrics"]["observed_rss"]
        summary = evidence.build_summary(
            experiment_id="m2-missing-observability",
            rows=[
                {
                    "status": "ok",
                    "wall_time_nanoseconds": 10,
                    "correctness_projection_matches_gate": True,
                    "child_page_faults": {"major": 0, "minor": 1},
                    "demo": output,
                }
            ],
            gate_projection_digest="sha256:" + "0" * 64,
            observations_digest="1" * 64,
            binary_unchanged=True,
            bootstrap_seed=1,
            bootstrap_resamples=100,
        )
        self.assertFalse(
            summary["correctness"]["all_successful_trials_have_observability"]
        )
        self.assertEqual(summary["correctness"]["outcome"], "fail")


if __name__ == "__main__":
    unittest.main()
