from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from oracle import cache_policy


def _compact(record: dict[str, object]) -> bytes:
    return json.dumps(record, ensure_ascii=True, separators=(",", ":")).encode("ascii")


def _trace_bytes(
    geometry: list[tuple[int, int]],
    demands: list[int],
) -> bytes:
    records: list[dict[str, object]] = [
        {
            "kind": "header",
            "schema": "runnel.cache-trace/1",
            "trace_id": "python-oracle-test",
            "page_count": len(geometry),
            "event_count": len(demands),
            "charge_quantum": 1,
            "prefetch_model": "instant-between-events-v1",
        }
    ]
    records.extend(
        {
            "id": page_id,
            "logical_bytes": logical_bytes,
            "charge_bytes": charge_bytes,
            "class": {"kind": "shared"},
        }
        for page_id, (logical_bytes, charge_bytes) in enumerate(geometry)
    )
    records.extend(
        {
            "kind": "demand",
            "sequence": sequence,
            "request": 0,
            "step": sequence,
            "page": page_id,
        }
        for sequence, page_id in enumerate(demands)
    )
    return b"\n".join(_compact(record) for record in records) + b"\n"


class CachePolicyOracleTests(unittest.TestCase):
    def parse(self, payload: bytes) -> cache_policy.Trace:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_bytes(payload)
            return cache_policy.parse_trace(path)

    def run_main(self, payload: bytes, policy: str) -> tuple[int, str, str]:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_bytes(payload)
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                status = cache_policy.main(
                    [
                        "--trace",
                        str(path),
                        "--policy",
                        policy,
                        "--capacity-bytes",
                        "1",
                    ]
                )
            return status, stdout.getvalue(), stderr.getvalue()

    def test_lru_uses_byte_capacity_and_atomic_multi_victim_eviction(self) -> None:
        trace = self.parse(
            _trace_bytes([(1, 2), (2, 3), (4, 5)], [0, 1, 0, 2, 2])
        )

        self.assertEqual(
            cache_policy.simulate(trace, "lru", 5),
            {
                "demand_accesses": 5,
                "ordinary_demand_hits": 2,
                "demand_misses": 3,
                "admissions": 3,
                "bypasses": 0,
                "evictions": 2,
                "demand_load_bytes": 7,
                "final_resident_charge_bytes": 5,
                "peak_resident_charge_bytes": 5,
            },
        )

    def test_slru_demotion_becomes_probation_mru(self) -> None:
        trace = self.parse(
            _trace_bytes(
                [(3, 4), (3, 4), (3, 4), (3, 4), (2, 3), (4, 4)],
                [0, 1, 2, 3, 4, 0, 1, 2, 3, 5, 0],
            )
        )

        self.assertEqual(
            cache_policy.simulate(
                trace, "slru", 20, protected_fraction_ppm=750_000
            ),
            {
                "demand_accesses": 11,
                "ordinary_demand_hits": 5,
                "demand_misses": 6,
                "admissions": 6,
                "bypasses": 0,
                "evictions": 1,
                "demand_load_bytes": 18,
                "final_resident_charge_bytes": 20,
                "peak_resident_charge_bytes": 20,
            },
        )

    def test_tinylfu_observes_candidate_before_strict_admission(self) -> None:
        trace = self.parse(_trace_bytes([(7, 8)] * 3, [0, 1, 0, 1, 2, 0]))

        self.assertEqual(
            cache_policy.simulate(
                trace,
                "tiny-lfu",
                16,
                sketch_depth=4,
                sketch_width=64,
                sample_accesses=100,
            ),
            {
                "demand_accesses": 6,
                "ordinary_demand_hits": 3,
                "demand_misses": 3,
                "admissions": 2,
                "bypasses": 1,
                "evictions": 0,
                "demand_load_bytes": 21,
                "final_resident_charge_bytes": 16,
                "peak_resident_charge_bytes": 16,
            },
        )

    def test_tinylfu_aging_changes_a_multi_victim_decision(self) -> None:
        trace = self.parse(
            _trace_bytes([(2, 2), (1, 2), (4, 5), (3, 3)], [0, 1, 0, 1, 3, 0, 2])
        )

        self.assertEqual(
            cache_policy.simulate(
                trace,
                "tiny-lfu",
                8,
                sketch_depth=4,
                sketch_width=64,
                sample_accesses=4,
            ),
            {
                "demand_accesses": 7,
                "ordinary_demand_hits": 3,
                "demand_misses": 4,
                "admissions": 4,
                "bypasses": 0,
                "evictions": 2,
                "demand_load_bytes": 10,
                "final_resident_charge_bytes": 7,
                "peak_resident_charge_bytes": 7,
            },
        )

    def test_parser_preserves_router_records_and_expert_page_identity(self) -> None:
        records: list[dict[str, object]] = [
            {
                "kind": "header",
                "schema": "runnel.cache-trace/1",
                "trace_id": "ignored-router-signal",
                "page_count": 2,
                "event_count": 3,
                "charge_quantum": 1,
                "prefetch_model": "instant-between-events-v1",
            },
            {
                "id": 0,
                "logical_bytes": 1,
                "charge_bytes": 1,
                "class": {"kind": "expert", "layer": 0, "expert": 0, "ordinal": 0},
            },
            {
                "id": 1,
                "logical_bytes": 1,
                "charge_bytes": 1,
                "class": {"kind": "expert", "layer": 0, "expert": 1, "ordinal": 0},
            },
            {"kind": "demand", "sequence": 0, "request": 7, "step": 0, "page": 0},
            {
                "kind": "router_signal",
                "sequence": 1,
                "request": 7,
                "target_step": 1,
                "layer": 0,
                "predictions": [{"expert": 1, "score_ppm": 900_000}],
            },
            {"kind": "demand", "sequence": 2, "request": 7, "step": 1, "page": 1},
        ]
        trace = self.parse(b"\n".join(_compact(record) for record in records) + b"\n")

        self.assertEqual(trace.demands, (0, 1))
        self.assertEqual(
            (trace.pages[1].layer, trace.pages[1].expert, trace.pages[1].ordinal),
            (0, 1, 0),
        )
        self.assertIsInstance(trace.events[1], cache_policy.RouterSignalEvent)
        self.assertEqual(
            trace.events[1],
            cache_policy.RouterSignalEvent(
                sequence=1,
                request=7,
                target_step=1,
                layer=0,
                predictions=((1, 900_000),),
            ),
        )
        self.assertEqual(cache_policy.simulate(trace, "lru", 1)["demand_misses"], 2)

    def test_router_admit_keeps_layers_and_future_targets_independent(self) -> None:
        pages = [
            {
                "id": 0,
                "logical_bytes": 1,
                "charge_bytes": 1,
                "class": {"kind": "expert", "layer": 0, "expert": 0, "ordinal": 0},
            },
            {
                "id": 1,
                "logical_bytes": 1,
                "charge_bytes": 1,
                "class": {"kind": "expert", "layer": 0, "expert": 1, "ordinal": 0},
            },
            {
                "id": 2,
                "logical_bytes": 1,
                "charge_bytes": 1,
                "class": {"kind": "expert", "layer": 1, "expert": 0, "ordinal": 0},
            },
            {
                "id": 3,
                "logical_bytes": 1,
                "charge_bytes": 1,
                "class": {"kind": "shared"},
            },
        ]
        events: list[dict[str, object]] = [
            {"kind": "demand", "sequence": 0, "request": 7, "step": 0, "page": 0},
            {"kind": "demand", "sequence": 1, "request": 7, "step": 0, "page": 2},
            {
                "kind": "router_signal",
                "sequence": 2,
                "request": 7,
                "target_step": 2,
                "layer": 0,
                "predictions": [{"expert": 0, "score_ppm": 900_000}],
            },
            {
                "kind": "router_signal",
                "sequence": 3,
                "request": 7,
                "target_step": 1,
                "layer": 0,
                "predictions": [{"expert": 0, "score_ppm": 600_000}],
            },
            {
                "kind": "router_signal",
                "sequence": 4,
                "request": 7,
                "target_step": 1,
                "layer": 1,
                "predictions": [{"expert": 0, "score_ppm": 300_000}],
            },
            {"kind": "demand", "sequence": 5, "request": 7, "step": 1, "page": 3},
            {"kind": "demand", "sequence": 6, "request": 7, "step": 2, "page": 1},
            {"kind": "demand", "sequence": 7, "request": 7, "step": 2, "page": 0},
        ]
        records: list[dict[str, object]] = [
            {
                "kind": "header",
                "schema": "runnel.cache-trace/1",
                "trace_id": "router-multitarget-layers",
                "page_count": len(pages),
                "event_count": len(events),
                "charge_quantum": 1,
                "prefetch_model": "instant-between-events-v1",
            },
            *pages,
            *events,
        ]
        trace = self.parse(b"\n".join(_compact(record) for record in records) + b"\n")

        self.assertEqual(
            cache_policy.simulate(trace, "router-admit", 2),
            {
                "demand_accesses": 5,
                "ordinary_demand_hits": 1,
                "demand_misses": 4,
                "admissions": 3,
                "bypasses": 1,
                "evictions": 1,
                "demand_load_bytes": 4,
                "final_resident_charge_bytes": 2,
                "peak_resident_charge_bytes": 2,
            },
        )

    def test_parser_rejects_noncanonical_and_invalid_inputs(self) -> None:
        valid = _trace_bytes([(1, 1)], [0])
        lines = valid.splitlines()
        duplicate = lines[0].replace(
            b'"schema":',
            b'"schema":"runnel.cache-trace/1","schema":',
            1,
        )
        unknown_page = _trace_bytes([(1, 1)], [1])
        negative_size = _trace_bytes([(-1, 1)], [0])
        unknown_field = lines[1][:-1] + b',"extra":0}'
        cases = {
            "missing final LF": valid[:-1],
            "duplicate field": duplicate + b"\n" + b"\n".join(lines[1:]) + b"\n",
            "unknown page": unknown_page,
            "negative size": negative_size,
            "unknown field": lines[0] + b"\n" + unknown_field + b"\n" + lines[2] + b"\n",
        }
        for label, payload in cases.items():
            with self.subTest(label=label), self.assertRaises(cache_policy.OracleError):
                self.parse(payload)

        with self.assertRaisesRegex(
            cache_policy.OracleError, "duplicate JSON field 'schema'"
        ):
            self.parse(cases["duplicate field"])

    def test_parser_rejects_record_amplification_before_decoding(self) -> None:
        payload = b"{}\n" * (cache_policy.MAX_RECORDS + 1)

        with mock.patch.object(
            cache_policy,
            "_unique_object",
            side_effect=AssertionError("decoder should not run"),
        ), self.assertRaisesRegex(cache_policy.OracleError, "record limit"):
            self.parse(payload)

    def test_parser_wraps_json_value_and_recursion_failures(self) -> None:
        oversized_integer = b'{"kind":' + (b"9" * 5_000) + b"}\n"
        status, stdout, stderr = self.run_main(oversized_integer, "lru")
        self.assertEqual(status, 2)
        self.assertEqual(stdout, "")
        self.assertIn("invalid JSON value", stderr)
        self.assertNotIn("Traceback", stderr)

        with mock.patch.object(
            cache_policy.json,
            "loads",
            side_effect=RecursionError("synthetic recursion exhaustion"),
        ), self.assertRaisesRegex(cache_policy.OracleError, "nesting limit"):
            self.parse(b"{}\n")

    def test_parser_rejects_duplicate_expert_page_ordinal(self) -> None:
        records: list[dict[str, object]] = [
            {
                "kind": "header",
                "schema": "runnel.cache-trace/1",
                "trace_id": "duplicate-expert-ordinal",
                "page_count": 2,
                "event_count": 0,
                "charge_quantum": 1,
                "prefetch_model": "instant-between-events-v1",
            },
            {
                "id": 0,
                "logical_bytes": 1,
                "charge_bytes": 1,
                "class": {"kind": "expert", "layer": 3, "expert": 7, "ordinal": 2},
            },
            {
                "id": 1,
                "logical_bytes": 1,
                "charge_bytes": 1,
                "class": {"kind": "expert", "layer": 3, "expert": 7, "ordinal": 2},
            },
        ]
        payload = b"\n".join(_compact(record) for record in records) + b"\n"

        with self.assertRaisesRegex(
            cache_policy.OracleError,
            r"duplicate expert page tuple \(3, 7, 2\)",
        ):
            self.parse(payload)

    def test_empty_cli_traces_only_derive_a_tinylfu_default(self) -> None:
        payload = _trace_bytes([], [])
        expected = {
            "admissions": 0,
            "bypasses": 0,
            "demand_accesses": 0,
            "demand_load_bytes": 0,
            "demand_misses": 0,
            "evictions": 0,
            "final_resident_charge_bytes": 0,
            "ordinary_demand_hits": 0,
            "peak_resident_charge_bytes": 0,
        }

        for policy in ("lru", "slru", "router-admit", "tiny-lfu"):
            with self.subTest(policy=policy), mock.patch.object(
                cache_policy,
                "simulate",
                wraps=cache_policy.simulate,
            ) as simulate:
                status, stdout, stderr = self.run_main(payload, policy)
                self.assertEqual(status, 0)
                self.assertEqual(json.loads(stdout), expected)
                self.assertEqual(stderr, "")
                if policy == "tiny-lfu":
                    self.assertEqual(simulate.call_args.kwargs["sample_accesses"], 10)
                else:
                    self.assertNotIn("sample_accesses", simulate.call_args.kwargs)

    def test_configuration_bounds_are_enforced(self) -> None:
        trace = self.parse(_trace_bytes([(1, 1)], [0]))
        with self.assertRaisesRegex(cache_policy.OracleError, "one million"):
            cache_policy.simulate(trace, "slru", 1, protected_fraction_ppm=1_000_001)
        with self.assertRaisesRegex(cache_policy.OracleError, "sketch_width"):
            cache_policy.simulate(trace, "tiny-lfu", 1, sketch_width=63)
        with self.assertRaisesRegex(cache_policy.OracleError, "sample_accesses"):
            cache_policy.simulate(trace, "tiny-lfu", 1, sample_accesses=0)
        with self.assertRaisesRegex(cache_policy.OracleError, "minimum_score_ppm"):
            cache_policy.simulate(
                trace, "router-admit", 1, minimum_score_ppm=1_000_001
            )
        with self.assertRaisesRegex(cache_policy.OracleError, "max_experts_per_signal"):
            cache_policy.simulate(trace, "router-admit", 1, max_experts_per_signal=0)

    def test_parser_does_not_follow_a_trace_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "target.jsonl"
            target.write_bytes(_trace_bytes([(1, 1)], [0]))
            link = Path(directory) / "trace.jsonl"
            link.symlink_to(target)

            with self.assertRaises(cache_policy.OracleError):
                cache_policy.parse_trace(link)


if __name__ == "__main__":
    unittest.main()
