from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import copy
import hashlib
import io
from itertools import islice
from pathlib import Path
import tempfile
import unittest

from oracle import scheduler


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = REPOSITORY_ROOT / "fixtures" / "scheduler" / "actor-stress-v1.json"
EXPECTED_DESCRIPTOR_DIGEST = (
    "sha256:d902ecf3377310de99471f41287f62730671263b8e339ac03b87ee5d6edef42b"
)
EXPECTED_ACTION_DIGEST = (
    "sha256:430810784f31659367ecc4fecb5cf8693b4758176debe4a4769dc7cc62611b73"
)
EXPECTED_FIXTURE_ID = (
    "sha256:297827cef0bde04a5a1e6af549c565b5548d223a79dd9b80aacb58142c364436"
)
EXPECTED_FILE_SHA256 = "eb4a170a70c734471f6be0d33767f6d0d7102662307bd8860f9bc527d317e9b1"


class SchedulerOracleTests(unittest.TestCase):
    def test_word_stream_has_literal_little_endian_blocks(self) -> None:
        self.assertEqual(
            scheduler.word_block(0),
            (
                0xD88E_6BEC_D50C_3978,
                0x7304_8CF5_1D49_9E2A,
                0xB0F7_AE47_B624_EE9C,
                0x1444_D59A_7F44_B09F,
            ),
        )
        self.assertEqual(
            scheduler.word_block(1),
            (
                0xFD3B_92DD_897D_4E60,
                0xD6EE_05D5_C3AF_E173,
                0x239D_87B1_0FE7_E5A3,
                0x6A4D_F2C3_12E2_E4ED,
            ),
        )
        self.assertEqual(
            tuple(islice(scheduler.word_stream(), 8)),
            scheduler.word_block(0) + scheduler.word_block(1),
        )

    def test_unbiased_mapping_rejects_only_the_tail_and_checks_bounds(self) -> None:
        self.assertEqual(scheduler.unbiased_value(0, 5), 0)
        self.assertEqual(scheduler.unbiased_value(scheduler.MAX_U64 - 1, 5), 4)
        self.assertIsNone(scheduler.unbiased_value(scheduler.MAX_U64, 5))
        self.assertEqual(scheduler.unbiased_value(scheduler.MAX_U64, 64), 63)
        self.assertEqual(
            scheduler.unbiased_value(scheduler.MAX_U64, scheduler.UINT64_RANGE),
            scheduler.MAX_U64,
        )
        for word, bound in ((-1, 5), (0, 0), (0, scheduler.UINT64_RANGE + 1), (True, 5)):
            with self.subTest(word=word, bound=bound):
                with self.assertRaises(scheduler.SchedulerFixtureError):
                    scheduler.unbiased_value(word, bound)

    def test_action_stream_consumes_rejections_and_wake_selector(self) -> None:
        actions = scheduler.build_actions(
            3,
            words=(
                scheduler.MAX_U64,
                4,
                63,
                0,
                1,
                7,
            ),
        )
        self.assertEqual(
            actions,
            [
                {
                    "kind": "wake",
                    "ordinal": 0,
                    "producer": 0,
                    "selector_index": 63,
                },
                {
                    "exhausted": False,
                    "kind": "submit",
                    "ordinal": 1,
                    "producer": 0,
                    "request_index": 0,
                    "submit_attempt": 0,
                },
                {
                    "kind": "cancel",
                    "ordinal": 2,
                    "producer": 0,
                    "request_index": 7,
                },
            ],
        )

    def test_descriptors_have_literal_vectors_and_closed_bounds(self) -> None:
        descriptors = scheduler.build_descriptors()
        self.assertEqual(len(descriptors), 64)
        self.assertEqual(
            descriptors[:2],
            [
                {
                    "deadline_ns": None,
                    "index": 0,
                    "max_new_tokens": 15,
                    "prompt": [1, 22],
                    "sampling": "greedy",
                },
                {
                    "deadline_ns": None,
                    "index": 1,
                    "max_new_tokens": 13,
                    "prompt": [10, 21, 25],
                    "sampling": "greedy",
                },
            ],
        )
        self.assertEqual(
            descriptors[-2:],
            [
                {
                    "deadline_ns": None,
                    "index": 62,
                    "max_new_tokens": 10,
                    "prompt": [23, 8, 14, 29],
                    "sampling": "greedy",
                },
                {
                    "deadline_ns": None,
                    "index": 63,
                    "max_new_tokens": 13,
                    "prompt": [29],
                    "sampling": "greedy",
                },
            ],
        )
        for index, descriptor in enumerate(descriptors):
            with self.subTest(index=index):
                self.assertEqual(descriptor["index"], index)
                self.assertIn(len(descriptor["prompt"]), range(1, 5))
                self.assertTrue(all(1 <= token <= 31 for token in descriptor["prompt"]))
                self.assertGreaterEqual(descriptor["max_new_tokens"], 1)
                self.assertLessEqual(
                    len(descriptor["prompt"])
                    + descriptor["max_new_tokens"]
                    - 1,
                    16,
                )
                self.assertEqual(descriptor["sampling"], "greedy")
                self.assertIsNone(descriptor["deadline_ns"])

    def test_actions_have_literal_edges_counts_and_producer_mapping(self) -> None:
        actions = scheduler.build_actions()
        self.assertEqual(len(actions), 1_024)
        self.assertEqual(
            actions[:4],
            [
                {
                    "kind": "drain",
                    "ordinal": 0,
                    "producer": 0,
                    "request_index": 42,
                },
                {
                    "exhausted": False,
                    "kind": "submit",
                    "ordinal": 1,
                    "producer": 0,
                    "request_index": 0,
                    "submit_attempt": 0,
                },
                {
                    "exhausted": False,
                    "kind": "submit",
                    "ordinal": 2,
                    "producer": 1,
                    "request_index": 1,
                    "submit_attempt": 1,
                },
                {
                    "exhausted": False,
                    "kind": "submit",
                    "ordinal": 3,
                    "producer": 0,
                    "request_index": 2,
                    "submit_attempt": 2,
                },
            ],
        )
        self.assertEqual(
            actions[-4:],
            [
                {
                    "kind": "cancel",
                    "ordinal": 1_020,
                    "producer": 0,
                    "request_index": 1,
                },
                {
                    "kind": "drain",
                    "ordinal": 1_021,
                    "producer": 1,
                    "request_index": 7,
                },
                {
                    "kind": "drain",
                    "ordinal": 1_022,
                    "producer": 0,
                    "request_index": 28,
                },
                {
                    "kind": "wake",
                    "ordinal": 1_023,
                    "producer": 1,
                    "selector_index": 49,
                },
            ],
        )

        submits = [action for action in actions if action["kind"] == "submit"]
        self.assertEqual([action["submit_attempt"] for action in submits], list(range(206)))
        for action in actions:
            with self.subTest(ordinal=action["ordinal"]):
                self.assertIn(action["producer"], (0, 1))
                kind = action["kind"]
                if kind == "submit":
                    attempt = action["submit_attempt"]
                    self.assertEqual(action["producer"], attempt % 2)
                    self.assertEqual(action["exhausted"], attempt >= 64)
                    self.assertEqual(
                        action["request_index"], None if attempt >= 64 else attempt
                    )
                elif kind == "wake":
                    self.assertIn(action["selector_index"], range(64))
                    self.assertEqual(action["producer"], action["ordinal"] % 2)
                else:
                    request_index = action["request_index"]
                    self.assertIn(request_index, range(64))
                    home = request_index % 2
                    expected = 1 - home if kind == "cancel" else home
                    self.assertEqual(action["producer"], expected)

    def test_fixture_freezes_hashes_pins_and_input_only_scope(self) -> None:
        fixture = scheduler.build_fixture()
        self.assertEqual(
            fixture["actor_config"],
            {
                "adapter": "tiny-v3",
                "admission_reserve_bytes": 1_048_576,
                "backend": "scalar",
                "batch_width": 8,
                "command_capacity": 8,
                "logical_memory_limit_bytes": 8_388_608,
                "max_active_requests": 8,
                "max_context_tokens": 19,
                "max_new_tokens": 16,
                "max_outstanding_requests": 16,
                "max_prompt_tokens": 4,
                "max_queued_requests": 16,
                "max_retained_terminal_results": 16,
                "output_capacity_per_request": 2,
                "page_pool_partition_bytes": 0,
                "state_page_tokens": 4,
                "trace_capacity": 1_024,
                "waves_per_step": 4,
                "worker_count": 1,
            },
        )
        self.assertEqual(
            fixture["action_counts"],
            {
                "by_kind": {
                    "submit": 206,
                    "cancel": 220,
                    "drop": 222,
                    "drain": 237,
                    "wake": 139,
                },
                "exhausted_submits": 142,
                "total": 1_024,
            },
        )
        self.assertEqual(
            fixture["producer_counts"],
            {"producer_0": 518, "producer_1": 506},
        )
        self.assertEqual(fixture["descriptor_vectors"]["digest"], EXPECTED_DESCRIPTOR_DIGEST)
        self.assertEqual(fixture["action_vectors"]["digest"], EXPECTED_ACTION_DIGEST)
        self.assertEqual(fixture["fixture_id"], EXPECTED_FIXTURE_ID)
        self.assertEqual(
            fixture["descriptor_vectors"]["first"],
            scheduler.build_descriptors()[: scheduler.PIN_COUNT],
        )
        self.assertEqual(
            fixture["action_vectors"]["last"],
            scheduler.build_actions()[-scheduler.PIN_COUNT :],
        )
        encoded = scheduler.canonical_bytes(fixture)
        self.assertNotIn(b"terminal_digest", encoded)
        self.assertNotIn(b"runtime_result", encoded)
        self.assertNotIn(b"golden_outcome", encoded)

        self.assertEqual(fixture["actor_config"]["max_context_tokens"], 19)
        for descriptor in scheduler.build_descriptors():
            positions = len(descriptor["prompt"]) + descriptor["max_new_tokens"] - 1
            self.assertLessEqual(positions, 16)

    def test_committed_fixture_equals_regeneration_and_frozen_hashes(self) -> None:
        raw = FIXTURE_PATH.read_bytes()
        document = scheduler.parse_document_bytes(raw)
        self.assertEqual(raw, scheduler.generated_bytes())
        self.assertEqual(hashlib.sha256(raw).hexdigest(), EXPECTED_FILE_SHA256)
        self.assertEqual(document["fixture_id"], EXPECTED_FIXTURE_ID)
        self.assertEqual(scheduler.check_path(FIXTURE_PATH), EXPECTED_FILE_SHA256)

    def test_loader_rejects_unknown_duplicate_noncanonical_and_stale_data(self) -> None:
        fixture = scheduler.build_fixture()

        unknown = copy.deepcopy(fixture)
        unknown["unknown"] = 1
        with self.assertRaises(scheduler.SchedulerFixtureError):
            scheduler.parse_document_bytes(scheduler.canonical_bytes(unknown))

        stale = copy.deepcopy(fixture)
        stale["action_vectors"]["digest"] = "sha256:" + "0" * 64
        stale["fixture_id"] = scheduler.fixture_identity(stale)
        with self.assertRaises(scheduler.SchedulerFixtureError):
            scheduler.parse_document_bytes(scheduler.canonical_bytes(stale))

        boolean_integer = copy.deepcopy(fixture)
        boolean_integer["actor_config"]["worker_count"] = True
        boolean_integer["fixture_id"] = scheduler.fixture_identity(boolean_integer)
        with self.assertRaises(scheduler.SchedulerFixtureError):
            scheduler.parse_document_bytes(
                scheduler.canonical_bytes(boolean_integer)
            )

        raw = scheduler.generated_bytes()
        duplicate = raw.replace(
            b'{\n',
            b'{\n  "schema": "runnel.actor-stress-vectors/1",\n',
            1,
        )
        cases = (
            raw[:-1],
            raw.replace(b"\n", b" \n", 1),
            duplicate,
            raw.replace(b'"schema": "runnel.actor-stress-vectors/1"', b'"schema": NaN'),
        )
        for payload in cases:
            with self.subTest(payload=payload[:40]):
                with self.assertRaises(scheduler.SchedulerFixtureError):
                    scheduler.parse_document_bytes(payload)

    def test_cli_write_check_and_read_only_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "actor-stress-v1.json"
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                self.assertEqual(scheduler.main(["--write", "--path", str(path)]), 0)
                self.assertEqual(scheduler.main(["--check", "--path", str(path)]), 0)
            self.assertEqual(stderr.getvalue(), "")
            self.assertIn(f"sha256:{EXPECTED_FILE_SHA256}", stdout.getvalue())
            self.assertEqual(path.read_bytes(), scheduler.generated_bytes())

            path.write_bytes(b"{}\n")
            before = path.read_bytes()
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                self.assertEqual(scheduler.main(["--check", "--path", str(path)]), 1)
            self.assertEqual(stdout.getvalue(), "")
            self.assertTrue(stderr.getvalue().startswith("scheduler oracle:"))
            self.assertEqual(path.read_bytes(), before)

    def test_check_path_enforces_bounded_regular_file_custody(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            oversized = root / "oversized.json"
            oversized.write_bytes(b"x" * (scheduler.MAX_FIXTURE_BYTES + 1))
            with self.assertRaisesRegex(
                scheduler.SchedulerFixtureError, "byte limit"
            ):
                scheduler.check_path(oversized)

            target = root / "target.json"
            target.write_bytes(scheduler.generated_bytes())
            link = root / "link.json"
            link.symlink_to(target)
            with self.assertRaisesRegex(
                scheduler.SchedulerFixtureError, "without following links"
            ):
                scheduler.check_path(link)

    def test_invalid_public_generation_inputs_fail_closed(self) -> None:
        for index in (-1, scheduler.MAX_U32 + 1, True):
            with self.subTest(index=index):
                with self.assertRaises(scheduler.SchedulerFixtureError):
                    scheduler.request_descriptor(index)
        with self.assertRaises(scheduler.SchedulerFixtureError):
            scheduler.build_actions(-1)
        with self.assertRaises(scheduler.SchedulerFixtureError):
            scheduler.build_actions(1, words=())


if __name__ == "__main__":
    unittest.main()
