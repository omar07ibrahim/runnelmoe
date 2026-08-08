"""Tests for the candidate visual-evidence capture and renderer."""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import sys
import tempfile
import tomllib
import unittest
import xml.etree.ElementTree as ET

from scripts import generate_visual_evidence as evidence


ROOT = pathlib.Path(__file__).resolve().parents[2]
PIL_AVAILABLE = evidence.Image is not None
REQUIRE_PILLOW = os.environ.get("RUNNEL_REQUIRE_VISUAL_DEPS") == "1"
CANDIDATE = os.environ.get("RUNNEL_VISUAL_EVIDENCE_CANDIDATE")


class CommandContractTests(unittest.TestCase):
    def test_command_vectors_are_the_documented_m1_m4_sequence(self) -> None:
        self.assertEqual(
            [item.capture_id for item in evidence.COMMANDS],
            [
                "m1-demo",
                "m2-data-plane",
                "m3-cache-matrix",
                "m4-model-check",
            ],
        )
        self.assertEqual(
            evidence.COMMANDS[0].argv,
            (
                "cargo", "run", "--locked", "-p", "runnel", "--", "demo",
                "--prompt", "moe", "--max-new-tokens", "4", "--json",
            ),
        )
        self.assertEqual(
            evidence.COMMANDS[2].argv[-7:],
            (
                "matrix", "--family", "markov_clusters", "--replicate", "0",
                "--measured-steps", "64",
            ),
        )
        for command in evidence.COMMANDS:
            self.assertNotIn("sh", command.argv)
            self.assertNotIn("bash", command.argv)

    def test_cli_package_selects_the_documented_default_binary(self) -> None:
        manifest = tomllib.loads(
            (ROOT / "crates" / "runnel-cli" / "Cargo.toml").read_text()
        )
        self.assertEqual(manifest["package"]["default-run"], "runnel")
        self.assertEqual(
            {target["name"] for target in manifest["bin"]},
            {"runnel", "runnel-m4-model-check"},
        )

    def test_strict_json_rejects_duplicate_and_nonfinite_values(self) -> None:
        with self.assertRaises(evidence.ContractError):
            evidence.strict_json(b'{"a":1,"a":2}\n', "duplicate")
        with self.assertRaises(evidence.ContractError):
            evidence.strict_json(b'{"a":NaN}\n', "nonfinite")

    def test_secret_and_private_path_markers_are_rejected(self) -> None:
        samples = [
            b"/home/" + b"runner/work/project",
            b"gh" + b"o_example",
            b"AK" + b"IAEXAMPLE",
            b"-----BEGIN " + b"PRIVATE KEY-----",
        ]
        for index, sample in enumerate(samples):
            with self.subTest(index=index):
                with self.assertRaises(evidence.ContractError):
                    evidence.scan(sample, f"sample-{index}")

    def test_email_like_pii_is_rejected_without_a_published_literal(self) -> None:
        sample = b"portfolio.owner" + b"@" + b"example" + b".invalid"
        with self.assertRaises(evidence.ContractError):
            evidence.scan(sample, "email-like")

    def test_failed_command_does_not_echo_captured_stderr(self) -> None:
        environment = {"PATH": os.environ["PATH"], "LANG": "C.UTF-8"}
        with self.assertRaises(evidence.ContractError) as caught:
            evidence.bounded_run(
                (
                    sys.executable,
                    "-c",
                    "import sys;sys.stderr.write('sensitive details');raise SystemExit(7)",
                ),
                ROOT,
                environment,
                "failing-test",
            )
        self.assertEqual(str(caught.exception), "failing-test exited 7")
        self.assertNotIn("sensitive", str(caught.exception))

    def test_capture_size_is_bounded(self) -> None:
        oversized = b"x" * (evidence.MAX_STREAM + 1)
        with self.assertRaises(evidence.ContractError):
            evidence.strict_json(oversized, "oversized")

    def test_stable_demo_parser_accepts_only_real_vector(self) -> None:
        valid = {
            "schema_version": 1,
            "artifact_id": "sha256:" + "a" * 64,
            "adapter": {"id": "runnel.tiny-causal-moe", "version": 1},
            "input_token_count": 3,
            "generated_ids": [15, 11, 20, 9],
            "text": "njsh",
            "stop_reason": "max_new_tokens",
        }
        encoded = (json.dumps(valid, separators=(",", ":")) + "\n").encode()
        self.assertEqual(evidence.parse(evidence.COMMANDS[0], encoded), valid)
        valid["generated_ids"] = [0]
        with self.assertRaises(evidence.ContractError):
            evidence.parse(
                evidence.COMMANDS[0],
                (json.dumps(valid, separators=(",", ":")) + "\n").encode(),
            )

    def test_derived_workflow_text_ignores_unallowlisted_extra_fields(self) -> None:
        sentinel = "UNALLOWLISTED_SENTINEL"
        parsed = {
            "m1-demo": {
                "generated_ids": [15, 11, 20, 9],
                "text": "njsh",
                "stop_reason": "max_new_tokens",
                "extra": sentinel,
            },
            "m2-data-plane": {
                "parity": True,
                "forced_eviction_generation": {
                    "full_generation_parity": True
                },
                "metrics": {"physical_read_bytes": 64},
                "extra": sentinel,
            },
            "m3-cache-matrix": {
                "results": [
                    {
                        "capacity_bytes": 2 * 1024 * 1024,
                        "metrics": {"total_physical_load_bytes": 128},
                        "extra": sentinel,
                    }
                ]
            },
            "m4-model-check": [
                {
                    "check_id": "tiny-v1-preservation",
                    "status": "ok",
                    "backend": "v1-f32",
                    "extra": sentinel,
                }
            ],
        }
        rendered = json.dumps(evidence.summaries(parsed), sort_keys=True)
        self.assertNotIn(sentinel, rendered)

    def test_small_matrix_parser_checks_closed_grid_and_accounting(self) -> None:
        policies = (
            "lru", "slru", "tiny-lfu", "router-admit",
            "router-prefetch", "belady",
        )
        results = []
        for capacity in (2, 4, 8):
            for index, policy in enumerate(policies):
                results.append(
                    {
                        "schema": "runnel.cache-result/1",
                        "trace_sha256": "b" * 64,
                        "policy": policy,
                        "capacity_bytes": capacity * 1024 * 1024,
                        "metrics": {
                            "demand_load_bytes": 100 + index,
                            "prefetch_load_bytes": index,
                            "total_physical_load_bytes": 100 + index * 2,
                        },
                        "decision_sha256": "c" * 64,
                    }
                )
        value = {
            "schema": "runnel.cache-matrix/1",
            "family": "markov_clusters",
            "replicate": 0,
            "measured_steps": 64,
            "trace_sha256": "b" * 64,
            "results": results,
        }
        encoded = (json.dumps(value, separators=(",", ":")) + "\n").encode()
        self.assertEqual(evidence.parse(evidence.COMMANDS[2], encoded), value)
        results.pop()
        value["results"] = results
        with self.assertRaises(evidence.ContractError):
            evidence.parse(
                evidence.COMMANDS[2],
                (json.dumps(value, separators=(",", ":")) + "\n").encode(),
            )


class VectorAndArchitectureTests(unittest.TestCase):
    def test_svg_has_accessible_bounded_standalone_frame(self) -> None:
        payload, canvas, bounds = evidence.svg_card(
            "Title",
            "M1 / TEST",
            "Subtitle",
            [("Evidence", ["actual value 1", "actual value 2"])],
            "No performance claim.",
            "a" * 40,
            "b" * 64,
        )
        root = ET.fromstring(payload)
        self.assertEqual(root.tag, f"{{{evidence.SVG_NS}}}svg")
        self.assertEqual(root.attrib["viewBox"], f"0 0 {canvas[0]} {canvas[1]}")
        self.assertGreaterEqual(bounds[0], evidence.MARGIN)
        self.assertGreaterEqual(bounds[1], evidence.MARGIN)
        self.assertLessEqual(bounds[2], canvas[0] - evidence.MARGIN)
        self.assertLessEqual(bounds[3], canvas[1] - evidence.MARGIN)
        names = {element.tag.rsplit("}", 1)[-1] for element in root.iter()}
        self.assertTrue({"title", "desc", "metadata"} <= names)
        self.assertFalse(
            names & {
                "script", "foreignObject", "image", "use", "animate",
                "set", "clipPath", "mask",
            }
        )

    def test_architecture_is_actual_crate_graph_without_scheduler(self) -> None:
        nodes, edges, sources = evidence.crate_graph(ROOT)
        names = {node["name"] for node in nodes}
        self.assertEqual(
            names,
            {
                "runnel",
                "runnel-fixture",
                "runnel-format",
                "runnel-kernels",
                "runnel-runtime",
                "runnel-sim",
                "runnel-store",
            },
        )
        self.assertNotIn("scheduler", " ".join(sorted(names)).lower())
        self.assertEqual(len(sources), 8)
        self.assertIn(("runnel", "runnel-runtime"), edges)
        self.assertTrue(all(evidence.SHA256.fullmatch(row["sha256"]) for row in sources))

    def test_architecture_caption_explicitly_excludes_m5(self) -> None:
        payload, canvas, bounds, nodes, _edges, sources = (
            evidence.architecture_svg(ROOT, "a" * 40)
        )
        text = payload.decode()
        self.assertIn("No scheduler node or M5 result is inferred.", text)
        self.assertEqual(len(nodes), 7)
        self.assertEqual(len(sources), 8)
        self.assertLessEqual(canvas[0], evidence.MAX_WIDTH)
        self.assertLessEqual(canvas[1], evidence.MAX_HEIGHT)
        self.assertGreaterEqual(bounds[0], evidence.MARGIN)


class RasterTests(unittest.TestCase):
    def test_pillow_is_present_when_hosted_contract_requires_it(self) -> None:
        if REQUIRE_PILLOW:
            self.assertTrue(PIL_AVAILABLE)
            self.assertEqual(
                evidence.importlib.metadata.version("Pillow"),
                evidence.EXPECTED_PILLOW,
            )

    @unittest.skipUnless(PIL_AVAILABLE, "locked visual dependency is optional in ordinary CI")
    def test_transcript_png_is_actual_stdout_and_full_frame(self) -> None:
        output = {
            "schema_version": 1,
            "artifact_id": "sha256:" + "a" * 64,
            "adapter": {"id": "runnel.tiny-causal-moe", "version": 1},
            "input_token_count": 3,
            "generated_ids": [15, 11, 20, 9],
            "text": "njsh",
            "stop_reason": "max_new_tokens",
        }
        stdout = (json.dumps(output, separators=(",", ":")) + "\n").encode()
        source = evidence.digest(stdout)
        temp_root = os.environ.get("RUNNER_TEMP")
        with tempfile.TemporaryDirectory(dir=temp_root) as temporary:
            target = pathlib.Path(temporary) / "transcript.png"
            canvas, bounds = evidence.transcript_png(
                evidence.COMMANDS[0], stdout, "a" * 40, source, target
            )
            with evidence.Image.open(target) as image:
                self.assertEqual(image.format, "PNG")
                self.assertEqual(list(image.size), canvas)
                self.assertEqual(image.info["SourceStdoutSHA256"], source)
                self.assertIn("not an OS screenshot", image.info["RenderContract"])
                self.assertTrue(evidence.border_clear(image))
            self.assertGreaterEqual(bounds[0], evidence.MARGIN)
            self.assertLessEqual(bounds[2], canvas[0] - evidence.MARGIN)


@unittest.skipUnless(CANDIDATE, "hosted candidate artifact not supplied")
class HostedCandidateTests(unittest.TestCase):
    def test_candidate_passes_closed_verifier(self) -> None:
        result = evidence.verify(pathlib.Path(CANDIDATE))
        self.assertEqual(result["capture_count"], 4)
        self.assertEqual(result["asset_count"], 7)
        self.assertTrue(evidence.SHA256.fullmatch(result["manifest_sha256"]))

    def test_gif_has_four_exact_full_opaque_frames(self) -> None:
        gif = pathlib.Path(CANDIDATE) / "visuals" / "m1-m4-workflow.gif"
        with evidence.Image.open(gif) as image:
            self.assertEqual(image.info.get("loop"), 0)
            self.assertNotIn("transparency", image.info)
            self.assertEqual(image.n_frames, 4)
            for index, duration in enumerate((1400, 1400, 1400, 1600)):
                image.seek(index)
                tiles = list(image.tile)
                self.assertEqual(len(tiles), 1)
                self.assertEqual(tiles[0][1], (0, 0, *image.size))
                self.assertEqual(image.info.get("duration"), duration)
                self.assertEqual(image.disposal_method, 2)
                self.assertNotIn("transparency", image.info)
                self.assertTrue(evidence.border_clear(image))

    def test_raw_tamper_is_rejected(self) -> None:
        temp_root = os.environ.get("RUNNER_TEMP")
        with tempfile.TemporaryDirectory(dir=temp_root) as temporary:
            clone = pathlib.Path(temporary) / "candidate"
            shutil.copytree(CANDIDATE, clone)
            with (clone / "raw" / "m1-demo.stdout").open("ab") as output:
                output.write(b" ")
            with self.assertRaises(evidence.ContractError):
                evidence.verify(clone)

    def test_supplied_root_symlink_is_rejected(self) -> None:
        if not hasattr(os, "symlink"):
            self.skipTest("symlinks unavailable")
        temp_root = os.environ.get("RUNNER_TEMP")
        with tempfile.TemporaryDirectory(dir=temp_root) as temporary:
            link = pathlib.Path(temporary) / "candidate-link"
            os.symlink(pathlib.Path(CANDIDATE).resolve(), link, target_is_directory=True)
            with self.assertRaises(evidence.ContractError):
                evidence.verify(link)

    def test_unexpected_directory_is_rejected(self) -> None:
        temp_root = os.environ.get("RUNNER_TEMP")
        with tempfile.TemporaryDirectory(dir=temp_root) as temporary:
            clone = pathlib.Path(temporary) / "candidate"
            shutil.copytree(CANDIDATE, clone)
            (clone / "unexpected-directory").mkdir()
            with self.assertRaises(evidence.ContractError):
                evidence.verify(clone)

    def test_nonregular_entry_is_rejected(self) -> None:
        if not hasattr(os, "mkfifo"):
            self.skipTest("FIFOs unavailable")
        temp_root = os.environ.get("RUNNER_TEMP")
        with tempfile.TemporaryDirectory(dir=temp_root) as temporary:
            clone = pathlib.Path(temporary) / "candidate"
            shutil.copytree(CANDIDATE, clone)
            os.mkfifo(clone / "unexpected-pipe")
            with self.assertRaises(evidence.ContractError):
                evidence.verify(clone)

    def test_unlisted_symlink_is_rejected(self) -> None:
        if not hasattr(os, "symlink"):
            self.skipTest("symlinks unavailable")
        temp_root = os.environ.get("RUNNER_TEMP")
        with tempfile.TemporaryDirectory(dir=temp_root) as temporary:
            clone = pathlib.Path(temporary) / "candidate"
            shutil.copytree(CANDIDATE, clone)
            os.symlink("README.md", clone / "unexpected-link")
            with self.assertRaises(evidence.ContractError):
                evidence.verify(clone)


if __name__ == "__main__":
    unittest.main()
