"""Regression tests for repository binary/text classification."""

from pathlib import Path
import unittest

from scripts import verify_repository as contract


PNG = Path(
    "docs/visual-evidence/m1-m4-0aca4a7/visuals/m1-demo-transcript.png"
)
GIF = Path(
    "docs/visual-evidence/m1-m4-0aca4a7/visuals/m1-m4-workflow.gif"
)


class SourceTextTests(unittest.TestCase):
    def classify(self, path: Path, data: bytes) -> tuple[str | None, list[str]]:
        failures: list[str] = []
        text = contract.source_text(path, data, failures)
        return text, failures

    def test_only_expected_binary_paths_are_allowlisted(self) -> None:
        self.assertEqual(set(contract.APPROVED_BINARY_FILES), {PNG, GIF})

    def test_approved_png_and_gif_magic_are_accepted(self) -> None:
        for path, data in (
            (PNG, b"\x89PNG\r\n\x1a\n\x00\xff"),
            (GIF, b"GIF89a\x00\xff"),
        ):
            with self.subTest(path=path):
                text, failures = self.classify(path, data)
                self.assertIsNone(text)
                self.assertEqual(failures, [])

    def test_approved_path_with_mismatched_magic_is_rejected(self) -> None:
        text, failures = self.classify(PNG, b"GIF89a\x00\xff")
        self.assertIsNone(text)
        self.assertEqual(
            failures,
            [f"{PNG}: approved binary suffix or magic does not match"],
        )

    def test_binary_suffix_at_unknown_path_is_rejected(self) -> None:
        path = Path("docs/visual-evidence/rogue.png")
        text, failures = self.classify(
            path, b"\x89PNG\r\n\x1a\n\x00\xff"
        )
        self.assertIsNone(text)
        self.assertEqual(
            failures, [f"{path}: binary asset path is not allowlisted"]
        )

    def test_malformed_utf8_text_is_rejected(self) -> None:
        path = Path("docs/malformed.md")
        text, failures = self.classify(path, b"# heading\n\xff")
        self.assertIsNone(text)
        self.assertEqual(
            failures,
            [f"{path}: malformed UTF-8 is not an allowlisted binary"],
        )

    def test_utf8_text_still_requires_final_newline(self) -> None:
        path = Path("docs/no-newline.md")
        text, failures = self.classify(path, b"# heading")
        self.assertEqual(text, "# heading")
        self.assertEqual(failures, [f"{path}: missing final newline"])


class MarkdownLinkTests(unittest.TestCase):
    def test_image_and_text_destinations_are_parsed(self) -> None:
        text = "![result](visual.svg) and [evidence](raw.json)"
        self.assertEqual(
            contract.MARKDOWN_LINK.findall(text),
            ["visual.svg", "raw.json"],
        )


if __name__ == "__main__":
    unittest.main()
