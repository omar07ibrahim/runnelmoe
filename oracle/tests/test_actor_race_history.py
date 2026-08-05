from __future__ import annotations

import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

from oracle import actor_race_history


class CanonicalCursorTests(unittest.TestCase):
    def parse_pair(self, raw: bytes) -> tuple[int, str]:
        cursor = actor_race_history.CanonicalCursor(raw)
        cursor.begin_object()
        cursor.object_key("a")
        number = cursor.parse_u64("a")
        cursor.object_key("b")
        spelling = cursor.parse_raw_string("b", {"ok", "with space"})
        cursor.end_object()
        cursor.finish()
        return number, spelling

    def parse_two_numbers(self, raw: bytes) -> tuple[int, int]:
        cursor = actor_race_history.CanonicalCursor(raw)
        cursor.begin_object()
        cursor.object_key("a")
        first = cursor.parse_u64("a")
        cursor.object_key("b")
        second = cursor.parse_u64("b")
        cursor.end_object()
        cursor.finish()
        return first, second

    def test_accepts_exact_compact_ascii_primitives(self) -> None:
        maximum = actor_race_history.MAX_U64
        self.assertEqual(
            self.parse_pair(f'{{"a":{maximum},"b":"with space"}}\n'.encode()),
            (maximum, "with space"),
        )

    def test_requires_exact_single_lf_object_framing(self) -> None:
        invalid = (
            b"",
            b'{}',
            b'{}\n\n',
            b'{}\r\n',
            b'\xef\xbb\xbf{}\n',
            b'[]\n',
            b'{}{}\n',
            b'{}x\n',
        )
        for raw in invalid:
            with self.subTest(raw=raw):
                with self.assertRaises(actor_race_history.ActorRaceHistoryError):
                    actor_race_history.CanonicalCursor(raw)

    def test_rejects_whitespace_outside_strings(self) -> None:
        for raw in (
            b'{ "a":1,"b":"ok"}\n',
            b'{"a": 1,"b":"ok"}\n',
            b'{"a":1,\t"b":"ok"}\n',
            b'{"a":1,"b":"ok" }\n',
        ):
            with self.subTest(raw=raw):
                with self.assertRaises(actor_race_history.ActorRaceHistoryError):
                    self.parse_pair(raw)

    def test_rejects_escape_spellings_and_non_ascii_strings(self) -> None:
        for raw in (
            b'{"a":1,"b":"with\\/escape"}\n',
            b'{"a":1,"b":"with\\u0020escape"}\n',
            '{"a":1,"b":"café"}\n'.encode(),
            b'{"a":1,"b":"unterminated}\n',
        ):
            with self.subTest(raw=raw):
                with self.assertRaises(actor_race_history.ActorRaceHistoryError):
                    self.parse_pair(raw)

    def test_enforces_raw_string_bound_and_closed_choices(self) -> None:
        accepted = b'a' * actor_race_history.MAX_STRING_BYTES
        cursor = actor_race_history.CanonicalCursor(
            b'{"a":0,"b":"' + accepted + b'"}\n'
        )
        cursor.begin_object()
        cursor.object_key("a")
        self.assertEqual(cursor.parse_u64("a"), 0)
        cursor.object_key("b")
        self.assertEqual(cursor.parse_raw_string("b"), accepted.decode())
        cursor.end_object()
        cursor.finish()

        too_long = accepted + b'a'
        with self.assertRaises(actor_race_history.ActorRaceHistoryError):
            actor_race_history.CanonicalCursor(
                b'{"a":0,"b":"' + too_long + b'"}\n'
            )
        with self.assertRaises(actor_race_history.ActorRaceHistoryError):
            self.parse_pair(b'{"a":0,"b":"unknown"}\n')

    def test_rejects_noncanonical_and_out_of_range_u64_tokens(self) -> None:
        invalid_tokens = (
            b"-1",
            b"+1",
            b"01",
            b"1.0",
            b"1e0",
            str(actor_race_history.MAX_U64 + 1).encode(),
            b"999999999999999999999",
        )
        for token in invalid_tokens:
            raw = b'{"a":' + token + b',"b":"ok"}\n'
            with self.subTest(token=token):
                with self.assertRaises(actor_race_history.ActorRaceHistoryError):
                    self.parse_pair(raw)
        self.assertEqual(self.parse_pair(b'{"a":0,"b":"ok"}\n')[0], 0)

    def test_enforces_exact_punctuation_unique_sorted_keys(self) -> None:
        self.assertEqual(self.parse_two_numbers(b'{"a":1,"b":2}\n'), (1, 2))
        invalid = (
            b'{"a":1,"a":2}\n',
            b'{"b":2,"a":1}\n',
            b'{"a":1,"c":2}\n',
            b'{"a":1"b":2}\n',
            b'{"a"1,"b":2}\n',
            b'{"a":1,"b":2,"c":3}\n',
        )
        for raw in invalid:
            with self.subTest(raw=raw):
                with self.assertRaises(actor_race_history.ActorRaceHistoryError):
                    self.parse_two_numbers(raw)

    def test_enforces_array_item_punctuation(self) -> None:
        def parse_array(raw: bytes, count: int) -> list[int]:
            cursor = actor_race_history.CanonicalCursor(raw)
            cursor.begin_object()
            cursor.object_key("a")
            cursor.begin_array()
            values = []
            for _ in range(count):
                cursor.array_item()
                values.append(cursor.parse_u64("array item"))
            cursor.end_array()
            cursor.end_object()
            cursor.finish()
            return values

        self.assertEqual(parse_array(b'{"a":[1,2]}\n', 2), [1, 2])
        for raw, count in (
            (b'{"a":[1 2]}\n', 2),
            (b'{"a":[1,]}\n', 1),
            (b'{"a":[,1]}\n', 1),
            (b'{"a":[1,,2]}\n', 2),
        ):
            with self.subTest(raw=raw):
                with self.assertRaises(actor_race_history.ActorRaceHistoryError):
                    parse_array(raw, count)

    def test_enforces_nesting_depth_at_exact_boundary(self) -> None:
        allowed_arrays = actor_race_history.MAX_NESTING_DEPTH - 1
        valid = b'{"a":' + b"[" * allowed_arrays + b"0" + b"]" * allowed_arrays + b'}\n'
        actor_race_history.CanonicalCursor(valid)

        excessive_arrays = actor_race_history.MAX_NESTING_DEPTH
        invalid = (
            b'{"a":'
            + b"[" * excessive_arrays
            + b"0"
            + b"]" * excessive_arrays
            + b'}\n'
        )
        with self.assertRaises(actor_race_history.ActorRaceHistoryError):
            actor_race_history.CanonicalCursor(invalid)

    def test_boolean_and_null_tokens_are_closed(self) -> None:
        cursor = actor_race_history.CanonicalCursor(b'{"a":false,"b":null}\n')
        cursor.begin_object()
        cursor.object_key("a")
        self.assertFalse(cursor.parse_bool("a"))
        cursor.object_key("b")
        self.assertIsNone(cursor.parse_null("b"))
        cursor.end_object()
        cursor.finish()

        for raw in (b'{"a":falsex}\n', b'{"a":nullx}\n'):
            with self.subTest(raw=raw):
                cursor = actor_race_history.CanonicalCursor(raw)
                cursor.begin_object()
                cursor.object_key("a")
                with self.assertRaises(actor_race_history.ActorRaceHistoryError):
                    if b"false" in raw:
                        cursor.parse_bool("a")
                    else:
                        cursor.parse_null("a")

    def test_cursor_requires_exactly_one_value_per_requested_position(self) -> None:
        missing = actor_race_history.CanonicalCursor(b'{"a":}\n')
        missing.begin_object()
        missing.object_key("a")
        with self.assertRaises(actor_race_history.ActorRaceHistoryError):
            missing.end_object()

        adjacent = actor_race_history.CanonicalCursor(b'{"a":[{}{}]}\n')
        adjacent.begin_object()
        adjacent.object_key("a")
        adjacent.begin_array()
        adjacent.array_item()
        adjacent.begin_object()
        adjacent.end_object()
        with self.assertRaises(actor_race_history.ActorRaceHistoryError):
            adjacent.begin_object()

    def test_raw_string_choices_reject_substring_collections(self) -> None:
        cursor = actor_race_history.CanonicalCursor(b'{"a":"completed"}\n')
        cursor.begin_object()
        cursor.object_key("a")
        with self.assertRaises(actor_race_history.ActorRaceHistoryError):
            cursor.parse_raw_string("a", "completedcancelled")


class SafeCaptureReaderTests(unittest.TestCase):
    def test_reads_a_stable_regular_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.json"
            path.write_bytes(b'{}\n')
            self.assertEqual(actor_race_history.read_capture_bytes(path), b'{}\n')

    def test_rejects_symlink_directory_and_fifo(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "capture.json"
            target.write_bytes(b'{}\n')
            symlink = root / "capture-link.json"
            symlink.symlink_to(target)
            with self.assertRaises(actor_race_history.ActorRaceHistoryError):
                actor_race_history.read_capture_bytes(symlink)
            with self.assertRaises(actor_race_history.ActorRaceHistoryError):
                actor_race_history.read_capture_bytes(root)

            fifo = root / "capture.fifo"
            os.mkfifo(fifo)
            with self.assertRaises(actor_race_history.ActorRaceHistoryError):
                actor_race_history.read_capture_bytes(fifo)

    def test_rejects_sparse_file_over_byte_cap_without_reading_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "oversize.json"
            with path.open("wb") as stream:
                stream.truncate(actor_race_history.MAX_CAPTURE_BYTES + 1)
            with self.assertRaises(actor_race_history.ActorRaceHistoryError):
                actor_race_history.read_capture_bytes(path)

    def test_rejects_pre_and_post_read_identity_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.json"
            path.write_bytes(b'{}\n')
            metadata = path.stat()
            changed = types.SimpleNamespace(
                st_dev=metadata.st_dev,
                st_ino=metadata.st_ino,
                st_mode=metadata.st_mode,
                st_size=metadata.st_size,
                st_mtime_ns=metadata.st_mtime_ns,
                st_ctime_ns=metadata.st_ctime_ns + 1,
            )
            with mock.patch.object(
                actor_race_history.os,
                "fstat",
                side_effect=(metadata, changed),
            ):
                with self.assertRaises(actor_race_history.ActorRaceHistoryError):
                    actor_race_history.read_capture_bytes(path)

    def test_reader_rejects_noncanonical_file_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.json"
            path.write_bytes(b'{ "a":1}\n')
            with self.assertRaises(actor_race_history.ActorRaceHistoryError):
                actor_race_history.read_capture_bytes(path)


if __name__ == "__main__":
    unittest.main()
