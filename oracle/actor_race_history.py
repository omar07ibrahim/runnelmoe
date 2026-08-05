#!/usr/bin/env python3
"""Bounded Linux canonical-byte primitives for actor race-history captures.

This module deliberately stops below the semantic race model.  It provides a
fail-closed file reader and a small cursor for the capture's compact,
closed-schema JSON subset.  A later semantic checker can therefore consume
bytes directly without first handing attacker-controlled input to a general
JSON decoder.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
import os
from pathlib import Path
import stat
from typing import NoReturn


MAX_CAPTURE_BYTES = 32 * 1024 * 1024
MAX_NESTING_DEPTH = 16
MAX_STRING_BYTES = 128
MAX_U64 = (1 << 64) - 1
_READ_CHUNK_BYTES = 64 * 1024


class ActorRaceHistoryError(ValueError):
    """A custody, framing, or canonical-token failure."""


def _fail(message: str) -> NoReturn:
    raise ActorRaceHistoryError(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _stat_identity(metadata: os.stat_result) -> tuple[int, ...]:
    """Return the metadata fields that must remain stable during a read."""

    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def read_capture_bytes(path: str | Path) -> bytes:
    """Read and frame-check one bounded, immutable, regular-file capture.

    The descriptor is opened nonblocking without following a leaf symlink.
    The reader checks descriptor identity and size before and after the
    bounded read so descriptor-visible mutation is rejected.  An already-open
    descriptor deliberately remains authoritative if its pathname is later
    replaced.
    """

    required_flags = ("O_NONBLOCK", "O_CLOEXEC", "O_NOFOLLOW")
    _require(
        all(hasattr(os, name) for name in required_flags),
        "safe capture reads require Linux open flags",
    )
    flags = os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(os.fspath(path), flags)
    except OSError as error:
        _fail(f"cannot safely open race-history capture: {error}")

    try:
        try:
            before = os.fstat(descriptor)
        except OSError as error:
            _fail(f"cannot inspect race-history capture: {error}")
        _require(stat.S_ISREG(before.st_mode), "capture must be a regular file")
        _require(before.st_size > 0, "capture must not be empty")
        _require(
            before.st_size <= MAX_CAPTURE_BYTES,
            "capture exceeds the 32 MiB byte limit",
        )

        chunks: list[bytes] = []
        bytes_read = 0
        while True:
            remaining_probe = MAX_CAPTURE_BYTES + 1 - bytes_read
            if remaining_probe <= 0:
                _fail("capture exceeds the 32 MiB byte limit")
            try:
                chunk = os.read(
                    descriptor,
                    min(_READ_CHUNK_BYTES, remaining_probe),
                )
            except OSError as error:
                _fail(f"cannot read race-history capture: {error}")
            if not chunk:
                break
            chunks.append(chunk)
            bytes_read += len(chunk)
            if bytes_read > MAX_CAPTURE_BYTES:
                _fail("capture exceeds the 32 MiB byte limit")

        try:
            after = os.fstat(descriptor)
        except OSError as error:
            _fail(f"cannot re-inspect race-history capture: {error}")
        _require(
            _stat_identity(before) == _stat_identity(after),
            "capture changed while it was being read",
        )
        _require(
            bytes_read == after.st_size,
            "capture byte count does not match its file size",
        )
    finally:
        os.close(descriptor)

    raw = b"".join(chunks)
    CanonicalCursor(raw)
    return raw


@dataclass
class _Frame:
    kind: str
    item_count: int = 0
    last_key: bytes | None = None
    value_pending: bool = False


class CanonicalCursor:
    """Cursor over the actor capture's exact compact ASCII JSON subset."""

    def __init__(self, raw: bytes):
        _require(type(raw) is bytes, "capture must be supplied as exact bytes")
        self._validate_frame(raw)
        self.raw = raw
        self.position = 0
        self._frames: list[_Frame] = []
        self._root_started = False
        self._root_complete = False

    @staticmethod
    def _validate_frame(raw: bytes) -> None:
        _require(raw, "capture must not be empty")
        _require(
            len(raw) <= MAX_CAPTURE_BYTES,
            "capture exceeds the 32 MiB byte limit",
        )
        _require(raw.isascii(), "capture must contain only ASCII bytes")
        _require(raw.endswith(b"\n"), "capture must end with one LF byte")
        _require(
            b"\n" not in raw[:-1] and b"\r" not in raw,
            "capture must contain exactly one final LF and no CR bytes",
        )

        payload = raw[:-1]
        _require(
            payload.startswith(b"{") and payload.endswith(b"}"),
            "capture root must be an object",
        )

        stack: list[int] = []
        in_string = False
        string_bytes = 0
        pairs = {ord("}"): ord("{"), ord("]"): ord("[")}
        for index, byte in enumerate(payload):
            if in_string:
                if byte == ord('"'):
                    in_string = False
                    continue
                _require(byte != ord("\\"), "escape spellings are forbidden")
                _require(
                    0x20 <= byte <= 0x7E,
                    "string contains a non-printable ASCII byte",
                )
                string_bytes += 1
                _require(
                    string_bytes <= MAX_STRING_BYTES,
                    "string exceeds the 128-byte limit",
                )
                continue

            if byte == ord('"'):
                in_string = True
                string_bytes = 0
            elif byte in (ord("{"), ord("[")):
                stack.append(byte)
                _require(
                    len(stack) <= MAX_NESTING_DEPTH,
                    "capture exceeds the nesting-depth limit",
                )
            elif byte in pairs:
                _require(
                    bool(stack) and stack[-1] == pairs[byte],
                    "capture has mismatched structural punctuation",
                )
                stack.pop()
                _require(
                    bool(stack) or index == len(payload) - 1,
                    "capture contains bytes after its root object",
                )
            else:
                _require(byte != ord("\\"), "escape spellings are forbidden")
                _require(
                    0x21 <= byte <= 0x7E,
                    "whitespace and control bytes are forbidden outside strings",
                )

        _require(not in_string, "capture contains an unclosed string")
        _require(not stack, "capture contains unclosed structural punctuation")

    def _peek(self) -> int:
        _require(self.position < len(self.raw) - 1, "unexpected end of capture")
        return self.raw[self.position]

    def _expect_byte(self, expected: int, label: str) -> None:
        actual = self._peek()
        _require(actual == expected, f"expected {label} at byte {self.position}")
        self.position += 1

    def _push(self, kind: str) -> None:
        _require(
            len(self._frames) < MAX_NESTING_DEPTH,
            "capture exceeds the nesting-depth limit",
        )
        self._frames.append(_Frame(kind))

    def _begin_value(self) -> None:
        if not self._frames:
            _require(not self._root_started, "capture contains a second root value")
            self._root_started = True
            return
        frame = self._frames[-1]
        _require(frame.value_pending, f"{frame.kind} value was not requested")
        frame.value_pending = False

    def _finish_container(self) -> None:
        self._frames.pop()
        if not self._frames:
            self._root_complete = True

    def _top(self, kind: str) -> _Frame:
        _require(
            bool(self._frames) and self._frames[-1].kind == kind,
            f"cursor is not inside an {kind}",
        )
        return self._frames[-1]

    def _require_value_delimiter(self, label: str) -> None:
        _require(self.position < len(self.raw) - 1, f"{label} is not closed")
        _require(
            self.raw[self.position] in b",}]",
            f"{label} has a non-canonical token suffix",
        )

    def _parse_string_literal(self, label: str) -> tuple[str, bytes]:
        self._expect_byte(ord('"'), f'an opening quote for {label}')
        start = self.position
        while True:
            _require(self.position < len(self.raw) - 1, f"{label} is unclosed")
            byte = self.raw[self.position]
            if byte == ord('"'):
                value_bytes = self.raw[start : self.position]
                self.position += 1
                return value_bytes.decode("ascii"), value_bytes
            _require(byte != ord("\\"), f"{label} may not contain escapes")
            _require(
                0x20 <= byte <= 0x7E,
                f"{label} contains a non-printable ASCII byte",
            )
            _require(
                self.position - start < MAX_STRING_BYTES,
                f"{label} exceeds the 128-byte limit",
            )
            self.position += 1

    def begin_object(self) -> None:
        self._begin_value()
        self._expect_byte(ord("{"), "an object opening brace")
        self._push("object")

    def read_object_key(self) -> str:
        """Read the next colon-terminated key and enforce strict key order."""

        frame = self._top("object")
        _require(not frame.value_pending, "object key is missing its value")
        if frame.item_count:
            self._expect_byte(ord(","), "a comma between object members")
        key, key_bytes = self._parse_string_literal("object key")
        if frame.last_key is not None:
            _require(
                key_bytes > frame.last_key,
                "object keys must be unique and strictly ASCII-sorted",
            )
        self._expect_byte(ord(":"), "a colon after an object key")
        frame.item_count += 1
        frame.last_key = key_bytes
        frame.value_pending = True
        return key

    def object_key(self, expected: str) -> None:
        """Read one exact key in the caller's closed-schema field order."""

        try:
            expected_bytes = expected.encode("ascii")
        except UnicodeEncodeError:
            _fail("expected object key must be ASCII")
        _require(
            len(expected_bytes) <= MAX_STRING_BYTES,
            "expected object key exceeds the 128-byte limit",
        )
        actual = self.read_object_key()
        _require(actual == expected, f"expected object key {expected!r}")

    def end_object(self) -> None:
        frame = self._top("object")
        _require(not frame.value_pending, "object key is missing its value")
        self._expect_byte(ord("}"), "an object closing brace")
        self._finish_container()

    def begin_array(self) -> None:
        self._begin_value()
        self._expect_byte(ord("["), "an array opening bracket")
        self._push("array")

    def array_item(self) -> None:
        """Advance over the separator before the next array item."""

        frame = self._top("array")
        _require(not frame.value_pending, "previous array item is missing its value")
        if frame.item_count:
            self._expect_byte(ord(","), "a comma between array items")
        _require(self._peek() != ord("]"), "array item is missing")
        frame.item_count += 1
        frame.value_pending = True

    def at_array_end(self) -> bool:
        self._top("array")
        return self._peek() == ord("]")

    def end_array(self) -> None:
        frame = self._top("array")
        _require(not frame.value_pending, "array item is missing its value")
        self._expect_byte(ord("]"), "an array closing bracket")
        self._finish_container()

    def parse_u64(self, label: str) -> int:
        """Parse one canonical base-10 unsigned 64-bit integer token."""

        self._begin_value()
        first = self._peek()
        _require(ord("0") <= first <= ord("9"), f"{label} must be a u64")
        if first == ord("0"):
            self.position += 1
            _require(
                not (
                    self.position < len(self.raw) - 1
                    and ord("0") <= self.raw[self.position] <= ord("9")
                ),
                f"{label} has a leading zero",
            )
            self._require_value_delimiter(label)
            return 0

        value = 0
        while self.position < len(self.raw) - 1:
            byte = self.raw[self.position]
            if not ord("0") <= byte <= ord("9"):
                break
            digit = byte - ord("0")
            _require(
                value <= (MAX_U64 - digit) // 10,
                f"{label} exceeds the u64 range",
            )
            value = value * 10 + digit
            self.position += 1
        self._require_value_delimiter(label)
        return value

    def parse_raw_string(
        self,
        label: str,
        choices: Collection[str] | None = None,
    ) -> str:
        """Parse one bounded, closed, unescaped printable-ASCII string."""

        self._begin_value()
        value, _ = self._parse_string_literal(label)
        self._require_value_delimiter(label)
        if choices is not None:
            _require(
                not isinstance(choices, (str, bytes)),
                f"{label} choices must be a collection of complete strings",
            )
            _require(value in choices, f"{label} has an unsupported spelling")
        return value

    def parse_bool(self, label: str) -> bool:
        self._begin_value()
        for spelling, value in ((b"true", True), (b"false", False)):
            if self.raw.startswith(spelling, self.position):
                self.position += len(spelling)
                self._require_value_delimiter(label)
                return value
        _fail(f"{label} must be a boolean")

    def parse_null(self, label: str) -> None:
        self._begin_value()
        _require(
            self.raw.startswith(b"null", self.position),
            f"{label} must be null",
        )
        self.position += len(b"null")
        self._require_value_delimiter(label)

    def finish(self) -> None:
        """Require that the root value and sole final LF were consumed."""

        _require(not self._frames, "capture ended with an open container")
        _require(
            self._root_started and self._root_complete,
            "capture root object was not completely consumed",
        )
        _require(
            self.position == len(self.raw) - 1,
            "capture has unconsumed bytes after the root object",
        )
        _require(self.raw[self.position] == ord("\n"), "capture lacks final LF")
        self.position += 1
        _require(self.position == len(self.raw), "capture has trailing bytes")
