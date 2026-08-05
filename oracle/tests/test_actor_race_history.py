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


_CONTROL_SENTINEL = '["none",false,0,0,"0000000000000000","0000000000000000",null]'
_COMMAND_SENTINEL = "[false,0,0,0]"
_ACCEPTED_SENTINEL = "[false,0,0,0,0,0]"
_PRIMARY_SENTINEL = '["primary",false,0,0,0,0,null]'
_OPPORTUNISTIC_SENTINEL = '["opportunistic_eof",false,0,0,0,0,null]'
_WAKE_SENTINEL = "[false,false,0,0]"


def _action_json(ordinal: int, full_shape: bool) -> str:
    producer = ordinal % 2
    invocation = ordinal * 2 + 1
    response = invocation + 1
    fields: list[str]
    if full_shape and ordinal == 0:
        fields = [
            str(ordinal),
            str(producer),
            '"submit"',
            "0",
            "0",
            str(invocation),
            str(response),
            '"submit_accepted"',
            "null",
            "1",
            "null",
            "[true,0,1,1]",
            "[true,1,0,1,0,1]",
            _CONTROL_SENTINEL,
            _PRIMARY_SENTINEL,
            _OPPORTUNISTIC_SENTINEL,
            "false",
            _WAKE_SENTINEL,
        ]
    elif full_shape and ordinal == 1:
        fields = [
            str(ordinal),
            str(producer),
            '"cancel"',
            "null",
            "0",
            str(invocation),
            str(response),
            '"cancel_requested"',
            "null",
            "1",
            "null",
            _COMMAND_SENTINEL,
            _ACCEPTED_SENTINEL,
            '["cancel",true,0,1,"0000000000000008","0000000000000009","requested"]',
            _PRIMARY_SENTINEL,
            _OPPORTUNISTIC_SENTINEL,
            "false",
            _WAKE_SENTINEL,
        ]
    elif full_shape and ordinal == 2:
        fields = [
            str(ordinal),
            str(producer),
            '"receiver_drop"',
            "null",
            "0",
            str(invocation),
            str(response),
            '"receiver_dropped"',
            "null",
            "1",
            "null",
            _COMMAND_SENTINEL,
            _ACCEPTED_SENTINEL,
            '["disconnect",true,0,1,"0000000000000009","000000000000000b","requested"]',
            _PRIMARY_SENTINEL,
            _OPPORTUNISTIC_SENTINEL,
            "false",
            _WAKE_SENTINEL,
        ]
    elif full_shape and ordinal in (3, 6):
        output_index = 0 if ordinal == 3 else 1
        token = 7 if ordinal == 3 else 8
        before = output_index
        after = before + 1
        output = f"[1,{output_index},{token}]"
        opportunistic = (
            _OPPORTUNISTIC_SENTINEL
            if ordinal == 3
            else f'["opportunistic_eof",true,0,1,{after},{after},null]'
        )
        fields = [
            str(ordinal),
            str(producer),
            '"drain"',
            "null",
            "0",
            str(invocation),
            str(response),
            '"drain_output"',
            "null",
            "1",
            output,
            _COMMAND_SENTINEL,
            _ACCEPTED_SENTINEL,
            _CONTROL_SENTINEL,
            f'["primary",true,0,1,{before},{after},{output}]',
            opportunistic,
            "false",
            _WAKE_SENTINEL,
        ]
    elif full_shape and ordinal in (4, 5):
        primary = (
            '["primary",true,0,1,2,2,null]'
            if ordinal == 4
            else _PRIMARY_SENTINEL
        )
        fields = [
            str(ordinal),
            str(producer),
            '"drain"',
            "null",
            "0",
            str(invocation),
            str(response),
            '"drain_eof"',
            "null",
            "1",
            "null",
            _COMMAND_SENTINEL,
            _ACCEPTED_SENTINEL,
            _CONTROL_SENTINEL,
            primary,
            _OPPORTUNISTIC_SENTINEL,
            "true" if ordinal == 5 else "false",
            _WAKE_SENTINEL,
        ]
    else:
        fields = [
            str(ordinal),
            str(producer),
            '"wake"',
            "null",
            str(ordinal % 64),
            str(invocation),
            str(response),
            '"wake_signaled"',
            "null",
            "null",
            "null",
            _COMMAND_SENTINEL,
            _ACCEPTED_SENTINEL,
            _CONTROL_SENTINEL,
            _PRIMARY_SENTINEL,
            _OPPORTUNISTIC_SENTINEL,
            "false",
            "[true,false,0,1]",
        ]
    return "[" + ",".join(fields) + "]"


def _probe_json(
    *,
    engine_steps: int,
    pump_entries: int,
    outstanding_requests: int,
    post_shutdown: bool = False,
    pump_hold_epoch: int = 0,
    request_bytes: int = 0,
) -> str:
    return (
        '{"command_in_flight":0,"command_ready":0,"command_reserved":0,'
        '"command_responded":0,'
        f'"dirty":{"true" if post_shutdown else "false"},'
        f'"engine_steps":{engine_steps},'
        f'"outstanding_requests":{outstanding_requests},'
        f'"owner_done":{"true" if post_shutdown else "false"},'
        '"park_epoch":1,'
        f'"parked":{"false" if post_shutdown else "true"},'
        f'"pump_entries":{pump_entries},'
        f'"pump_hold_observed":{pump_hold_epoch},'
        f'"pump_hold_released":{pump_hold_epoch},'
        f'"pump_hold_requested":{pump_hold_epoch},'
        '"pump_in_flight":false,'
        f'"request_bytes":{request_bytes},"shared_bytes":0}}'
    )


def _recorder_json(observation_count: int) -> str:
    return (
        '{"allocated_capacity":675,'
        f'"observation_count":{observation_count},'
        '"observation_limit":675,"overflowed":false,"poisoned":false}'
    )


def _repetition_json(repetition: int, full_shape: bool) -> str:
    actions = ",".join(
        _action_json(ordinal, full_shape)
        for ordinal in range(actor_race_history.ACTION_COUNT)
    )
    if full_shape:
        cleanup_authorities = (
            '[[1,2,0,1,null,'
            '["cancel",true,0,1,"000000000000000c",'
            '"000000000000000c","already_terminal"]]]'
        )
        cleanup_receivers = (
            '[[3,4,0,1,[1,"completed",2,2],[[1,1,8]],true,'
            + _probe_json(
                engine_steps=12,
                pump_entries=22,
                outstanding_requests=0,
                pump_hold_epoch=1,
            )
            + "]]"
        )
        observations = (
            '[["output",1,0,7,null,null,null],'
            '["output",1,1,8,null,null,null],'
            '["terminal",1,null,null,"completed",2,2],'
            '["output_eof",1,null,null,null,null,null]]'
        )
        cleanup_counter = 4
        observation_count = 4
        accepted = 1
        rejected = 63
        pre_cleanup_outstanding = 1
    else:
        cleanup_authorities = "[]"
        cleanup_receivers = "[]"
        observations = "[]"
        cleanup_counter = 0
        observation_count = 0
        accepted = 0
        rejected = 64
        pre_cleanup_outstanding = 0
    diagnostics = (
        '{"action_counter_final":2048,"barrier_released":true,'
        f'"cleanup_counter_final":{cleanup_counter},'
        '"engine_steps_delta":2,"final_engine_steps":12,'
        '"final_pump_entries":22,"initial_engine_steps":10,'
        '"initial_pump_entries":20,"overlap_pair":[0,"command",1,"control"],'
        '"pump_entries_delta":2,'
        f'"recorder_final":{_recorder_json(observation_count)},'
        f'"recorder_initial":{_recorder_json(0)}}}'
    )
    shutdown = (
        f'{{"accepted_submissions":{accepted},"discarded_output_events":0,'
        f'"engine_steps":12,"rejected_submissions":{rejected},'
        '"released_request_bytes":0,"remaining_shared_bytes":0,'
        '"shutdown_cancellations":0,"terminated_requests":0}'
    )
    post_shutdown = _probe_json(
        engine_steps=12,
        pump_entries=22,
        outstanding_requests=0,
        post_shutdown=True,
        pump_hold_epoch=1,
    )
    pre_cleanup = _probe_json(
        engine_steps=12,
        pump_entries=22,
        outstanding_requests=pre_cleanup_outstanding,
        request_bytes=pre_cleanup_outstanding,
    )
    pre_shutdown = _probe_json(
        engine_steps=12,
        pump_entries=22,
        outstanding_requests=0,
        pump_hold_epoch=1,
    )
    return (
        f'{{"actions":[{actions}],'
        f'"cleanup_authorities":{cleanup_authorities},'
        f'"cleanup_receivers":{cleanup_receivers},'
        f'"diagnostics":{diagnostics},'
        f'"observations":{observations},'
        f'"post_shutdown":{post_shutdown},'
        f'"pre_cleanup":{pre_cleanup},'
        f'"pre_shutdown":{pre_shutdown},'
        f'"repetition":{repetition},"shutdown":{shutdown}}}'
    )


def _capture_bytes(full_shape: bool = False) -> bytes:
    repetitions = ",".join(
        _repetition_json(repetition, full_shape)
        for repetition in range(actor_race_history.REPETITION_COUNT)
    )
    workload = (
        '{"artifact_id":"sha256:382856e13f688b5176ad1e5f06c26bcd85bcaeb719a10a60e9adc9ff387c945c",'
        '"artifact_object_sha256":"sha256:275f985b05a85d4f85d78fc290c10c9d39e9169c46449f4d3a3ed6513a8965ab",'
        '"artifact_page_table_sha256":"sha256:7d660764b861f97afbc800efb361bdd59ac38fdea5fc424c62f9fb6a30b2896c",'
        '"model_spec_sha256":"sha256:ed57d7961e65c76223c169cabebaff9c02d8293da026abb0c0c0a22d38079845",'
        '"specification":"runnel-m5-actor-stress-v1",'
        '"vector_file_sha256":"sha256:eca1faeee91a41d19d98be7ffdad6fc5cebb9027f3e7a634c01ea1cc394fb574",'
        '"vector_id":"sha256:5010492fb74eda207511b26811992ed4779814185b9f184663b37a37747bd051",'
        '"vector_schema":"runnel.actor-stress-vectors/2"}'
    )
    return (
        f'{{"repetition_count":32,"repetitions":[{repetitions}],'
        f'"schema":"runnel.actor-race-history/1","workload":{workload}}}\n'
    ).encode("ascii")


class ActorRaceHistoryDecoderTests(unittest.TestCase):
    @staticmethod
    def parse_action(value: str, ordinal: int = 0) -> actor_race_history.Action:
        cursor = actor_race_history.CanonicalCursor(
            f'{{"value":{value}}}\n'.encode("ascii")
        )
        cursor.begin_object()
        cursor.object_key("value")
        action = actor_race_history._parse_action(cursor, ordinal)
        cursor.end_object()
        cursor.finish()
        return action

    @staticmethod
    def parse_probe(value: str) -> actor_race_history.ProbeSnapshot:
        cursor = actor_race_history.CanonicalCursor(
            f'{{"value":{value}}}\n'.encode("ascii")
        )
        cursor.begin_object()
        cursor.object_key("value")
        probe = actor_race_history._parse_probe_snapshot(cursor, "probe")
        cursor.end_object()
        cursor.finish()
        return probe

    @staticmethod
    def parse_cleanup_receiver(value: str) -> actor_race_history.CleanupReceiver:
        cursor = actor_race_history.CanonicalCursor(
            f'{{"value":{value}}}\n'.encode("ascii")
        )
        cursor.begin_object()
        cursor.object_key("value")
        receiver = actor_race_history._parse_cleanup_receiver(cursor, 0)
        cursor.end_object()
        cursor.finish()
        return receiver

    def test_decodes_exact_minimal_closed_capture_to_immutable_types(self) -> None:
        raw = _capture_bytes()
        self.assertLessEqual(len(raw), actor_race_history.MAX_CAPTURE_BYTES)
        capture = actor_race_history.decode_capture(raw)
        self.assertEqual(capture.repetition_count, 32)
        self.assertEqual(len(capture.repetitions), 32)
        self.assertEqual(len(capture.repetitions[0].actions), 1024)
        self.assertEqual(capture.repetitions[-1].repetition, 31)
        self.assertEqual(capture.repetitions[0].actions[1].producer, 1)
        self.assertIsInstance(capture.repetitions, tuple)
        with self.assertRaises(AttributeError):
            capture.schema = "changed"  # type: ignore[misc]

    def test_decodes_full_tuple_and_variable_array_shapes(self) -> None:
        capture = actor_race_history.decode_capture(_capture_bytes(full_shape=True))
        repetition = capture.repetitions[0]
        self.assertTrue(repetition.actions[0].command.boundary)
        self.assertEqual(repetition.actions[0].accepted.request_id, 1)
        self.assertEqual(repetition.actions[1].control.operation, "cancel")
        self.assertEqual(repetition.actions[2].control.operation, "disconnect")
        self.assertEqual(repetition.actions[3].output.token_id, 7)
        self.assertTrue(repetition.actions[5].cached_eof)
        self.assertTrue(repetition.actions[6].opportunistic_eof_pop.boundary)
        self.assertEqual(len(repetition.cleanup_authorities), 1)
        self.assertEqual(len(repetition.cleanup_receivers), 1)
        self.assertEqual(len(repetition.observations), 4)

    def test_rejects_unknown_missing_and_misordered_object_keys(self) -> None:
        raw = _capture_bytes()
        unknown_top = raw.replace(
            b'"repetition_count":32', b'"repetition_counu":32', 1
        )
        missing_workload = raw.replace(
            b',"vector_schema":"runnel.actor-stress-vectors/2"', b"", 1
        )
        misordered_probe = _probe_json(
            engine_steps=12, pump_entries=22, outstanding_requests=0
        ).replace(
            '"command_in_flight":0,"command_ready":0',
            '"command_ready":0,"command_in_flight":0',
            1,
        )
        for invalid in (unknown_top, missing_workload):
            with self.subTest(case=invalid[:40]):
                with self.assertRaises(actor_race_history.ActorRaceHistoryError):
                    actor_race_history.decode_capture(invalid)

        cursor = actor_race_history.CanonicalCursor(
            f'{{"value":{misordered_probe}}}\n'.encode("ascii")
        )
        cursor.begin_object()
        cursor.object_key("value")
        with self.assertRaises(actor_race_history.ActorRaceHistoryError):
            actor_race_history._parse_probe_snapshot(cursor, "probe")

    def test_rejects_wrong_tuple_width_enums_sentinels_and_ranges(self) -> None:
        valid = _action_json(0, False)
        invalid = (
            valid.replace("[true,false,0,1]]", "[true,false,0,1,0]]"),
            valid.replace('"wake"', '"unknown"', 1),
            valid.replace("[false,0,0,0]", "[false,1,0,0]", 1),
            valid.replace("[0,0,\"wake\"", "[0,2,\"wake\"", 1),
            valid.replace("[true,false,0,1]", "[false,true,0,0]", 1),
        )
        for action in invalid:
            with self.subTest(action=action[:80]):
                with self.assertRaises(actor_race_history.ActorRaceHistoryError):
                    self.parse_action(action)

    def test_rejects_hex_sha_and_u32_pattern_or_range_failures(self) -> None:
        control_bad_hex = _action_json(0, False).replace(
            '"0000000000000000"', '"g000000000000000"', 1
        )
        with self.assertRaises(actor_race_history.ActorRaceHistoryError):
            self.parse_action(control_bad_hex)

        too_large_token = _action_json(3, True).replace(
            "[1,0,7]", f"[1,0,{actor_race_history.UINT32_MAX + 1}]"
        )
        with self.assertRaises(actor_race_history.ActorRaceHistoryError):
            self.parse_action(too_large_token, 3)

        cursor = actor_race_history.CanonicalCursor(
            b'{"value":"sha256:ABCDEF0000000000000000000000000000000000000000000000000000abcdef"}\n'
        )
        cursor.begin_object()
        cursor.object_key("value")
        with self.assertRaises(actor_race_history.ActorRaceHistoryError):
            actor_race_history._parse_sha256(cursor, "digest")

    def test_reached_control_words_and_wakes_require_live_generations(self) -> None:
        invalid_actions = (
            (
                _action_json(1, True).replace(
                    '"0000000000000008"', '"0000000000000000"', 1
                ),
                1,
            ),
            (
                _action_json(1, True).replace(
                    '["cancel",true,0,1,',
                    f'["cancel",true,0,{actor_race_history.MAX_CONTROL_GENERATION + 1},',
                    1,
                ),
                1,
            ),
            (
                _action_json(0, True).replace(
                    "[true,1,0,1,0,1]",
                    "[true,1,0,2305843009213693952,0,1]",
                    1,
                ),
                0,
            ),
            (
                _action_json(0, False).replace(
                    "[true,false,0,1]", "[true,false,1,1]", 1
                ),
                0,
            ),
        )
        for action, ordinal in invalid_actions:
            with self.subTest(ordinal=ordinal, action=action[:96]):
                with self.assertRaises(actor_race_history.ActorRaceHistoryError):
                    self.parse_action(action, ordinal)

    def test_unavailable_receiver_lookup_may_retain_published_request_id(self) -> None:
        action = _action_json(2, True).replace(
            '"receiver_dropped"', '"target_unavailable"', 1
        ).replace(
            '["disconnect",true,0,1,"0000000000000009",'
            '"000000000000000b","requested"]',
            _CONTROL_SENTINEL,
            1,
        )
        decoded = self.parse_action(action, 2)
        self.assertEqual(decoded.result, "target_unavailable")
        self.assertEqual(decoded.request_id, 1)

    def test_quiescence_requires_zero_commands_and_completed_hold_epoch(self) -> None:
        valid = _probe_json(
            engine_steps=12,
            pump_entries=22,
            outstanding_requests=0,
        )
        actor_race_history._require_quiescent(self.parse_probe(valid), "probe")
        invalid = (
            valid.replace('"command_in_flight":0', '"command_in_flight":1', 1),
            valid.replace('"pump_hold_observed":0', '"pump_hold_observed":1', 1),
        )
        for value in invalid:
            with self.subTest(value=value[:96]):
                with self.assertRaises(actor_race_history.ActorRaceHistoryError):
                    actor_race_history._require_quiescent(
                        self.parse_probe(value), "probe"
                    )

        probe = self.parse_probe(valid)
        actor_race_history._require_pre_cleanup_not_before_initial(
            probe, 12, 22, "repetition"
        )
        for initial_engine_steps, initial_pump_entries in ((13, 22), (12, 23)):
            with self.subTest(
                initial_engine_steps=initial_engine_steps,
                initial_pump_entries=initial_pump_entries,
            ):
                with self.assertRaises(actor_race_history.ActorRaceHistoryError):
                    actor_race_history._require_pre_cleanup_not_before_initial(
                        probe,
                        initial_engine_steps,
                        initial_pump_entries,
                        "repetition",
                    )

    def test_cleanup_suffix_is_bounded_consecutive_and_terminal_aligned(self) -> None:
        snapshot = _probe_json(
            engine_steps=12,
            pump_entries=22,
            outstanding_requests=0,
            pump_hold_epoch=1,
        )
        valid = (
            '[1,2,0,1,[1,"completed",4,4],'
            f'[[1,2,7],[1,3,8]],true,{snapshot}]'
        )
        decoded = self.parse_cleanup_receiver(valid)
        self.assertEqual([output.output_index for output in decoded.outputs], [2, 3])

        nonconsecutive = valid.replace("[1,3,8]", "[1,4,8]", 1).replace(
            '"completed",4,4', '"completed",5,5', 1
        )
        wrong_terminal_end = valid.replace('"completed",4,4', '"completed",5,5', 1)
        seventeen_outputs = ",".join(
            f"[1,{index},7]" for index in range(actor_race_history.MAX_NEW_TOKENS + 1)
        )
        too_many = (
            '[1,2,0,1,[1,"completed",17,17],'
            f'[{seventeen_outputs}],true,{snapshot}]'
        )
        for value in (nonconsecutive, wrong_terminal_end, too_many):
            with self.subTest(value=value[:120]):
                with self.assertRaises(actor_race_history.ActorRaceHistoryError):
                    self.parse_cleanup_receiver(value)

    def test_cleanup_indexes_and_observations_require_canonical_order(self) -> None:
        with self.assertRaises(actor_race_history.ActorRaceHistoryError):
            actor_race_history._require_strictly_ascending(
                (0, 2, 1), "cleanup indexes"
            )

        output = actor_race_history.Observation(
            "output", 1, 0, 7, None, None, None
        )
        terminal = actor_race_history.Observation(
            "terminal", 1, None, None, "completed", 1, 1
        )
        eof = actor_race_history.Observation(
            "output_eof", 1, None, None, None, None, None
        )
        actor_race_history._validate_observation_order(
            (output, terminal, eof), "observations"
        )
        for observations in (
            (terminal, output, eof),
            (output, eof, terminal),
            (output, output, terminal, eof),
        ):
            with self.subTest(observations=observations):
                with self.assertRaises(actor_race_history.ActorRaceHistoryError):
                    actor_race_history._validate_observation_order(
                        observations, "observations"
                    )

    def test_requires_exact_repetition_count_and_indexes(self) -> None:
        raw = _capture_bytes()
        wrong_count = raw.replace(b'"repetition_count":32', b'"repetition_count":31', 1)
        wrong_index = raw.replace(b'"repetition":0,"shutdown"', b'"repetition":32,"shutdown"', 1)
        for invalid in (wrong_count, wrong_index):
            with self.subTest(case=invalid[:60]):
                with self.assertRaises(actor_race_history.ActorRaceHistoryError):
                    actor_race_history.decode_capture(invalid)


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
