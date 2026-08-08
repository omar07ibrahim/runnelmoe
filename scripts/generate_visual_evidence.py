#!/usr/bin/env python3
"""Build and verify a candidate-only, source-backed visual evidence bundle."""

from __future__ import annotations

import argparse
import hashlib
import html
import importlib.metadata
import json
import os
import pathlib
import platform
import re
import selectors
import shutil
import signal
import subprocess
import sys
import time
import tomllib
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

try:
    from PIL import Image, ImageDraw, ImageFont, PngImagePlugin
except ImportError:
    Image = ImageDraw = ImageFont = PngImagePlugin = None


SCHEMA = "runnel.visual-evidence-manifest/1"
ASSET_SCHEMA = "runnel.visual-evidence-asset/1"
EXPECTED_PYTHON = (3, 12, 11)
EXPECTED_PILLOW = "11.3.0"
MAX_STREAM = 1024 * 1024
MAX_FILE = 12 * 1024 * 1024
MAX_WIDTH = 1600
MAX_HEIGHT = 2000
MARGIN = 24
SVG_NS = "http://www.w3.org/2000/svg"
SHA256 = re.compile(r"^[0-9a-f]{64}$")
REVISION = re.compile(r"^[0-9a-f]{40}$")
SAFE_PATH = re.compile(r"^[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*$")
SAFE_REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
EMAIL_LIKE = re.compile(
    rb"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
)

COLORS = {
    "bg": "#07131f",
    "panel": "#102538",
    "panel2": "#142d40",
    "ink": "#eff8f6",
    "muted": "#a3bac0",
    "teal": "#5de0c2",
    "blue": "#77aaff",
    "amber": "#ffc66d",
    "border": "#2a4a60",
}

FORBIDDEN = {
    b"/home/" + b"runner/": "hosted-runner path",
    b"/home/" + b"ubuntu/": "workstation path",
    b"/Us" + b"ers/": "macOS path",
    b"C:\\" + b"Users\\": "Windows path",
    b"-----BEGIN " + b"PRIVATE KEY-----": "private key",
    b"-----BEGIN OPEN" + b"SSH PRIVATE KEY-----": "SSH key",
    b"gh" + b"o_": "GitHub token",
    b"gh" + b"p_": "GitHub token",
    b"gh" + b"u_": "GitHub token",
    b"gh" + b"s_": "GitHub token",
    b"gh" + b"r_": "GitHub token",
    b"github_" + b"pat_": "GitHub token",
    b"AK" + b"IA": "AWS access key",
    b"AS" + b"IA": "AWS temporary key",
}


class ContractError(RuntimeError):
    """Evidence contract violation."""


@dataclass(frozen=True)
class Command:
    capture_id: str
    milestone: str
    argv: tuple[str, ...]
    jsonl: bool = False


COMMANDS = (
    Command(
        "m1-demo",
        "M1",
        (
            "cargo", "run", "--locked", "-p", "runnel", "--", "demo",
            "--prompt", "moe", "--max-new-tokens", "4", "--json",
        ),
    ),
    Command(
        "m2-data-plane",
        "M2",
        (
            "cargo", "run", "--locked", "-p", "runnel", "--",
            "data-plane-demo", "--json",
        ),
    ),
    Command(
        "m3-cache-matrix",
        "M3",
        (
            "cargo", "run", "--locked", "-p", "runnel-sim", "--bin",
            "runnel-cache-sim", "--", "matrix", "--family",
            "markov_clusters", "--replicate", "0", "--measured-steps", "64",
        ),
    ),
    Command(
        "m4-model-check",
        "M4",
        (
            "cargo", "run", "--locked", "-p", "runnel", "--bin",
            "runnel-m4-model-check",
        ),
        True,
    ),
)


def need(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_digest(path: pathlib.Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(131072):
            value.update(block)
    return value.hexdigest()


def scan(data: bytes, label: str) -> str:
    need(len(data) <= MAX_FILE, f"{label} exceeds the file bound")
    for marker, description in FORBIDDEN.items():
        need(marker not in data, f"{label} contains {description}")
    need(b"\0" not in data, f"{label} contains NUL")
    need(EMAIL_LIKE.search(data) is None, f"{label} contains email-like PII")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ContractError(f"{label} is not UTF-8") from error


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        need(key not in result, f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def strict_json(data: bytes, label: str) -> Any:
    need(len(data) <= MAX_STREAM, f"{label} exceeds the stream bound")
    try:
        return json.loads(
            scan(data, label),
            object_pairs_hook=_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ContractError(f"non-finite JSON value {value}")
            ),
        )
    except json.JSONDecodeError as error:
        raise ContractError(f"{label} is not strict JSON: {error}") from error


def jsonl(data: bytes, label: str) -> list[Any]:
    need(data.endswith(b"\n"), f"{label} lacks a final newline")
    return [strict_json(line, f"{label}:{index}") for index, line in enumerate(data.splitlines(), 1)]


def keys(value: Any, required: Iterable[str], label: str) -> dict[str, Any]:
    need(isinstance(value, dict), f"{label} is not an object")
    missing = set(required) - set(value)
    need(not missing, f"{label} lacks {sorted(missing)}")
    return value


def parse(command: Command, data: bytes) -> Any:
    value = jsonl(data, command.capture_id) if command.jsonl else strict_json(data, command.capture_id)
    if command.capture_id == "m1-demo":
        row = keys(value, ("schema_version", "artifact_id", "adapter", "input_token_count", "generated_ids", "text", "stop_reason"), "M1")
        need(row["schema_version"] == 1, "M1 schema changed")
        need(row["generated_ids"] == [15, 11, 20, 9] and row["text"] == "njsh", "M1 stable vector changed")
        need(row["adapter"]["version"] == 1, "M1 adapter changed")
    elif command.capture_id == "m2-data-plane":
        row = keys(value, ("schema_version", "generated_ids", "generated_text", "parity", "generation_cache", "forced_eviction_generation", "metrics"), "M2")
        need(row["schema_version"] == 2 and row["parity"] is True, "M2 parity failed")
        need(row["generated_ids"] == [15, 11, 20, 9] and row["generated_text"] == "njsh", "M2 stable vector changed")
        need(row["forced_eviction_generation"]["full_generation_parity"] is True, "M2 forced-eviction parity failed")
        need(row["metrics"]["trace_events_dropped"] == 0, "M2 trace dropped events")
    elif command.capture_id == "m3-cache-matrix":
        row = keys(value, ("schema", "family", "replicate", "measured_steps", "trace_sha256", "results"), "M3")
        need(row["schema"] == "runnel.cache-matrix/1", "M3 schema changed")
        need(row["family"] == "markov_clusters" and row["replicate"] == 0 and row["measured_steps"] == 64, "M3 smoke parameters changed")
        need(SHA256.fullmatch(row["trace_sha256"]) is not None, "M3 trace digest changed")
        need(isinstance(row["results"], list) and len(row["results"]) == 18, "M3 matrix is not 18 cells")
        expected = {"lru", "slru", "tiny-lfu", "router-admit", "router-prefetch", "belady"}
        groups: dict[int, set[str]] = {}
        for result in row["results"]:
            metric = result["metrics"]
            need(metric["total_physical_load_bytes"] == metric["demand_load_bytes"] + metric["prefetch_load_bytes"], "M3 byte identity failed")
            groups.setdefault(result["capacity_bytes"], set()).add(result["policy"])
        need(len(groups) == 3 and all(group == expected for group in groups.values()), "M3 grid changed")
    else:
        need(isinstance(value, list) and len(value) == 3, "M4 ledger is not three rows")
        expected = ("tiny-v1-preservation", "tiny-v2-scalar", "tiny-v2-avx2")
        for index, row in enumerate(value):
            keys(row, ("schema", "check_id", "backend", "status", "metrics"), f"M4:{index}")
            need(row["schema"] == "runnel.m4-correctness/1" and row["check_id"] == expected[index], "M4 row changed")
            need(row["status"] == "ok" if index < 2 else row["status"] in {"ok", "unsupported"}, "M4 status failed")
            if row["status"] == "ok":
                metric = row["metrics"]
                need(metric["tokens_exact"] and metric["expert_ids_exact"] and metric["deterministic"], "M4 exactness failed")
        need(b'"timing"' not in data and b'"elapsed"' not in data, "M4 ledger contains timing")
    return value


def bounded_run(
    argv: Sequence[str],
    root: pathlib.Path,
    environment: dict[str, str],
    command_id: str,
) -> tuple[bytes, bytes]:
    need(re.fullmatch(r"[a-z0-9-]+", command_id) is not None, "command ID is unsafe")
    process = subprocess.Popen(
        list(argv),
        cwd=root,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    assert process.stdout is not None and process.stderr is not None
    selector = selectors.DefaultSelector()
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    for name, stream in (("stdout", process.stdout), ("stderr", process.stderr)):
        os.set_blocking(stream.fileno(), False)
        selector.register(stream, selectors.EVENT_READ, name)
    deadline = time.monotonic() + 120
    failure = None
    while selector.get_map() and failure is None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            failure = f"{command_id} exceeded 120 seconds"
            break
        events = selector.select(min(remaining, 0.5))
        if not events and process.poll() is not None:
            events = [(item, selectors.EVENT_READ) for item in selector.get_map().values()]
        for item, _ in events:
            try:
                chunk = os.read(item.fileobj.fileno(), 65536)
            except BlockingIOError:
                continue
            if not chunk:
                selector.unregister(item.fileobj)
                item.fileobj.close()
                continue
            buffers[item.data].extend(chunk)
            if len(buffers[item.data]) > MAX_STREAM:
                failure = f"{command_id} {item.data} exceeded {MAX_STREAM} bytes"
                break
    selector.close()
    if failure is not None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        raise ContractError(failure)
    code = process.wait(timeout=5)
    stdout, stderr = bytes(buffers["stdout"]), bytes(buffers["stderr"])
    if code:
        raise ContractError(f"{command_id} exited {code}")
    return stdout, stderr


def small(argv: Sequence[str], root: pathlib.Path) -> str:
    result = subprocess.run(
        list(argv),
        cwd=root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=20,
        env={"HOME": os.environ.get("HOME", ""), "PATH": os.environ["PATH"], "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"},
    )
    need(result.returncode == 0 and not result.stderr and len(result.stdout) < 65536, f"{argv[0]} provenance command failed")
    return result.stdout.decode().strip()


def write_new(path: pathlib.Path, data: bytes) -> None:
    need(not path.exists(), f"refusing to replace {path.name}")
    with path.open("xb") as output:
        output.write(data)
    path.chmod(0o600)


def wrap(text: Any, columns: int = 100) -> list[str]:
    remaining = str(text)
    output = []
    while len(remaining) > columns:
        point = remaining.rfind(" ", 0, columns + 1)
        point = columns if point < 1 else point
        output.append(remaining[:point])
        remaining = remaining[point:].lstrip()
    output.append(remaining)
    return output


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def svg_card(
    title: str,
    kicker: str,
    subtitle: str,
    sections: list[tuple[str, list[str]]],
    footer: str,
    revision: str,
    source: str,
) -> tuple[bytes, list[int], list[int]]:
    width = 1440
    line_count = sum(1 + sum(len(wrap(line)) for line in lines) for _, lines in sections)
    height = 275 + len(sections) * 78 + line_count * 24 + len(wrap(footer, 115)) * 21
    need(height <= MAX_HEIGHT, "SVG height exceeded")
    body = [
        f'<text x="58" y="66" fill="{COLORS["teal"]}" font-family="sans-serif" font-size="15" font-weight="700">{esc(kicker)}</text>',
        f'<text x="58" y="110" fill="{COLORS["ink"]}" font-family="sans-serif" font-size="32" font-weight="700">{esc(title)}</text>',
        f'<text x="58" y="141" fill="{COLORS["muted"]}" font-family="sans-serif" font-size="15">{esc(subtitle)}</text>',
    ]
    y = 177
    for heading, raw_lines in sections:
        lines = [part for line in raw_lines for part in wrap(line)]
        panel_height = 57 + len(lines) * 24
        body.append(f'<rect x="48" y="{y}" width="{width - 96}" height="{panel_height}" rx="14" fill="{COLORS["panel"]}" stroke="{COLORS["border"]}"/>')
        body.append(f'<text x="72" y="{y + 29}" fill="{COLORS["blue"]}" font-family="sans-serif" font-size="16" font-weight="700">{esc(heading)}</text>')
        line_y = y + 56
        for line in lines:
            body.append(f'<text x="72" y="{line_y}" fill="{COLORS["ink"]}" font-family="monospace" font-size="15">{esc(line)}</text>')
            line_y += 24
        y += panel_height + 17
    y += 8
    for line in wrap(footer, 115):
        body.append(f'<text x="58" y="{y}" fill="{COLORS["amber"]}" font-family="sans-serif" font-size="14">{esc(line)}</text>')
        y += 21
    body.append(f'<text x="{width - 58}" y="{height - 35}" text-anchor="end" fill="{COLORS["muted"]}" font-family="monospace" font-size="12">source {revision[:12]} / stdout {source[:16]}...</text>')
    bounds = [48, 48, width - 48, height - 30]
    metadata = {"schema": ASSET_SCHEMA, "revision": revision, "source_stdout_sha256": source, "content_bounds": bounds}
    return svg_document(title, subtitle, width, height, body, metadata), [width, height], bounds


def svg_document(title: str, description: str, width: int, height: int, body: list[str], metadata: dict[str, Any]) -> bytes:
    need(width <= MAX_WIDTH and height <= MAX_HEIGHT, "SVG canvas exceeded")
    encoded = esc(json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False))
    joined = "\n  ".join(body)
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="{SVG_NS}" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
  <title id="title">{esc(title)}</title>
  <desc id="desc">{esc(description)}</desc>
  <metadata>{encoded}</metadata>
  <defs>
    <linearGradient id="surface" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#10273a"/><stop offset="1" stop-color="#091725"/></linearGradient>
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="{COLORS["muted"]}"/></marker>
  </defs>
  <rect width="{width}" height="{height}" rx="24" fill="{COLORS["bg"]}"/>
  <rect x="20" y="20" width="{width - 40}" height="{height - 40}" rx="20" fill="url(#surface)" stroke="{COLORS["border"]}"/>
  {joined}
</svg>
'''.encode()


def human_bytes(value: int) -> str:
    if value % 1048576 == 0:
        return f"{value // 1048576} MiB"
    return f"{value / 1048576:.3f} MiB"


def crate_graph(root: pathlib.Path) -> tuple[list[dict[str, str]], list[tuple[str, str]], list[dict[str, Any]]]:
    workspace_path = root / "Cargo.toml"
    workspace = tomllib.loads(workspace_path.read_text())
    members = workspace["workspace"]["members"]
    workspace_deps = workspace["workspace"]["dependencies"]
    packages: dict[str, tuple[pathlib.Path, dict[str, Any]]] = {}
    source_files = []
    for relative in ["Cargo.toml", *[f"{member}/Cargo.toml" for member in members]]:
        path = root / relative
        parsed = tomllib.loads(path.read_text())
        source_files.append({"path": relative, "bytes": path.stat().st_size, "sha256": file_digest(path)})
        if relative != "Cargo.toml":
            packages[parsed["package"]["name"]] = (path, parsed)
    edges = set()
    for consumer, (_, manifest) in packages.items():
        for alias, setting in manifest.get("dependencies", {}).items():
            name = alias
            if isinstance(setting, dict):
                name = setting.get("package", alias)
                if setting.get("workspace") is True and isinstance(workspace_deps.get(alias), dict):
                    name = workspace_deps[alias].get("package", alias)
            if name in packages:
                edges.add((consumer, name))
    nodes = [{"name": name, "description": parsed["package"].get("description", "workspace crate")} for name, (_, parsed) in packages.items()]
    return sorted(nodes, key=lambda row: row["name"]), sorted(edges), source_files


def architecture_svg(root: pathlib.Path, revision: str) -> tuple[bytes, list[int], list[int], list[dict[str, str]], list[tuple[str, str]], list[dict[str, Any]]]:
    nodes, edges, source_files = crate_graph(root)
    names = [node["name"] for node in nodes]
    indegree = {name: 0 for name in names}
    outgoing = {name: [] for name in names}
    for consumer, dependency in edges:
        outgoing[consumer].append(dependency)
        indegree[dependency] += 1
    levels = []
    current = sorted(name for name in names if indegree[name] == 0)
    seen = set()
    while current:
        levels.append(current)
        next_level = []
        for name in current:
            seen.add(name)
            for target in outgoing[name]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    next_level.append(target)
        current = sorted(next_level)
    need(seen == set(names), "crate graph has a cycle")
    width, node_w, node_h, gap_x, gap_y, top = 1520, 330, 108, 42, 82, 185
    positions = {}
    for row, level in enumerate(levels):
        total = len(level) * node_w + max(0, len(level) - 1) * gap_x
        start = (width - total) / 2
        for column, name in enumerate(level):
            positions[name] = (start + column * (node_w + gap_x), top + row * (node_h + gap_y))
    height = int(top + len(levels) * node_h + max(0, len(levels) - 1) * gap_y + 145)
    descriptions = {node["name"]: node["description"] for node in nodes}
    body = [
        f'<text x="58" y="66" fill="{COLORS["teal"]}" font-family="sans-serif" font-size="15" font-weight="700">SOURCE-BACKED ARCHITECTURE</text>',
        f'<text x="58" y="110" fill="{COLORS["ink"]}" font-family="sans-serif" font-size="32" font-weight="700">M1-M4 crate topology</text>',
        f'<text x="58" y="141" fill="{COLORS["muted"]}" font-family="sans-serif" font-size="15">Actual dependencies parsed from the workspace and seven crate Cargo manifests</text>',
    ]
    for consumer, dependency in edges:
        x1, y1 = positions[consumer]
        x2, y2 = positions[dependency]
        x1 += node_w / 2
        x2 += node_w / 2
        y1 += node_h
        middle = (y1 + y2) / 2
        body.append(f'<path d="M {x1:.1f} {y1:.1f} C {x1:.1f} {middle:.1f}, {x2:.1f} {middle:.1f}, {x2:.1f} {y2:.1f}" fill="none" stroke="{COLORS["muted"]}" stroke-width="2" marker-end="url(#arrow)"/>')
    for name, (x, y) in sorted(positions.items()):
        body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{node_w}" height="{node_h}" rx="14" fill="{COLORS["panel"]}" stroke="{COLORS["border"]}"/>')
        body.append(f'<text x="{x + 18:.1f}" y="{y + 36:.1f}" fill="{COLORS["blue"]}" font-family="monospace" font-size="17" font-weight="700">{esc(name)}</text>')
        for index, line in enumerate(wrap(descriptions[name], 45)[:2]):
            body.append(f'<text x="{x + 18:.1f}" y="{y + 67 + index * 20:.1f}" fill="{COLORS["muted"]}" font-family="sans-serif" font-size="13">{esc(line)}</text>')
    body.append(f'<text x="58" y="{height - 61}" fill="{COLORS["amber"]}" font-family="sans-serif" font-size="14">No scheduler node or M5 result is inferred.</text>')
    body.append(f'<text x="{width - 58}" y="{height - 35}" text-anchor="end" fill="{COLORS["muted"]}" font-family="monospace" font-size="12">source {revision[:12]} / {len(source_files)} manifest digests</text>')
    bounds = [48, 48, width - 48, height - 30]
    metadata = {"schema": ASSET_SCHEMA, "revision": revision, "source_manifest_files": source_files, "content_bounds": bounds}
    return svg_document("M1-M4 crate topology", "Committed internal Cargo dependency graph with no M5 claim.", width, height, body, metadata), [width, height], bounds, nodes, edges, source_files


def pillow() -> None:
    need(Image is not None, "Pillow is required")
    need(importlib.metadata.version("Pillow") == EXPECTED_PILLOW, f"Pillow must be {EXPECTED_PILLOW}")


def font(size: int) -> Any:
    pillow()
    return ImageFont.load_default(size=size)


def pixel_wrap(draw: Any, text: str, selected_font: Any, max_width: int) -> list[str]:
    output = []
    for logical in text.splitlines() or [""]:
        current = ""
        for character in logical:
            candidate = current + character
            box = draw.textbbox((0, 0), candidate, font=selected_font)
            if current and box[2] > max_width:
                output.append(current)
                current = character
            else:
                current = candidate
        output.append(current)
    return output


def draw_text(draw: Any, xy: tuple[int, int], text: str, selected_font: Any, fill: str, size: tuple[int, int], bounds: list[int]) -> None:
    box = draw.textbbox(xy, text, font=selected_font)
    need(box[0] >= 0 and box[1] >= 0 and box[2] <= size[0] and box[3] <= size[1], "raster text clipped")
    draw.text(xy, text, font=selected_font, fill=fill)
    bounds[0] = min(bounds[0], box[0])
    bounds[1] = min(bounds[1], box[1])
    bounds[2] = max(bounds[2], box[2])
    bounds[3] = max(bounds[3], box[3])


def transcript_png(command: Command, stdout: bytes, revision: str, source: str, path: pathlib.Path) -> tuple[list[int], list[int]]:
    pillow()
    need(stdout.endswith(b"\n"), "M1 stdout lacks newline")
    raw = stdout[:-1].decode()
    title_font, body_font, small_font = font(28), font(17), font(14)
    scratch = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    command_lines = pixel_wrap(scratch, "$ " + " ".join(command.argv), body_font, 1300)
    output_lines = pixel_wrap(scratch, raw, body_font, 1300)
    need("".join(output_lines) == raw.replace("\n", ""), "PNG wrapping changed stdout")
    width = 1440
    height = 340 + 27 * (len(command_lines) + len(output_lines))
    need(height <= MAX_HEIGHT, "PNG canvas exceeded")
    image = Image.new("RGB", (width, height), COLORS["bg"])
    draw = ImageDraw.Draw(image)
    bounds = [width, height, 0, 0]
    draw.rounded_rectangle((24, 24, width - 24, height - 24), radius=22, fill=COLORS["panel"], outline=COLORS["border"], width=2)
    bounds[:] = [24, 24, width - 24, height - 24]
    draw_text(draw, (58, 56), "ACTUAL CLI STDOUT / M1", small_font, COLORS["teal"], image.size, bounds)
    draw_text(draw, (58, 91), "Artifact-to-token transcript", title_font, COLORS["ink"], image.size, bounds)
    draw_text(draw, (58, 132), "Deterministic rendering of captured stdout - not an OS screenshot.", small_font, COLORS["muted"], image.size, bounds)
    y = 181
    for line in command_lines:
        draw_text(draw, (58, y), line, body_font, COLORS["blue"], image.size, bounds)
        y += 27
    y += 17
    draw.line((58, y, width - 58, y), fill=COLORS["border"], width=2)
    y += 27
    for line in output_lines:
        draw_text(draw, (58, y), line, body_font, COLORS["ink"], image.size, bounds)
        y += 27
    draw_text(draw, (58, height - 60), f"revision {revision[:12]} / stdout sha256 {source}", small_font, COLORS["muted"], image.size, bounds)
    need(bounds[0] >= MARGIN and bounds[1] >= MARGIN and bounds[2] <= width - MARGIN and bounds[3] <= height - MARGIN, "PNG bounds escaped")
    info = PngImagePlugin.PngInfo()
    for key, value in {
        "Schema": ASSET_SCHEMA,
        "SourceRevision": revision,
        "SourceStdoutSHA256": source,
        "RenderContract": "deterministic rendering of captured stdout; not an OS screenshot",
        "ContentBounds": json.dumps(bounds, separators=(",", ":")),
    }.items():
        info.add_text(key, value)
    image.save(path, format="PNG", pnginfo=info, optimize=False, compress_level=9)
    path.chmod(0o600)
    return [width, height], bounds


def summaries(parsed: dict[str, Any]) -> dict[str, list[str]]:
    demo, plane, matrix, ledger = (parsed[item.capture_id] for item in COMMANDS)
    groups: dict[int, list[int]] = {}
    for result in matrix["results"]:
        groups.setdefault(result["capacity_bytes"], []).append(result["metrics"]["total_physical_load_bytes"])
    return {
        "m1-demo": [f"generated IDs {demo['generated_ids']}", f"decoded text {demo['text']!r}", f"stop reason {demo['stop_reason']}"],
        "m2-data-plane": [f"sync/cache parity {plane['parity']}", f"forced-eviction parity {plane['forced_eviction_generation']['full_generation_parity']}", f"physical reads {human_bytes(plane['metrics']['physical_read_bytes'])}"],
        "m3-cache-matrix": [f"{human_bytes(capacity)} / 6 policies / {min(values) / 1048576:.3f}-{max(values) / 1048576:.3f} MiB modeled" for capacity, values in sorted(groups.items())],
        "m4-model-check": [f"{row['check_id']} / {row['status']} / {row['backend']}" for row in ledger],
    }


def workflow_gif(parsed: dict[str, Any], sources: dict[str, str], revision: str, path: pathlib.Path) -> tuple[list[int], list[list[int]]]:
    pillow()
    width, height = 1280, 720
    title_font, body_font, small_font = font(29), font(19), font(14)
    values = summaries(parsed)
    frames, all_bounds = [], []
    titles = ("Artifact to deterministic tokens", "Verified bounded data plane", "Offline policy smoke matrix", "Tiny-model correctness ledger")
    for index, command in enumerate(COMMANDS):
        image = Image.new("RGB", (width, height), COLORS["bg"])
        draw = ImageDraw.Draw(image)
        bounds = [24, 24, width - 24, height - 24]
        draw.rounded_rectangle(tuple(bounds), radius=22, fill=COLORS["panel"], outline=COLORS["border"], width=2)
        draw_text(draw, (58, 56), f"FRAME {index + 1}/4 / {command.milestone}", small_font, COLORS["teal"], image.size, bounds)
        draw_text(draw, (58, 94), titles[index], title_font, COLORS["ink"], image.size, bounds)
        y = 150
        for line in pixel_wrap(draw, "$ " + " ".join(command.argv), small_font, width - 116):
            draw_text(draw, (58, y), line, small_font, COLORS["blue"], image.size, bounds)
            y += 23
        y += 18
        draw.rounded_rectangle((58, y, width - 58, y + 280), radius=16, fill=COLORS["panel2"], outline=COLORS["border"], width=2)
        y += 47
        for line in values[command.capture_id]:
            draw_text(draw, (86, y), line, body_font, COLORS["ink"], image.size, bounds)
            y += 39
        draw_text(draw, (58, height - 91), "Actual captured stdout / deterministic rendering / not a screen recording or timing benchmark", small_font, COLORS["amber"], image.size, bounds)
        draw_text(draw, (58, height - 59), f"source {revision[:12]} / stdout {sources[command.capture_id][:24]}...", small_font, COLORS["muted"], image.size, bounds)
        need(bounds[0] >= MARGIN and bounds[1] >= MARGIN and bounds[2] <= width - MARGIN and bounds[3] <= height - MARGIN, "GIF bounds escaped")
        frames.append(image.convert("P", palette=Image.Palette.ADAPTIVE, colors=64))
        all_bounds.append(bounds.copy())
    comment = json.dumps({"schema": ASSET_SCHEMA, "revision": revision, "sources": sources, "rendering": "actual stdout; deterministic frames; not a screen recording"}, sort_keys=True, separators=(",", ":")).encode()
    frames[0].save(path, format="GIF", save_all=True, append_images=frames[1:], duration=[1400, 1400, 1400, 1600], loop=0, disposal=2, optimize=False, comment=comment)
    path.chmod(0o600)
    return [width, height], all_bounds


def record(root: pathlib.Path, path: pathlib.Path, role: str) -> dict[str, Any]:
    relative = path.relative_to(root).as_posix()
    need(SAFE_PATH.fullmatch(relative) is not None, f"unsafe artifact path {relative}")
    return {"path": relative, "role": role, "bytes": path.stat().st_size, "sha256": file_digest(path)}


def tool(executable: str, argv: Sequence[str], root: pathlib.Path) -> dict[str, str]:
    resolved = shutil.which(executable)
    need(resolved is not None, f"{executable} is unavailable")
    path = pathlib.Path(resolved).resolve()
    return {"version": small(argv, root), "executable_sha256": file_digest(path)}


def candidate_readme(revision: str) -> bytes:
    return f"""# RunnelMoE visual-evidence candidate

Source revision: {revision}

Candidate only: these files have not been adopted into the repository. Raw
stdout/stderr is authoritative. Every visual joins to raw or committed source
by SHA-256 in manifest.json.

The PNG is a deterministic rendering of actual captured stdout, not an OS
screenshot. The four-frame GIF is derived from the four actual stdout streams,
not a screen recording. The M3 view is a 64-step functional smoke test, not the
accepted 30-seed experiment or a timing benchmark. The M4 ledger is
correctness-only. Volatile M2 timing/RSS values remain in raw stdout and are
not visualized as performance evidence. The architecture is parsed from
committed M1-M4 Cargo manifests and contains no scheduler or M5 claim.

SHA256SUMS covers every other artifact file, including manifest.json.
""".encode()


def build(root: pathlib.Path, output: pathlib.Path, repository: str, revision: str, run_id: str, run_attempt: str) -> dict[str, Any]:
    need(sys.version_info[:3] == EXPECTED_PYTHON, "Python patch version is not pinned")
    pillow()
    need(SAFE_REPOSITORY.fullmatch(repository) is not None, "repository name is unsafe")
    need(REVISION.fullmatch(revision) is not None and run_id.isdigit() and run_attempt.isdigit(), "workflow provenance is invalid")
    root, output = root.resolve(), output.resolve()
    need(root.is_dir() and not output.exists() and not output.is_relative_to(root), "capture paths are invalid")
    need(small(("git", "rev-parse", "HEAD"), root) == revision, "HEAD differs from revision")
    need(small(("git", "status", "--porcelain=v1", "--untracked-files=all"), root) == "", "checkout is dirty")
    tree = small(("git", "show", "-s", "--format=%T", "HEAD"), root)
    timestamp = small(("git", "show", "-s", "--format=%ct", "HEAD"), root)
    need(REVISION.fullmatch(tree) is not None and timestamp.isdigit(), "Git source state is invalid")
    output.mkdir(mode=0o700)
    (output / "raw").mkdir(mode=0o700)
    (output / "visuals").mkdir(mode=0o700)
    environment = {
        "CARGO_NET_OFFLINE": "true",
        "CARGO_TERM_COLOR": "never",
        "HOME": os.environ.get("HOME", ""),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": os.environ["PATH"],
        "RUST_BACKTRACE": "0",
    }
    recorded_env = {key: environment[key] for key in ("CARGO_NET_OFFLINE", "CARGO_TERM_COLOR", "LANG", "LC_ALL", "RUST_BACKTRACE")}
    parsed, command_records, sources = {}, [], {}
    for command in COMMANDS:
        stdout, stderr = bounded_run(
            command.argv, root, environment, command.capture_id
        )
        scan(stdout, command.capture_id + ".stdout")
        scan(stderr, command.capture_id + ".stderr")
        parsed[command.capture_id] = parse(command, stdout)
        stdout_path = output / "raw" / (command.capture_id + ".stdout")
        stderr_path = output / "raw" / (command.capture_id + ".stderr")
        write_new(stdout_path, stdout)
        write_new(stderr_path, stderr)
        stdout_record, stderr_record = record(output, stdout_path, "raw-stdout"), record(output, stderr_path, "raw-stderr")
        sources[command.capture_id] = stdout_record["sha256"]
        command_records.append({"id": command.capture_id, "milestone": command.milestone, "argv": list(command.argv), "cwd": "$REPOSITORY", "environment": recorded_env, "exit_code": 0, "stdout": stdout_record, "stderr": stderr_record})

    assets = []
    demo = parsed["m1-demo"]
    svg, canvas, bounds = svg_card(
        "Artifact-to-token demo", "M1 / ACTUAL CLI RESULT",
        "Generated fixture to authenticated adapter to deterministic tokens",
        [("Adapter", [f"{demo['adapter']['id']} / version {demo['adapter']['version']}", f"artifact {demo['artifact_id']}"]),
         ("Generation", [f"input tokens {demo['input_token_count']}", f"generated IDs {demo['generated_ids']}", f"decoded text {demo['text']!r}", f"stop reason {demo['stop_reason']}"])],
        "Deterministic systems-test vector only; not language-quality or performance evidence.",
        revision, sources["m1-demo"],
    )
    path = output / "visuals" / "m1-demo.svg"
    write_new(path, svg)
    assets.append({"path": "visuals/m1-demo.svg", "kind": "svg", "canvas": canvas, "content_bounds": bounds, "source_capture_ids": ["m1-demo"], "source_stdout_sha256": sources["m1-demo"]})

    plane = parsed["m2-data-plane"]
    metric, forced = plane["metrics"], plane["forced_eviction_generation"]
    svg, canvas, bounds = svg_card(
        "Verified bounded data plane", "M2 / ACTUAL CLI RESULT",
        "Sync/cache parity and a forced-eviction demand trace",
        [("Parity", [f"sync/cache parity {plane['parity']}", f"forced-eviction generation parity {forced['full_generation_parity']}", f"generated IDs {plane['generated_ids']} / text {plane['generated_text']!r}"]),
         ("Stable accounting", [f"physical reads {human_bytes(metric['physical_read_bytes'])}", f"hits {metric['hits']} / misses {metric['misses']} / admissions {metric['admissions']} / evictions {metric['evictions']}", f"trace events dropped {metric['trace_events_dropped']}", f"forced cache capacity {human_bytes(forced['cache_capacity_bytes'])}"])],
        "Correctness/observability only. Volatile wait, I/O, and RSS remain in raw stdout and are not visualized as performance.",
        revision, sources["m2-data-plane"],
    )
    path = output / "visuals" / "m2-data-plane.svg"
    write_new(path, svg)
    assets.append({"path": "visuals/m2-data-plane.svg", "kind": "svg", "canvas": canvas, "content_bounds": bounds, "source_capture_ids": ["m2-data-plane"], "source_stdout_sha256": sources["m2-data-plane"]})

    matrix = parsed["m3-cache-matrix"]
    groups: dict[int, dict[str, int]] = {}
    for result in matrix["results"]:
        groups.setdefault(result["capacity_bytes"], {})[result["policy"]] = result["metrics"]["total_physical_load_bytes"]
    matrix_sections = []
    order = ("lru", "slru", "tiny-lfu", "router-admit", "router-prefetch", "belady")
    for capacity, values in sorted(groups.items()):
        baseline = values["lru"]
        matrix_sections.append((human_bytes(capacity), [f"{name}: {human_bytes(values[name])} / {values[name] / baseline:.3f} x LRU" for name in order]))
    svg, canvas, bounds = svg_card(
        "Small offline cache matrix", "M3 / FUNCTIONAL SMOKE",
        "markov_clusters replicate 0 / 64 measured steps / modeled physical bytes",
        matrix_sections,
        "Actual 18-cell CLI smoke. Not the accepted 30-seed experiment, host timing, or a general policy ranking.",
        revision, sources["m3-cache-matrix"],
    )
    path = output / "visuals" / "m3-cache-matrix.svg"
    write_new(path, svg)
    assets.append({"path": "visuals/m3-cache-matrix.svg", "kind": "svg", "canvas": canvas, "content_bounds": bounds, "source_capture_ids": ["m3-cache-matrix"], "source_stdout_sha256": sources["m3-cache-matrix"]})

    ledger = parsed["m4-model-check"]
    lines = []
    for row in ledger:
        lines.append(f"{row['check_id']} / {row['status']} / {row['backend']} / {row['metrics']['representation']}")
        if row["status"] == "ok":
            ratios = [row["metrics"][name]["max_tolerance_ratio"] for name in ("logits_error", "router_score_error", "route_weight_error") if row["metrics"][name] is not None]
            lines.append(f"  tokens and experts exact / deterministic / max tolerance ratio {max(ratios):.6f}")
        else:
            lines.append(f"  explicitly unsupported: {row['failure']}")
    svg, canvas, bounds = svg_card(
        "Tiny-model correctness ledger", "M4 / ACTUAL CLI JSONL",
        "Three complete-model checks; AVX2 may be explicitly unsupported",
        [("Closed three-row protocol", lines)],
        "Correctness only. This command emits no timing and supports no inference, serving, model, or storage speed claim.",
        revision, sources["m4-model-check"],
    )
    path = output / "visuals" / "m4-model-check.svg"
    write_new(path, svg)
    assets.append({"path": "visuals/m4-model-check.svg", "kind": "svg", "canvas": canvas, "content_bounds": bounds, "source_capture_ids": ["m4-model-check"], "source_stdout_sha256": sources["m4-model-check"]})

    svg, canvas, bounds, nodes, edges, graph_sources = architecture_svg(root, revision)
    path = output / "visuals" / "m1-m4-architecture.svg"
    write_new(path, svg)
    assets.append({"path": "visuals/m1-m4-architecture.svg", "kind": "svg", "canvas": canvas, "content_bounds": bounds, "source_capture_ids": [], "source_manifest_files": graph_sources})

    path = output / "visuals" / "m1-demo-transcript.png"
    canvas, bounds = transcript_png(COMMANDS[0], (output / "raw" / "m1-demo.stdout").read_bytes(), revision, sources["m1-demo"], path)
    assets.append({"path": "visuals/m1-demo-transcript.png", "kind": "png", "canvas": canvas, "content_bounds": bounds, "source_capture_ids": ["m1-demo"], "source_stdout_sha256": sources["m1-demo"], "rendering": "actual stdout; deterministic rendering; not an OS screenshot"})

    path = output / "visuals" / "m1-m4-workflow.gif"
    canvas, bounds = workflow_gif(parsed, sources, revision, path)
    assets.append({"path": "visuals/m1-m4-workflow.gif", "kind": "gif", "canvas": canvas, "content_bounds": bounds, "frames": 4, "source_capture_ids": [item.capture_id for item in COMMANDS], "source_stdout_sha256": sources, "rendering": "actual stdout; deterministic frames; not a screen recording"})

    write_new(output / "README.md", candidate_readme(revision))
    requirements = (root / "scripts" / "visual-requirements.txt").read_bytes()
    scan(requirements, "visual-requirements.txt")
    wheel_hash = re.search(rb"--hash=sha256:([0-9a-f]{64})", requirements)
    need(wheel_hash is not None, "Pillow wheel hash is missing")

    file_records = []
    for candidate in sorted(output.rglob("*")):
        if candidate.is_file():
            role = "candidate-visual" if candidate.parent.name == "visuals" else "raw-stream" if candidate.parent.name == "raw" else "candidate-documentation"
            file_records.append(record(output, candidate, role))
    source_paths = (
        ".github/workflows/ci.yml", "Cargo.lock", "rust-toolchain.toml",
        "scripts/generate_visual_evidence.py", "scripts/visual-requirements.txt",
    )
    source_files = []
    for relative in source_paths:
        path = root / relative
        source_files.append({"path": relative, "bytes": path.stat().st_size, "sha256": file_digest(path)})
    manifest = {
        "schema": SCHEMA,
        "candidate_only": True,
        "claim_scope": "accepted M1-M4 only; no M5 claim",
        "repository": repository,
        "source": {"revision": revision, "tree": tree, "commit_unix_timestamp": int(timestamp), "clean_checkout": True, "files": source_files},
        "workflow": {"event": "workflow_dispatch", "run_id": int(run_id), "run_attempt": int(run_attempt)},
        "commands": command_records,
        "architecture": {"derivation": "committed Cargo manifests", "nodes": nodes, "edges": [{"consumer": left, "dependency": right} for left, right in edges], "source_files": graph_sources, "excludes_unaccepted_scheduler_claim": True},
        "assets": assets,
        "tools": {
            "python": {"version": platform.python_version(), "executable_sha256": file_digest(pathlib.Path(sys.executable).resolve())},
            "pillow": {"version": importlib.metadata.version("Pillow"), "wheel_sha256": wheel_hash.group(1).decode(), "requirements_sha256": digest(requirements)},
            "cargo": tool("cargo", ("cargo", "--version", "--verbose"), root),
            "rustc": tool("rustc", ("rustc", "--version", "--verbose"), root),
            "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
        },
        "files": file_records,
    }
    write_new(output / "manifest.json", (json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())
    paths = sorted(path for path in output.rglob("*") if path.is_file())
    write_new(output / "SHA256SUMS", "".join(f"{file_digest(path)}  {path.relative_to(output).as_posix()}\n" for path in paths).encode())
    need(small(("git", "status", "--porcelain=v1", "--untracked-files=all"), root) == "", "capture dirtied checkout")
    return verify(output, revision)


def sums(data: bytes) -> dict[str, str]:
    result = {}
    for index, line in enumerate(scan(data, "SHA256SUMS").splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*)", line)
        need(match is not None and match.group(2) not in result, f"bad SHA256SUMS line {index}")
        result[match.group(2)] = match.group(1)
    return result


def border_clear(image: Any) -> bool:
    rgb = image.convert("RGB")
    color = rgb.getpixel((0, 0))
    width, height = rgb.size
    points = [(x, 0) for x in range(width)] + [(x, height - 1) for x in range(width)]
    points += [(0, y) for y in range(height)] + [(width - 1, y) for y in range(height)]
    return all(rgb.getpixel(point) == color for point in points)


def verify(root: pathlib.Path, expected_revision: str | None = None) -> dict[str, Any]:
    pillow()
    need(not root.is_symlink(), "artifact root must not be a symlink")
    root = root.resolve()
    need(root.is_dir(), "artifact root is unsafe")
    entries = list(root.rglob("*"))
    for entry in entries:
        need(
            not entry.is_symlink() and (entry.is_file() or entry.is_dir()),
            f"artifact contains unsafe entry {entry.relative_to(root)}",
        )
    directories = {
        entry.relative_to(root).as_posix() for entry in entries if entry.is_dir()
    }
    need(directories == {"raw", "visuals"}, "artifact directory topology changed")
    manifest = strict_json((root / "manifest.json").read_bytes(), "manifest.json")
    need(manifest["schema"] == SCHEMA and manifest["candidate_only"] is True, "manifest schema changed")
    revision = manifest["source"]["revision"]
    need(REVISION.fullmatch(revision) is not None and (expected_revision is None or revision == expected_revision), "revision changed")
    need([row["argv"] for row in manifest["commands"]] == [list(item.argv) for item in COMMANDS], "command vectors changed")
    declared = {row["path"]: row for row in manifest["files"]}
    regular_files = [entry for entry in entries if entry.is_file()]
    actual = {
        path.relative_to(root).as_posix()
        for path in regular_files
        if path.name not in {"manifest.json", "SHA256SUMS"}
    }
    need(set(declared) == actual, "manifest inventory is not closed")
    for relative, row in declared.items():
        need(SAFE_PATH.fullmatch(relative) is not None, "unsafe artifact path")
        path = root / relative
        need(path.is_file() and not path.is_symlink() and path.stat().st_size == row["bytes"] <= MAX_FILE and file_digest(path) == row["sha256"], f"file custody failed for {relative}")
    checksums = sums((root / "SHA256SUMS").read_bytes())
    expected_sums = {
        path.relative_to(root).as_posix()
        for path in regular_files
        if path.name != "SHA256SUMS"
    }
    need(set(checksums) == expected_sums, "checksum inventory is not closed")
    for relative, value in checksums.items():
        need(file_digest(root / relative) == value, f"checksum failed for {relative}")

    for command, row in zip(COMMANDS, manifest["commands"], strict=True):
        stdout = (root / row["stdout"]["path"]).read_bytes()
        stderr = (root / row["stderr"]["path"]).read_bytes()
        scan(stdout, row["stdout"]["path"])
        scan(stderr, row["stderr"]["path"])
        need(digest(stdout) == row["stdout"]["sha256"] and digest(stderr) == row["stderr"]["sha256"], "raw stream join failed")
        parse(command, stdout)

    expected_assets = {
        "visuals/m1-demo.svg", "visuals/m2-data-plane.svg",
        "visuals/m3-cache-matrix.svg", "visuals/m4-model-check.svg",
        "visuals/m1-m4-architecture.svg", "visuals/m1-demo-transcript.png",
        "visuals/m1-m4-workflow.gif",
    }
    need({row["path"] for row in manifest["assets"]} == expected_assets, "asset inventory changed")
    forbidden_tags = {"script", "foreignObject", "image", "use", "animate", "set", "clipPath", "mask"}
    for asset in manifest["assets"]:
        path = root / asset["path"]
        width, height = asset["canvas"]
        need(width <= MAX_WIDTH and height <= MAX_HEIGHT, "asset canvas exceeded")
        bounds = asset["content_bounds"]
        bound_rows = bounds if asset["kind"] == "gif" else [bounds]
        need(all(row[0] >= MARGIN and row[1] >= MARGIN and row[2] <= width - MARGIN and row[3] <= height - MARGIN for row in bound_rows), "asset content bounds escaped")
        if asset["kind"] == "svg":
            data = path.read_bytes()
            scan(data, asset["path"])
            document = ET.fromstring(data)
            need(document.tag == f"{{{SVG_NS}}}svg" and document.attrib["viewBox"] == f"0 0 {width} {height}", "SVG frame changed")
            for element in document.iter():
                local = element.tag.rsplit("}", 1)[-1]
                need(local not in forbidden_tags, f"forbidden SVG element {local}")
                for key, value in element.attrib.items():
                    lowered = value.lower()
                    need(not key.rsplit("}", 1)[-1].lower().startswith("on") and "javascript:" not in lowered and "data:" not in lowered and "http://" not in lowered and "https://" not in lowered, "unsafe SVG reference")
                if element.text:
                    need("url(" not in element.text.lower(), "unsafe SVG text")
        elif asset["kind"] == "png":
            with Image.open(path) as image:
                need(image.format == "PNG" and image.n_frames == 1 and list(image.size) == asset["canvas"], "PNG frame changed")
                need(image.info["Schema"] == ASSET_SCHEMA and image.info["SourceRevision"] == revision and image.info["SourceStdoutSHA256"] == asset["source_stdout_sha256"], "PNG provenance changed")
                need("not an OS screenshot" in image.info["RenderContract"] and json.loads(image.info["ContentBounds"]) == bounds and border_clear(image), "PNG bounds/render contract changed")
        else:
            with Image.open(path) as image:
                need(
                    image.format == "GIF"
                    and image.n_frames == 4
                    and list(image.size) == asset["canvas"]
                    and image.info.get("loop") == 0
                    and "transparency" not in image.info,
                    "GIF frame contract changed",
                )
                comment = json.loads(image.info["comment"].decode())
                need(comment["schema"] == ASSET_SCHEMA and comment["revision"] == revision and "not a screen recording" in comment["rendering"], "GIF provenance changed")
                durations = (1400, 1400, 1400, 1600)
                for index, duration in enumerate(durations):
                    image.seek(index)
                    tiles = list(image.tile)
                    need(
                        image.size == tuple(asset["canvas"])
                        and len(tiles) == 1
                        and tiles[0][1] == (0, 0, *asset["canvas"])
                        and image.info.get("duration") == duration
                        and getattr(image, "disposal_method", None) == 2
                        and "transparency" not in image.info
                        and border_clear(image),
                        f"GIF frame {index} is not a full opaque frame",
                    )
    for path in [root / "README.md", root / "manifest.json", root / "SHA256SUMS", *sorted((root / "raw").glob("*")), *sorted((root / "visuals").glob("*.svg"))]:
        scan(path.read_bytes(), path.relative_to(root).as_posix())
    return {"revision": revision, "manifest_sha256": file_digest(root / "manifest.json"), "capture_count": len(COMMANDS), "asset_count": len(manifest["assets"]), "file_count": len(checksums) + 1}


def arguments(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    capture = commands.add_parser("capture")
    capture.add_argument("--repo-root", required=True, type=pathlib.Path)
    capture.add_argument("--output", required=True, type=pathlib.Path)
    capture.add_argument("--repository", required=True)
    capture.add_argument("--revision", required=True)
    capture.add_argument("--run-id", required=True)
    capture.add_argument("--run-attempt", required=True)
    check = commands.add_parser("verify")
    check.add_argument("--input", required=True, type=pathlib.Path)
    check.add_argument("--expected-revision")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    options = arguments(sys.argv[1:] if argv is None else argv)
    try:
        if options.action == "capture":
            result = build(options.repo_root, options.output, options.repository, options.revision, options.run_id, options.run_attempt)
        else:
            result = verify(options.input, options.expected_revision)
    except (ContractError, KeyError, OSError, TypeError, ValueError, subprocess.SubprocessError, ET.ParseError) as error:
        print(f"visual evidence: FAILED: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"status": "ok", **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
