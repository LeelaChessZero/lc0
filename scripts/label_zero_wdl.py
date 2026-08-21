#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later

"""Label a partitioned Parquet chess dataset with lc0 depth-zero WDL."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


WDL_RE = re.compile(r"\bdepth 0\b.*\bwdl (\d+) (\d+) (\d+)\b")
ENGINE_MARKER = "zero-wdl"
WDL_COLUMNS = ("wdl_win", "wdl_draw", "wdl_loss")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class Lc0:
    def __init__(self, binary: Path, weights: Path, backend: str) -> None:
        self.binary = binary.resolve()
        self.weights = weights.resolve()
        self.backend = backend
        self.stderr_lines: collections.deque[str] = collections.deque(maxlen=200)
        self.backend_ready = threading.Event()
        self.backend_verified = False
        self._cuda_runtime_seen = False
        self.process = subprocess.Popen(
            [
                str(self.binary),
                f"--weights={self.weights}",
                f"--backend={backend}",
                "--minibatch-size=1",
                "--nncache=0",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert self.process.stdin and self.process.stdout and self.process.stderr
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()
        self.version = self._initialize()
        if ENGINE_MARKER not in self.version:
            self.close()
            raise RuntimeError(
                f"lc0 identity {self.version!r} does not contain {ENGINE_MARKER!r}"
            )

    def _drain_stderr(self) -> None:
        assert self.process.stderr
        for raw in self.process.stderr:
            line = raw.rstrip()
            self.stderr_lines.append(line)
            if self.backend == "metal" and "Initialized metal backend on device" in line:
                self.backend_ready.set()
            if self.backend == "cuda":
                if line.startswith("CUDA Runtime version:"):
                    self._cuda_runtime_seen = True
                elif self._cuda_runtime_seen and line.startswith("GPU: "):
                    self.backend_ready.set()

    def _send(self, command: str) -> None:
        if self.process.poll() is not None:
            raise RuntimeError(f"lc0 exited with status {self.process.returncode}")
        assert self.process.stdin
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def _read_until(self, terminator: str) -> list[str]:
        assert self.process.stdout
        lines: list[str] = []
        while True:
            raw = self.process.stdout.readline()
            if raw == "":
                tail = "\n".join(self.stderr_lines)
                raise RuntimeError(f"lc0 closed stdout before {terminator!r}\n{tail}")
            line = raw.rstrip()
            lines.append(line)
            if line == terminator or line.startswith(terminator + " "):
                return lines

    def _initialize(self) -> str:
        self._send("uci")
        lines = self._read_until("uciok")
        names = [line.removeprefix("id name ") for line in lines if line.startswith("id name ")]
        if len(names) != 1:
            raise RuntimeError(f"expected one UCI engine identity, received {names}")
        self._send("isready")
        self._read_until("readyok")
        return names[0]

    def evaluate(self, fen: str) -> tuple[int, int, int]:
        self._send("position fen " + fen)
        self._send("go depth 0")
        lines = self._read_until("bestmove")
        matches = [WDL_RE.search(line) for line in lines]
        values = [tuple(map(int, match.groups())) for match in matches if match]
        if len(values) != 1:
            raise RuntimeError(f"expected one depth-zero WDL response for {fen!r}: {lines}")
        wdl = values[0]
        if sum(wdl) != 1000:
            raise RuntimeError(f"invalid WDL sum for {fen!r}: {wdl}")
        if not self.backend_verified:
            self.require_accelerator()
        return wdl

    def require_accelerator(self) -> None:
        if not self.backend_ready.wait(timeout=2):
            tail = "\n".join(self.stderr_lines)
            raise RuntimeError(
                f"lc0 did not confirm {self.backend} initialization; "
                "refusing CPU fallback\n"
                + tail
            )
        self.backend_verified = True

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self._send("quit")
                self.process.wait(timeout=5)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                self.process.terminate()
                self.process.wait(timeout=5)

    def __enter__(self) -> "Lc0":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def add_wdl(table: pa.Table, values: list[tuple[int, int, int]], metadata: dict[bytes, bytes]) -> pa.Table:
    if any(name in table.column_names for name in WDL_COLUMNS):
        raise RuntimeError("input already contains WDL columns")
    for index, name in enumerate(WDL_COLUMNS):
        table = table.append_column(
            name, pa.array((wdl[index] for wdl in values), type=pa.uint16())
        )
    return table.replace_schema_metadata(metadata)


def chunk_is_valid(path: Path, rows: int, metadata: dict[bytes, bytes]) -> bool:
    if not path.is_file():
        return False
    parquet = pq.ParquetFile(path)
    actual = parquet.schema_arrow.metadata or {}
    return parquet.metadata.num_rows == rows and all(
        actual.get(key) == value for key, value in metadata.items()
    )


def output_is_valid(path: Path, rows: int, metadata: dict[bytes, bytes]) -> bool:
    return chunk_is_valid(path, rows, metadata) and (
        pq.ParquetFile(path).schema_arrow.metadata or {}
    ).get(b"zero_wdl_complete") == b"true"


def label_file(
    source: Path,
    destination: Path,
    work: Path,
    engine: Lc0,
    provenance: dict[bytes, bytes],
    progress: dict[str, float],
) -> None:
    source_hash = sha256(source)
    metadata = dict(pq.ParquetFile(source).schema_arrow.metadata or {})
    metadata.update(provenance)
    metadata[b"source_parquet_sha256"] = source_hash.encode()
    metadata[b"zero_wdl_complete"] = b"false"
    parquet = pq.ParquetFile(source)
    rows = parquet.metadata.num_rows
    final_metadata = dict(metadata)
    final_metadata[b"zero_wdl_complete"] = b"true"
    if output_is_valid(destination, rows, final_metadata):
        progress["done"] += rows
        print(f"skip complete {destination}: {rows:,} rows", flush=True)
        return

    work.mkdir(parents=True, exist_ok=True)
    chunks: list[Path] = []
    for group in range(parquet.num_row_groups):
        table = parquet.read_row_group(group)
        chunk = work / f"row-group-{group:05d}.parquet"
        chunks.append(chunk)
        if chunk_is_valid(chunk, table.num_rows, metadata):
            progress["done"] += table.num_rows
            continue
        fens = table.column("fen").to_pylist()
        wdls: list[tuple[int, int, int]] = []
        for fen in fens:
            wdls.append(engine.evaluate(fen))
            progress["done"] += 1
            if int(progress["done"]) % 100 == 0:
                elapsed = time.monotonic() - progress["started"]
                rate = progress["done"] / elapsed
                remaining = (progress["total"] - progress["done"]) / rate
                print(
                    f"{int(progress['done']):,}/{int(progress['total']):,} "
                    f"({rate:.2f} positions/s, ETA {remaining / 3600:.2f} h)",
                    flush=True,
                )
        labeled = add_wdl(table, wdls, metadata)
        temporary = chunk.with_suffix(".tmp")
        pq.write_table(labeled, temporary, compression="zstd", compression_level=6)
        os.replace(temporary, chunk)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    writer: pq.ParquetWriter | None = None
    try:
        for chunk in chunks:
            table = pq.read_table(chunk).replace_schema_metadata(final_metadata)
            if writer is None:
                writer = pq.ParquetWriter(
                    temporary,
                    table.schema,
                    compression="zstd",
                    compression_level=6,
                    write_statistics=True,
                )
            writer.write_table(table, row_group_size=table.num_rows)
        if writer is None:
            raise RuntimeError(f"source has no row groups: {source}")
        writer.close()
        writer = None
        os.replace(temporary, destination)
    finally:
        if writer is not None:
            writer.close()
    if not output_is_valid(destination, rows, final_metadata):
        raise RuntimeError(f"completed output failed validation: {destination}")
    print(f"completed {destination}: {rows:,} rows", flush=True)


def run(args: argparse.Namespace) -> None:
    source_root = Path(args.input).resolve()
    output_root = Path(args.output).resolve()
    binary = Path(args.lc0).resolve()
    weights = Path(args.weights).resolve()
    if source_root == output_root:
        raise RuntimeError("input and output must differ for atomic, recoverable labeling")
    files = sorted(source_root.glob("source_split=*/phase=*/*.parquet"))
    if not files:
        raise RuntimeError(f"no partitioned Parquet files found under {source_root}")
    for path in (binary, weights):
        if not path.is_file():
            raise RuntimeError(f"missing required file: {path}")

    total = sum(pq.ParquetFile(path).metadata.num_rows for path in files)
    progress = {"done": 0.0, "total": float(total), "started": time.monotonic()}
    binary_hash = sha256(binary)
    weights_hash = sha256(weights)
    with Lc0(binary, weights, args.backend) as engine:
        provenance = {
            b"zero_wdl_command": b"go depth 0",
            b"zero_wdl_perspective": b"side_to_move",
            b"zero_wdl_scale": b"1000",
            b"lc0_backend": args.backend.encode(),
            b"lc0_version": engine.version.encode(),
            b"lc0_binary_sha256": binary_hash.encode(),
            b"lc0_weights_filename": weights.name.encode(),
            b"lc0_weights_sha256": weights_hash.encode(),
        }
        print(
            json.dumps(
                {
                    "rows": total,
                    "lc0_version": engine.version,
                    "lc0_binary_sha256": binary_hash,
                    "weights_sha256": weights_hash,
                    "backend": args.backend,
                },
                indent=2,
            ),
            flush=True,
        )
        for source in files:
            relative = source.relative_to(source_root)
            destination = output_root / relative
            work = output_root / "_work" / relative.parent
            label_file(source, destination, work, engine, provenance, progress)

    partitions = []
    output_total = 0
    for source in files:
        relative = source.relative_to(source_root)
        destination = output_root / relative
        parquet = pq.ParquetFile(destination)
        rows = parquet.metadata.num_rows
        output_total += rows
        partitions.append(
            {
                "file": relative.as_posix(),
                "rows": rows,
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
            }
        )
    if output_total != total:
        raise RuntimeError(f"output row mismatch: {output_total} != {total}")
    manifest = {
        "format": "VPD1-Parquet-ZeroWDL",
        "rows": output_total,
        "wdl_columns": list(WDL_COLUMNS),
        "wdl_scale": 1000,
        "wdl_perspective": "side_to_move",
        "lc0_version": engine.version,
        "lc0_binary_sha256": binary_hash,
        "lc0_weights_filename": weights.name,
        "lc0_weights_sha256": weights_hash,
        "backend": args.backend,
        "partitions": partitions,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = output_root / "_manifest.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output_root / "_manifest.json")
    print(f"complete: {output_total:,} rows", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="unlabeled Parquet dataset directory")
    parser.add_argument("--output", required=True, help="labeled Parquet dataset directory")
    parser.add_argument("--lc0", required=True, help="custom zero-wdl lc0 binary")
    parser.add_argument("--weights", required=True, help="lc0 network weights")
    parser.add_argument(
        "--backend",
        required=True,
        choices=("cuda", "metal"),
        help="hardware accelerator; CPU backends are intentionally unsupported",
    )
    try:
        run(parser.parse_args())
    except (KeyboardInterrupt, BrokenPipeError):
        print("interrupted; completed row-group checkpoints are retained", file=sys.stderr)
        raise SystemExit(130)


if __name__ == "__main__":
    main()
