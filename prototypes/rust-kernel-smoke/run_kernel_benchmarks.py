#!/usr/bin/env python3
"""Run repeated kernel smoke benchmarks and summarize median speedups."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
RUST_BIN = HERE / "target" / "release" / "evas-rust-kernel-smoke"
ELAPSED_RE = re.compile(r"elapsed_s=([0-9.]+)")
STEPS_RE = re.compile(r"processed_steps=([0-9]+)")
EVENTS_RE = re.compile(r"events=([0-9]+)")
CHECKSUM_RE = re.compile(r"checksum=([-0-9.]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--pfd-steps", type=int, default=60_062)
    parser.add_argument("--models", type=int, default=64)
    parser.add_argument("--record-stride", type=int, default=16)
    parser.add_argument("--output", type=Path, default=HERE / "benchmark_results_20260602.json")
    return parser.parse_args()


def run_command(command: list[str]) -> dict:
    completed = subprocess.run(
        command,
        cwd=HERE,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout = completed.stdout.strip()
    elapsed_match = ELAPSED_RE.search(stdout)
    steps_match = STEPS_RE.search(stdout)
    events_match = EVENTS_RE.search(stdout)
    checksum_match = CHECKSUM_RE.search(stdout)
    if elapsed_match is None or steps_match is None:
        raise RuntimeError(f"could not parse output: {stdout}")
    return {
        "elapsed_s": float(elapsed_match.group(1)),
        "processed_steps": int(steps_match.group(1)),
        "events": int(events_match.group(1)) if events_match else None,
        "checksum": float(checksum_match.group(1)) if checksum_match else None,
        "stdout": stdout,
    }


def summarize(samples: list[dict]) -> dict:
    elapsed = [sample["elapsed_s"] for sample in samples]
    return {
        "runs": len(samples),
        "median_s": statistics.median(elapsed),
        "min_s": min(elapsed),
        "max_s": max(elapsed),
        "processed_steps": samples[-1]["processed_steps"],
        "events": samples[-1]["events"],
        "checksum": samples[-1]["checksum"],
        "samples": samples,
    }


def main() -> None:
    args = parse_args()
    if args.repeats <= 0:
        raise SystemExit("repeats must be positive")

    subprocess.run(["cargo", "build", "--release"], cwd=HERE, check=True)

    configs = [
        (
            "python_measurement_dict",
            [
                "python3",
                "python_kernel_variants.py",
                "--kernel",
                "measurement-dict",
                "--steps",
                str(args.steps),
                "--models",
                str(args.models),
                "--record-stride",
                str(args.record_stride),
            ],
        ),
        (
            "python_measurement_indexed",
            [
                "python3",
                "python_kernel_variants.py",
                "--kernel",
                "measurement-indexed",
                "--steps",
                str(args.steps),
                "--models",
                str(args.models),
                "--record-stride",
                str(args.record_stride),
            ],
        ),
        (
            "rust_measurement_indexed",
            [
                str(RUST_BIN),
                "--kernel",
                "measurement-indexed",
                "--steps",
                str(args.steps),
                "--models",
                str(args.models),
                "--record-stride",
                str(args.record_stride),
            ],
        ),
        (
            "python_pfd_fixed_step",
            [
                "python3",
                "python_kernel_variants.py",
                "--kernel",
                "pfd-fixed-step",
                "--steps",
                str(args.pfd_steps),
                "--models",
                str(args.models),
                "--record-stride",
                str(args.record_stride),
            ],
        ),
        (
            "python_pfd_event_queue",
            [
                "python3",
                "python_kernel_variants.py",
                "--kernel",
                "pfd-event-queue",
                "--steps",
                str(args.pfd_steps),
                "--models",
                str(args.models),
                "--record-stride",
                str(args.record_stride),
            ],
        ),
        (
            "rust_pfd_fixed_step",
            [
                str(RUST_BIN),
                "--kernel",
                "pfd-fixed-step",
                "--steps",
                str(args.pfd_steps),
                "--models",
                str(args.models),
                "--record-stride",
                str(args.record_stride),
            ],
        ),
        (
            "rust_pfd_event_queue",
            [
                str(RUST_BIN),
                "--kernel",
                "pfd-event-queue",
                "--steps",
                str(args.pfd_steps),
                "--models",
                str(args.models),
                "--record-stride",
                str(args.record_stride),
            ],
        ),
    ]

    results: dict[str, dict] = {}
    for name, command in configs:
        samples = [run_command(command) for _ in range(args.repeats)]
        results[name] = summarize(samples)

    speedups = {
        "python_indexed_over_python_dict": (
            results["python_measurement_dict"]["median_s"]
            / results["python_measurement_indexed"]["median_s"]
        ),
        "rust_indexed_over_python_dict": (
            results["python_measurement_dict"]["median_s"]
            / results["rust_measurement_indexed"]["median_s"]
        ),
        "rust_indexed_over_python_indexed": (
            results["python_measurement_indexed"]["median_s"]
            / results["rust_measurement_indexed"]["median_s"]
        ),
        "python_pfd_event_over_fixed": (
            results["python_pfd_fixed_step"]["median_s"]
            / results["python_pfd_event_queue"]["median_s"]
        ),
        "rust_pfd_event_over_fixed": (
            results["rust_pfd_fixed_step"]["median_s"]
            / results["rust_pfd_event_queue"]["median_s"]
        ),
        "rust_pfd_event_over_python_pfd_fixed": (
            results["python_pfd_fixed_step"]["median_s"]
            / results["rust_pfd_event_queue"]["median_s"]
        ),
    }

    output = {
        "repeats": args.repeats,
        "steps": args.steps,
        "pfd_steps": args.pfd_steps,
        "models": args.models,
        "record_stride": args.record_stride,
        "results": results,
        "speedups": speedups,
    }
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
