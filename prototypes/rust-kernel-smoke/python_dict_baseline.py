#!/usr/bin/env python3
"""Python dict/object baseline for the Rust kernel smoke test.

This intentionally mirrors EVAS' current hot-path shape more than an optimized
Python numeric loop: string node names, dict snapshots, and tuple model maps.
"""

from __future__ import annotations

import argparse
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--models", type=int, default=64)
    parser.add_argument("--record-stride", type=int, default=16)
    return parser.parse_args()


def run_kernel(steps: int, models: int, record_stride: int) -> tuple[int, float, float]:
    nodes = {f"n{i}": 0.0 for i in range(models * 3 + 4)}
    states = {f"s{i}": 0.0 for i in range(models)}
    last_diff = {f"d{i}": 0.0 for i in range(models)}
    model_specs = []
    for model in range(models):
        base = 4 + model * 3
        inp = "n0" if model == 0 else f"n{base - 1}"
        inn = "n1" if model % 2 == 0 else "n3"
        model_specs.append(
            (
                model,
                inp,
                inn,
                f"n{base}",
                f"n{base + 1}",
                f"n{base + 2}",
                f"s{model}",
                f"d{model}",
                -0.1 + 0.2 * (model % 7) / 6.0,
            )
        )

    phase = 0.0
    events = 0
    checksum = 0.0
    err_acc = 0.0
    last_node = f"n{models * 3 + 3}"

    for step in range(steps):
        prev = dict(nodes)

        phase += 0.000_013
        if phase >= 1.0:
            phase -= 1.0
        nodes["n0"] = phase
        nodes["n1"] = 1.0 - phase
        nodes["n2"] = 1.0 if phase > 0.5 else 0.0
        nodes["n3"] = 1.0 if 0.25 < phase < 0.75 else 0.0

        for _, inp, inn, diff_node, state_node, out, state_key, diff_key, threshold in model_specs:
            diff = nodes[inp] - nodes[inn]
            states[state_key] += 0.0025 * (diff - states[state_key])
            nodes[diff_node] = diff
            nodes[state_node] = states[state_key]
            nodes[out] = 0.55 + 0.30 * states[state_key] + 0.05 * nodes["n2"]
            if last_diff[diff_key] <= threshold < diff:
                events += 1
            last_diff[diff_key] = diff

        for node, value in nodes.items():
            err_acc += abs(value - prev[node])

        if step % record_stride == 0:
            checksum += nodes[last_node] + nodes["n0"] * 0.125 + err_acc * 1.0e-12

    return events, checksum, err_acc


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.models <= 0 or args.record_stride <= 0:
        raise SystemExit("steps, models, and record-stride must be positive")
    started = time.perf_counter()
    events, checksum, err_acc = run_kernel(args.steps, args.models, args.record_stride)
    elapsed = time.perf_counter() - started
    print(
        "engine=python_dict "
        f"steps={args.steps} models={args.models} record_stride={args.record_stride} "
        f"elapsed_s={elapsed:.6f} events={events} checksum={checksum:.9f} err_acc={err_acc:.9f}"
    )


if __name__ == "__main__":
    main()
