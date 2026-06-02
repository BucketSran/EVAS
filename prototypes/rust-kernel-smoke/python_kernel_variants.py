#!/usr/bin/env python3
"""Python-side kernel variants for EVAS Rust planning.

The goal is to separate three effects:

1. current-style string/dict node access;
2. Python indexed-array access;
3. event-queue scheduling versus fixed-step scheduling.
"""

from __future__ import annotations

import argparse
import heapq
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kernel",
        choices=[
            "measurement-dict",
            "measurement-indexed",
            "pfd-fixed-step",
            "pfd-event-queue",
        ],
        default="measurement-dict",
    )
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--models", type=int, default=64)
    parser.add_argument("--record-stride", type=int, default=16)
    return parser.parse_args()


def run_measurement_dict(steps: int, models: int, record_stride: int) -> tuple[int, int, float, float]:
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

        for inp, inn, diff_node, state_node, out, state_key, diff_key, threshold in model_specs:
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

    return steps, events, checksum, err_acc


def run_measurement_indexed(steps: int, models: int, record_stride: int) -> tuple[int, int, float, float]:
    node_count = models * 3 + 4
    prev = [0.0] * node_count
    curr = [0.0] * node_count
    state = [0.0] * models
    last_diff = [0.0] * models
    events = 0
    checksum = 0.0
    phase = 0.0
    err_acc = 0.0

    for step in range(steps):
        prev[:] = curr
        phase += 0.000_013
        if phase >= 1.0:
            phase -= 1.0
        curr[0] = phase
        curr[1] = 1.0 - phase
        curr[2] = 1.0 if phase > 0.5 else 0.0
        curr[3] = 1.0 if 0.25 < phase < 0.75 else 0.0

        for model in range(models):
            base = 4 + model * 3
            inp = 0 if model == 0 else base - 1
            inn = 1 if model % 2 == 0 else 3
            out = base + 2
            diff = curr[inp] - curr[inn]
            state[model] += 0.0025 * (diff - state[model])
            curr[base] = diff
            curr[base + 1] = state[model]
            curr[out] = 0.55 + 0.30 * state[model] + 0.05 * curr[2]
            threshold = -0.1 + 0.2 * (model % 7) / 6.0
            if last_diff[model] <= threshold < diff:
                events += 1
            last_diff[model] = diff

        for idx, value in enumerate(curr):
            err_acc += abs(value - prev[idx])

        if step % record_stride == 0:
            checksum += curr[node_count - 1] + curr[0] * 0.125 + err_acc * 1.0e-12

    return steps, events, checksum, err_acc


def run_pfd_fixed_step(steps: int, models: int, record_stride: int) -> tuple[int, int, float, float]:
    stop_ticks = 300_000
    ref_period = 20_000
    div_period = 20_000
    div_phase = 700
    state = [0.0] * max(1, models)
    last_ref = False
    last_div = False
    up = False
    dn = False
    events = 0
    checksum = 0.0
    err_acc = 0.0

    for step in range(steps):
        tick = step * stop_ticks // steps
        ref_clk = tick % ref_period < ref_period // 2
        div_clk = (tick + div_phase) % div_period < div_period // 2
        ref_rise = ref_clk and not last_ref
        div_rise = div_clk and not last_div

        if ref_rise:
            up = True
            events += 1
        if div_rise:
            dn = True
            events += 1
        if up and dn:
            up = False
            dn = False

        drive = (1.0 if up else 0.0) - (1.0 if dn else 0.0)
        for idx, value in enumerate(state):
            gain = 0.001 + (idx % 5) * 0.0002
            new_value = value + gain * (drive - value)
            state[idx] = new_value
            err_acc += abs(new_value - value)

        if step % record_stride == 0:
            checksum += state[-1] + (0.25 if up else 0.0)
        last_ref = ref_clk
        last_div = div_clk

    return steps, events, checksum, err_acc


def run_pfd_event_queue(steps: int, models: int, record_stride: int) -> tuple[int, int, float, float]:
    stop_ticks = 300_000
    ref_period = 20_000
    div_period = 20_000
    div_phase = 700
    fixed_dt = max(1, stop_ticks // max(1, steps))
    record_period = max(1, fixed_dt * record_stride)
    heap: list[tuple[int, int]] = []
    state = [0.0] * max(1, models)
    up = False
    dn = False
    events = 0
    processed_steps = 0
    checksum = 0.0
    err_acc = 0.0

    tick = 0
    while tick <= stop_ticks:
        heapq.heappush(heap, (tick, 0))
        tick += ref_period
    tick = div_phase
    while tick <= stop_ticks:
        heapq.heappush(heap, (tick, 1))
        tick += div_period
    tick = 0
    while tick <= stop_ticks:
        heapq.heappush(heap, (tick, 2))
        tick += record_period

    last_tick = 0
    while heap:
        tick, kind = heapq.heappop(heap)
        if tick > stop_ticks:
            continue
        dt = max(1, tick - last_tick)
        alpha = min(1.0, dt / 20_000.0) * 0.025
        drive = (1.0 if up else 0.0) - (1.0 if dn else 0.0)
        for idx, value in enumerate(state):
            gain = alpha * (1.0 + (idx % 5) * 0.1)
            new_value = value + gain * (drive - value)
            state[idx] = new_value
            err_acc += abs(new_value - value)

        if kind == 0:
            up = True
            events += 1
        elif kind == 1:
            dn = True
            events += 1
        else:
            checksum += state[-1] + (0.25 if up else 0.0)
        if up and dn:
            up = False
            dn = False
        last_tick = tick
        processed_steps += 1

    return processed_steps, events, checksum, err_acc


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.models <= 0 or args.record_stride <= 0:
        raise SystemExit("steps, models, and record-stride must be positive")

    kernels = {
        "measurement-dict": run_measurement_dict,
        "measurement-indexed": run_measurement_indexed,
        "pfd-fixed-step": run_pfd_fixed_step,
        "pfd-event-queue": run_pfd_event_queue,
    }
    started = time.perf_counter()
    processed_steps, events, checksum, err_acc = kernels[args.kernel](
        args.steps,
        args.models,
        args.record_stride,
    )
    elapsed = time.perf_counter() - started
    print(
        "engine=python_variant "
        f"kernel={args.kernel} requested_steps={args.steps} processed_steps={processed_steps} "
        f"models={args.models} record_stride={args.record_stride} elapsed_s={elapsed:.6f} "
        f"events={events} checksum={checksum:.9f} err_acc={err_acc:.9f}"
    )


if __name__ == "__main__":
    main()
