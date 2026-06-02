use std::env;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::time::Instant;

#[derive(Clone, Debug)]
struct Config {
    kernel: Kernel,
    steps: usize,
    models: usize,
    record_stride: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Kernel {
    MeasurementIndexed,
    PfdFixedStep,
    PfdEventQueue,
}

impl Kernel {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "measurement-indexed" => Ok(Self::MeasurementIndexed),
            "pfd-fixed-step" => Ok(Self::PfdFixedStep),
            "pfd-event-queue" => Ok(Self::PfdEventQueue),
            _ => Err(format!(
                "unknown kernel: {value}; expected measurement-indexed, pfd-fixed-step, or pfd-event-queue"
            )),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::MeasurementIndexed => "measurement-indexed",
            Self::PfdFixedStep => "pfd-fixed-step",
            Self::PfdEventQueue => "pfd-event-queue",
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            kernel: Kernel::MeasurementIndexed,
            steps: 200_000,
            models: 64,
            record_stride: 16,
        }
    }
}

impl Config {
    fn from_args() -> Result<Self, String> {
        let mut cfg = Self::default();
        let mut args = env::args().skip(1);
        while let Some(arg) = args.next() {
            let value = args
                .next()
                .ok_or_else(|| format!("missing value for {arg}"))?;
            match arg.as_str() {
                "--kernel" => cfg.kernel = Kernel::parse(&value)?,
                "--steps" => cfg.steps = parse_positive(&value, "steps")?,
                "--models" => cfg.models = parse_positive(&value, "models")?,
                "--record-stride" => cfg.record_stride = parse_positive(&value, "record-stride")?,
                _ => return Err(format!("unknown argument: {arg}")),
            }
        }
        Ok(cfg)
    }
}

fn parse_positive(value: &str, name: &str) -> Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|err| format!("invalid {name}: {err}"))?;
    if parsed == 0 {
        return Err(format!("{name} must be positive"));
    }
    Ok(parsed)
}

fn run_measurement_indexed(cfg: &Config) -> (usize, u64, f64, f64) {
    let node_count = cfg.models * 3 + 4;
    let mut prev = vec![0.0_f64; node_count];
    let mut curr = vec![0.0_f64; node_count];
    let mut state = vec![0.0_f64; cfg.models];
    let mut last_diff = vec![0.0_f64; cfg.models];
    let mut events = 0_u64;
    let mut checksum = 0.0_f64;
    let mut phase = 0.0_f64;
    let mut err_acc = 0.0_f64;

    for step in 0..cfg.steps {
        prev.copy_from_slice(&curr);

        phase += 0.000_013;
        if phase >= 1.0 {
            phase -= 1.0;
        }
        curr[0] = phase;
        curr[1] = 1.0 - phase;
        curr[2] = if phase > 0.5 { 1.0 } else { 0.0 };
        curr[3] = if phase > 0.25 && phase < 0.75 { 1.0 } else { 0.0 };

        for model in 0..cfg.models {
            let base = 4 + model * 3;
            let inp = if model == 0 { 0 } else { base - 1 };
            let inn = if model % 2 == 0 { 1 } else { 3 };
            let out = base + 2;
            let diff = curr[inp] - curr[inn];
            state[model] += 0.0025 * (diff - state[model]);
            curr[base] = diff;
            curr[base + 1] = state[model];
            curr[out] = 0.55 + 0.30 * state[model] + 0.05 * curr[2];

            let threshold = -0.1 + 0.2 * ((model % 7) as f64) / 6.0;
            if last_diff[model] <= threshold && diff > threshold {
                events += 1;
            }
            last_diff[model] = diff;
        }

        for idx in 0..node_count {
            err_acc += (curr[idx] - prev[idx]).abs();
        }

        if step % cfg.record_stride == 0 {
            checksum += curr[node_count - 1] + curr[0] * 0.125 + err_acc * 1.0e-12;
        }
    }

    (cfg.steps, events, checksum, err_acc)
}

fn run_pfd_fixed_step(cfg: &Config) -> (usize, u64, f64, f64) {
    let stop_ticks = 300_000_u64;
    let ref_period = 20_000_u64;
    let div_period = 20_000_u64;
    let div_phase = 700_u64;
    let mut state = vec![0.0_f64; cfg.models.max(1)];
    let mut last_ref = false;
    let mut last_div = false;
    let mut up = false;
    let mut dn = false;
    let mut events = 0_u64;
    let mut checksum = 0.0_f64;
    let mut err_acc = 0.0_f64;

    for step in 0..cfg.steps {
        let tick = ((step as u128 * stop_ticks as u128) / cfg.steps as u128) as u64;
        let ref_clk = (tick % ref_period) < ref_period / 2;
        let div_clk = ((tick + div_phase) % div_period) < div_period / 2;
        let ref_rise = ref_clk && !last_ref;
        let div_rise = div_clk && !last_div;

        if ref_rise {
            up = true;
            events += 1;
        }
        if div_rise {
            dn = true;
            events += 1;
        }
        if up && dn {
            up = false;
            dn = false;
        }

        let drive = (if up { 1.0 } else { 0.0 }) - (if dn { 1.0 } else { 0.0 });
        for (idx, value) in state.iter_mut().enumerate() {
            let old = *value;
            let gain = 0.001 + (idx % 5) as f64 * 0.0002;
            *value += gain * (drive - *value);
            err_acc += (*value - old).abs();
        }

        if step % cfg.record_stride == 0 {
            checksum += state[state.len() - 1] + if up { 0.25 } else { 0.0 };
        }
        last_ref = ref_clk;
        last_div = div_clk;
    }

    (cfg.steps, events, checksum, err_acc)
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct PfdEvent {
    tick: u64,
    kind: u8,
}

fn run_pfd_event_queue(cfg: &Config) -> (usize, u64, f64, f64) {
    let stop_ticks = 300_000_u64;
    let ref_period = 20_000_u64;
    let div_period = 20_000_u64;
    let div_phase = 700_u64;
    let fixed_dt = (stop_ticks / cfg.steps.max(1) as u64).max(1);
    let record_period = fixed_dt.saturating_mul(cfg.record_stride as u64).max(1);
    let mut heap: BinaryHeap<Reverse<PfdEvent>> = BinaryHeap::new();
    let mut state = vec![0.0_f64; cfg.models.max(1)];
    let mut up = false;
    let mut dn = false;
    let mut events = 0_u64;
    let mut processed_steps = 0_usize;
    let mut checksum = 0.0_f64;
    let mut err_acc = 0.0_f64;

    let mut tick = 0_u64;
    while tick <= stop_ticks {
        heap.push(Reverse(PfdEvent { tick, kind: 0 }));
        tick += ref_period;
    }
    tick = div_phase;
    while tick <= stop_ticks {
        heap.push(Reverse(PfdEvent { tick, kind: 1 }));
        tick += div_period;
    }
    tick = 0;
    while tick <= stop_ticks {
        heap.push(Reverse(PfdEvent { tick, kind: 2 }));
        tick += record_period;
    }

    let mut last_tick = 0_u64;
    while let Some(Reverse(event)) = heap.pop() {
        if event.tick > stop_ticks {
            continue;
        }
        let dt = (event.tick.saturating_sub(last_tick) as f64).max(1.0);
        let alpha = (dt / 20_000.0).min(1.0) * 0.025;
        let drive = (if up { 1.0 } else { 0.0 }) - (if dn { 1.0 } else { 0.0 });

        for (idx, value) in state.iter_mut().enumerate() {
            let old = *value;
            let gain = alpha * (1.0 + (idx % 5) as f64 * 0.1);
            *value += gain * (drive - *value);
            err_acc += (*value - old).abs();
        }

        match event.kind {
            0 => {
                up = true;
                events += 1;
            }
            1 => {
                dn = true;
                events += 1;
            }
            _ => {
                checksum += state[state.len() - 1] + if up { 0.25 } else { 0.0 };
            }
        }
        if up && dn {
            up = false;
            dn = false;
        }
        last_tick = event.tick;
        processed_steps += 1;
    }

    (processed_steps, events, checksum, err_acc)
}

fn main() {
    let cfg = Config::from_args().unwrap_or_else(|err| {
        eprintln!("{err}");
        std::process::exit(2);
    });
    let started = Instant::now();
    let (processed_steps, events, checksum, err_acc) = match cfg.kernel {
        Kernel::MeasurementIndexed => run_measurement_indexed(&cfg),
        Kernel::PfdFixedStep => run_pfd_fixed_step(&cfg),
        Kernel::PfdEventQueue => run_pfd_event_queue(&cfg),
    };
    let elapsed = started.elapsed().as_secs_f64();
    println!(
        "engine=rust_indexed kernel={} requested_steps={} processed_steps={} models={} record_stride={} elapsed_s={:.6} events={} checksum={:.9} err_acc={:.9}",
        cfg.kernel.as_str(), cfg.steps, processed_steps, cfg.models, cfg.record_stride, elapsed, events, checksum, err_acc
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toy_kernel_is_deterministic() {
        let cfg = Config {
            kernel: Kernel::MeasurementIndexed,
            steps: 1_000,
            models: 8,
            record_stride: 4,
        };
        let first = run_measurement_indexed(&cfg);
        let second = run_measurement_indexed(&cfg);
        assert_eq!(first.0, second.0);
        assert_eq!(first.1, second.1);
        assert!((first.2 - second.2).abs() < 1e-15);
        assert!((first.3 - second.3).abs() < 1e-15);
    }

    #[test]
    fn pfd_event_queue_uses_fewer_steps_than_fixed_step() {
        let cfg = Config {
            kernel: Kernel::PfdFixedStep,
            steps: 60_062,
            models: 16,
            record_stride: 16,
        };
        let fixed = run_pfd_fixed_step(&cfg);
        let queued = run_pfd_event_queue(&cfg);
        assert!(queued.0 < fixed.0 / 10);
        assert_eq!(fixed.1, queued.1);
    }
}
