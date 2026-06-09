// Transition state and breakpoint kernels.

pub fn next_transition_breakpoint_for_arrays(
    start_times: &[f64],
    start_values: &[f64],
    target_values: &[f64],
    delays: &[f64],
    rise_times: &[f64],
    fall_times: &[f64],
    active_flags: &[u8],
    time: f64,
    min_ramp_time: f64,
) -> Result<Option<f64>, i32> {
    let count = start_times.len();
    if start_values.len() != count
        || target_values.len() != count
        || delays.len() != count
        || rise_times.len() != count
        || fall_times.len() != count
        || active_flags.len() != count
    {
        return Err(-51);
    }

    let mut best: Option<f64> = None;
    let min_ramp = min_ramp_time.max(0.0);
    for idx in 0..count {
        if active_flags[idx] == 0 {
            continue;
        }
        let going_up = target_values[idx] > start_values[idx];
        let ramp_time = if going_up {
            rise_times[idx]
        } else {
            fall_times[idx]
        };
        let t_begin = start_times[idx] + delays[idx] + spectre_ramp_origin_offset(ramp_time);
        let t_end = t_begin + ramp_time;
        let mut candidate: Option<f64> = None;

        if t_begin > time {
            candidate = Some(t_begin);
        }
        if ramp_time > min_ramp {
            for frac in [0.25_f64, 0.5_f64, 0.75_f64] {
                let t_inner = t_begin + frac * ramp_time;
                if t_inner > time && t_inner < t_end {
                    if candidate.map_or(true, |best_candidate| t_inner < best_candidate) {
                        candidate = Some(t_inner);
                    }
                }
            }
        }
        if t_end > time && candidate.map_or(true, |best_candidate| t_end < best_candidate) {
            candidate = Some(t_end);
        }
        if let Some(bp) = candidate {
            if best.map_or(true, |best_bp| bp < best_bp) {
                best = Some(bp);
            }
        }
    }
    Ok(best)
}

#[inline]
fn spectre_ramp_origin_offset(ramp_time: f64) -> f64 {
    if ramp_time > 0.0 {
        0.25 * ramp_time
    } else {
        0.0
    }
}

pub fn transition_state_step_for_arrays(
    current_values: &mut [f64],
    target_values: &mut [f64],
    start_times: &mut [f64],
    start_values: &mut [f64],
    delays: &mut [f64],
    rise_times: &mut [f64],
    fall_times: &mut [f64],
    active_flags: &mut [u8],
    initialized_flags: &mut [u8],
    input_targets: &[f64],
    input_delays: &[f64],
    input_rises: &[f64],
    input_falls: &[f64],
    output_values: &mut [f64],
    time: f64,
    default_transition: f64,
    initial_condition_mode: bool,
) -> Result<(), i32> {
    let count = current_values.len();
    if target_values.len() != count
        || start_times.len() != count
        || start_values.len() != count
        || delays.len() != count
        || rise_times.len() != count
        || fall_times.len() != count
        || active_flags.len() != count
        || initialized_flags.len() != count
        || input_targets.len() != count
        || input_delays.len() != count
        || input_rises.len() != count
        || input_falls.len() != count
        || output_values.len() != count
    {
        return Err(-111);
    }

    for idx in 0..count {
        let target = input_targets[idx];
        let delay = input_delays[idx];
        let rise = if input_rises[idx] > 0.0 {
            input_rises[idx]
        } else {
            default_transition
        };
        let fall = if input_falls[idx] > 0.0 {
            input_falls[idx]
        } else {
            default_transition
        };

        if initial_condition_mode || initialized_flags[idx] == 0 {
            current_values[idx] = target;
            target_values[idx] = target;
            start_values[idx] = target;
            start_times[idx] = time;
            delays[idx] = delay;
            rise_times[idx] = rise;
            fall_times[idx] = fall;
            active_flags[idx] = 0;
            initialized_flags[idx] = 1;
            output_values[idx] = target;
            continue;
        }

        transition_evaluate_one(
            &mut current_values[idx],
            target_values[idx],
            start_times[idx],
            start_values[idx],
            delays[idx],
            rise_times[idx],
            fall_times[idx],
            &mut active_flags[idx],
            time,
        );
        transition_set_target_one(
            &mut current_values[idx],
            &mut target_values[idx],
            &mut start_times[idx],
            &mut start_values[idx],
            &mut delays[idx],
            &mut rise_times[idx],
            &mut fall_times[idx],
            &mut active_flags[idx],
            time,
            target,
            delay,
            rise,
            fall,
        );
        output_values[idx] = transition_evaluate_one(
            &mut current_values[idx],
            target_values[idx],
            start_times[idx],
            start_values[idx],
            delays[idx],
            rise_times[idx],
            fall_times[idx],
            &mut active_flags[idx],
            time,
        );
    }
    Ok(())
}

pub(crate) fn transition_evaluate_one(
    current_value: &mut f64,
    target_value: f64,
    start_time: f64,
    start_value: f64,
    delay: f64,
    rise_time: f64,
    fall_time: f64,
    active_flag: &mut u8,
    time: f64,
) -> f64 {
    if *active_flag == 0 {
        return *current_value;
    }

    let going_up = target_value > start_value;
    let ramp_time = if going_up { rise_time } else { fall_time };
    let t_begin = start_time + delay + spectre_ramp_origin_offset(ramp_time);

    if time < t_begin {
        start_value
    } else if time >= t_begin + ramp_time {
        *current_value = target_value;
        *active_flag = 0;
        target_value
    } else {
        let mut frac = if ramp_time > 0.0 {
            (time - t_begin) / ramp_time
        } else {
            1.0
        };
        frac = frac.max(0.0).min(1.0);
        let value = start_value + frac * (target_value - start_value);
        *current_value = value;
        value
    }
}

pub(crate) fn transition_set_target_one(
    current_value: &mut f64,
    target_value: &mut f64,
    start_time: &mut f64,
    start_value: &mut f64,
    delay_value: &mut f64,
    rise_time: &mut f64,
    fall_time: &mut f64,
    active_flag: &mut u8,
    time: f64,
    target: f64,
    delay: f64,
    rise: f64,
    fall: f64,
) {
    let changed = (target - *target_value).abs() > 1.0e-15;
    if changed && *active_flag != 0 {
        let going_up = *target_value > *start_value;
        let ramp_time = if going_up { *rise_time } else { *fall_time };
        let t_begin = *start_time + *delay_value + spectre_ramp_origin_offset(ramp_time);
        if time < t_begin {
            *target_value = target;
            *delay_value = delay;
            *rise_time = rise;
            *fall_time = fall;
            return;
        }
        let in_active_region = time >= t_begin && time < t_begin + ramp_time;
        if in_active_region {
            let vi = *current_value;
            let (basis, readjust_time) = if going_up {
                if target < vi {
                    (*target_value, fall)
                } else {
                    (*start_value, rise)
                }
            } else if target < vi {
                (*start_value, fall)
            } else {
                (*target_value, rise)
            };
            let slope = if readjust_time > 0.0 {
                (target - basis) / readjust_time
            } else {
                0.0
            };
            if slope.abs() > 1.0e-30 {
                *start_value = basis;
                *start_time = time - spectre_ramp_origin_offset(readjust_time) - (vi - basis) / slope;
                *delay_value = 0.0;
                *target_value = target;
                *rise_time = rise;
                *fall_time = fall;
                *current_value = vi;
                *active_flag = 1;
                return;
            }
        }
    }

    if changed || *active_flag == 0 {
        if (target - *current_value).abs() > 1.0e-15 {
            *start_value = *current_value;
            *target_value = target;
            *start_time = time;
            *delay_value = delay;
            *rise_time = rise;
            *fall_time = fall;
            *active_flag = 1;
        } else {
            *target_value = target;
            *current_value = target;
            *active_flag = 0;
        }
    }
}
