// Timer, cross, and above detector kernels.

pub fn next_timer_breakpoint_for_arrays(
    next_fire_times: &[f64],
    last_fired_times: &[f64],
    has_last_fired_flags: &[u8],
    time: f64,
) -> Result<Option<f64>, i32> {
    let count = next_fire_times.len();
    if last_fired_times.len() != count || has_last_fired_flags.len() != count {
        return Err(-81);
    }

    let mut best: Option<f64> = None;
    for idx in 0..count {
        let next_fire = next_fire_times[idx];
        if has_last_fired_flags[idx] != 0 && (last_fired_times[idx] - next_fire).abs() <= 1.0e-18 {
            continue;
        }
        if next_fire > time && best.map_or(true, |best_bp| next_fire < best_bp) {
            best = Some(next_fire);
        }
    }
    Ok(best)
}

pub fn timer_periodic_step_for_arrays(
    next_fire_times: &mut [f64],
    has_state_flags: &mut [u8],
    periods: &[f64],
    starts: &[f64],
    has_start_flags: &[u8],
    due_flags: &mut [u8],
    skipped_flags: &mut [u8],
    time: f64,
    reschedule_on_due: bool,
    eps: f64,
) -> Result<(), i32> {
    let count = next_fire_times.len();
    if has_state_flags.len() != count
        || periods.len() != count
        || starts.len() != count
        || has_start_flags.len() != count
        || due_flags.len() != count
        || skipped_flags.len() != count
    {
        return Err(-131);
    }

    let tolerance = eps.abs();
    for idx in 0..count {
        due_flags[idx] = 0;
        skipped_flags[idx] = 0;
        let period = periods[idx];
        if period <= 0.0 || !period.is_finite() {
            continue;
        }

        if has_state_flags[idx] == 0 {
            let mut next_fire = if has_start_flags[idx] != 0 {
                starts[idx]
            } else {
                period
            };
            if !next_fire.is_finite() {
                next_fire = period;
            }
            next_fire_times[idx] = next_fire;
            has_state_flags[idx] = 1;
        }

        let mut next_fire = next_fire_times[idx];
        if time > next_fire + tolerance {
            let missed = ((time - next_fire) / period).floor() + 1.0;
            next_fire += missed * period;
            next_fire_times[idx] = next_fire;
            skipped_flags[idx] = 1;
            continue;
        }

        if time >= next_fire - tolerance {
            due_flags[idx] = 1;
            if reschedule_on_due {
                next_fire_times[idx] = next_fire + period;
            }
        }
    }
    Ok(())
}

pub fn timer_absolute_step_for_arrays(
    next_fire_times: &mut [f64],
    has_state_flags: &mut [u8],
    last_fired_times: &mut [f64],
    has_last_fired_flags: &mut [u8],
    targets: &[f64],
    due_flags: &mut [u8],
    expired_flags: &mut [u8],
    time: f64,
    eps: f64,
) -> Result<(), i32> {
    let count = next_fire_times.len();
    if has_state_flags.len() != count
        || last_fired_times.len() != count
        || has_last_fired_flags.len() != count
        || targets.len() != count
        || due_flags.len() != count
        || expired_flags.len() != count
    {
        return Err(-141);
    }

    let tolerance = eps.abs();
    for idx in 0..count {
        due_flags[idx] = 0;
        expired_flags[idx] = 0;
        let target = targets[idx];
        if !target.is_finite() {
            continue;
        }

        let first_seen = has_state_flags[idx] == 0;
        if first_seen || (next_fire_times[idx] - target).abs() > tolerance {
            next_fire_times[idx] = target;
            has_state_flags[idx] = 1;
        }
        if first_seen && time > target + tolerance {
            last_fired_times[idx] = target;
            has_last_fired_flags[idx] = 1;
            expired_flags[idx] = 1;
            continue;
        }

        let armed_target = next_fire_times[idx];
        if has_last_fired_flags[idx] != 0
            && (last_fired_times[idx] - armed_target).abs() <= tolerance
        {
            continue;
        }
        if time >= armed_target - tolerance {
            last_fired_times[idx] = armed_target;
            has_last_fired_flags[idx] = 1;
            due_flags[idx] = 1;
        }
    }
    Ok(())
}

pub fn cross_detector_step_for_arrays(
    prev_values: &mut [f64],
    prev_times: &mut [f64],
    pprev_values: &mut [f64],
    pprev_times: &mut [f64],
    initialized_flags: &mut [u8],
    directions: &[i32],
    last_cross_times: &mut [f64],
    current_values: &[f64],
    triggered_flags: &mut [u8],
    cross_times: &mut [f64],
    trigger_directions: &mut [i32],
    went_beyond_flags: &mut [u8],
    time: f64,
    time_tol: f64,
    expr_tol: f64,
) -> Result<(), i32> {
    let count = prev_values.len();
    if prev_times.len() != count
        || pprev_values.len() != count
        || pprev_times.len() != count
        || initialized_flags.len() != count
        || directions.len() != count
        || last_cross_times.len() != count
        || current_values.len() != count
        || triggered_flags.len() != count
        || cross_times.len() != count
        || trigger_directions.len() != count
        || went_beyond_flags.len() != count
    {
        return Err(-161);
    }

    let e_tol = expr_tol.abs();
    let t_tol = time_tol.max(0.0);
    for idx in 0..count {
        triggered_flags[idx] = 0;
        trigger_directions[idx] = 0;
        went_beyond_flags[idx] = 0;
        let mut value = current_values[idx];

        if initialized_flags[idx] == 0 {
            prev_values[idx] = value;
            pprev_values[idx] = value;
            prev_times[idx] = time;
            pprev_times[idx] = time;
            initialized_flags[idx] = 1;
            continue;
        }

        let previous_value = prev_values[idx];
        let previous_time = prev_times[idx];
        let direction = directions[idx];
        let mut triggered = false;
        let mut trigger_direction = 0;
        let mut went_beyond = false;
        let mut cross_time = 0.0;

        if direction >= 0 && previous_value < -e_tol {
            if value > e_tol {
                triggered = true;
                trigger_direction = 1;
                went_beyond = true;
                cross_time = interpolate_cross_time(previous_time, previous_value, time, value);
            } else if value.abs() <= e_tol {
                triggered = true;
                trigger_direction = 1;
                went_beyond = false;
                cross_time = interpolate_cross_time(previous_time, previous_value, time, value);
            }
        }
        if !triggered && direction <= 0 && previous_value > e_tol {
            if value < -e_tol {
                triggered = true;
                trigger_direction = -1;
                went_beyond = true;
                cross_time = interpolate_cross_time(previous_time, previous_value, time, value);
            } else if value.abs() <= e_tol {
                triggered = true;
                trigger_direction = -1;
                went_beyond = false;
                cross_time = interpolate_cross_time(previous_time, previous_value, time, value);
            }
        }

        if triggered {
            cross_times[idx] = cross_time;
            if last_cross_times[idx] >= 0.0 && (cross_time - last_cross_times[idx]).abs() <= t_tol {
                triggered = false;
            } else {
                last_cross_times[idx] = cross_time;
            }
            let sign_eps = e_tol.max(1.0e-18);
            if went_beyond {
                if trigger_direction < 0 {
                    value = value.min(-sign_eps);
                } else if trigger_direction > 0 {
                    value = value.max(sign_eps);
                }
            }
        }

        pprev_values[idx] = previous_value;
        pprev_times[idx] = previous_time;
        prev_values[idx] = value;
        prev_times[idx] = time;
        if triggered {
            triggered_flags[idx] = 1;
            trigger_directions[idx] = trigger_direction;
            went_beyond_flags[idx] = if went_beyond { 1 } else { 0 };
        }
    }
    Ok(())
}

pub fn above_detector_step_for_arrays(
    prev_values: &mut [f64],
    prev_times: &mut [f64],
    pprev_values: &mut [f64],
    pprev_times: &mut [f64],
    initialized_flags: &mut [u8],
    directions: &[i32],
    current_values: &[f64],
    triggered_flags: &mut [u8],
    cross_times: &mut [f64],
    time: f64,
) -> Result<(), i32> {
    let count = prev_values.len();
    if prev_times.len() != count
        || pprev_values.len() != count
        || pprev_times.len() != count
        || initialized_flags.len() != count
        || directions.len() != count
        || current_values.len() != count
        || triggered_flags.len() != count
        || cross_times.len() != count
    {
        return Err(-181);
    }

    for idx in 0..count {
        triggered_flags[idx] = 0;
        let mut value = current_values[idx];
        let direction = directions[idx];

        if initialized_flags[idx] == 0 {
            pprev_values[idx] = value;
            prev_values[idx] = value;
            pprev_times[idx] = time;
            prev_times[idx] = time;
            initialized_flags[idx] = 1;
            if direction >= 0 && value >= -1.0e-12 {
                triggered_flags[idx] = 1;
                cross_times[idx] = time;
                prev_values[idx] = value.max(0.0);
            }
            continue;
        }

        let previous_value = prev_values[idx];
        let previous_time = prev_times[idx];
        let mut triggered = false;
        if direction >= 0 && previous_value < 0.0 && value >= -1.0e-12 {
            triggered = true;
        }

        if triggered {
            cross_times[idx] = interpolate_cross_time(previous_time, previous_value, time, value);
            value = value.max(0.0);
        }

        pprev_values[idx] = previous_value;
        pprev_times[idx] = previous_time;
        prev_times[idx] = time;
        prev_values[idx] = value;
        triggered_flags[idx] = if triggered { 1 } else { 0 };
    }
    Ok(())
}

pub(crate) fn interpolate_cross_time(
    prev_time: f64,
    prev_value: f64,
    time: f64,
    value: f64,
) -> f64 {
    let dv = value - prev_value;
    let frac = if dv.abs() > 1.0e-30 {
        (-prev_value / dv).clamp(0.0, 1.0)
    } else {
        0.0
    };
    prev_time + frac * (time - prev_time)
}
