use crate::abi::*;
use crate::event::*;
use crate::expr::*;
use crate::specialized::*;
use crate::transition::*;

// RustSimProgram owned source/event/transition/record simulation loop.

pub(crate) fn rust_sim_source_value(
    source: &EvasRustSimSourceSpec,
    source_data: &[f64],
    time: f64,
) -> Result<f64, i32> {
    match source.kind {
        RUST_SIM_SOURCE_DC => Ok(source.p0),
        RUST_SIM_SOURCE_PULSE => Ok(pulse_value(
            source.p0,
            source.p1,
            source.p2,
            source.p3,
            source.p4,
            source.p5,
            source.p6,
            source.p7,
            source.flags & RUST_SIM_SOURCE_FLAG_HAS_WIDTH != 0,
            time,
        )),
        RUST_SIM_SOURCE_SINE => Ok(source.p0
            + source.p1 * (2.0 * std::f64::consts::PI * source.p2 * time + source.p3).sin()),
        RUST_SIM_SOURCE_PWL => {
            let count = source.data_count;
            if count == 0 {
                return Err(-812);
            }
            let times_start = source.data_start;
            let values_start = times_start.checked_add(count).ok_or(-813)?;
            let values_end = values_start.checked_add(count).ok_or(-814)?;
            if values_end > source_data.len() {
                return Err(-815);
            }
            let times = &source_data[times_start..values_start];
            let values = &source_data[values_start..values_end];
            if time <= times[0] {
                return Ok(values[0]);
            }
            if time >= times[count - 1] {
                return Ok(values[count - 1]);
            }
            for idx in 0..count - 1 {
                if times[idx] <= time && time < times[idx + 1] {
                    let dt = times[idx + 1] - times[idx];
                    if dt <= 0.0 {
                        return Err(-816);
                    }
                    let frac = (time - times[idx]) / dt;
                    return Ok(values[idx] + frac * (values[idx + 1] - values[idx]));
                }
            }
            Ok(values[count - 1])
        }
        _ => Err(-811),
    }
}

pub(crate) fn rust_sim_source_next_breakpoint(
    source: &EvasRustSimSourceSpec,
    source_data: &[f64],
    time: f64,
) -> Result<Option<f64>, i32> {
    let eps = 1.0e-18;
    match source.kind {
        RUST_SIM_SOURCE_DC | RUST_SIM_SOURCE_SINE => Ok(None),
        RUST_SIM_SOURCE_PULSE => {
            let period = source.p2;
            let duty = source.p3;
            let rise = source.p4;
            let fall = source.p5;
            let delay = source.p6;
            let width = source.p7;
            let has_width = source.flags & RUST_SIM_SOURCE_FLAG_HAS_WIDTH != 0;
            let one_shot = period <= 0.0 || source.flags & RUST_SIM_SOURCE_FLAG_ONE_SHOT != 0;
            if delay > time + eps {
                return Ok(Some(delay));
            }
            let fall_start = pulse_fall_start(period, duty, rise, width, has_width);
            let mut knees = [0.0_f64; 6];
            let mut knee_count = 0_usize;
            knees[knee_count] = 0.0;
            knee_count += 1;
            knees[knee_count] = rise;
            knee_count += 1;
            if fall_start.is_finite() {
                knees[knee_count] = fall_start;
                knee_count += 1;
                knees[knee_count] = fall_start + fall;
                knee_count += 1;
                if fall > 0.0 {
                    knees[knee_count] = fall_start + 0.5 * fall;
                    knee_count += 1;
                }
            }
            if rise > 0.0 {
                knees[knee_count] = 0.5 * rise;
                knee_count += 1;
            }

            let t_eff = time - delay;
            let mut cycle = if !one_shot && period > 0.0 {
                (t_eff / period).floor().max(0.0) as usize
            } else {
                0
            };
            for _ in 0..2 {
                let base = delay
                    + if !one_shot && period > 0.0 {
                        period * cycle as f64
                    } else {
                        0.0
                    };
                let mut best: Option<f64> = None;
                for offset in &knees[..knee_count] {
                    if !offset.is_finite() {
                        continue;
                    }
                    let candidate = base + *offset;
                    if candidate > time + eps
                        && best.map_or(true, |best_value| candidate < best_value)
                    {
                        best = Some(candidate);
                    }
                }
                if best.is_some() {
                    return Ok(best);
                }
                if one_shot {
                    return Ok(None);
                }
                cycle = cycle.checked_add(1).ok_or(-817)?;
            }
            Ok(None)
        }
        RUST_SIM_SOURCE_PWL => {
            let count = source.data_count;
            if count == 0 {
                return Err(-818);
            }
            let times_start = source.data_start;
            let times_end = times_start.checked_add(count).ok_or(-819)?;
            let values_end = times_end.checked_add(count).ok_or(-820)?;
            if values_end > source_data.len() {
                return Err(-821);
            }
            for idx in times_start..times_end {
                let candidate = source_data[idx];
                if candidate > time + eps {
                    return Ok(Some(candidate));
                }
            }
            Ok(None)
        }
        _ => Err(-822),
    }
}

pub(crate) fn rust_sim_write_sources(
    sources: &[EvasRustSimSourceSpec],
    source_data: &[f64],
    node_values: &mut [f64],
    time: f64,
) -> Result<(), i32> {
    for source in sources {
        if source.node_id >= node_values.len() {
            return Err(-831);
        }
        node_values[source.node_id] = rust_sim_source_value(source, source_data, time)?;
    }
    Ok(())
}

pub(crate) fn rust_sim_next_source_breakpoint_after(
    sources: &[EvasRustSimSourceSpec],
    source_data: &[f64],
    time: f64,
) -> Result<Option<f64>, i32> {
    let mut best: Option<f64> = None;
    for source in sources {
        if let Some(bp) = rust_sim_source_next_breakpoint(source, source_data, time)? {
            if bp > time && (best.is_none() || bp < best.unwrap()) {
                best = Some(bp);
            }
        }
    }
    Ok(best)
}

pub(crate) fn rust_sim_record_point(
    time_values: &mut [f64],
    signal_values: &mut [f64],
    step_values: &mut [f64],
    count: usize,
    time: f64,
    step: f64,
    node_values: &[f64],
    record_node_ids: &[usize],
) -> Result<(), i32> {
    if count >= time_values.len() || count >= step_values.len() {
        return Err(-841);
    }
    let record_count = record_node_ids.len();
    let row_start = count.checked_mul(record_count).ok_or(-842)?;
    let row_end = row_start.checked_add(record_count).ok_or(-843)?;
    if row_end > signal_values.len() {
        return Err(-844);
    }
    time_values[count] = time;
    step_values[count] = step;
    for (idx, node_id) in record_node_ids.iter().enumerate() {
        signal_values[row_start + idx] = if *node_id < node_values.len() {
            node_values[*node_id]
        } else {
            0.0
        };
    }
    Ok(())
}

pub(crate) fn rust_sim_record_point_dedup(
    time_values: &mut [f64],
    signal_values: &mut [f64],
    step_values: &mut [f64],
    count: &mut usize,
    time: f64,
    step: f64,
    node_values: &[f64],
    record_node_ids: &[usize],
) -> Result<(), i32> {
    let mut record_time = time;
    if *count > 0 && record_time < time_values[*count - 1] - 1.0e-18 {
        record_time = time_values[*count - 1];
    }
    if *count > 0 && (time_values[*count - 1] - record_time).abs() <= 1.0e-18 {
        return rust_sim_record_point(
            time_values,
            signal_values,
            step_values,
            *count - 1,
            record_time,
            step,
            node_values,
            record_node_ids,
        );
    }
    rust_sim_record_point(
        time_values,
        signal_values,
        step_values,
        *count,
        record_time,
        step,
        node_values,
        record_node_ids,
    )?;
    *count += 1;
    Ok(())
}

pub fn rust_sim_source_record_trace(
    sources: &[EvasRustSimSourceSpec],
    source_data: &[f64],
    node_values: &mut [f64],
    record_node_ids: &[usize],
    time_values: &mut [f64],
    signal_values: &mut [f64],
    step_values: &mut [f64],
    tstop: f64,
    tstep: f64,
    max_step: f64,
    record_step: f64,
    use_record_step: bool,
) -> Result<(usize, usize), i32> {
    if !tstop.is_finite() || !tstep.is_finite() || !max_step.is_finite() {
        return Err(-851);
    }
    if tstop < 0.0 || tstep <= 0.0 || max_step <= 0.0 {
        return Err(-852);
    }
    if use_record_step && (!record_step.is_finite() || record_step <= 0.0) {
        return Err(-853);
    }
    if time_values.len() != step_values.len() {
        return Err(-854);
    }
    let record_count = record_node_ids.len();
    if record_count == 0 {
        return Err(-855);
    }
    if signal_values.len() < time_values.len().saturating_mul(record_count) {
        return Err(-856);
    }

    let eps = 1.0e-18;
    let mut source_breakpoints = 0_usize;
    let mut count = 0_usize;
    let mut time = 0.0_f64;
    let mut next_source_breakpoint =
        rust_sim_next_source_breakpoint_after(sources, source_data, 0.0)?;
    let mut next_record_time = if use_record_step {
        record_step
    } else {
        f64::INFINITY
    };

    rust_sim_write_sources(sources, source_data, node_values, 0.0)?;
    rust_sim_record_point(
        time_values,
        signal_values,
        step_values,
        count,
        0.0,
        0.0,
        node_values,
        record_node_ids,
    )?;
    count += 1;

    let mut loop_iterations = 0_usize;
    let max_loop_iterations = time_values.len().saturating_mul(16).max(1024);
    while time < tstop {
        loop_iterations = loop_iterations.checked_add(1).ok_or(-2145)?;
        if loop_iterations > max_loop_iterations {
            return Err(-2146);
        }
        let mut force_record = false;
        let mut dt = tstep.min(max_step).min(tstop - time);
        if dt <= 0.0 {
            break;
        }
        while next_source_breakpoint.map_or(false, |bp| bp <= time + eps) {
            next_source_breakpoint =
                rust_sim_next_source_breakpoint_after(sources, source_data, time)?;
        }
        if let Some(bp) = next_source_breakpoint {
            if bp < time + dt {
                dt = bp - time;
                force_record = true;
                source_breakpoints += 1;
                if dt < eps {
                    dt = eps;
                }
            }
        }
        if use_record_step && next_record_time > time && next_record_time < time + dt {
            dt = next_record_time - time;
            force_record = true;
        }
        if dt <= 0.0 {
            return Err(-857);
        }
        time += dt;
        if time > tstop && time < tstop + eps {
            time = tstop;
        }
        rust_sim_write_sources(sources, source_data, node_values, time)?;

        let should_record = if use_record_step {
            force_record || time >= next_record_time - eps || time >= tstop - eps
        } else {
            true
        };
        if should_record {
            rust_sim_record_point(
                time_values,
                signal_values,
                step_values,
                count,
                time,
                dt,
                node_values,
                record_node_ids,
            )?;
            count += 1;
            if use_record_step {
                while next_record_time <= time + eps {
                    next_record_time += record_step;
                }
            }
        }
    }
    Ok((count, source_breakpoints))
}

pub fn rust_sim_source_linear_record_trace(
    sources: &[EvasRustSimSourceSpec],
    source_data: &[f64],
    linear_ops: &[EvasRustLinearOp],
    linear_terms: &[EvasRustLinearTerm],
    linear_conditions: &[EvasRustLinearCondition],
    node_values: &mut [f64],
    state_values: &mut [f64],
    record_node_ids: &[usize],
    time_values: &mut [f64],
    signal_values: &mut [f64],
    step_values: &mut [f64],
    tstop: f64,
    tstep: f64,
    max_step: f64,
    record_step: f64,
    use_record_step: bool,
) -> Result<(usize, usize), i32> {
    if !tstop.is_finite() || !tstep.is_finite() || !max_step.is_finite() {
        return Err(-881);
    }
    if tstop < 0.0 || tstep <= 0.0 || max_step <= 0.0 {
        return Err(-882);
    }
    if use_record_step && (!record_step.is_finite() || record_step <= 0.0) {
        return Err(-883);
    }
    if time_values.len() != step_values.len() {
        return Err(-884);
    }
    let record_count = record_node_ids.len();
    if record_count == 0 {
        return Err(-885);
    }
    if signal_values.len() < time_values.len().saturating_mul(record_count) {
        return Err(-886);
    }

    let eps = 1.0e-18;
    let mut source_breakpoints = 0_usize;
    let mut count = 0_usize;
    let mut time = 0.0_f64;
    let mut next_source_breakpoint =
        rust_sim_next_source_breakpoint_after(sources, source_data, 0.0)?;
    let mut next_record_time = if use_record_step {
        record_step
    } else {
        f64::INFINITY
    };

    rust_sim_write_sources(sources, source_data, node_values, 0.0)?;
    evaluate_static_linear_ops(
        linear_ops,
        linear_terms,
        linear_conditions,
        node_values,
        state_values,
    )?;
    rust_sim_record_point(
        time_values,
        signal_values,
        step_values,
        count,
        0.0,
        0.0,
        node_values,
        record_node_ids,
    )?;
    count += 1;

    let mut loop_iterations = 0_usize;
    let max_loop_iterations = time_values.len().saturating_mul(16).max(1024);
    while time < tstop {
        loop_iterations = loop_iterations.checked_add(1).ok_or(-887)?;
        if loop_iterations > max_loop_iterations {
            return Err(-888);
        }
        let mut force_record = false;
        let mut dt = tstep.min(max_step).min(tstop - time);
        if dt <= 0.0 {
            break;
        }
        while next_source_breakpoint.map_or(false, |bp| bp <= time + eps) {
            next_source_breakpoint =
                rust_sim_next_source_breakpoint_after(sources, source_data, time)?;
        }
        if let Some(bp) = next_source_breakpoint {
            if bp < time + dt {
                dt = bp - time;
                force_record = true;
                source_breakpoints += 1;
                if dt < eps {
                    dt = eps;
                }
            }
        }
        if use_record_step && next_record_time > time && next_record_time < time + dt {
            dt = next_record_time - time;
            force_record = true;
        }
        if dt <= 0.0 {
            return Err(-887);
        }
        time += dt;
        if time > tstop && time < tstop + eps {
            time = tstop;
        }
        rust_sim_write_sources(sources, source_data, node_values, time)?;
        evaluate_static_linear_ops(
            linear_ops,
            linear_terms,
            linear_conditions,
            node_values,
            state_values,
        )?;

        let should_record = if use_record_step {
            force_record || time >= next_record_time - eps || time >= tstop - eps
        } else {
            true
        };
        if should_record {
            rust_sim_record_point(
                time_values,
                signal_values,
                step_values,
                count,
                time,
                dt,
                node_values,
                record_node_ids,
            )?;
            count += 1;
            if use_record_step {
                while next_record_time <= time + eps {
                    next_record_time += record_step;
                }
            }
        }
    }
    Ok((count, source_breakpoints))
}

pub(crate) fn rust_sim_eval_expr_segment(
    expr_ops: &[EvasRustBodyExprOp],
    start: usize,
    count: usize,
    node_values: &[f64],
    state_values: &[f64],
    param_values: &[f64],
    time: f64,
    default_value: f64,
) -> Result<f64, i32> {
    if count == 0 {
        return Ok(default_value);
    }
    let end = start.checked_add(count).ok_or(-931)?;
    if end > expr_ops.len() {
        return Err(-932);
    }
    evaluate_body_expr_ops_at_time(
        &expr_ops[start..end],
        node_values,
        state_values,
        param_values,
        time,
    )
}

pub(crate) fn rust_sim_execute_event_body(
    event: &EvasRustSimEventSpec,
    body_stmt_ops: &[EvasRustBodyStmtOp],
    body_expr_ops: &[EvasRustBodyExprOp],
    node_values: &mut [f64],
    state_values: &mut [f64],
    param_values: &[f64],
    time: f64,
    bound_step_limit: &mut f64,
    side_effect_log: Option<&mut RustSideEffectLog<'_>>,
) -> Result<(), i32> {
    let end = event
        .body_stmt_start
        .checked_add(event.body_stmt_count)
        .ok_or(-941)?;
    if end > body_stmt_ops.len() {
        return Err(-942);
    }
    evaluate_body_ir_ops_at_time_impl(
        &body_stmt_ops[event.body_stmt_start..end],
        body_expr_ops,
        node_values,
        state_values,
        param_values,
        time,
        bound_step_limit,
        true,
        side_effect_log,
    )
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RustSimCrossCandidate {
    pub(crate) event_idx: usize,
    pub(crate) event_time: f64,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn rust_sim_collect_cross_events_into(
    events: &[EvasRustSimEventSpec],
    body_expr_ops: &[EvasRustBodyExprOp],
    node_values: &[f64],
    state_values: &[f64],
    param_values: &[f64],
    cross_prev_values: &mut [f64],
    cross_prev_times: &mut [f64],
    cross_pprev_values: &mut [f64],
    cross_pprev_times: &mut [f64],
    cross_initialized: &mut [u8],
    cross_last_times: &mut [f64],
    time: f64,
    initial_condition_mode: bool,
    phase: u8,
    candidates: &mut Vec<RustSimCrossCandidate>,
) -> Result<(), i32> {
    if cross_prev_values.len() != events.len()
        || cross_prev_times.len() != events.len()
        || cross_pprev_values.len() != events.len()
        || cross_pprev_times.len() != events.len()
        || cross_initialized.len() != events.len()
        || cross_last_times.len() != events.len()
    {
        return Err(-954);
    }

    candidates.clear();
    for event_idx in 0..events.len() {
        let event = events[event_idx];
        if event.phase != phase || event.kind != RUST_SIM_EVENT_CROSS {
            continue;
        }
        let expr_value = rust_sim_eval_expr_segment(
            body_expr_ops,
            event.expr_start,
            event.expr_count,
            node_values,
            state_values,
            param_values,
            time,
            0.0,
        )?;
        let time_tol = rust_sim_eval_expr_segment(
            body_expr_ops,
            event.time_tol_start,
            event.time_tol_count,
            node_values,
            state_values,
            param_values,
            time,
            0.0,
        )?;
        let expr_tol = rust_sim_eval_expr_segment(
            body_expr_ops,
            event.expr_tol_start,
            event.expr_tol_count,
            node_values,
            state_values,
            param_values,
            time,
            1.0e-12,
        )?;
        let current_values = [expr_value];
        let directions = [event.direction];
        let mut triggered = [0_u8];
        let mut cross_times = [0.0_f64];
        let mut trigger_dirs = [0_i32];
        let mut went_beyond = [0_u8];
        cross_detector_step_for_arrays(
            &mut cross_prev_values[event_idx..event_idx + 1],
            &mut cross_prev_times[event_idx..event_idx + 1],
            &mut cross_pprev_values[event_idx..event_idx + 1],
            &mut cross_pprev_times[event_idx..event_idx + 1],
            &mut cross_initialized[event_idx..event_idx + 1],
            &directions,
            &mut cross_last_times[event_idx..event_idx + 1],
            &current_values,
            &mut triggered,
            &mut cross_times,
            &mut trigger_dirs,
            &mut went_beyond,
            time,
            time_tol,
            expr_tol,
        )?;
        if !initial_condition_mode && triggered[0] != 0 && cross_times[0].is_finite() {
            candidates.push(RustSimCrossCandidate {
                event_idx,
                event_time: cross_times[0],
            });
        }
    }
    candidates.sort_by(|left, right| {
        left.event_time
            .partial_cmp(&right.event_time)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.event_idx.cmp(&right.event_idx))
    });
    Ok(())
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn rust_sim_collect_cross_events(
    events: &[EvasRustSimEventSpec],
    body_expr_ops: &[EvasRustBodyExprOp],
    node_values: &[f64],
    state_values: &[f64],
    param_values: &[f64],
    cross_prev_values: &mut [f64],
    cross_prev_times: &mut [f64],
    cross_pprev_values: &mut [f64],
    cross_pprev_times: &mut [f64],
    cross_initialized: &mut [u8],
    cross_last_times: &mut [f64],
    time: f64,
    initial_condition_mode: bool,
    phase: u8,
) -> Result<Vec<RustSimCrossCandidate>, i32> {
    let mut candidates = Vec::new();
    rust_sim_collect_cross_events_into(
        events,
        body_expr_ops,
        node_values,
        state_values,
        param_values,
        cross_prev_values,
        cross_prev_times,
        cross_pprev_values,
        cross_pprev_times,
        cross_initialized,
        cross_last_times,
        time,
        initial_condition_mode,
        phase,
        &mut candidates,
    )?;
    Ok(candidates)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn rust_sim_execute_events(
    events: &[EvasRustSimEventSpec],
    body_stmt_ops: &[EvasRustBodyStmtOp],
    body_expr_ops: &[EvasRustBodyExprOp],
    node_values: &mut [f64],
    state_values: &mut [f64],
    param_values: &[f64],
    cross_prev_values: &mut [f64],
    cross_prev_times: &mut [f64],
    cross_pprev_values: &mut [f64],
    cross_pprev_times: &mut [f64],
    cross_initialized: &mut [u8],
    cross_last_times: &mut [f64],
    above_prev_values: &mut [f64],
    above_prev_times: &mut [f64],
    above_pprev_values: &mut [f64],
    above_pprev_times: &mut [f64],
    above_initialized: &mut [u8],
    timer_next_fire_times: &mut [f64],
    timer_has_state_flags: &mut [u8],
    timer_last_fired_times: &mut [f64],
    timer_has_last_flags: &mut [u8],
    bound_step_limit: &mut f64,
    side_effect_log: &mut RustSideEffectLog<'_>,
    time: f64,
    initial_condition_mode: bool,
    phase: u8,
) -> Result<usize, i32> {
    if cross_prev_values.len() != events.len()
        || cross_prev_times.len() != events.len()
        || cross_pprev_values.len() != events.len()
        || cross_pprev_times.len() != events.len()
        || cross_initialized.len() != events.len()
        || cross_last_times.len() != events.len()
        || above_prev_values.len() != events.len()
        || above_prev_times.len() != events.len()
        || above_pprev_values.len() != events.len()
        || above_pprev_times.len() != events.len()
        || above_initialized.len() != events.len()
        || timer_next_fire_times.len() != events.len()
        || timer_has_state_flags.len() != events.len()
        || timer_last_fired_times.len() != events.len()
        || timer_has_last_flags.len() != events.len()
    {
        return Err(-951);
    }

    let mut fired_count = 0_usize;
    for event_idx in 0..events.len() {
        let event = events[event_idx];
        if event.phase != phase {
            continue;
        }
        let fired: bool;
        match event.kind {
            RUST_SIM_EVENT_ALWAYS => {
                fired = true;
            }
            RUST_SIM_EVENT_INITIAL_STEP => {
                fired = initial_condition_mode;
            }
            RUST_SIM_EVENT_CROSS => {
                continue;
            }
            RUST_SIM_EVENT_ABOVE => {
                let expr_value = rust_sim_eval_expr_segment(
                    body_expr_ops,
                    event.expr_start,
                    event.expr_count,
                    node_values,
                    state_values,
                    param_values,
                    time,
                    0.0,
                )?;
                let current_values = [expr_value];
                let directions = [event.direction];
                let mut triggered = [0_u8];
                let mut cross_times = [0.0_f64];
                above_detector_step_for_arrays(
                    &mut above_prev_values[event_idx..event_idx + 1],
                    &mut above_prev_times[event_idx..event_idx + 1],
                    &mut above_pprev_values[event_idx..event_idx + 1],
                    &mut above_pprev_times[event_idx..event_idx + 1],
                    &mut above_initialized[event_idx..event_idx + 1],
                    &directions,
                    &current_values,
                    &mut triggered,
                    &mut cross_times,
                    time,
                )?;
                fired = triggered[0] != 0;
            }
            RUST_SIM_EVENT_TIMER => {
                let start = rust_sim_eval_expr_segment(
                    body_expr_ops,
                    event.timer_start_expr_start,
                    event.timer_start_expr_count,
                    node_values,
                    state_values,
                    param_values,
                    time,
                    0.0,
                )?;
                if event.timer_period_expr_count > 0 {
                    let period = rust_sim_eval_expr_segment(
                        body_expr_ops,
                        event.timer_period_expr_start,
                        event.timer_period_expr_count,
                        node_values,
                        state_values,
                        param_values,
                        time,
                        0.0,
                    )?;
                    let periods = [period];
                    let starts = [start];
                    let has_start = [1_u8];
                    let mut due = [0_u8];
                    let mut skipped = [0_u8];
                    timer_periodic_step_for_arrays(
                        &mut timer_next_fire_times[event_idx..event_idx + 1],
                        &mut timer_has_state_flags[event_idx..event_idx + 1],
                        &periods,
                        &starts,
                        &has_start,
                        &mut due,
                        &mut skipped,
                        time,
                        true,
                        1.0e-18,
                    )?;
                    fired = due[0] != 0;
                } else {
                    let targets = [start];
                    let mut due = [0_u8];
                    let mut expired = [0_u8];
                    timer_absolute_step_for_arrays(
                        &mut timer_next_fire_times[event_idx..event_idx + 1],
                        &mut timer_has_state_flags[event_idx..event_idx + 1],
                        &mut timer_last_fired_times[event_idx..event_idx + 1],
                        &mut timer_has_last_flags[event_idx..event_idx + 1],
                        &targets,
                        &mut due,
                        &mut expired,
                        time,
                        1.0e-18,
                    )?;
                    fired = due[0] != 0;
                }
            }
            RUST_SIM_EVENT_FINAL_STEP => {
                continue;
            }
            _ => return Err(-952),
        }

        if fired {
            rust_sim_execute_event_body(
                &event,
                body_stmt_ops,
                body_expr_ops,
                node_values,
                state_values,
                param_values,
                time,
                bound_step_limit,
                Some(&mut *side_effect_log),
            )?;
            if event.kind == RUST_SIM_EVENT_TIMER && event.timer_period_expr_count == 0 {
                let next_target = rust_sim_eval_expr_segment(
                    body_expr_ops,
                    event.timer_start_expr_start,
                    event.timer_start_expr_count,
                    node_values,
                    state_values,
                    param_values,
                    time,
                    timer_next_fire_times[event_idx],
                )?;
                if next_target.is_finite() {
                    timer_next_fire_times[event_idx] = next_target;
                    timer_has_state_flags[event_idx] = 1;
                }
            }
            fired_count += 1;
        }
    }
    Ok(fired_count)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn rust_sim_execute_final_step_events(
    events: &[EvasRustSimEventSpec],
    body_stmt_ops: &[EvasRustBodyStmtOp],
    body_expr_ops: &[EvasRustBodyExprOp],
    node_values: &mut [f64],
    state_values: &mut [f64],
    param_values: &[f64],
    time: f64,
    bound_step_limit: &mut f64,
    side_effect_log: &mut RustSideEffectLog<'_>,
) -> Result<usize, i32> {
    let mut fired_count = 0_usize;
    for event in events {
        if event.kind != RUST_SIM_EVENT_FINAL_STEP {
            continue;
        }
        rust_sim_execute_event_body(
            event,
            body_stmt_ops,
            body_expr_ops,
            node_values,
            state_values,
            param_values,
            time,
            bound_step_limit,
            Some(&mut *side_effect_log),
        )?;
        fired_count += 1;
    }
    Ok(fired_count)
}

#[allow(clippy::too_many_arguments)]
fn rust_sim_transition_target_values_after_pre_always(
    sources: &[EvasRustSimSourceSpec],
    source_data: &[f64],
    linear_ops: &[EvasRustLinearOp],
    linear_terms: &[EvasRustLinearTerm],
    linear_conditions: &[EvasRustLinearCondition],
    body_stmt_ops: &[EvasRustBodyStmtOp],
    body_expr_ops: &[EvasRustBodyExprOp],
    events: &[EvasRustSimEventSpec],
    transitions: &[EvasRustSimTransitionSpec],
    param_values: &[f64],
    node_values: &[f64],
    state_values: &[f64],
    time: f64,
) -> Result<Vec<f64>, i32> {
    let mut scratch_nodes = node_values.to_vec();
    let mut scratch_states = state_values.to_vec();
    let mut scratch_bound_step = f64::INFINITY;

    rust_sim_write_sources(sources, source_data, &mut scratch_nodes, time)?;
    for event in events {
        if event.phase != RUST_SIM_EVENT_PHASE_PRE || event.kind != RUST_SIM_EVENT_ALWAYS {
            continue;
        }
        rust_sim_execute_event_body(
            event,
            body_stmt_ops,
            body_expr_ops,
            &mut scratch_nodes,
            &mut scratch_states,
            param_values,
            time,
            &mut scratch_bound_step,
            None,
        )?;
    }
    evaluate_static_linear_ops(
        linear_ops,
        linear_terms,
        linear_conditions,
        &mut scratch_nodes,
        &mut scratch_states,
    )?;

    let mut values = Vec::<f64>::with_capacity(transitions.len());
    for transition in transitions {
        values.push(rust_sim_eval_expr_segment(
            body_expr_ops,
            transition.target_expr_start,
            transition.target_expr_count,
            &scratch_nodes,
            &scratch_states,
            param_values,
            time,
            0.0,
        )?);
    }
    Ok(values)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn rust_sim_next_transition_target_change_breakpoint(
    sources: &[EvasRustSimSourceSpec],
    source_data: &[f64],
    linear_ops: &[EvasRustLinearOp],
    linear_terms: &[EvasRustLinearTerm],
    linear_conditions: &[EvasRustLinearCondition],
    body_stmt_ops: &[EvasRustBodyStmtOp],
    body_expr_ops: &[EvasRustBodyExprOp],
    events: &[EvasRustSimEventSpec],
    transitions: &[EvasRustSimTransitionSpec],
    param_values: &[f64],
    node_values: &[f64],
    state_values: &[f64],
    time: f64,
    dt: f64,
) -> Result<Option<f64>, i32> {
    if transitions.is_empty() || dt <= 0.0 {
        return Ok(None);
    }
    let horizon = time + dt;
    if !horizon.is_finite() || horizon <= time {
        return Ok(None);
    }

    let start_values = rust_sim_transition_target_values_after_pre_always(
        sources,
        source_data,
        linear_ops,
        linear_terms,
        linear_conditions,
        body_stmt_ops,
        body_expr_ops,
        events,
        transitions,
        param_values,
        node_values,
        state_values,
        time,
    )?;
    let min_sep = 1.0e-18_f64.max(dt.abs() * 1.0e-12);
    let mut changes_immediately = vec![false; start_values.len()];
    let near_time = (time + min_sep).min(horizon);
    if near_time > time && near_time < horizon {
        let near_values = rust_sim_transition_target_values_after_pre_always(
            sources,
            source_data,
            linear_ops,
            linear_terms,
            linear_conditions,
            body_stmt_ops,
            body_expr_ops,
            events,
            transitions,
            param_values,
            node_values,
            state_values,
            near_time,
        )?;
        if start_values.len() != near_values.len() {
            return Err(-973);
        }
        for idx in 0..start_values.len() {
            let start = start_values[idx];
            let near = near_values[idx];
            let immediate_delta = (start - near).abs();
            if start.is_finite()
                && near.is_finite()
                && immediate_delta > 1.0e-15
                && immediate_delta <= 1.0e-6_f64.max(1.0e-9 * start.abs().max(near.abs()).max(1.0))
            {
                changes_immediately[idx] = true;
            }
        }
    }
    let end_values = rust_sim_transition_target_values_after_pre_always(
        sources,
        source_data,
        linear_ops,
        linear_terms,
        linear_conditions,
        body_stmt_ops,
        body_expr_ops,
        events,
        transitions,
        param_values,
        node_values,
        state_values,
        horizon,
    )?;
    if start_values.len() != end_values.len() {
        return Err(-971);
    }

    let eps = 1.0e-15;
    let mut best: Option<f64> = None;
    for idx in 0..start_values.len() {
        if changes_immediately.get(idx).copied().unwrap_or(false) {
            continue;
        }
        let start = start_values[idx];
        let end = end_values[idx];
        if !start.is_finite() || !end.is_finite() || (start - end).abs() <= eps {
            continue;
        }

        let mut lo = time;
        let mut hi = horizon;
        for _ in 0..60 {
            let mid = 0.5 * (lo + hi);
            if mid <= lo || mid >= hi {
                break;
            }
            let mid_values = rust_sim_transition_target_values_after_pre_always(
                sources,
                source_data,
                linear_ops,
                linear_terms,
                linear_conditions,
                body_stmt_ops,
                body_expr_ops,
                events,
                transitions,
                param_values,
                node_values,
                state_values,
                mid,
            )?;
            let mid_value = *mid_values.get(idx).ok_or(-972)?;
            if mid_value.is_finite() && (mid_value - start).abs() <= eps {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let mut candidate = hi.max(time + min_sep);
        if candidate > time && candidate < horizon {
            let after_time = (candidate + min_sep).min(horizon);
            if after_time > candidate && after_time < horizon {
                let candidate_values = rust_sim_transition_target_values_after_pre_always(
                    sources,
                    source_data,
                    linear_ops,
                    linear_terms,
                    linear_conditions,
                    body_stmt_ops,
                    body_expr_ops,
                    events,
                    transitions,
                    param_values,
                    node_values,
                    state_values,
                    candidate,
                )?;
                let after_values = rust_sim_transition_target_values_after_pre_always(
                    sources,
                    source_data,
                    linear_ops,
                    linear_terms,
                    linear_conditions,
                    body_stmt_ops,
                    body_expr_ops,
                    events,
                    transitions,
                    param_values,
                    node_values,
                    state_values,
                    after_time,
                )?;
                let candidate_value = *candidate_values.get(idx).ok_or(-974)?;
                let after_value = *after_values.get(idx).ok_or(-975)?;
                if candidate_value.is_finite()
                    && after_value.is_finite()
                    && (candidate_value - after_value).abs() > eps
                    && (after_value - start).abs() > eps
                {
                    candidate = after_time;
                }
            }
        }
        if candidate > time
            && candidate < horizon
            && best.map_or(true, |best_value| candidate < best_value)
        {
            best = Some(candidate);
        }
    }
    Ok(best)
}

pub(crate) fn rust_sim_next_timer_event_breakpoint(
    events: &[EvasRustSimEventSpec],
    next_fire_times: &[f64],
    has_state_flags: &[u8],
    last_fired_times: &[f64],
    has_last_fired_flags: &[u8],
    time: f64,
    horizon: f64,
) -> Result<Option<f64>, i32> {
    if next_fire_times.len() != events.len()
        || has_state_flags.len() != events.len()
        || last_fired_times.len() != events.len()
        || has_last_fired_flags.len() != events.len()
    {
        return Err(-961);
    }
    let mut best: Option<f64> = None;
    for idx in 0..events.len() {
        if events[idx].phase != RUST_SIM_EVENT_PHASE_PRE
            || events[idx].kind != RUST_SIM_EVENT_TIMER
            || has_state_flags[idx] == 0
        {
            continue;
        }
        let candidate = next_fire_times[idx];
        if !candidate.is_finite() {
            continue;
        }
        if events[idx].timer_period_expr_count == 0
            && has_last_fired_flags[idx] != 0
            && (last_fired_times[idx] - candidate).abs() <= 1.0e-18
        {
            continue;
        }
        if candidate > time
            && candidate < horizon
            && best.map_or(true, |best_value| candidate < best_value)
        {
            best = Some(candidate);
        }
    }
    Ok(best)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn rust_sim_apply_transitions(
    transitions: &[EvasRustSimTransitionSpec],
    body_expr_ops: &[EvasRustBodyExprOp],
    node_values: &mut [f64],
    state_values: &[f64],
    param_values: &[f64],
    current_values: &mut [f64],
    target_values: &mut [f64],
    start_times: &mut [f64],
    start_values: &mut [f64],
    delays: &mut [f64],
    rise_times: &mut [f64],
    fall_times: &mut [f64],
    active_flags: &mut [u8],
    initialized_flags: &mut [u8],
    output_values: &mut [f64],
    time: f64,
    default_transition: f64,
    initial_condition_mode: bool,
) -> Result<(), i32> {
    let count = transitions.len();
    if current_values.len() != count
        || target_values.len() != count
        || start_times.len() != count
        || start_values.len() != count
        || delays.len() != count
        || rise_times.len() != count
        || fall_times.len() != count
        || active_flags.len() != count
        || initialized_flags.len() != count
        || output_values.len() != count
    {
        return Err(-971);
    }
    if count == 0 {
        return Ok(());
    }

    let mut input_targets = vec![0.0_f64; count];
    let mut input_delays = vec![0.0_f64; count];
    let mut input_rises = vec![0.0_f64; count];
    let mut input_falls = vec![0.0_f64; count];
    for idx in 0..count {
        let spec = transitions[idx];
        input_targets[idx] = rust_sim_eval_expr_segment(
            body_expr_ops,
            spec.target_expr_start,
            spec.target_expr_count,
            node_values,
            state_values,
            param_values,
            time,
            0.0,
        )?;
        input_delays[idx] = rust_sim_eval_expr_segment(
            body_expr_ops,
            spec.delay_expr_start,
            spec.delay_expr_count,
            node_values,
            state_values,
            param_values,
            time,
            0.0,
        )?;
        input_rises[idx] = rust_sim_eval_expr_segment(
            body_expr_ops,
            spec.rise_expr_start,
            spec.rise_expr_count,
            node_values,
            state_values,
            param_values,
            time,
            0.0,
        )?;
        input_falls[idx] = rust_sim_eval_expr_segment(
            body_expr_ops,
            spec.fall_expr_start,
            spec.fall_expr_count,
            node_values,
            state_values,
            param_values,
            time,
            input_rises[idx],
        )?;
        let spec_default_transition =
            if spec.default_transition.is_finite() && spec.default_transition > 0.0 {
                spec.default_transition
            } else {
                default_transition
            };
        if input_rises[idx] <= 0.0 {
            input_rises[idx] = spec_default_transition;
        }
        if input_falls[idx] <= 0.0 {
            input_falls[idx] = spec_default_transition;
        }
    }

    transition_state_step_for_arrays(
        current_values,
        target_values,
        start_times,
        start_values,
        delays,
        rise_times,
        fall_times,
        active_flags,
        initialized_flags,
        &input_targets,
        &input_delays,
        &input_rises,
        &input_falls,
        output_values,
        time,
        default_transition,
        initial_condition_mode,
    )?;

    for idx in 0..count {
        let spec = transitions[idx];
        if spec.output_node_id >= node_values.len() {
            return Err(-972);
        }
        let reference = if spec.reference_node_id == CONDITION_NONE {
            0.0
        } else {
            if spec.reference_node_id >= node_values.len() {
                return Err(-973);
            }
            node_values[spec.reference_node_id]
        };
        let output_bias = rust_sim_eval_expr_segment(
            body_expr_ops,
            spec.output_bias_expr_start,
            spec.output_bias_expr_count,
            node_values,
            state_values,
            param_values,
            time,
            0.0,
        )?;
        let output_scale = rust_sim_eval_expr_segment(
            body_expr_ops,
            spec.output_scale_expr_start,
            spec.output_scale_expr_count,
            node_values,
            state_values,
            param_values,
            time,
            1.0,
        )?;
        node_values[spec.output_node_id] =
            reference + output_bias + output_scale * output_values[idx];
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn rust_sim_record_transition_breakpoints_until(
    sources: &[EvasRustSimSourceSpec],
    source_data: &[f64],
    linear_ops: &[EvasRustLinearOp],
    linear_terms: &[EvasRustLinearTerm],
    linear_conditions: &[EvasRustLinearCondition],
    transitions: &[EvasRustSimTransitionSpec],
    body_stmt_ops: &[EvasRustBodyStmtOp],
    body_expr_ops: &[EvasRustBodyExprOp],
    events: &[EvasRustSimEventSpec],
    param_values: &[f64],
    node_values: &mut [f64],
    state_values: &mut [f64],
    record_node_ids: &[usize],
    time_values: &mut [f64],
    signal_values: &mut [f64],
    step_values: &mut [f64],
    count: &mut usize,
    transition_current_values: &mut [f64],
    transition_target_values: &mut [f64],
    transition_start_times: &mut [f64],
    transition_start_values: &mut [f64],
    transition_delays: &mut [f64],
    transition_rise_times: &mut [f64],
    transition_fall_times: &mut [f64],
    transition_active_flags: &mut [u8],
    transition_initialized_flags: &mut [u8],
    transition_output_values: &mut [f64],
    cross_prev_values: &mut [f64],
    cross_prev_times: &mut [f64],
    cross_pprev_values: &mut [f64],
    cross_pprev_times: &mut [f64],
    cross_initialized: &mut [u8],
    cross_last_times: &mut [f64],
    bound_step_limit: &mut f64,
    side_effect_log: &mut RustSideEffectLog<'_>,
    drain_post_cross_events: bool,
    event_fires: &mut usize,
    cursor_time: &mut f64,
    horizon: f64,
    min_ramp_time: f64,
    default_transition: f64,
) -> Result<usize, i32> {
    let eps = 1.0e-18;
    let mut added = 0_usize;
    let mut guard = 0_usize;
    let mut post_cross_candidates = Vec::<RustSimCrossCandidate>::with_capacity(events.len());
    loop {
        let Some(bp) = next_transition_breakpoint_for_arrays(
            transition_start_times,
            transition_start_values,
            transition_target_values,
            transition_delays,
            transition_rise_times,
            transition_fall_times,
            transition_active_flags,
            *cursor_time,
            min_ramp_time,
        )?
        else {
            break;
        };
        if bp <= *cursor_time + eps || bp >= horizon - eps {
            break;
        }
        rust_sim_write_sources(sources, source_data, node_values, bp)?;
        evaluate_static_linear_ops(
            linear_ops,
            linear_terms,
            linear_conditions,
            node_values,
            state_values,
        )?;
        rust_sim_apply_transitions(
            transitions,
            body_expr_ops,
            node_values,
            state_values,
            param_values,
            transition_current_values,
            transition_target_values,
            transition_start_times,
            transition_start_values,
            transition_delays,
            transition_rise_times,
            transition_fall_times,
            transition_active_flags,
            transition_initialized_flags,
            transition_output_values,
            bp,
            default_transition,
            false,
        )?;
        if drain_post_cross_events {
            rust_sim_collect_cross_events_into(
                events,
                body_expr_ops,
                node_values,
                state_values,
                param_values,
                cross_prev_values,
                cross_prev_times,
                cross_pprev_values,
                cross_pprev_times,
                cross_initialized,
                cross_last_times,
                bp,
                false,
                RUST_SIM_EVENT_PHASE_POST,
                &mut post_cross_candidates,
            )?;
            for candidate in &post_cross_candidates {
                if candidate.event_idx >= events.len() {
                    return Err(-990);
                }
                let event_time = bp;
                if event_time < *cursor_time - eps || event_time > bp + eps {
                    continue;
                }
                rust_sim_write_sources(sources, source_data, node_values, event_time)?;
                evaluate_static_linear_ops(
                    linear_ops,
                    linear_terms,
                    linear_conditions,
                    node_values,
                    state_values,
                )?;
                rust_sim_apply_transitions(
                    transitions,
                    body_expr_ops,
                    node_values,
                    state_values,
                    param_values,
                    transition_current_values,
                    transition_target_values,
                    transition_start_times,
                    transition_start_values,
                    transition_delays,
                    transition_rise_times,
                    transition_fall_times,
                    transition_active_flags,
                    transition_initialized_flags,
                    transition_output_values,
                    event_time,
                    default_transition,
                    false,
                )?;
                rust_sim_execute_event_body(
                    &events[candidate.event_idx],
                    body_stmt_ops,
                    body_expr_ops,
                    node_values,
                    state_values,
                    param_values,
                    event_time,
                    bound_step_limit,
                    Some(&mut *side_effect_log),
                )?;
                evaluate_static_linear_ops(
                    linear_ops,
                    linear_terms,
                    linear_conditions,
                    node_values,
                    state_values,
                )?;
                rust_sim_apply_transitions(
                    transitions,
                    body_expr_ops,
                    node_values,
                    state_values,
                    param_values,
                    transition_current_values,
                    transition_target_values,
                    transition_start_times,
                    transition_start_values,
                    transition_delays,
                    transition_rise_times,
                    transition_fall_times,
                    transition_active_flags,
                    transition_initialized_flags,
                    transition_output_values,
                    event_time,
                    default_transition,
                    false,
                )?;
                rust_sim_record_point_dedup(
                    time_values,
                    signal_values,
                    step_values,
                    count,
                    event_time,
                    event_time - *cursor_time,
                    node_values,
                    record_node_ids,
                )?;
                *event_fires = event_fires.checked_add(1).ok_or(-991)?;
            }
            rust_sim_write_sources(sources, source_data, node_values, bp)?;
            evaluate_static_linear_ops(
                linear_ops,
                linear_terms,
                linear_conditions,
                node_values,
                state_values,
            )?;
            rust_sim_apply_transitions(
                transitions,
                body_expr_ops,
                node_values,
                state_values,
                param_values,
                transition_current_values,
                transition_target_values,
                transition_start_times,
                transition_start_values,
                transition_delays,
                transition_rise_times,
                transition_fall_times,
                transition_active_flags,
                transition_initialized_flags,
                transition_output_values,
                bp,
                default_transition,
                false,
            )?;
        }
        rust_sim_record_point_dedup(
            time_values,
            signal_values,
            step_values,
            count,
            bp,
            bp - *cursor_time,
            node_values,
            record_node_ids,
        )?;
        *cursor_time = bp;
        added += 1;
        guard += 1;
        if guard > 4096 {
            return Err(-988);
        }
    }
    Ok(added)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn rust_sim_execute_ordered_cross_events(
    sources: &[EvasRustSimSourceSpec],
    source_data: &[f64],
    linear_ops: &[EvasRustLinearOp],
    linear_terms: &[EvasRustLinearTerm],
    linear_conditions: &[EvasRustLinearCondition],
    body_stmt_ops: &[EvasRustBodyStmtOp],
    body_expr_ops: &[EvasRustBodyExprOp],
    events: &[EvasRustSimEventSpec],
    transitions: &[EvasRustSimTransitionSpec],
    param_values: &[f64],
    node_values: &mut [f64],
    state_values: &mut [f64],
    record_node_ids: &[usize],
    time_values: &mut [f64],
    signal_values: &mut [f64],
    step_values: &mut [f64],
    count: &mut usize,
    transition_current_values: &mut [f64],
    transition_target_values: &mut [f64],
    transition_start_times: &mut [f64],
    transition_start_values: &mut [f64],
    transition_delays: &mut [f64],
    transition_rise_times: &mut [f64],
    transition_fall_times: &mut [f64],
    transition_active_flags: &mut [u8],
    transition_initialized_flags: &mut [u8],
    transition_output_values: &mut [f64],
    candidates: &[RustSimCrossCandidate],
    step_start_time: f64,
    step_end_time: f64,
    min_ramp_time: f64,
    default_transition: f64,
    drain_transition_post_cross_events: bool,
    bound_step_limit: &mut f64,
    side_effect_log: &mut RustSideEffectLog<'_>,
    cross_prev_values: &mut [f64],
    cross_prev_times: &mut [f64],
    cross_pprev_values: &mut [f64],
    cross_pprev_times: &mut [f64],
    cross_initialized: &mut [u8],
    cross_last_times: &mut [f64],
) -> Result<(usize, usize), i32> {
    let eps = 1.0e-18;
    let mut fired = 0_usize;
    let mut transition_breakpoints = 0_usize;
    let mut cursor_time = step_start_time;

    for candidate in candidates {
        if candidate.event_idx >= events.len() {
            return Err(-989);
        }
        let mut event_time = candidate.event_time;
        if event_time < step_start_time && event_time + eps >= step_start_time {
            event_time = step_start_time;
        }
        if event_time > step_end_time && event_time - eps <= step_end_time {
            event_time = step_end_time;
        }
        if event_time < step_start_time - eps || event_time > step_end_time + eps {
            continue;
        }
        if event_time < cursor_time {
            event_time = cursor_time;
        }

        transition_breakpoints += rust_sim_record_transition_breakpoints_until(
            sources,
            source_data,
            linear_ops,
            linear_terms,
            linear_conditions,
            transitions,
            body_stmt_ops,
            body_expr_ops,
            events,
            param_values,
            node_values,
            state_values,
            record_node_ids,
            time_values,
            signal_values,
            step_values,
            count,
            transition_current_values,
            transition_target_values,
            transition_start_times,
            transition_start_values,
            transition_delays,
            transition_rise_times,
            transition_fall_times,
            transition_active_flags,
            transition_initialized_flags,
            transition_output_values,
            cross_prev_values,
            cross_prev_times,
            cross_pprev_values,
            cross_pprev_times,
            cross_initialized,
            cross_last_times,
            bound_step_limit,
            side_effect_log,
            drain_transition_post_cross_events && fired > 0,
            &mut fired,
            &mut cursor_time,
            event_time,
            min_ramp_time,
            default_transition,
        )?;
        if *count > 0 {
            let last_recorded_time = time_values[*count - 1];
            if event_time < last_recorded_time - eps {
                event_time = last_recorded_time;
            }
        }

        rust_sim_write_sources(sources, source_data, node_values, event_time)?;
        evaluate_static_linear_ops(
            linear_ops,
            linear_terms,
            linear_conditions,
            node_values,
            state_values,
        )?;
        rust_sim_apply_transitions(
            transitions,
            body_expr_ops,
            node_values,
            state_values,
            param_values,
            transition_current_values,
            transition_target_values,
            transition_start_times,
            transition_start_values,
            transition_delays,
            transition_rise_times,
            transition_fall_times,
            transition_active_flags,
            transition_initialized_flags,
            transition_output_values,
            event_time,
            default_transition,
            false,
        )?;
        rust_sim_execute_event_body(
            &events[candidate.event_idx],
            body_stmt_ops,
            body_expr_ops,
            node_values,
            state_values,
            param_values,
            event_time,
            bound_step_limit,
            Some(&mut *side_effect_log),
        )?;
        evaluate_static_linear_ops(
            linear_ops,
            linear_terms,
            linear_conditions,
            node_values,
            state_values,
        )?;
        rust_sim_apply_transitions(
            transitions,
            body_expr_ops,
            node_values,
            state_values,
            param_values,
            transition_current_values,
            transition_target_values,
            transition_start_times,
            transition_start_values,
            transition_delays,
            transition_rise_times,
            transition_fall_times,
            transition_active_flags,
            transition_initialized_flags,
            transition_output_values,
            event_time,
            default_transition,
            false,
        )?;
        rust_sim_record_point_dedup(
            time_values,
            signal_values,
            step_values,
            count,
            event_time,
            event_time - cursor_time,
            node_values,
            record_node_ids,
        )?;
        cursor_time = event_time;
        fired += 1;
    }

    transition_breakpoints += rust_sim_record_transition_breakpoints_until(
        sources,
        source_data,
        linear_ops,
        linear_terms,
        linear_conditions,
        transitions,
        body_stmt_ops,
        body_expr_ops,
        events,
        param_values,
        node_values,
        state_values,
        record_node_ids,
        time_values,
        signal_values,
        step_values,
        count,
        transition_current_values,
        transition_target_values,
        transition_start_times,
        transition_start_values,
        transition_delays,
        transition_rise_times,
        transition_fall_times,
        transition_active_flags,
        transition_initialized_flags,
        transition_output_values,
        cross_prev_values,
        cross_prev_times,
        cross_pprev_values,
        cross_pprev_times,
        cross_initialized,
        cross_last_times,
        bound_step_limit,
        side_effect_log,
        drain_transition_post_cross_events && fired > 0,
        &mut fired,
        &mut cursor_time,
        step_end_time,
        min_ramp_time,
        default_transition,
    )?;

    Ok((fired, transition_breakpoints))
}

#[allow(clippy::too_many_arguments)]
pub fn rust_sim_event_transition_record_trace(
    sources: &[EvasRustSimSourceSpec],
    source_data: &[f64],
    linear_ops: &[EvasRustLinearOp],
    linear_terms: &[EvasRustLinearTerm],
    linear_conditions: &[EvasRustLinearCondition],
    body_stmt_ops: &[EvasRustBodyStmtOp],
    body_expr_ops: &[EvasRustBodyExprOp],
    events: &[EvasRustSimEventSpec],
    transitions: &[EvasRustSimTransitionSpec],
    side_effect_kinds: &mut [u8],
    side_effect_spec_ids: &mut [usize],
    side_effect_arg_starts: &mut [usize],
    side_effect_arg_counts: &mut [usize],
    side_effect_times: &mut [f64],
    side_effect_count: &mut usize,
    side_effect_values: &mut [f64],
    side_effect_value_count: &mut usize,
    param_values: &[f64],
    node_values: &mut [f64],
    state_values: &mut [f64],
    record_node_ids: &[usize],
    time_values: &mut [f64],
    signal_values: &mut [f64],
    step_values: &mut [f64],
    tstop: f64,
    tstep: f64,
    max_step: f64,
    record_step: f64,
    use_record_step: bool,
    min_ramp_time: f64,
    default_transition: f64,
) -> Result<(usize, usize, usize, usize), i32> {
    if !tstop.is_finite() || !tstep.is_finite() || !max_step.is_finite() {
        return Err(-981);
    }
    if tstop < 0.0 || tstep <= 0.0 || max_step <= 0.0 {
        return Err(-982);
    }
    if use_record_step && (!record_step.is_finite() || record_step <= 0.0) {
        return Err(-983);
    }
    if time_values.len() != step_values.len() {
        return Err(-984);
    }
    let record_count = record_node_ids.len();
    if record_count == 0 {
        return Err(-985);
    }
    if signal_values.len() < time_values.len().saturating_mul(record_count) {
        return Err(-986);
    }

    let eps = 1.0e-18;
    let event_count = events.len();
    let transition_count = transitions.len();
    let has_pre_cross_events = events
        .iter()
        .any(|event| event.phase == RUST_SIM_EVENT_PHASE_PRE && event.kind == RUST_SIM_EVENT_CROSS);
    let has_post_cross_events = events.iter().any(|event| {
        event.phase == RUST_SIM_EVENT_PHASE_POST && event.kind == RUST_SIM_EVENT_CROSS
    });
    let has_pre_initial_runtime_events = events.iter().any(|event| {
        event.phase == RUST_SIM_EVENT_PHASE_PRE
            && event.kind != RUST_SIM_EVENT_CROSS
            && event.kind != RUST_SIM_EVENT_FINAL_STEP
    });
    let has_post_initial_runtime_events = events.iter().any(|event| {
        event.phase == RUST_SIM_EVENT_PHASE_POST
            && event.kind != RUST_SIM_EVENT_CROSS
            && event.kind != RUST_SIM_EVENT_FINAL_STEP
    });
    let has_pre_step_runtime_events = events.iter().any(|event| {
        event.phase == RUST_SIM_EVENT_PHASE_PRE
            && (event.kind == RUST_SIM_EVENT_ABOVE
                || event.kind == RUST_SIM_EVENT_TIMER
                || event.kind == RUST_SIM_EVENT_ALWAYS)
    });
    let has_post_step_runtime_events = events.iter().any(|event| {
        event.phase == RUST_SIM_EVENT_PHASE_POST
            && (event.kind == RUST_SIM_EVENT_ABOVE
                || event.kind == RUST_SIM_EVENT_TIMER
                || event.kind == RUST_SIM_EVENT_ALWAYS)
    });
    let has_pre_timer_events = events
        .iter()
        .any(|event| event.phase == RUST_SIM_EVENT_PHASE_PRE && event.kind == RUST_SIM_EVENT_TIMER);
    let has_final_step_events = events
        .iter()
        .any(|event| event.kind == RUST_SIM_EVENT_FINAL_STEP);
    let mut source_breakpoints = 0_usize;
    let mut event_fires = 0_usize;
    let mut transition_breakpoints = 0_usize;
    let mut count = 0_usize;
    let mut time = 0.0_f64;
    let mut next_source_breakpoint =
        rust_sim_next_source_breakpoint_after(sources, source_data, 0.0)?;
    let mut next_record_time = if use_record_step {
        record_step
    } else {
        f64::INFINITY
    };
    let mut pre_cross_candidates = Vec::<RustSimCrossCandidate>::with_capacity(event_count);
    let mut post_cross_candidates = Vec::<RustSimCrossCandidate>::with_capacity(event_count);

    let mut cross_prev_values = vec![0.0_f64; event_count];
    let mut cross_prev_times = vec![0.0_f64; event_count];
    let mut cross_pprev_values = vec![0.0_f64; event_count];
    let mut cross_pprev_times = vec![0.0_f64; event_count];
    let mut cross_initialized = vec![0_u8; event_count];
    let mut cross_last_times = vec![-1.0_f64; event_count];
    let mut above_prev_values = vec![0.0_f64; event_count];
    let mut above_prev_times = vec![0.0_f64; event_count];
    let mut above_pprev_values = vec![0.0_f64; event_count];
    let mut above_pprev_times = vec![0.0_f64; event_count];
    let mut above_initialized = vec![0_u8; event_count];
    let mut timer_next_fire_times = vec![0.0_f64; event_count];
    let mut timer_has_state_flags = vec![0_u8; event_count];
    let mut timer_last_fired_times = vec![0.0_f64; event_count];
    let mut timer_has_last_flags = vec![0_u8; event_count];

    let mut transition_current_values = vec![0.0_f64; transition_count];
    let mut transition_target_values = vec![0.0_f64; transition_count];
    let mut transition_start_times = vec![0.0_f64; transition_count];
    let mut transition_start_values = vec![0.0_f64; transition_count];
    let mut transition_delays = vec![0.0_f64; transition_count];
    let mut transition_rise_times = vec![0.0_f64; transition_count];
    let mut transition_fall_times = vec![0.0_f64; transition_count];
    let mut transition_active_flags = vec![0_u8; transition_count];
    let mut transition_initialized_flags = vec![0_u8; transition_count];
    let mut transition_output_values = vec![0.0_f64; transition_count];
    let mut bound_step_limit = f64::INFINITY;
    let mut side_effect_log = RustSideEffectLog {
        kinds: side_effect_kinds,
        spec_ids: side_effect_spec_ids,
        arg_starts: side_effect_arg_starts,
        arg_counts: side_effect_arg_counts,
        times: side_effect_times,
        values: side_effect_values,
        count: side_effect_count,
        value_count: side_effect_value_count,
    };

    rust_sim_write_sources(sources, source_data, node_values, 0.0)?;
    if has_pre_initial_runtime_events {
        event_fires += rust_sim_execute_events(
            events,
            body_stmt_ops,
            body_expr_ops,
            node_values,
            state_values,
            param_values,
            &mut cross_prev_values,
            &mut cross_prev_times,
            &mut cross_pprev_values,
            &mut cross_pprev_times,
            &mut cross_initialized,
            &mut cross_last_times,
            &mut above_prev_values,
            &mut above_prev_times,
            &mut above_pprev_values,
            &mut above_pprev_times,
            &mut above_initialized,
            &mut timer_next_fire_times,
            &mut timer_has_state_flags,
            &mut timer_last_fired_times,
            &mut timer_has_last_flags,
            &mut bound_step_limit,
            &mut side_effect_log,
            0.0,
            true,
            RUST_SIM_EVENT_PHASE_PRE,
        )?;
    }
    if has_pre_cross_events {
        rust_sim_collect_cross_events_into(
            events,
            body_expr_ops,
            node_values,
            state_values,
            param_values,
            &mut cross_prev_values,
            &mut cross_prev_times,
            &mut cross_pprev_values,
            &mut cross_pprev_times,
            &mut cross_initialized,
            &mut cross_last_times,
            0.0,
            true,
            RUST_SIM_EVENT_PHASE_PRE,
            &mut pre_cross_candidates,
        )?;
    }
    evaluate_static_linear_ops(
        linear_ops,
        linear_terms,
        linear_conditions,
        node_values,
        state_values,
    )?;
    rust_sim_apply_transitions(
        transitions,
        body_expr_ops,
        node_values,
        state_values,
        param_values,
        &mut transition_current_values,
        &mut transition_target_values,
        &mut transition_start_times,
        &mut transition_start_values,
        &mut transition_delays,
        &mut transition_rise_times,
        &mut transition_fall_times,
        &mut transition_active_flags,
        &mut transition_initialized_flags,
        &mut transition_output_values,
        0.0,
        default_transition,
        true,
    )?;
    if has_post_cross_events {
        rust_sim_collect_cross_events_into(
            events,
            body_expr_ops,
            node_values,
            state_values,
            param_values,
            &mut cross_prev_values,
            &mut cross_prev_times,
            &mut cross_pprev_values,
            &mut cross_pprev_times,
            &mut cross_initialized,
            &mut cross_last_times,
            0.0,
            false,
            RUST_SIM_EVENT_PHASE_POST,
            &mut post_cross_candidates,
        )?;
    }
    let post_event_fires = if has_post_initial_runtime_events {
        rust_sim_execute_events(
            events,
            body_stmt_ops,
            body_expr_ops,
            node_values,
            state_values,
            param_values,
            &mut cross_prev_values,
            &mut cross_prev_times,
            &mut cross_pprev_values,
            &mut cross_pprev_times,
            &mut cross_initialized,
            &mut cross_last_times,
            &mut above_prev_values,
            &mut above_prev_times,
            &mut above_pprev_values,
            &mut above_pprev_times,
            &mut above_initialized,
            &mut timer_next_fire_times,
            &mut timer_has_state_flags,
            &mut timer_last_fired_times,
            &mut timer_has_last_flags,
            &mut bound_step_limit,
            &mut side_effect_log,
            0.0,
            false,
            RUST_SIM_EVENT_PHASE_POST,
        )?
    } else {
        0
    };
    event_fires += post_event_fires;
    if post_event_fires > 0 {
        evaluate_static_linear_ops(
            linear_ops,
            linear_terms,
            linear_conditions,
            node_values,
            state_values,
        )?;
        rust_sim_apply_transitions(
            transitions,
            body_expr_ops,
            node_values,
            state_values,
            param_values,
            &mut transition_current_values,
            &mut transition_target_values,
            &mut transition_start_times,
            &mut transition_start_values,
            &mut transition_delays,
            &mut transition_rise_times,
            &mut transition_fall_times,
            &mut transition_active_flags,
            &mut transition_initialized_flags,
            &mut transition_output_values,
            0.0,
            default_transition,
            false,
        )?;
    }
    rust_sim_record_point(
        time_values,
        signal_values,
        step_values,
        count,
        0.0,
        0.0,
        node_values,
        record_node_ids,
    )?;
    count += 1;

    let mut loop_iterations = 0_usize;
    let max_loop_iterations = time_values.len().saturating_mul(16).max(1024);
    while time < tstop {
        loop_iterations = loop_iterations.checked_add(1).ok_or(-2145)?;
        if loop_iterations > max_loop_iterations {
            return Err(-2146);
        }
        let mut force_record = false;
        let mut dt = tstep.min(max_step).min(tstop - time);
        if dt <= 0.0 {
            break;
        }
        let horizon = time + dt;
        while next_source_breakpoint.map_or(false, |bp| bp <= time + eps) {
            next_source_breakpoint =
                rust_sim_next_source_breakpoint_after(sources, source_data, time)?;
        }
        if let Some(bp) = next_source_breakpoint {
            if bp < horizon {
                dt = bp - time;
                force_record = true;
                source_breakpoints += 1;
            }
        }
        if let Some(bp) = next_transition_breakpoint_for_arrays(
            &transition_start_times,
            &transition_start_values,
            &transition_target_values,
            &transition_delays,
            &transition_rise_times,
            &transition_fall_times,
            &transition_active_flags,
            time,
            min_ramp_time,
        )? {
            if bp > time && bp < time + dt {
                dt = bp - time;
                force_record = true;
                transition_breakpoints += 1;
            }
        }
        if has_pre_timer_events {
            if let Some(bp) = rust_sim_next_timer_event_breakpoint(
                events,
                &timer_next_fire_times,
                &timer_has_state_flags,
                &timer_last_fired_times,
                &timer_has_last_flags,
                time,
                time + dt,
            )? {
                dt = bp - time;
                force_record = true;
            }
        }
        if bound_step_limit.is_finite() && bound_step_limit > 0.0 && dt > bound_step_limit {
            dt = bound_step_limit;
            force_record = true;
        }
        if let Some(bp) = rust_sim_next_transition_target_change_breakpoint(
            sources,
            source_data,
            linear_ops,
            linear_terms,
            linear_conditions,
            body_stmt_ops,
            body_expr_ops,
            events,
            transitions,
            param_values,
            node_values,
            state_values,
            time,
            dt,
        )? {
            if bp > time && bp < time + dt {
                dt = bp - time;
                force_record = true;
                transition_breakpoints += 1;
            }
        }
        if use_record_step && next_record_time > time && next_record_time < time + dt {
            dt = next_record_time - time;
            force_record = true;
        }
        if dt <= 0.0 {
            dt = eps;
        }
        let step_start_time = time;
        time += dt;
        if time > tstop && time < tstop + eps {
            time = tstop;
        }
        bound_step_limit = f64::INFINITY;

        rust_sim_write_sources(sources, source_data, node_values, time)?;
        if has_pre_cross_events {
            rust_sim_collect_cross_events_into(
                events,
                body_expr_ops,
                node_values,
                state_values,
                param_values,
                &mut cross_prev_values,
                &mut cross_prev_times,
                &mut cross_pprev_values,
                &mut cross_pprev_times,
                &mut cross_initialized,
                &mut cross_last_times,
                time,
                false,
                RUST_SIM_EVENT_PHASE_PRE,
                &mut pre_cross_candidates,
            )?;
            let (pre_cross_fires, pre_cross_transition_bps) =
                rust_sim_execute_ordered_cross_events(
                    sources,
                    source_data,
                    linear_ops,
                    linear_terms,
                    linear_conditions,
                    body_stmt_ops,
                    body_expr_ops,
                    events,
                    transitions,
                    param_values,
                    node_values,
                    state_values,
                    record_node_ids,
                    time_values,
                    signal_values,
                    step_values,
                    &mut count,
                    &mut transition_current_values,
                    &mut transition_target_values,
                    &mut transition_start_times,
                    &mut transition_start_values,
                    &mut transition_delays,
                    &mut transition_rise_times,
                    &mut transition_fall_times,
                    &mut transition_active_flags,
                    &mut transition_initialized_flags,
                    &mut transition_output_values,
                    &pre_cross_candidates,
                    step_start_time,
                    time,
                    min_ramp_time,
                    default_transition,
                    false,
                    &mut bound_step_limit,
                    &mut side_effect_log,
                    &mut cross_prev_values,
                    &mut cross_prev_times,
                    &mut cross_pprev_values,
                    &mut cross_pprev_times,
                    &mut cross_initialized,
                    &mut cross_last_times,
                )?;
            if pre_cross_fires > 0 || pre_cross_transition_bps > 0 {
                event_fires += pre_cross_fires;
                transition_breakpoints += pre_cross_transition_bps;
                force_record = true;
                rust_sim_write_sources(sources, source_data, node_values, time)?;
            }
        }
        if has_pre_step_runtime_events {
            event_fires += rust_sim_execute_events(
                events,
                body_stmt_ops,
                body_expr_ops,
                node_values,
                state_values,
                param_values,
                &mut cross_prev_values,
                &mut cross_prev_times,
                &mut cross_pprev_values,
                &mut cross_pprev_times,
                &mut cross_initialized,
                &mut cross_last_times,
                &mut above_prev_values,
                &mut above_prev_times,
                &mut above_pprev_values,
                &mut above_pprev_times,
                &mut above_initialized,
                &mut timer_next_fire_times,
                &mut timer_has_state_flags,
                &mut timer_last_fired_times,
                &mut timer_has_last_flags,
                &mut bound_step_limit,
                &mut side_effect_log,
                time,
                false,
                RUST_SIM_EVENT_PHASE_PRE,
            )?;
        }
        evaluate_static_linear_ops(
            linear_ops,
            linear_terms,
            linear_conditions,
            node_values,
            state_values,
        )?;
        rust_sim_apply_transitions(
            transitions,
            body_expr_ops,
            node_values,
            state_values,
            param_values,
            &mut transition_current_values,
            &mut transition_target_values,
            &mut transition_start_times,
            &mut transition_start_values,
            &mut transition_delays,
            &mut transition_rise_times,
            &mut transition_fall_times,
            &mut transition_active_flags,
            &mut transition_initialized_flags,
            &mut transition_output_values,
            time,
            default_transition,
            false,
        )?;
        if has_post_cross_events {
            rust_sim_collect_cross_events_into(
                events,
                body_expr_ops,
                node_values,
                state_values,
                param_values,
                &mut cross_prev_values,
                &mut cross_prev_times,
                &mut cross_pprev_values,
                &mut cross_pprev_times,
                &mut cross_initialized,
                &mut cross_last_times,
                time,
                false,
                RUST_SIM_EVENT_PHASE_POST,
                &mut post_cross_candidates,
            )?;
            let (post_cross_fires, post_cross_transition_bps) =
                rust_sim_execute_ordered_cross_events(
                    sources,
                    source_data,
                    linear_ops,
                    linear_terms,
                    linear_conditions,
                    body_stmt_ops,
                    body_expr_ops,
                    events,
                    transitions,
                    param_values,
                    node_values,
                    state_values,
                    record_node_ids,
                    time_values,
                    signal_values,
                    step_values,
                    &mut count,
                    &mut transition_current_values,
                    &mut transition_target_values,
                    &mut transition_start_times,
                    &mut transition_start_values,
                    &mut transition_delays,
                    &mut transition_rise_times,
                    &mut transition_fall_times,
                    &mut transition_active_flags,
                    &mut transition_initialized_flags,
                    &mut transition_output_values,
                    &post_cross_candidates,
                    step_start_time,
                    time,
                    min_ramp_time,
                    default_transition,
                    true,
                    &mut bound_step_limit,
                    &mut side_effect_log,
                    &mut cross_prev_values,
                    &mut cross_prev_times,
                    &mut cross_pprev_values,
                    &mut cross_pprev_times,
                    &mut cross_initialized,
                    &mut cross_last_times,
                )?;
            if post_cross_fires > 0 || post_cross_transition_bps > 0 {
                event_fires += post_cross_fires;
                transition_breakpoints += post_cross_transition_bps;
                force_record = true;
                rust_sim_write_sources(sources, source_data, node_values, time)?;
                evaluate_static_linear_ops(
                    linear_ops,
                    linear_terms,
                    linear_conditions,
                    node_values,
                    state_values,
                )?;
                rust_sim_apply_transitions(
                    transitions,
                    body_expr_ops,
                    node_values,
                    state_values,
                    param_values,
                    &mut transition_current_values,
                    &mut transition_target_values,
                    &mut transition_start_times,
                    &mut transition_start_values,
                    &mut transition_delays,
                    &mut transition_rise_times,
                    &mut transition_fall_times,
                    &mut transition_active_flags,
                    &mut transition_initialized_flags,
                    &mut transition_output_values,
                    time,
                    default_transition,
                    false,
                )?;
            }
        }
        let post_event_fires = if has_post_step_runtime_events {
            rust_sim_execute_events(
                events,
                body_stmt_ops,
                body_expr_ops,
                node_values,
                state_values,
                param_values,
                &mut cross_prev_values,
                &mut cross_prev_times,
                &mut cross_pprev_values,
                &mut cross_pprev_times,
                &mut cross_initialized,
                &mut cross_last_times,
                &mut above_prev_values,
                &mut above_prev_times,
                &mut above_pprev_values,
                &mut above_pprev_times,
                &mut above_initialized,
                &mut timer_next_fire_times,
                &mut timer_has_state_flags,
                &mut timer_last_fired_times,
                &mut timer_has_last_flags,
                &mut bound_step_limit,
                &mut side_effect_log,
                time,
                false,
                RUST_SIM_EVENT_PHASE_POST,
            )?
        } else {
            0
        };
        if post_event_fires > 0 {
            event_fires += post_event_fires;
            force_record = true;
            evaluate_static_linear_ops(
                linear_ops,
                linear_terms,
                linear_conditions,
                node_values,
                state_values,
            )?;
            rust_sim_apply_transitions(
                transitions,
                body_expr_ops,
                node_values,
                state_values,
                param_values,
                &mut transition_current_values,
                &mut transition_target_values,
                &mut transition_start_times,
                &mut transition_start_values,
                &mut transition_delays,
                &mut transition_rise_times,
                &mut transition_fall_times,
                &mut transition_active_flags,
                &mut transition_initialized_flags,
                &mut transition_output_values,
                time,
                default_transition,
                false,
            )?;
        }

        let should_record = if use_record_step {
            force_record || time >= next_record_time - eps || time >= tstop - eps
        } else {
            true
        };
        if should_record {
            rust_sim_record_point_dedup(
                time_values,
                signal_values,
                step_values,
                &mut count,
                time,
                dt,
                node_values,
                record_node_ids,
            )?;
            if use_record_step {
                while next_record_time <= time + eps {
                    next_record_time += record_step;
                }
            }
        }
    }

    if has_final_step_events {
        event_fires += rust_sim_execute_final_step_events(
            events,
            body_stmt_ops,
            body_expr_ops,
            node_values,
            state_values,
            param_values,
            tstop,
            &mut bound_step_limit,
            &mut side_effect_log,
        )?;
    }

    Ok((
        count,
        source_breakpoints,
        event_fires,
        transition_breakpoints,
    ))
}
