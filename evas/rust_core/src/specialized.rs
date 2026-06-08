use crate::event::*;
use crate::transition::*;

// Specialized trace kernels retained for existing benchmark speed experiments.

pub(crate) fn pulse_value(
    v_lo: f64,
    v_hi: f64,
    period: f64,
    duty: f64,
    rise: f64,
    fall: f64,
    delay: f64,
    width: f64,
    has_width: bool,
    time: f64,
) -> f64 {
    let t_eff = time - delay;
    if t_eff < 0.0 {
        return v_lo;
    }

    let one_shot = period <= 0.0;
    let fall_start = if has_width {
        rise + width
    } else if one_shot {
        f64::INFINITY
    } else {
        period * duty
    };
    let fall_end = fall_start + fall;
    let t_mod = if one_shot { t_eff } else { t_eff % period };

    if t_mod < rise {
        let frac = if rise > 0.0 { t_mod / rise } else { 1.0 };
        v_lo + frac * (v_hi - v_lo)
    } else if t_mod < fall_start {
        v_hi
    } else if t_mod < fall_end {
        let frac = if fall > 0.0 {
            (t_mod - fall_start) / fall
        } else {
            1.0
        };
        v_hi - frac * (v_hi - v_lo)
    } else {
        v_lo
    }
}

pub(crate) fn pulse_fall_start(
    period: f64,
    duty: f64,
    rise: f64,
    width: f64,
    has_width: bool,
) -> f64 {
    if has_width {
        rise + width
    } else if period <= 0.0 {
        f64::INFINITY
    } else {
        period * duty
    }
}

pub(crate) fn add_pulse_schedule_times(
    times: &mut Vec<f64>,
    v_lo: f64,
    v_hi: f64,
    period: f64,
    duty: f64,
    rise: f64,
    fall: f64,
    delay: f64,
    width: f64,
    has_width: bool,
    edge_threshold: f64,
    edge_direction: i32,
    tstop: f64,
) -> usize {
    let fall_start = pulse_fall_start(period, duty, rise, width, has_width);
    let mut knees = vec![0.0, rise];
    if fall_start.is_finite() {
        knees.push(fall_start);
        knees.push(fall_start + fall);
        if fall > 0.0 {
            knees.push(fall_start + 0.5 * fall);
        }
    }
    if rise > 0.0 {
        knees.push(0.5 * rise);
    }

    let cycles = if period <= 0.0 {
        1_usize
    } else {
        ((0.0_f64.max(tstop - delay) / period).floor() as usize) + 2
    };
    let mut cross_count = 0_usize;
    for n in 0..cycles.max(1) {
        let base = delay + if period > 0.0 { period * n as f64 } else { 0.0 };
        for &offset in &knees {
            if offset.is_finite() {
                times.push(base + offset);
            }
        }
        if v_hi != v_lo && rise >= 0.0 && edge_direction >= 0 {
            let lo = v_lo.min(v_hi);
            let hi = v_lo.max(v_hi);
            if edge_threshold >= lo && edge_threshold <= hi {
                let frac = if rise <= 0.0 {
                    1.0
                } else {
                    (edge_threshold - v_lo) / (v_hi - v_lo)
                };
                let cross_time = base + frac.max(0.0).min(1.0) * rise;
                if cross_time >= -1.0e-18 && cross_time <= tstop + 1.0e-18 {
                    times.push(cross_time);
                    cross_count += 1;
                }
            }
        }
        if period <= 0.0 {
            break;
        }
    }
    cross_count
}

pub(crate) fn uniform_record_times(tstop: f64, sample_step: f64) -> Vec<f64> {
    if sample_step <= 0.0 {
        return vec![0.0, tstop];
    }
    let count = (tstop / sample_step + 1.0e-9).floor() as usize + 1;
    let mut times = Vec::with_capacity(count + 1);
    for idx in 0..count.max(1) {
        times.push(tstop.min(idx as f64 * sample_step));
    }
    if times.last().map_or(true, |last| *last < tstop - 1.0e-18) {
        times.push(tstop);
    }
    times
}

pub(crate) fn dedupe_trace_times(mut times: Vec<f64>, tstop: f64) -> Vec<f64> {
    let eps = 1.0e-18;
    times.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    let mut cleaned: Vec<f64> = Vec::with_capacity(times.len() + 2);
    for mut value in times {
        if value < -eps || value > tstop + eps {
            continue;
        }
        if value.abs() <= eps {
            value = 0.0;
        }
        if (value - tstop).abs() <= eps {
            value = tstop;
        }
        if cleaned
            .last()
            .map_or(false, |last| (value - *last).abs() <= eps)
        {
            continue;
        }
        cleaned.push(value);
    }
    if cleaned.first().copied() != Some(0.0) {
        cleaned.insert(0, 0.0);
    }
    if cleaned.last().map_or(true, |last| *last < tstop - eps) {
        cleaned.push(tstop);
    }
    cleaned
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn event_transition_core_trace_pulse_step(
    time: f64,
    initial_condition_mode: bool,
    v_lo: f64,
    v_hi: f64,
    period: f64,
    duty: f64,
    pulse_rise: f64,
    pulse_fall: f64,
    pulse_delay: f64,
    pulse_width: f64,
    pulse_has_width: bool,
    edge_threshold: f64,
    event_state_value: f64,
    default_transition: f64,
    state_value: &mut f64,
    fired_events: &mut usize,
    cross_prev_values: &mut [f64; 1],
    cross_prev_times: &mut [f64; 1],
    cross_pprev_values: &mut [f64; 1],
    cross_pprev_times: &mut [f64; 1],
    cross_initialized: &mut [u8; 1],
    cross_directions: &[i32; 1],
    cross_last_times: &mut [f64; 1],
    cross_triggered: &mut [u8; 1],
    cross_times: &mut [f64; 1],
    cross_trigger_dirs: &mut [i32; 1],
    cross_went_beyond: &mut [u8; 1],
    current_values: &mut [f64; 1],
    target_values: &mut [f64; 1],
    start_times: &mut [f64; 1],
    start_values: &mut [f64; 1],
    delays: &mut [f64; 1],
    rise_times: &mut [f64; 1],
    fall_times: &mut [f64; 1],
    active_flags: &mut [u8; 1],
    initialized_flags: &mut [u8; 1],
    input_targets: &mut [f64; 1],
    input_delays: &[f64; 1],
    input_rises: &[f64; 1],
    input_falls: &[f64; 1],
    output_values: &mut [f64; 1],
) -> Result<f64, i32> {
    let source_value = pulse_value(
        v_lo,
        v_hi,
        period,
        duty,
        pulse_rise,
        pulse_fall,
        pulse_delay,
        pulse_width,
        pulse_has_width,
        time,
    );
    let expr_value = [source_value - edge_threshold];
    cross_detector_step_for_arrays(
        cross_prev_values,
        cross_prev_times,
        cross_pprev_values,
        cross_pprev_times,
        cross_initialized,
        cross_directions,
        cross_last_times,
        &expr_value,
        cross_triggered,
        cross_times,
        cross_trigger_dirs,
        cross_went_beyond,
        time,
        0.0,
        1.0e-12,
    )?;
    if !initial_condition_mode && cross_triggered[0] != 0 {
        *state_value = event_state_value;
        *fired_events += 1;
    }
    input_targets[0] = *state_value;
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
        input_targets,
        input_delays,
        input_rises,
        input_falls,
        output_values,
        time,
        default_transition,
        initial_condition_mode,
    )?;
    Ok(output_values[0])
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn event_transition_core_trace_pulse(
    time_values: &mut [f64],
    out_values: &mut [f64],
    tstop: f64,
    sample_step: f64,
    tstep: f64,
    v_lo: f64,
    v_hi: f64,
    period: f64,
    duty: f64,
    pulse_rise: f64,
    pulse_fall: f64,
    pulse_delay: f64,
    pulse_width: f64,
    pulse_has_width: bool,
    edge_threshold: f64,
    edge_direction: i32,
    initial_state_value: f64,
    event_state_value: f64,
    transition_delay: f64,
    transition_rise: f64,
    transition_fall: f64,
    default_transition: f64,
) -> Result<(usize, f64, usize, usize, usize), i32> {
    if time_values.len() != out_values.len() {
        return Err(-701);
    }
    if tstop < 0.0 || sample_step <= 0.0 || tstep <= 0.0 {
        return Err(-702);
    }

    let mut planned = uniform_record_times(tstop, sample_step);
    let source_breakpoints = add_pulse_schedule_times(
        &mut planned,
        v_lo,
        v_hi,
        period,
        duty,
        pulse_rise,
        pulse_fall,
        pulse_delay,
        pulse_width,
        pulse_has_width,
        edge_threshold,
        edge_direction,
        tstop,
    );
    let planned = dedupe_trace_times(planned, tstop);

    let mut count = 0_usize;
    let mut fired_events = 1_usize;
    let mut transition_breakpoints = 0_usize;
    let mut state_value = initial_state_value;

    let mut cross_prev_values = [0.0_f64; 1];
    let mut cross_prev_times = [0.0_f64; 1];
    let mut cross_pprev_values = [0.0_f64; 1];
    let mut cross_pprev_times = [0.0_f64; 1];
    let mut cross_initialized = [0_u8; 1];
    let cross_directions = [edge_direction];
    let mut cross_last_times = [-1.0_f64; 1];
    let mut cross_triggered = [0_u8; 1];
    let mut cross_times = [0.0_f64; 1];
    let mut cross_trigger_dirs = [0_i32; 1];
    let mut cross_went_beyond = [0_u8; 1];

    let mut current_values = [initial_state_value; 1];
    let mut target_values = [initial_state_value; 1];
    let mut start_times = [0.0_f64; 1];
    let mut start_values = [initial_state_value; 1];
    let mut delays = [transition_delay; 1];
    let mut rise_times = [transition_rise; 1];
    let mut fall_times = [transition_fall; 1];
    let mut active_flags = [0_u8; 1];
    let mut initialized_flags = [0_u8; 1];
    let mut input_targets = [initial_state_value; 1];
    let input_delays = [transition_delay; 1];
    let input_rises = [transition_rise; 1];
    let input_falls = [transition_fall; 1];
    let mut output_values = [initial_state_value; 1];

    let mut record_at = |time: f64, value: f64, count_ref: &mut usize| -> Result<(), i32> {
        if *count_ref >= time_values.len() {
            return Err(-710);
        }
        time_values[*count_ref] = time;
        out_values[*count_ref] = value;
        *count_ref += 1;
        Ok(())
    };

    let first_value = event_transition_core_trace_pulse_step(
        0.0,
        true,
        v_lo,
        v_hi,
        period,
        duty,
        pulse_rise,
        pulse_fall,
        pulse_delay,
        pulse_width,
        pulse_has_width,
        edge_threshold,
        event_state_value,
        default_transition,
        &mut state_value,
        &mut fired_events,
        &mut cross_prev_values,
        &mut cross_prev_times,
        &mut cross_pprev_values,
        &mut cross_pprev_times,
        &mut cross_initialized,
        &cross_directions,
        &mut cross_last_times,
        &mut cross_triggered,
        &mut cross_times,
        &mut cross_trigger_dirs,
        &mut cross_went_beyond,
        &mut current_values,
        &mut target_values,
        &mut start_times,
        &mut start_values,
        &mut delays,
        &mut rise_times,
        &mut fall_times,
        &mut active_flags,
        &mut initialized_flags,
        &mut input_targets,
        &input_delays,
        &input_rises,
        &input_falls,
        &mut output_values,
    )?;
    record_at(0.0, first_value, &mut count)?;
    let mut prev_t = 0.0_f64;
    let min_ramp = 0.15 * tstep;

    for &target_t in &planned {
        if target_t <= prev_t + 1.0e-18 {
            continue;
        }
        loop {
            let bp = next_transition_breakpoint_for_arrays(
                &start_times,
                &start_values,
                &target_values,
                &delays,
                &rise_times,
                &fall_times,
                &active_flags,
                prev_t,
                min_ramp,
            )?;
            let Some(bp_value) = bp else {
                break;
            };
            if !(prev_t + 1.0e-18 < bp_value && bp_value < target_t - 1.0e-18) {
                break;
            }
            let value = event_transition_core_trace_pulse_step(
                bp_value,
                false,
                v_lo,
                v_hi,
                period,
                duty,
                pulse_rise,
                pulse_fall,
                pulse_delay,
                pulse_width,
                pulse_has_width,
                edge_threshold,
                event_state_value,
                default_transition,
                &mut state_value,
                &mut fired_events,
                &mut cross_prev_values,
                &mut cross_prev_times,
                &mut cross_pprev_values,
                &mut cross_pprev_times,
                &mut cross_initialized,
                &cross_directions,
                &mut cross_last_times,
                &mut cross_triggered,
                &mut cross_times,
                &mut cross_trigger_dirs,
                &mut cross_went_beyond,
                &mut current_values,
                &mut target_values,
                &mut start_times,
                &mut start_values,
                &mut delays,
                &mut rise_times,
                &mut fall_times,
                &mut active_flags,
                &mut initialized_flags,
                &mut input_targets,
                &input_delays,
                &input_rises,
                &input_falls,
                &mut output_values,
            )?;
            record_at(bp_value, value, &mut count)?;
            transition_breakpoints += 1;
            prev_t = bp_value;
        }
        let value = event_transition_core_trace_pulse_step(
            target_t,
            false,
            v_lo,
            v_hi,
            period,
            duty,
            pulse_rise,
            pulse_fall,
            pulse_delay,
            pulse_width,
            pulse_has_width,
            edge_threshold,
            event_state_value,
            default_transition,
            &mut state_value,
            &mut fired_events,
            &mut cross_prev_values,
            &mut cross_prev_times,
            &mut cross_pprev_values,
            &mut cross_pprev_times,
            &mut cross_initialized,
            &cross_directions,
            &mut cross_last_times,
            &mut cross_triggered,
            &mut cross_times,
            &mut cross_trigger_dirs,
            &mut cross_went_beyond,
            &mut current_values,
            &mut target_values,
            &mut start_times,
            &mut start_values,
            &mut delays,
            &mut rise_times,
            &mut fall_times,
            &mut active_flags,
            &mut initialized_flags,
            &mut input_targets,
            &input_delays,
            &input_rises,
            &input_falls,
            &mut output_values,
        )?;
        record_at(target_t, value, &mut count)?;
        prev_t = target_t;
    }

    Ok((
        count,
        state_value,
        fired_events,
        transition_breakpoints,
        source_breakpoints,
    ))
}

pub(crate) fn reset_prbs7_bits(bits: &mut [u8; 7], seed: i64) {
    for idx in 0..7 {
        bits[idx] = (((seed >> idx) & 1) != 0) as u8;
    }
    if bits.iter().all(|bit| *bit == 0) {
        bits[6] = 1;
    }
}

pub(crate) fn prbs7_shift(bits: &mut [u8; 7]) {
    let old = *bits;
    let feedback = old[6] ^ old[5];
    bits[6] = old[5];
    bits[5] = old[4];
    bits[4] = old[3];
    bits[3] = old[2];
    bits[2] = old[1];
    bits[1] = old[0];
    bits[0] = feedback;
}

pub(crate) fn prbs7_targets(bits: &[u8; 7], vdd: f64, targets: &mut [f64; 8]) {
    targets[0] = if bits[6] != 0 { vdd } else { 0.0 };
    for idx in 0..7 {
        targets[idx + 1] = if bits[idx] != 0 { vdd } else { 0.0 };
    }
}

pub(crate) fn reset_lfsr_bits(
    bits: &mut [u8],
    seed: i64,
    zero_guard_index: i32,
) -> Result<(), i32> {
    if bits.is_empty() || bits.len() > 63 {
        return Err(-341);
    }
    for (idx, bit) in bits.iter_mut().enumerate() {
        *bit = (((seed >> idx) & 1) != 0) as u8;
    }
    if zero_guard_index >= 0 && bits.iter().all(|bit| *bit == 0) {
        let guard_idx = zero_guard_index as usize;
        if guard_idx >= bits.len() {
            return Err(-342);
        }
        bits[guard_idx] = 1;
    }
    Ok(())
}

pub(crate) fn shift_lfsr_bits(
    bits: &mut [u8],
    taps: &[usize],
    shift_sources: &[i32],
) -> Result<(), i32> {
    if bits.is_empty() || shift_sources.len() != bits.len() || taps.is_empty() {
        return Err(-343);
    }
    let old = bits.to_vec();
    let mut feedback = 0_u8;
    for &tap in taps {
        if tap >= old.len() {
            return Err(-344);
        }
        feedback ^= old[tap];
    }
    for (idx, &source) in shift_sources.iter().enumerate() {
        bits[idx] = if source < 0 {
            feedback
        } else {
            let source_idx = source as usize;
            if source_idx >= old.len() {
                return Err(-345);
            }
            old[source_idx]
        };
    }
    Ok(())
}

pub(crate) fn lfsr_transition_targets(
    bits: &[u8],
    output_bits: &[usize],
    vdd: f64,
    targets: &mut [f64],
) -> Result<(), i32> {
    if output_bits.len() != targets.len() {
        return Err(-346);
    }
    for (idx, &bit_idx) in output_bits.iter().enumerate() {
        if bit_idx >= bits.len() {
            return Err(-347);
        }
        targets[idx] = if bits[bit_idx] != 0 { vdd } else { 0.0 };
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn lfsr_transition_trace_for_times(
    times: &[f64],
    values: &mut [f64],
    signal_count: usize,
    clk_vlo: f64,
    clk_vhi: f64,
    clk_period: f64,
    clk_duty: f64,
    clk_rise: f64,
    clk_fall: f64,
    clk_delay: f64,
    clk_width: f64,
    clk_has_width: bool,
    rst_vlo: f64,
    rst_vhi: f64,
    rst_period: f64,
    rst_duty: f64,
    rst_rise: f64,
    rst_fall: f64,
    rst_delay: f64,
    rst_width: f64,
    rst_has_width: bool,
    en_voltage: f64,
    vdd: f64,
    vth: f64,
    trf: f64,
    td: f64,
    seed: i64,
    width: usize,
    taps: &[usize],
    shift_sources: &[i32],
    output_bits: &[usize],
    zero_guard_index: i32,
) -> Result<usize, i32> {
    if width == 0 || width > 63 || taps.is_empty() || shift_sources.len() != width {
        return Err(-348);
    }
    if output_bits.is_empty() {
        return Err(-349);
    }
    let expected_signal_count = 3usize.checked_add(output_bits.len()).ok_or(-350)?;
    if signal_count != expected_signal_count {
        return Err(-351);
    }
    let expected = times.len().checked_mul(signal_count).ok_or(-352)?;
    if values.len() != expected {
        return Err(-353);
    }

    let mut bits = vec![0_u8; width];
    reset_lfsr_bits(&mut bits, seed, zero_guard_index)?;

    let transition_count = output_bits.len();
    let mut targets = vec![0.0_f64; transition_count];
    lfsr_transition_targets(&bits, output_bits, vdd, &mut targets)?;

    let mut current = targets.clone();
    let mut target_values = targets.clone();
    let mut start_times = vec![0.0_f64; transition_count];
    let mut start_values = targets.clone();
    let mut delays = vec![0.0_f64; transition_count];
    let mut rises = vec![0.0_f64; transition_count];
    let mut falls = vec![0.0_f64; transition_count];
    let mut active = vec![0_u8; transition_count];
    let mut initialized = vec![0_u8; transition_count];
    let input_delays = vec![td; transition_count];
    let input_rises = vec![trf; transition_count];
    let input_falls = vec![trf; transition_count];
    let mut transition_outputs = vec![0.0_f64; transition_count];

    let mut prev_time = 0.0;
    let mut prev_clk = pulse_value(
        clk_vlo,
        clk_vhi,
        clk_period,
        clk_duty,
        clk_rise,
        clk_fall,
        clk_delay,
        clk_width,
        clk_has_width,
        0.0,
    );
    let mut event_count = 0_usize;

    for (row_idx, &time) in times.iter().enumerate() {
        let clk = pulse_value(
            clk_vlo,
            clk_vhi,
            clk_period,
            clk_duty,
            clk_rise,
            clk_fall,
            clk_delay,
            clk_width,
            clk_has_width,
            time,
        );
        let rst = pulse_value(
            rst_vlo,
            rst_vhi,
            rst_period,
            rst_duty,
            rst_rise,
            rst_fall,
            rst_delay,
            rst_width,
            rst_has_width,
            time,
        );

        if row_idx == 0 {
            transition_state_step_for_arrays(
                &mut current,
                &mut target_values,
                &mut start_times,
                &mut start_values,
                &mut delays,
                &mut rises,
                &mut falls,
                &mut active,
                &mut initialized,
                &targets,
                &input_delays,
                &input_rises,
                &input_falls,
                &mut transition_outputs,
                time,
                1.0e-12,
                true,
            )?;
        } else {
            let crossed = prev_clk < vth && clk >= vth;
            if crossed {
                let dv = clk - prev_clk;
                let frac = if dv.abs() > 1.0e-30 {
                    ((vth - prev_clk) / dv).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let cross_time = prev_time + frac * (time - prev_time);
                let rst_at_cross = pulse_value(
                    rst_vlo,
                    rst_vhi,
                    rst_period,
                    rst_duty,
                    rst_rise,
                    rst_fall,
                    rst_delay,
                    rst_width,
                    rst_has_width,
                    cross_time,
                );
                if rst_at_cross < vth {
                    reset_lfsr_bits(&mut bits, seed, zero_guard_index)?;
                } else if en_voltage > vth {
                    shift_lfsr_bits(&mut bits, taps, shift_sources)?;
                }
                lfsr_transition_targets(&bits, output_bits, vdd, &mut targets)?;
                transition_state_step_for_arrays(
                    &mut current,
                    &mut target_values,
                    &mut start_times,
                    &mut start_values,
                    &mut delays,
                    &mut rises,
                    &mut falls,
                    &mut active,
                    &mut initialized,
                    &targets,
                    &input_delays,
                    &input_rises,
                    &input_falls,
                    &mut transition_outputs,
                    cross_time,
                    1.0e-12,
                    false,
                )?;
                event_count += 1;
            }

            transition_state_step_for_arrays(
                &mut current,
                &mut target_values,
                &mut start_times,
                &mut start_values,
                &mut delays,
                &mut rises,
                &mut falls,
                &mut active,
                &mut initialized,
                &targets,
                &input_delays,
                &input_rises,
                &input_falls,
                &mut transition_outputs,
                time,
                1.0e-12,
                false,
            )?;
        }

        let base = row_idx * signal_count;
        values[base] = clk;
        values[base + 1] = rst;
        values[base + 2] = en_voltage;
        for (idx, output) in transition_outputs.iter().enumerate() {
            values[base + 3 + idx] = *output;
        }

        prev_time = time;
        prev_clk = clk;
    }

    Ok(event_count)
}

#[allow(clippy::too_many_arguments)]
pub fn gain_timer_reduction_trace_for_arrays(
    times: &[f64],
    sample_times: &[f64],
    values: &mut [f64],
    signal_count: usize,
    point_vdd: &[f64],
    point_vss: &[f64],
    point_vinp: &[f64],
    point_vinn: &[f64],
    point_voutp: &[f64],
    point_voutn: &[f64],
    sample_vdd: &[f64],
    sample_vss: &[f64],
    sample_vinp: &[f64],
    sample_vinn: &[f64],
    sample_voutp: &[f64],
    sample_voutn: &[f64],
    start_time: f64,
    gain_scale: f64,
    min_input_span: f64,
    tedge: f64,
) -> Result<usize, i32> {
    const SIGNAL_COUNT: usize = 8;
    if signal_count != SIGNAL_COUNT {
        return Err(-371);
    }
    if gain_scale.abs() <= 1.0e-30 {
        return Err(-372);
    }
    let point_count = times.len();
    let sample_count = sample_times.len();
    let expected_values = point_count.checked_mul(signal_count).ok_or(-373)?;
    if values.len() != expected_values {
        return Err(-374);
    }
    for slice in [
        point_vdd,
        point_vss,
        point_vinp,
        point_vinn,
        point_voutp,
        point_voutn,
    ] {
        if slice.len() != point_count {
            return Err(-375);
        }
    }
    for slice in [
        sample_vdd,
        sample_vss,
        sample_vinp,
        sample_vinn,
        sample_voutp,
        sample_voutn,
    ] {
        if slice.len() != sample_count {
            return Err(-376);
        }
    }

    let mut in_min = 1.0e9_f64;
    let mut in_max = -1.0e9_f64;
    let mut out_min = 1.0e9_f64;
    let mut out_max = -1.0e9_f64;
    let mut gain_q = 0.0_f64;
    let mut valid_q = 0_u8;
    let mut sample_idx = 0_usize;
    let mut sample_events = 0_usize;

    let mut current = [0.0_f64; 2];
    let mut target_values = [0.0_f64; 2];
    let mut start_times = [0.0_f64; 2];
    let mut start_values = [0.0_f64; 2];
    let mut delays = [0.0_f64; 2];
    let mut rises = [tedge; 2];
    let mut falls = [tedge; 2];
    let mut active = [0_u8; 2];
    let mut initialized = [0_u8; 2];
    let mut input_targets = [0.0_f64; 2];
    let input_delays = [0.0_f64; 2];
    let input_rises = [tedge; 2];
    let input_falls = [tedge; 2];
    let mut transition_outputs = [0.0_f64; 2];

    for (row_idx, &time) in times.iter().enumerate() {
        while sample_idx < sample_count && sample_times[sample_idx] <= time + 1.0e-18 {
            let sample_time = sample_times[sample_idx];
            let vdd_val = sample_vdd[sample_idx];
            let vss_val = sample_vss[sample_idx];
            if sample_time >= start_time - 1.0e-18 {
                let vin_diff = sample_vinp[sample_idx] - sample_vinn[sample_idx];
                let vout_diff = sample_voutp[sample_idx] - sample_voutn[sample_idx];
                in_min = in_min.min(vin_diff);
                in_max = in_max.max(vin_diff);
                out_min = out_min.min(vout_diff);
                out_max = out_max.max(vout_diff);
                let in_span = in_max - in_min;
                let out_span = out_max - out_min;
                if in_span > min_input_span {
                    gain_q = out_span / in_span;
                    valid_q = 1;
                }
            }
            input_targets[0] = (vdd_val - vss_val) * gain_q / gain_scale;
            input_targets[1] = if valid_q != 0 { vdd_val - vss_val } else { 0.0 };
            transition_state_step_for_arrays(
                &mut current,
                &mut target_values,
                &mut start_times,
                &mut start_values,
                &mut delays,
                &mut rises,
                &mut falls,
                &mut active,
                &mut initialized,
                &input_targets,
                &input_delays,
                &input_rises,
                &input_falls,
                &mut transition_outputs,
                sample_time,
                1.0e-12,
                false,
            )?;
            sample_idx += 1;
            sample_events += 1;
        }

        transition_outputs[0] = transition_evaluate_one(
            &mut current[0],
            target_values[0],
            start_times[0],
            start_values[0],
            delays[0],
            rises[0],
            falls[0],
            &mut active[0],
            time,
        );
        transition_outputs[1] = transition_evaluate_one(
            &mut current[1],
            target_values[1],
            start_times[1],
            start_values[1],
            delays[1],
            rises[1],
            falls[1],
            &mut active[1],
            time,
        );

        let base = row_idx * signal_count;
        values[base] = point_vdd[row_idx];
        values[base + 1] = point_vss[row_idx];
        values[base + 2] = point_vinp[row_idx];
        values[base + 3] = point_vinn[row_idx];
        values[base + 4] = point_voutp[row_idx];
        values[base + 5] = point_voutn[row_idx];
        values[base + 6] = point_vss[row_idx] + transition_outputs[0];
        values[base + 7] = point_vss[row_idx] + transition_outputs[1];
    }

    Ok(sample_events)
}

#[allow(clippy::too_many_arguments)]
pub fn gain_measurement_flow_trace_for_arrays(
    times: &[f64],
    values: &mut [f64],
    signal_count: usize,
    vin_event_times: &[f64],
    vin_event_vinp: &[f64],
    vin_event_vinn: &[f64],
    lfsr_event_times: &[f64],
    vcm: f64,
    vth: f64,
    dither_amp: f64,
    actual_gain: f64,
    vin_transition: f64,
    lfsr_transition: f64,
    vdd: f64,
    vss: f64,
    lfsr_seed: i64,
) -> Result<(usize, usize), i32> {
    const SIGNAL_COUNT: usize = 4;
    if signal_count != SIGNAL_COUNT {
        return Err(-521);
    }
    if vin_event_vinp.len() != vin_event_times.len()
        || vin_event_vinn.len() != vin_event_times.len()
    {
        return Err(-522);
    }
    let expected_values = times.len().checked_mul(signal_count).ok_or(-523)?;
    if values.len() != expected_values {
        return Err(-524);
    }
    if !actual_gain.is_finite()
        || !dither_amp.is_finite()
        || !vcm.is_finite()
        || !vth.is_finite()
        || !vdd.is_finite()
        || !vss.is_finite()
    {
        return Err(-525);
    }

    let vin_tr = if vin_transition > 0.0 {
        vin_transition
    } else {
        1.0e-12
    };
    let lfsr_tr = if lfsr_transition > 0.0 {
        lfsr_transition
    } else {
        1.0e-12
    };

    let mut vin_current = [vcm, vcm];
    let mut vin_target_values = [vcm, vcm];
    let mut vin_start_times = [0.0_f64; 2];
    let mut vin_start_values = [vcm, vcm];
    let mut vin_delays = [0.0_f64; 2];
    let mut vin_rises = [vin_tr; 2];
    let mut vin_falls = [vin_tr; 2];
    let mut vin_active = [0_u8; 2];

    let mut dpn_current = [vss];
    let mut dpn_target_values = [vss];
    let mut dpn_start_times = [0.0_f64; 1];
    let mut dpn_start_values = [vss];
    let mut dpn_delays = [0.0_f64; 1];
    let mut dpn_rises = [lfsr_tr; 1];
    let mut dpn_falls = [lfsr_tr; 1];
    let mut dpn_active = [0_u8; 1];

    let mut lfsr = [0_u8; 32];
    let seed_bits = lfsr_seed as u64;
    for idx in 0..32 {
        lfsr[idx] = ((seed_bits >> idx) & 1) as u8;
    }
    for idx in (0..32).step_by(5) {
        lfsr[idx] = 1;
    }
    let initial_dpn = if lfsr[31] != 0 { vdd } else { vss };
    dpn_current[0] = initial_dpn;
    dpn_target_values[0] = initial_dpn;
    dpn_start_values[0] = initial_dpn;

    let mut vin_idx = 0_usize;
    let mut lfsr_idx = 0_usize;
    let mut vin_events = 0_usize;
    let mut lfsr_events = 0_usize;

    for (row_idx, &time) in times.iter().enumerate() {
        loop {
            let vin_due =
                vin_idx < vin_event_times.len() && vin_event_times[vin_idx] <= time + 1.0e-18;
            let lfsr_due =
                lfsr_idx < lfsr_event_times.len() && lfsr_event_times[lfsr_idx] <= time + 1.0e-18;
            if !vin_due && !lfsr_due {
                break;
            }

            if vin_due
                && (!lfsr_due || vin_event_times[vin_idx] <= lfsr_event_times[lfsr_idx] + 1.0e-18)
            {
                let event_time = vin_event_times[vin_idx];
                transition_drive_index(
                    &mut vin_current,
                    &mut vin_target_values,
                    &mut vin_start_times,
                    &mut vin_start_values,
                    &mut vin_delays,
                    &mut vin_rises,
                    &mut vin_falls,
                    &mut vin_active,
                    0,
                    event_time,
                    vin_event_vinp[vin_idx],
                    0.0,
                    vin_tr,
                    vin_tr,
                )?;
                transition_drive_index(
                    &mut vin_current,
                    &mut vin_target_values,
                    &mut vin_start_times,
                    &mut vin_start_values,
                    &mut vin_delays,
                    &mut vin_rises,
                    &mut vin_falls,
                    &mut vin_active,
                    1,
                    event_time,
                    vin_event_vinn[vin_idx],
                    0.0,
                    vin_tr,
                    vin_tr,
                )?;
                vin_idx += 1;
                vin_events += 1;
                continue;
            }

            let event_time = lfsr_event_times[lfsr_idx];
            let previous = lfsr;
            let feedback = previous[31] ^ previous[21] ^ previous[1] ^ previous[0];
            lfsr[0] = feedback;
            lfsr[1..32].copy_from_slice(&previous[0..31]);
            let target = if lfsr[31] != 0 { vdd } else { vss };
            transition_drive_index(
                &mut dpn_current,
                &mut dpn_target_values,
                &mut dpn_start_times,
                &mut dpn_start_values,
                &mut dpn_delays,
                &mut dpn_rises,
                &mut dpn_falls,
                &mut dpn_active,
                0,
                event_time,
                target,
                0.0,
                lfsr_tr,
                lfsr_tr,
            )?;
            lfsr_idx += 1;
            lfsr_events += 1;
        }

        let vinp = transition_evaluate_one(
            &mut vin_current[0],
            vin_target_values[0],
            vin_start_times[0],
            vin_start_values[0],
            vin_delays[0],
            vin_rises[0],
            vin_falls[0],
            &mut vin_active[0],
            time,
        );
        let vinn = transition_evaluate_one(
            &mut vin_current[1],
            vin_target_values[1],
            vin_start_times[1],
            vin_start_values[1],
            vin_delays[1],
            vin_rises[1],
            vin_falls[1],
            &mut vin_active[1],
            time,
        );
        let dpn = transition_evaluate_one(
            &mut dpn_current[0],
            dpn_target_values[0],
            dpn_start_times[0],
            dpn_start_values[0],
            dpn_delays[0],
            dpn_rises[0],
            dpn_falls[0],
            &mut dpn_active[0],
            time,
        );
        let dither_diff = if dpn > vth { dither_amp } else { -dither_amp };
        let vout_diff = actual_gain * (vinp - vinn + dither_diff);
        let vamp_p = vcm + 0.5 * vout_diff;
        let vamp_n = vcm - 0.5 * vout_diff;

        let base = row_idx * signal_count;
        values[base] = vinp;
        values[base + 1] = vinn;
        values[base + 2] = vamp_p;
        values[base + 3] = vamp_n;
    }

    Ok((vin_events, lfsr_events))
}

#[inline]
pub(crate) fn threshold_crossed(prev: f64, cur: f64, direction: i32, eps: f64) -> bool {
    if direction > 0 {
        return prev < -eps && cur >= -eps;
    }
    if direction < 0 {
        return prev > eps && cur <= eps;
    }
    (prev < -eps && cur >= -eps) || (prev > eps && cur <= eps)
}

#[inline]
fn threshold_cross_time(
    prev_time: f64,
    time: f64,
    prev: f64,
    cur: f64,
    direction: i32,
    eps: f64,
) -> Option<f64> {
    if threshold_crossed(prev, cur, direction, eps) {
        Some(interpolate_cross_time(prev_time, prev, time, cur))
    } else {
        None
    }
}

#[inline]
fn interpolate_sample_value(prev_time: f64, time: f64, prev: f64, cur: f64, at: f64) -> f64 {
    let dt = time - prev_time;
    if dt.abs() <= 1.0e-30 {
        return cur;
    }
    let frac = ((at - prev_time) / dt).clamp(0.0, 1.0);
    prev + frac * (cur - prev)
}

pub(crate) fn sar_current_code(bits: &[u8], width: usize) -> u64 {
    let mut code = 0_u64;
    for idx in 0..width {
        if bits[idx] != 0 {
            code += 1_u64 << (width - 1 - idx);
        }
    }
    code
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn transition_drive_index(
    current: &mut [f64],
    target_values: &mut [f64],
    start_times: &mut [f64],
    start_values: &mut [f64],
    delays: &mut [f64],
    rises: &mut [f64],
    falls: &mut [f64],
    active: &mut [u8],
    idx: usize,
    time: f64,
    target: f64,
    delay: f64,
    rise: f64,
    fall: f64,
) -> Result<(), i32> {
    if idx >= current.len()
        || idx >= target_values.len()
        || idx >= start_times.len()
        || idx >= start_values.len()
        || idx >= delays.len()
        || idx >= rises.len()
        || idx >= falls.len()
        || idx >= active.len()
    {
        return Err(-461);
    }
    transition_evaluate_one(
        &mut current[idx],
        target_values[idx],
        start_times[idx],
        start_values[idx],
        delays[idx],
        rises[idx],
        falls[idx],
        &mut active[idx],
        time,
    );
    transition_set_target_one(
        &mut current[idx],
        &mut target_values[idx],
        &mut start_times[idx],
        &mut start_values[idx],
        &mut delays[idx],
        &mut rises[idx],
        &mut falls[idx],
        &mut active[idx],
        time,
        target,
        delay,
        rise,
        fall,
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn sar_drive_outputs(
    current: &mut [f64],
    target_values: &mut [f64],
    start_times: &mut [f64],
    start_values: &mut [f64],
    delays: &mut [f64],
    rises: &mut [f64],
    falls: &mut [f64],
    active: &mut [u8],
    dout_bits: &[u8],
    width: usize,
    total_code: f64,
    vdd: f64,
    default_tr: f64,
    bit_index_v: f64,
    trial_code_v: f64,
    trial_vdac_state: f64,
    cmp_decision_v: f64,
    conv_done_v: f64,
    vsampled: f64,
    time: f64,
) -> Result<(), i32> {
    let code = sar_current_code(dout_bits, width) as f64;
    transition_drive_index(
        current,
        target_values,
        start_times,
        start_values,
        delays,
        rises,
        falls,
        active,
        1,
        time,
        code / total_code * vdd,
        0.0,
        default_tr,
        default_tr,
    )?;
    for (idx, target) in [
        (2_usize, bit_index_v),
        (3_usize, trial_code_v),
        (4_usize, trial_vdac_state),
        (5_usize, cmp_decision_v),
        (6_usize, conv_done_v),
        (7_usize, vsampled),
    ] {
        transition_drive_index(
            current,
            target_values,
            start_times,
            start_values,
            delays,
            rises,
            falls,
            active,
            idx,
            time,
            target,
            0.0,
            default_tr,
            default_tr,
        )?;
    }
    for (idx, bit) in dout_bits.iter().enumerate().take(width) {
        transition_drive_index(
            current,
            target_values,
            start_times,
            start_values,
            delays,
            rises,
            falls,
            active,
            8 + idx,
            time,
            if *bit != 0 { vdd } else { 0.0 },
            0.0,
            default_tr,
            default_tr,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn sar_loop_trace_for_arrays(
    times: &[f64],
    values: &mut [f64],
    signal_count: usize,
    point_vin: &[f64],
    point_clk: &[f64],
    point_rst: &[f64],
    vdd: f64,
    vth: f64,
    sh_tr: f64,
    default_tr: f64,
    width: usize,
) -> Result<usize, i32> {
    const FIXED_SIGNAL_COUNT: usize = 11;
    if width == 0 {
        return Err(-451);
    }
    let total_code_u64 = 1_u64.checked_shl(width as u32).ok_or(-452)?;
    let total_code_u64 = total_code_u64.checked_sub(1).ok_or(-453)?;
    let total_code = total_code_u64 as f64;
    let expected_signal_count = FIXED_SIGNAL_COUNT.checked_add(width).ok_or(-454)?;
    if signal_count != expected_signal_count {
        return Err(-455);
    }
    let point_count = times.len();
    let expected_values = point_count.checked_mul(signal_count).ok_or(-456)?;
    if values.len() != expected_values {
        return Err(-457);
    }
    for slice in [point_vin, point_clk, point_rst] {
        if slice.len() != point_count {
            return Err(-458);
        }
    }
    let transition_count = 8_usize.checked_add(width).ok_or(-459)?;
    let mut current = vec![0.0_f64; transition_count];
    let mut target_values = vec![0.0_f64; transition_count];
    let mut start_times = vec![0.0_f64; transition_count];
    let mut start_values = vec![0.0_f64; transition_count];
    let mut delays = vec![0.0_f64; transition_count];
    let mut rises = vec![default_tr; transition_count];
    let mut falls = vec![default_tr; transition_count];
    let mut active = vec![0_u8; transition_count];
    let mut dout_bits = vec![0_u8; width];
    let mut trial_bits = vec![0_u8; width];
    let mut bit_idx = 0_usize;
    let mut busy = 0_u8;
    let mut vsampled = 0.0_f64;
    let mut trial_vdac_state = 0.0_f64;
    let mut bit_index_v = 0.0_f64;
    let mut trial_code_v = 0.0_f64;
    let mut cmp_decision_v = 0.0_f64;
    let mut conv_done_v = 0.0_f64;
    let mut prev_clk_expr = point_clk.first().copied().unwrap_or(0.0) - vth;
    let mut clock_events = 0_usize;

    sar_drive_outputs(
        &mut current,
        &mut target_values,
        &mut start_times,
        &mut start_values,
        &mut delays,
        &mut rises,
        &mut falls,
        &mut active,
        &dout_bits,
        width,
        total_code,
        vdd,
        default_tr,
        bit_index_v,
        trial_code_v,
        trial_vdac_state,
        cmp_decision_v,
        conv_done_v,
        vsampled,
        0.0,
    )?;

    for (row_idx, &time) in times.iter().enumerate() {
        let vin_val = point_vin[row_idx];
        let clk_val = point_clk[row_idx];
        let rst_val = point_rst[row_idx];
        let clk_expr = clk_val - vth;
        if threshold_crossed(prev_clk_expr, clk_expr, 1, 1.0e-12) {
            if rst_val > vth {
                transition_drive_index(
                    &mut current,
                    &mut target_values,
                    &mut start_times,
                    &mut start_values,
                    &mut delays,
                    &mut rises,
                    &mut falls,
                    &mut active,
                    0,
                    time,
                    vin_val,
                    0.0,
                    sh_tr,
                    sh_tr,
                )?;
            }
            if rst_val < vth {
                dout_bits.fill(0);
                trial_bits.fill(0);
                bit_idx = 0;
                busy = 0;
                trial_vdac_state = 0.0;
                bit_index_v = 0.0;
                trial_code_v = 0.0;
                cmp_decision_v = 0.0;
                conv_done_v = 0.0;
            } else if busy == 1 && bit_idx > 0 {
                let bit_pos = width - bit_idx;
                if cmp_decision_v > vth && bit_pos < width {
                    dout_bits[bit_pos] = 1;
                }
                bit_idx -= 1;
                if bit_idx > 0 {
                    trial_bits.clone_from_slice(&dout_bits);
                    let next_pos = width - bit_idx;
                    if next_pos < width {
                        trial_bits[next_pos] = 1;
                    }
                    let trial_code_int = sar_current_code(&trial_bits, width);
                    trial_vdac_state = trial_code_int as f64 / total_code * vdd;
                    cmp_decision_v = if vsampled >= trial_vdac_state {
                        vdd
                    } else {
                        0.0
                    };
                    bit_index_v = bit_idx as f64 / width as f64 * vdd;
                    trial_code_v = trial_code_int as f64 / total_code * vdd;
                    conv_done_v = 0.0;
                } else {
                    let final_code = sar_current_code(&dout_bits, width);
                    trial_vdac_state = final_code as f64 / total_code * vdd;
                    trial_code_v = trial_vdac_state;
                    bit_index_v = 0.0;
                    cmp_decision_v = if vsampled >= trial_vdac_state {
                        vdd
                    } else {
                        0.0
                    };
                    conv_done_v = vdd;
                    busy = 0;
                }
            }
            sar_drive_outputs(
                &mut current,
                &mut target_values,
                &mut start_times,
                &mut start_values,
                &mut delays,
                &mut rises,
                &mut falls,
                &mut active,
                &dout_bits,
                width,
                total_code,
                vdd,
                default_tr,
                bit_index_v,
                trial_code_v,
                trial_vdac_state,
                cmp_decision_v,
                conv_done_v,
                vsampled,
                time,
            )?;
            clock_events += 1;
        } else if threshold_crossed(prev_clk_expr, clk_expr, -1, 1.0e-12) {
            if rst_val >= vth && busy == 0 {
                vsampled = transition_evaluate_one(
                    &mut current[0],
                    target_values[0],
                    start_times[0],
                    start_values[0],
                    delays[0],
                    rises[0],
                    falls[0],
                    &mut active[0],
                    time,
                );
                dout_bits.fill(0);
                trial_bits.fill(0);
                trial_bits[0] = 1;
                bit_idx = width;
                busy = 1;
                conv_done_v = 0.0;
                let trial_code_int = sar_current_code(&trial_bits, width);
                trial_vdac_state = trial_code_int as f64 / total_code * vdd;
                cmp_decision_v = if vsampled >= trial_vdac_state {
                    vdd
                } else {
                    0.0
                };
                bit_index_v = bit_idx as f64 / width as f64 * vdd;
                trial_code_v = trial_code_int as f64 / total_code * vdd;
                sar_drive_outputs(
                    &mut current,
                    &mut target_values,
                    &mut start_times,
                    &mut start_values,
                    &mut delays,
                    &mut rises,
                    &mut falls,
                    &mut active,
                    &dout_bits,
                    width,
                    total_code,
                    vdd,
                    default_tr,
                    bit_index_v,
                    trial_code_v,
                    trial_vdac_state,
                    cmp_decision_v,
                    conv_done_v,
                    vsampled,
                    time,
                )?;
            }
            clock_events += 1;
        }

        let vin_sh_val = transition_evaluate_one(
            &mut current[0],
            target_values[0],
            start_times[0],
            start_values[0],
            delays[0],
            rises[0],
            falls[0],
            &mut active[0],
            time,
        );
        let vout_val = transition_evaluate_one(
            &mut current[1],
            target_values[1],
            start_times[1],
            start_values[1],
            delays[1],
            rises[1],
            falls[1],
            &mut active[1],
            time,
        );
        let base = row_idx * signal_count;
        values[base] = vin_val;
        values[base + 1] = vin_sh_val;
        values[base + 2] = clk_val;
        values[base + 3] = rst_val;
        values[base + 4] = vout_val;
        for idx in 2..8 {
            values[base + 3 + idx] = transition_evaluate_one(
                &mut current[idx],
                target_values[idx],
                start_times[idx],
                start_values[idx],
                delays[idx],
                rises[idx],
                falls[idx],
                &mut active[idx],
                time,
            );
        }
        for idx in 0..width {
            values[base + FIXED_SIGNAL_COUNT + idx] = transition_evaluate_one(
                &mut current[8 + idx],
                target_values[8 + idx],
                start_times[8 + idx],
                start_values[8 + idx],
                delays[8 + idx],
                rises[8 + idx],
                falls[8 + idx],
                &mut active[8 + idx],
                time,
            );
        }
        prev_clk_expr = clk_expr;
    }

    Ok(clock_events)
}

#[allow(clippy::too_many_arguments)]
pub fn cmp_delay_trace_for_arrays(
    times: &[f64],
    values: &mut [f64],
    signal_count: usize,
    point_clk: &[f64],
    point_vinn: &[f64],
    point_vinp: &[f64],
    point_vdd: &[f64],
    voffset: f64,
    tau: f64,
    td0: f64,
    td_min: f64,
    td_max: f64,
    tedge: f64,
    edge_vth: f64,
) -> Result<usize, i32> {
    const SIGNAL_COUNT: usize = 6;
    if signal_count != SIGNAL_COUNT {
        return Err(-421);
    }
    let point_count = times.len();
    let expected_values = point_count.checked_mul(signal_count).ok_or(-422)?;
    if values.len() != expected_values {
        return Err(-423);
    }
    for slice in [point_clk, point_vinn, point_vinp, point_vdd] {
        if slice.len() != point_count {
            return Err(-424);
        }
    }

    let mut current = [0.0_f64; 3];
    let mut target_values = [0.0_f64; 3];
    let mut start_times = [0.0_f64; 3];
    let mut start_values = [0.0_f64; 3];
    let mut delays = [0.0_f64; 3];
    let mut rises = [tedge, tedge, 10.0e-12];
    let mut falls = [tedge, tedge, 10.0e-12];
    let mut active = [0_u8; 3];

    let mut prev_clk_expr = point_clk.first().copied().unwrap_or(0.0) - edge_vth;
    let mut prev_outp_expr = -edge_vth;
    let mut t_start = 0.0_f64;
    let mut armed = false;
    let mut clock_events = 0_usize;

    for (row_idx, &time) in times.iter().enumerate() {
        let prev_time = if row_idx > 0 {
            times[row_idx - 1]
        } else {
            time
        };
        let clk_val = point_clk[row_idx];
        let vinn_val = point_vinn[row_idx];
        let vinp_val = point_vinp[row_idx];
        let vdd_val = point_vdd[row_idx];
        let clk_expr = clk_val - edge_vth;

        if let Some(clk_cross_time) =
            threshold_cross_time(prev_time, time, prev_clk_expr, clk_expr, 1, 1.0e-12)
        {
            t_start = clk_cross_time;
            armed = true;
            let prev_vinp = if row_idx > 0 {
                point_vinp[row_idx - 1]
            } else {
                vinp_val
            };
            let prev_vinn = if row_idx > 0 {
                point_vinn[row_idx - 1]
            } else {
                vinn_val
            };
            let prev_vdd = if row_idx > 0 {
                point_vdd[row_idx - 1]
            } else {
                vdd_val
            };
            let vinp_at_cross =
                interpolate_sample_value(prev_time, time, prev_vinp, vinp_val, clk_cross_time);
            let vinn_at_cross =
                interpolate_sample_value(prev_time, time, prev_vinn, vinn_val, clk_cross_time);
            let vdd_at_cross =
                interpolate_sample_value(prev_time, time, prev_vdd, vdd_val, clk_cross_time);
            let vdiff = vinp_at_cross - vinn_at_cross - voffset;
            let vdiff_eff = vdiff.abs().max(1.0e-9);
            let mut td = td0 + tau * (vdd_at_cross / vdiff_eff).ln();
            td = td.min(td_max).max(td_min);
            for idx in 0..2 {
                transition_evaluate_one(
                    &mut current[idx],
                    target_values[idx],
                    start_times[idx],
                    start_values[idx],
                    delays[idx],
                    rises[idx],
                    falls[idx],
                    &mut active[idx],
                    clk_cross_time,
                );
            }
            let (outp_target, outn_target) = if vdiff > 0.0 {
                (vdd_at_cross, 0.0)
            } else {
                (0.0, vdd_at_cross)
            };
            transition_set_target_one(
                &mut current[0],
                &mut target_values[0],
                &mut start_times[0],
                &mut start_values[0],
                &mut delays[0],
                &mut rises[0],
                &mut falls[0],
                &mut active[0],
                clk_cross_time,
                outp_target,
                td,
                tedge,
                tedge,
            );
            transition_set_target_one(
                &mut current[1],
                &mut target_values[1],
                &mut start_times[1],
                &mut start_values[1],
                &mut delays[1],
                &mut rises[1],
                &mut falls[1],
                &mut active[1],
                clk_cross_time,
                outn_target,
                td,
                tedge,
                tedge,
            );
            clock_events += 1;
        } else if let Some(clk_cross_time) =
            threshold_cross_time(prev_time, time, prev_clk_expr, clk_expr, -1, 1.0e-12)
        {
            for idx in 0..2 {
                transition_evaluate_one(
                    &mut current[idx],
                    target_values[idx],
                    start_times[idx],
                    start_values[idx],
                    delays[idx],
                    rises[idx],
                    falls[idx],
                    &mut active[idx],
                    clk_cross_time,
                );
                transition_set_target_one(
                    &mut current[idx],
                    &mut target_values[idx],
                    &mut start_times[idx],
                    &mut start_values[idx],
                    &mut delays[idx],
                    &mut rises[idx],
                    &mut falls[idx],
                    &mut active[idx],
                    clk_cross_time,
                    0.0,
                    0.0,
                    tedge,
                    tedge,
                );
            }
            clock_events += 1;
        }

        let outp_val = transition_evaluate_one(
            &mut current[0],
            target_values[0],
            start_times[0],
            start_values[0],
            delays[0],
            rises[0],
            falls[0],
            &mut active[0],
            time,
        );
        let outn_val = transition_evaluate_one(
            &mut current[1],
            target_values[1],
            start_times[1],
            start_values[1],
            delays[1],
            rises[1],
            falls[1],
            &mut active[1],
            time,
        );
        let outp_expr = outp_val - edge_vth;
        if armed {
            if let Some(outp_cross_time) =
                threshold_cross_time(prev_time, time, prev_outp_expr, outp_expr, 1, 1.0e-12)
            {
                let delay_ps = (outp_cross_time - t_start) * 1.0e12;
                transition_evaluate_one(
                    &mut current[2],
                    target_values[2],
                    start_times[2],
                    start_values[2],
                    delays[2],
                    rises[2],
                    falls[2],
                    &mut active[2],
                    outp_cross_time,
                );
                transition_set_target_one(
                    &mut current[2],
                    &mut target_values[2],
                    &mut start_times[2],
                    &mut start_values[2],
                    &mut delays[2],
                    &mut rises[2],
                    &mut falls[2],
                    &mut active[2],
                    outp_cross_time,
                    delay_ps,
                    0.0,
                    10.0e-12,
                    10.0e-12,
                );
                armed = false;
            }
        }
        let delay_val = transition_evaluate_one(
            &mut current[2],
            target_values[2],
            start_times[2],
            start_values[2],
            delays[2],
            rises[2],
            falls[2],
            &mut active[2],
            time,
        );

        let base = row_idx * signal_count;
        values[base] = clk_val;
        values[base + 1] = vinn_val;
        values[base + 2] = vinp_val;
        values[base + 3] = outn_val;
        values[base + 4] = outp_val;
        values[base + 5] = delay_val;
        prev_clk_expr = clk_expr;
        prev_outp_expr = outp_expr;
    }

    Ok(clock_events)
}

#[allow(clippy::too_many_arguments)]
pub fn cppll_reacquire_trace_for_arrays(
    times: &[f64],
    values: &mut [f64],
    signal_count: usize,
    ref_event_times: &[f64],
    ref_event_values: &[f64],
    dco_event_times: &[f64],
    dco_event_values: &[f64],
    fb_event_times: &[f64],
    fb_event_values: &[f64],
    lock_event_times: &[f64],
    lock_event_values: &[f64],
    vctrl_event_times: &[f64],
    vctrl_event_values: &[f64],
    vh: f64,
    vl: f64,
    ref_tedge: f64,
    pll_tedge: f64,
) -> Result<(), i32> {
    const SIGNAL_COUNT: usize = 7;
    if signal_count != SIGNAL_COUNT {
        return Err(-491);
    }
    if ref_tedge <= 0.0 || pll_tedge <= 0.0 {
        return Err(-492);
    }
    let point_count = times.len();
    let expected_values = point_count.checked_mul(signal_count).ok_or(-493)?;
    if values.len() != expected_values {
        return Err(-494);
    }
    for (times_slice, values_slice) in [
        (ref_event_times, ref_event_values),
        (dco_event_times, dco_event_values),
        (fb_event_times, fb_event_values),
        (lock_event_times, lock_event_values),
        (vctrl_event_times, vctrl_event_values),
    ] {
        if times_slice.len() != values_slice.len() {
            return Err(-495);
        }
    }

    let mut current = [vl; 4];
    let mut target_values = [vl; 4];
    let mut start_times = [0.0_f64; 4];
    let mut start_values = [vl; 4];
    let mut delays = [0.0_f64; 4];
    let mut rises = [ref_tedge, pll_tedge, pll_tedge, pll_tedge];
    let mut falls = [ref_tedge, pll_tedge, pll_tedge, pll_tedge];
    let mut active = [0_u8; 4];

    let mut ref_idx = 0_usize;
    let mut dco_idx = 0_usize;
    let mut fb_idx = 0_usize;
    let mut lock_idx = 0_usize;
    let mut vctrl_idx = 0_usize;
    let mut vctrl_current = vctrl_event_values
        .first()
        .copied()
        .unwrap_or(0.5 * (vh + vl));

    for (row_idx, &time) in times.iter().enumerate() {
        while ref_idx < ref_event_times.len() && ref_event_times[ref_idx] <= time + 1.0e-18 {
            transition_drive_index(
                &mut current,
                &mut target_values,
                &mut start_times,
                &mut start_values,
                &mut delays,
                &mut rises,
                &mut falls,
                &mut active,
                0,
                ref_event_times[ref_idx],
                ref_event_values[ref_idx],
                0.0,
                ref_tedge,
                ref_tedge,
            )?;
            ref_idx += 1;
        }
        while dco_idx < dco_event_times.len() && dco_event_times[dco_idx] <= time + 1.0e-18 {
            transition_drive_index(
                &mut current,
                &mut target_values,
                &mut start_times,
                &mut start_values,
                &mut delays,
                &mut rises,
                &mut falls,
                &mut active,
                1,
                dco_event_times[dco_idx],
                dco_event_values[dco_idx],
                0.0,
                pll_tedge,
                pll_tedge,
            )?;
            dco_idx += 1;
        }
        while fb_idx < fb_event_times.len() && fb_event_times[fb_idx] <= time + 1.0e-18 {
            transition_drive_index(
                &mut current,
                &mut target_values,
                &mut start_times,
                &mut start_values,
                &mut delays,
                &mut rises,
                &mut falls,
                &mut active,
                2,
                fb_event_times[fb_idx],
                fb_event_values[fb_idx],
                0.0,
                pll_tedge,
                pll_tedge,
            )?;
            fb_idx += 1;
        }
        while lock_idx < lock_event_times.len() && lock_event_times[lock_idx] <= time + 1.0e-18 {
            transition_drive_index(
                &mut current,
                &mut target_values,
                &mut start_times,
                &mut start_values,
                &mut delays,
                &mut rises,
                &mut falls,
                &mut active,
                3,
                lock_event_times[lock_idx],
                lock_event_values[lock_idx],
                0.0,
                pll_tedge,
                pll_tedge,
            )?;
            lock_idx += 1;
        }
        while vctrl_idx < vctrl_event_times.len() && vctrl_event_times[vctrl_idx] <= time + 1.0e-18
        {
            vctrl_current = vctrl_event_values[vctrl_idx];
            vctrl_idx += 1;
        }

        let ref_val = transition_evaluate_one(
            &mut current[0],
            target_values[0],
            start_times[0],
            start_values[0],
            delays[0],
            rises[0],
            falls[0],
            &mut active[0],
            time,
        );
        let dco_val = transition_evaluate_one(
            &mut current[1],
            target_values[1],
            start_times[1],
            start_values[1],
            delays[1],
            rises[1],
            falls[1],
            &mut active[1],
            time,
        );
        let fb_val = transition_evaluate_one(
            &mut current[2],
            target_values[2],
            start_times[2],
            start_values[2],
            delays[2],
            rises[2],
            falls[2],
            &mut active[2],
            time,
        );
        let lock_val = transition_evaluate_one(
            &mut current[3],
            target_values[3],
            start_times[3],
            start_values[3],
            delays[3],
            rises[3],
            falls[3],
            &mut active[3],
            time,
        );
        let base = row_idx * signal_count;
        values[base] = vh;
        values[base + 1] = vl;
        values[base + 2] = ref_val;
        values[base + 3] = dco_val;
        values[base + 4] = fb_val;
        values[base + 5] = vctrl_current;
        values[base + 6] = lock_val;
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn prbs7_trace_for_times(
    times: &[f64],
    values: &mut [f64],
    signal_count: usize,
    clk_vlo: f64,
    clk_vhi: f64,
    clk_period: f64,
    clk_duty: f64,
    clk_rise: f64,
    clk_fall: f64,
    clk_delay: f64,
    clk_width: f64,
    clk_has_width: bool,
    rst_vlo: f64,
    rst_vhi: f64,
    rst_period: f64,
    rst_duty: f64,
    rst_rise: f64,
    rst_fall: f64,
    rst_delay: f64,
    rst_width: f64,
    rst_has_width: bool,
    en_voltage: f64,
    vdd: f64,
    vth: f64,
    trf: f64,
    td: f64,
    seed: i64,
) -> Result<usize, i32> {
    const PRBS7_SIGNAL_COUNT: usize = 11;
    const TRANSITION_COUNT: usize = 8;
    if signal_count != PRBS7_SIGNAL_COUNT {
        return Err(-301);
    }
    let expected = times.len().checked_mul(signal_count).ok_or(-302)?;
    if values.len() != expected {
        return Err(-303);
    }

    let mut bits = [0_u8; 7];
    reset_prbs7_bits(&mut bits, seed);

    let mut targets = [0.0_f64; TRANSITION_COUNT];
    prbs7_targets(&bits, vdd, &mut targets);

    let mut current = targets;
    let mut target_values = targets;
    let mut start_times = [0.0_f64; TRANSITION_COUNT];
    let mut start_values = targets;
    let mut delays = [0.0_f64; TRANSITION_COUNT];
    let mut rises = [0.0_f64; TRANSITION_COUNT];
    let mut falls = [0.0_f64; TRANSITION_COUNT];
    let mut active = [0_u8; TRANSITION_COUNT];
    let mut initialized = [0_u8; TRANSITION_COUNT];
    let input_delays = [td; TRANSITION_COUNT];
    let input_rises = [trf; TRANSITION_COUNT];
    let input_falls = [trf; TRANSITION_COUNT];
    let mut transition_outputs = [0.0_f64; TRANSITION_COUNT];

    let mut prev_time = 0.0;
    let mut prev_clk = pulse_value(
        clk_vlo,
        clk_vhi,
        clk_period,
        clk_duty,
        clk_rise,
        clk_fall,
        clk_delay,
        clk_width,
        clk_has_width,
        0.0,
    );
    let mut event_count = 0_usize;

    for (row_idx, &time) in times.iter().enumerate() {
        let clk = pulse_value(
            clk_vlo,
            clk_vhi,
            clk_period,
            clk_duty,
            clk_rise,
            clk_fall,
            clk_delay,
            clk_width,
            clk_has_width,
            time,
        );
        let rst = pulse_value(
            rst_vlo,
            rst_vhi,
            rst_period,
            rst_duty,
            rst_rise,
            rst_fall,
            rst_delay,
            rst_width,
            rst_has_width,
            time,
        );

        if row_idx == 0 {
            transition_state_step_for_arrays(
                &mut current,
                &mut target_values,
                &mut start_times,
                &mut start_values,
                &mut delays,
                &mut rises,
                &mut falls,
                &mut active,
                &mut initialized,
                &targets,
                &input_delays,
                &input_rises,
                &input_falls,
                &mut transition_outputs,
                time,
                1.0e-12,
                true,
            )?;
        } else {
            let crossed = prev_clk < vth && clk >= vth;
            if crossed {
                let dv = clk - prev_clk;
                let frac = if dv.abs() > 1.0e-30 {
                    ((vth - prev_clk) / dv).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let cross_time = prev_time + frac * (time - prev_time);
                let rst_at_cross = pulse_value(
                    rst_vlo,
                    rst_vhi,
                    rst_period,
                    rst_duty,
                    rst_rise,
                    rst_fall,
                    rst_delay,
                    rst_width,
                    rst_has_width,
                    cross_time,
                );
                if rst_at_cross < vth {
                    reset_prbs7_bits(&mut bits, seed);
                } else if en_voltage > vth {
                    prbs7_shift(&mut bits);
                }
                prbs7_targets(&bits, vdd, &mut targets);
                transition_state_step_for_arrays(
                    &mut current,
                    &mut target_values,
                    &mut start_times,
                    &mut start_values,
                    &mut delays,
                    &mut rises,
                    &mut falls,
                    &mut active,
                    &mut initialized,
                    &targets,
                    &input_delays,
                    &input_rises,
                    &input_falls,
                    &mut transition_outputs,
                    cross_time,
                    1.0e-12,
                    false,
                )?;
                event_count += 1;
            }

            transition_state_step_for_arrays(
                &mut current,
                &mut target_values,
                &mut start_times,
                &mut start_values,
                &mut delays,
                &mut rises,
                &mut falls,
                &mut active,
                &mut initialized,
                &targets,
                &input_delays,
                &input_rises,
                &input_falls,
                &mut transition_outputs,
                time,
                1.0e-12,
                false,
            )?;
        }

        let base = row_idx * signal_count;
        values[base] = clk;
        values[base + 1] = rst;
        values[base + 2] = en_voltage;
        values[base + 3] = transition_outputs[0];
        for idx in 0..7 {
            values[base + 4 + idx] = transition_outputs[idx + 1];
        }

        prev_time = time;
        prev_clk = clk;
    }

    Ok(event_count)
}
