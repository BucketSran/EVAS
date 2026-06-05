#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EvasRustStaticAffineOp {
    pub read_node_id: usize,
    pub write_node_id: usize,
    pub gain: f64,
    pub bias: f64,
}

const SOURCE_NODE: u8 = 0;
const SOURCE_STATE: u8 = 1;
const TARGET_NODE: u8 = 0;
const TARGET_STATE: u8 = 1;
const CONDITION_NONE: usize = usize::MAX;
const COND_GT: u8 = 1;
const COND_LT: u8 = 2;
const COND_GE: u8 = 3;
const COND_LE: u8 = 4;
const COND_EQ: u8 = 5;
const COND_NE: u8 = 6;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EvasRustLinearTerm {
    pub source_kind: u8,
    pub source_id: usize,
    pub gain: f64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EvasRustLinearCondition {
    pub op_kind: u8,
    pub left_term_start: usize,
    pub left_term_count: usize,
    pub left_bias: f64,
    pub right_term_start: usize,
    pub right_term_count: usize,
    pub right_bias: f64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EvasRustLinearOp {
    pub target_kind: u8,
    pub target_integer: u8,
    pub target_id: usize,
    pub term_start: usize,
    pub term_count: usize,
    pub bias: f64,
    pub condition_id: usize,
    pub false_term_start: usize,
    pub false_term_count: usize,
    pub false_bias: f64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EvasRustTransitionTargetOp {
    pub target_id: usize,
    pub term_start: usize,
    pub term_count: usize,
    pub bias: f64,
    pub condition_id: usize,
    pub false_term_start: usize,
    pub false_term_count: usize,
    pub false_bias: f64,
    pub delay: f64,
    pub rise: f64,
    pub fall: f64,
}

const BODY_EXPR_CONST: u8 = 0;
const BODY_EXPR_READ_NODE: u8 = 1;
const BODY_EXPR_READ_STATE: u8 = 2;
const BODY_EXPR_READ_PARAM: u8 = 3;
const BODY_EXPR_NEG: u8 = 10;
const BODY_EXPR_NOT: u8 = 11;
const BODY_EXPR_ADD: u8 = 20;
const BODY_EXPR_SUB: u8 = 21;
const BODY_EXPR_MUL: u8 = 22;
const BODY_EXPR_DIV: u8 = 23;
const BODY_EXPR_MOD: u8 = 24;
const BODY_EXPR_GT: u8 = 30;
const BODY_EXPR_LT: u8 = 31;
const BODY_EXPR_GE: u8 = 32;
const BODY_EXPR_LE: u8 = 33;
const BODY_EXPR_EQ: u8 = 34;
const BODY_EXPR_NE: u8 = 35;
const BODY_EXPR_LAND: u8 = 36;
const BODY_EXPR_LOR: u8 = 37;
const BODY_EXPR_BITAND: u8 = 38;
const BODY_EXPR_BITOR: u8 = 39;
const BODY_EXPR_BITXOR: u8 = 40;
const BODY_EXPR_SELECT: u8 = 50;
const BODY_EXPR_ABS: u8 = 60;
const BODY_EXPR_SQRT: u8 = 61;
const BODY_EXPR_EXP: u8 = 62;
const BODY_EXPR_LN: u8 = 63;
const BODY_EXPR_LOG10: u8 = 64;
const BODY_EXPR_SIN: u8 = 65;
const BODY_EXPR_COS: u8 = 66;
const BODY_EXPR_FLOOR: u8 = 67;
const BODY_EXPR_CEIL: u8 = 68;
const BODY_EXPR_MIN: u8 = 69;
const BODY_EXPR_MAX: u8 = 70;
const BODY_EXPR_POW: u8 = 71;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EvasRustBodyExprOp {
    pub op_kind: u8,
    pub index: usize,
    pub value: f64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EvasRustBodyStmtOp {
    pub target_kind: u8,
    pub target_integer: u8,
    pub target_id: usize,
    pub expr_start: usize,
    pub expr_count: usize,
}

pub fn evaluate_static_affine_ops(
    ops: &[EvasRustStaticAffineOp],
    values: &mut [f64],
) -> Result<(), i32> {
    for op in ops {
        if op.read_node_id >= values.len() || op.write_node_id >= values.len() {
            return Err(-3);
        }
        values[op.write_node_id] = op.bias + op.gain * values[op.read_node_id];
    }
    Ok(())
}

pub fn evaluate_static_linear_ops(
    ops: &[EvasRustLinearOp],
    terms: &[EvasRustLinearTerm],
    conditions: &[EvasRustLinearCondition],
    node_values: &mut [f64],
    state_values: &mut [f64],
) -> Result<(), i32> {
    for op in ops {
        let term_end = op.term_start.checked_add(op.term_count).ok_or(-6)?;
        if term_end > terms.len() {
            return Err(-7);
        }

        let mut value = evaluate_linear_value(
            op.bias,
            &terms[op.term_start..term_end],
            node_values,
            state_values,
        )?;
        if op.condition_id != CONDITION_NONE {
            if op.condition_id >= conditions.len() {
                return Err(-14);
            }
            let condition = &conditions[op.condition_id];
            if !evaluate_condition(condition, terms, node_values, state_values)? {
                let false_term_end = op
                    .false_term_start
                    .checked_add(op.false_term_count)
                    .ok_or(-15)?;
                if false_term_end > terms.len() {
                    return Err(-16);
                }
                value = evaluate_linear_value(
                    op.false_bias,
                    &terms[op.false_term_start..false_term_end],
                    node_values,
                    state_values,
                )?;
            }
        }

        match op.target_kind {
            TARGET_NODE => {
                if op.target_id >= node_values.len() {
                    return Err(-11);
                }
                node_values[op.target_id] = value;
            }
            TARGET_STATE => {
                if op.target_id >= state_values.len() {
                    return Err(-12);
                }
                state_values[op.target_id] = if op.target_integer != 0 {
                    to_veriloga_integer(value) as f64
                } else {
                    value
                };
            }
            _ => return Err(-13),
        }
    }
    Ok(())
}

pub fn evaluate_body_ir_ops(
    stmt_ops: &[EvasRustBodyStmtOp],
    expr_ops: &[EvasRustBodyExprOp],
    node_values: &mut [f64],
    state_values: &mut [f64],
    param_values: &[f64],
) -> Result<(), i32> {
    let mut stack: Vec<f64> = Vec::with_capacity(32);
    for stmt in stmt_ops {
        let expr_end = stmt.expr_start.checked_add(stmt.expr_count).ok_or(-2206)?;
        if expr_end > expr_ops.len() {
            return Err(-2207);
        }
        stack.clear();
        evaluate_body_expr_segment(
            &expr_ops[stmt.expr_start..expr_end],
            node_values,
            state_values,
            param_values,
            &mut stack,
        )?;
        let mut value = stack.pop().ok_or(-2208)?;
        if !stack.is_empty() {
            return Err(-2209);
        }
        match stmt.target_kind {
            TARGET_NODE => {
                if stmt.target_id >= node_values.len() {
                    return Err(-2210);
                }
                node_values[stmt.target_id] = value;
            }
            TARGET_STATE => {
                if stmt.target_id >= state_values.len() {
                    return Err(-2211);
                }
                if stmt.target_integer != 0 {
                    value = to_veriloga_integer(value) as f64;
                }
                state_values[stmt.target_id] = value;
            }
            _ => return Err(-2212),
        }
    }
    Ok(())
}

pub fn evaluate_body_expr_ops(
    expr_ops: &[EvasRustBodyExprOp],
    node_values: &[f64],
    state_values: &[f64],
    param_values: &[f64],
) -> Result<f64, i32> {
    let mut stack: Vec<f64> = Vec::with_capacity(32);
    evaluate_body_expr_segment(
        expr_ops,
        node_values,
        state_values,
        param_values,
        &mut stack,
    )?;
    let value = stack.pop().ok_or(-2260)?;
    if !stack.is_empty() {
        return Err(-2261);
    }
    Ok(value)
}

pub fn evaluate_body_expr_segments(
    expr_ops: &[EvasRustBodyExprOp],
    expr_starts: &[usize],
    expr_counts: &[usize],
    node_values: &[f64],
    state_values: &[f64],
    param_values: &[f64],
    output_values: &mut [f64],
) -> Result<(), i32> {
    if expr_starts.len() != expr_counts.len() || expr_starts.len() != output_values.len() {
        return Err(-2270);
    }

    let mut stack: Vec<f64> = Vec::with_capacity(32);
    for idx in 0..expr_starts.len() {
        let expr_end = expr_starts[idx]
            .checked_add(expr_counts[idx])
            .ok_or(-2271)?;
        if expr_end > expr_ops.len() {
            return Err(-2272);
        }
        stack.clear();
        evaluate_body_expr_segment(
            &expr_ops[expr_starts[idx]..expr_end],
            node_values,
            state_values,
            param_values,
            &mut stack,
        )?;
        output_values[idx] = stack.pop().ok_or(-2273)?;
        if !stack.is_empty() {
            return Err(-2274);
        }
    }
    Ok(())
}

pub fn evaluate_transition_target_ops(
    ops: &[EvasRustTransitionTargetOp],
    terms: &[EvasRustLinearTerm],
    conditions: &[EvasRustLinearCondition],
    node_values: &[f64],
    state_values: &[f64],
    target_values: &mut [f64],
    delay_values: &mut [f64],
    rise_values: &mut [f64],
    fall_values: &mut [f64],
) -> Result<(), i32> {
    for op in ops {
        if op.target_id >= target_values.len()
            || op.target_id >= delay_values.len()
            || op.target_id >= rise_values.len()
            || op.target_id >= fall_values.len()
        {
            return Err(-61);
        }
        let term_end = op.term_start.checked_add(op.term_count).ok_or(-62)?;
        if term_end > terms.len() {
            return Err(-63);
        }

        let mut value = evaluate_linear_value(
            op.bias,
            &terms[op.term_start..term_end],
            node_values,
            state_values,
        )?;
        if op.condition_id != CONDITION_NONE {
            if op.condition_id >= conditions.len() {
                return Err(-64);
            }
            let condition = &conditions[op.condition_id];
            if !evaluate_condition(condition, terms, node_values, state_values)? {
                let false_term_end = op
                    .false_term_start
                    .checked_add(op.false_term_count)
                    .ok_or(-65)?;
                if false_term_end > terms.len() {
                    return Err(-66);
                }
                value = evaluate_linear_value(
                    op.false_bias,
                    &terms[op.false_term_start..false_term_end],
                    node_values,
                    state_values,
                )?;
            }
        }

        target_values[op.target_id] = value;
        delay_values[op.target_id] = op.delay;
        rise_values[op.target_id] = op.rise;
        fall_values[op.target_id] = op.fall;
    }
    Ok(())
}

pub fn evaluate_ordered_transition_segment(
    linear_ops: &[EvasRustLinearOp],
    linear_terms: &[EvasRustLinearTerm],
    linear_conditions: &[EvasRustLinearCondition],
    transition_ops: &[EvasRustTransitionTargetOp],
    transition_terms: &[EvasRustLinearTerm],
    transition_conditions: &[EvasRustLinearCondition],
    node_values: &mut [f64],
    state_values: &mut [f64],
    target_values: &mut [f64],
    delay_values: &mut [f64],
    rise_values: &mut [f64],
    fall_values: &mut [f64],
) -> Result<(), i32> {
    evaluate_static_linear_ops(
        linear_ops,
        linear_terms,
        linear_conditions,
        node_values,
        state_values,
    )?;
    evaluate_transition_target_ops(
        transition_ops,
        transition_terms,
        transition_conditions,
        node_values,
        state_values,
        target_values,
        delay_values,
        rise_values,
        fall_values,
    )
}

pub fn timer_static_linear_trace_for_arrays(
    times: &[f64],
    source_node_ids: &[usize],
    source_values: &[f64],
    node_values: &mut [f64],
    state_values: &mut [f64],
    event_ops: &[EvasRustLinearOp],
    event_terms: &[EvasRustLinearTerm],
    event_conditions: &[EvasRustLinearCondition],
    evaluate_ops: &[EvasRustLinearOp],
    evaluate_terms: &[EvasRustLinearTerm],
    evaluate_conditions: &[EvasRustLinearCondition],
    record_node_ids: &[usize],
    out_values: &mut [f64],
    timer_start: f64,
    timer_period: f64,
    has_start: bool,
    eps: f64,
) -> Result<usize, i32> {
    let point_count = times.len();
    let source_count = source_node_ids.len();
    let record_count = record_node_ids.len();
    if source_values.len() != point_count.checked_mul(source_count).ok_or(-701)? {
        return Err(-702);
    }
    if out_values.len() != point_count.checked_mul(record_count).ok_or(-703)? {
        return Err(-704);
    }
    if timer_period <= 0.0 || !timer_period.is_finite() {
        return Err(-705);
    }
    if has_start && !timer_start.is_finite() {
        return Err(-706);
    }
    for id in source_node_ids {
        if *id >= node_values.len() {
            return Err(-707);
        }
    }
    for id in record_node_ids {
        if *id >= node_values.len() {
            return Err(-708);
        }
    }

    let tolerance = eps.abs();
    let mut next_fire = if has_start { timer_start } else { timer_period };
    if !next_fire.is_finite() {
        next_fire = timer_period;
    }
    let mut event_count = 0usize;
    let mut previous_time = f64::NEG_INFINITY;

    for point_idx in 0..point_count {
        let time = times[point_idx];
        if !time.is_finite() {
            return Err(-709);
        }
        if time + tolerance < previous_time {
            return Err(-710);
        }

        while time >= next_fire - tolerance {
            evaluate_static_linear_ops(
                event_ops,
                event_terms,
                event_conditions,
                node_values,
                state_values,
            )?;
            event_count = event_count.checked_add(1).ok_or(-711)?;
            next_fire += timer_period;
            if !next_fire.is_finite() {
                return Err(-712);
            }
        }

        let source_offset = point_idx.checked_mul(source_count).ok_or(-713)?;
        for source_idx in 0..source_count {
            node_values[source_node_ids[source_idx]] =
                source_values[source_offset + source_idx];
        }

        evaluate_static_linear_ops(
            evaluate_ops,
            evaluate_terms,
            evaluate_conditions,
            node_values,
            state_values,
        )?;

        let out_offset = point_idx.checked_mul(record_count).ok_or(-714)?;
        for record_idx in 0..record_count {
            out_values[out_offset + record_idx] = node_values[record_node_ids[record_idx]];
        }
        previous_time = time;
    }
    Ok(event_count)
}

pub fn timer_static_linear_queue_trace_for_arrays(
    times: &[f64],
    source_node_ids: &[usize],
    source_values: &[f64],
    node_values: &mut [f64],
    state_values: &mut [f64],
    timer_starts: &[f64],
    timer_periods: &[f64],
    event_op_starts: &[usize],
    event_op_counts: &[usize],
    event_ops: &[EvasRustLinearOp],
    event_terms: &[EvasRustLinearTerm],
    event_conditions: &[EvasRustLinearCondition],
    evaluate_ops: &[EvasRustLinearOp],
    evaluate_terms: &[EvasRustLinearTerm],
    evaluate_conditions: &[EvasRustLinearCondition],
    record_node_ids: &[usize],
    out_values: &mut [f64],
    eps: f64,
) -> Result<usize, i32> {
    let point_count = times.len();
    let source_count = source_node_ids.len();
    let record_count = record_node_ids.len();
    let timer_count = timer_starts.len();
    if timer_count == 0 {
        return Err(-741);
    }
    if timer_periods.len() != timer_count
        || event_op_starts.len() != timer_count
        || event_op_counts.len() != timer_count
    {
        return Err(-742);
    }
    if source_values.len() != point_count.checked_mul(source_count).ok_or(-743)? {
        return Err(-744);
    }
    if out_values.len() != point_count.checked_mul(record_count).ok_or(-745)? {
        return Err(-746);
    }
    for id in source_node_ids {
        if *id >= node_values.len() {
            return Err(-747);
        }
    }
    for id in record_node_ids {
        if *id >= node_values.len() {
            return Err(-748);
        }
    }
    for idx in 0..timer_count {
        if !timer_starts[idx].is_finite() || timer_starts[idx] < -eps.abs() {
            return Err(-749);
        }
        if timer_periods[idx] <= 0.0 || !timer_periods[idx].is_finite() {
            return Err(-750);
        }
        let op_end = event_op_starts[idx]
            .checked_add(event_op_counts[idx])
            .ok_or(-751)?;
        if op_end > event_ops.len() {
            return Err(-752);
        }
    }

    let tolerance = eps.abs();
    let mut next_fires: Vec<f64> = timer_starts
        .iter()
        .map(|start| if *start < 0.0 { 0.0 } else { *start })
        .collect();
    let mut event_count = 0usize;
    let mut previous_time = f64::NEG_INFINITY;

    for point_idx in 0..point_count {
        let time = times[point_idx];
        if !time.is_finite() {
            return Err(-753);
        }
        if time + tolerance < previous_time {
            return Err(-754);
        }

        loop {
            let mut next_due_time = f64::INFINITY;
            for fire_time in &next_fires {
                if *fire_time <= time + tolerance && *fire_time < next_due_time {
                    next_due_time = *fire_time;
                }
            }
            if !next_due_time.is_finite() {
                break;
            }

            for timer_idx in 0..timer_count {
                if next_fires[timer_idx] <= next_due_time + tolerance
                    && next_fires[timer_idx] <= time + tolerance
                {
                    let op_start = event_op_starts[timer_idx];
                    let op_end = op_start + event_op_counts[timer_idx];
                    evaluate_static_linear_ops(
                        &event_ops[op_start..op_end],
                        event_terms,
                        event_conditions,
                        node_values,
                        state_values,
                    )?;
                    event_count = event_count.checked_add(1).ok_or(-755)?;
                    next_fires[timer_idx] += timer_periods[timer_idx];
                    if !next_fires[timer_idx].is_finite() {
                        return Err(-756);
                    }
                }
            }
        }

        let source_offset = point_idx.checked_mul(source_count).ok_or(-757)?;
        for source_idx in 0..source_count {
            node_values[source_node_ids[source_idx]] =
                source_values[source_offset + source_idx];
        }

        evaluate_static_linear_ops(
            evaluate_ops,
            evaluate_terms,
            evaluate_conditions,
            node_values,
            state_values,
        )?;

        let out_offset = point_idx.checked_mul(record_count).ok_or(-758)?;
        for record_idx in 0..record_count {
            out_values[out_offset + record_idx] = node_values[record_node_ids[record_idx]];
        }
        previous_time = time;
    }
    Ok(event_count)
}

pub fn event_lfsr_shift_xor_step(
    state_values: &mut [f64],
    node_values: &[f64],
    lfsr_slots: &[usize],
    tmp_slots: &[usize],
    tap_slots: &[usize],
    gate_node_id: usize,
    gate_threshold: f64,
    high_node_id: usize,
    low_node_id: usize,
    output_state_id: usize,
    loop_state_id: usize,
    loop_final_value: f64,
) -> Result<bool, i32> {
    if lfsr_slots.is_empty() {
        return Err(-201);
    }
    if tmp_slots.len() < lfsr_slots.len() + 1 {
        return Err(-202);
    }
    for slot in lfsr_slots.iter().chain(tmp_slots).chain(tap_slots) {
        if *slot >= state_values.len() {
            return Err(-203);
        }
    }
    if output_state_id != CONDITION_NONE && output_state_id >= state_values.len() {
        return Err(-204);
    }
    if loop_state_id != CONDITION_NONE && loop_state_id >= state_values.len() {
        return Err(-205);
    }
    if gate_node_id != CONDITION_NONE {
        if gate_node_id >= node_values.len() {
            return Err(-206);
        }
        if node_values[gate_node_id] <= gate_threshold {
            return Ok(false);
        }
    }
    if output_state_id != CONDITION_NONE {
        if high_node_id >= node_values.len() || low_node_id >= node_values.len() {
            return Err(-207);
        }
    }

    let old_bits: Vec<i64> = lfsr_slots
        .iter()
        .map(|slot| to_veriloga_integer(state_values[*slot]) & 1)
        .collect();
    let mut feedback: i64 = 0;
    for slot in tap_slots {
        feedback ^= to_veriloga_integer(state_values[*slot]) & 1;
    }

    state_values[tmp_slots[0]] = feedback as f64;
    for (idx, bit) in old_bits.iter().enumerate() {
        state_values[tmp_slots[idx + 1]] = *bit as f64;
    }
    state_values[lfsr_slots[0]] = feedback as f64;
    for idx in 1..lfsr_slots.len() {
        state_values[lfsr_slots[idx]] = old_bits[idx - 1] as f64;
    }

    if output_state_id != CONDITION_NONE {
        let last_bit = to_veriloga_integer(state_values[*lfsr_slots.last().unwrap()]) & 1;
        state_values[output_state_id] = if last_bit > 0 {
            node_values[high_node_id]
        } else {
            node_values[low_node_id]
        };
    }
    if loop_state_id != CONDITION_NONE {
        state_values[loop_state_id] = to_veriloga_integer(loop_final_value) as f64;
    }
    Ok(true)
}

pub fn timer_lfsr_output_step_for_arrays(
    state_values: &mut [f64],
    node_values: &mut [f64],
    next_fire_times: &mut [f64],
    has_state_flags: &mut [u8],
    period: f64,
    start: f64,
    has_start: u8,
    time: f64,
    eps: f64,
    lfsr_slots: &[usize],
    tmp_slots: &[usize],
    tap_slots: &[usize],
    gate_node_id: usize,
    gate_threshold: f64,
    high_node_id: usize,
    low_node_id: usize,
    output_state_id: usize,
    output_node_id: usize,
    loop_state_id: usize,
    loop_final_value: f64,
    due: &mut u8,
    skipped: &mut u8,
    executed: &mut u8,
    output_written: &mut u8,
) -> Result<(), i32> {
    if next_fire_times.len() != 1 || has_state_flags.len() != 1 {
        return Err(-251);
    }
    *due = 0;
    *skipped = 0;
    *executed = 0;
    *output_written = 0;

    let periods = [period];
    let starts = [start];
    let has_starts = [has_start];
    let mut due_flags = [0_u8];
    let mut skipped_flags = [0_u8];
    timer_periodic_step_for_arrays(
        next_fire_times,
        has_state_flags,
        &periods,
        &starts,
        &has_starts,
        &mut due_flags,
        &mut skipped_flags,
        time,
        true,
        eps,
    )?;
    *due = due_flags[0];
    *skipped = skipped_flags[0];
    if due_flags[0] == 0 {
        return Ok(());
    }

    let did_execute = event_lfsr_shift_xor_step(
        state_values,
        node_values,
        lfsr_slots,
        tmp_slots,
        tap_slots,
        gate_node_id,
        gate_threshold,
        high_node_id,
        low_node_id,
        output_state_id,
        loop_state_id,
        loop_final_value,
    )?;
    if !did_execute {
        return Ok(());
    }
    *executed = 1;
    if output_node_id != CONDITION_NONE && output_state_id != CONDITION_NONE {
        if output_node_id >= node_values.len() || output_state_id >= state_values.len() {
            return Err(-252);
        }
        node_values[output_node_id] = state_values[output_state_id];
        *output_written = 1;
    }
    Ok(())
}

fn pulse_value(
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
        let frac = if fall > 0.0 { (t_mod - fall_start) / fall } else { 1.0 };
        v_hi - frac * (v_hi - v_lo)
    } else {
        v_lo
    }
}

fn reset_prbs7_bits(bits: &mut [u8; 7], seed: i64) {
    for idx in 0..7 {
        bits[idx] = (((seed >> idx) & 1) != 0) as u8;
    }
    if bits.iter().all(|bit| *bit == 0) {
        bits[6] = 1;
    }
}

fn prbs7_shift(bits: &mut [u8; 7]) {
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

fn prbs7_targets(bits: &[u8; 7], vdd: f64, targets: &mut [f64; 8]) {
    targets[0] = if bits[6] != 0 { vdd } else { 0.0 };
    for idx in 0..7 {
        targets[idx + 1] = if bits[idx] != 0 { vdd } else { 0.0 };
    }
}

fn reset_lfsr_bits(bits: &mut [u8], seed: i64, zero_guard_index: i32) -> Result<(), i32> {
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

fn shift_lfsr_bits(
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

fn lfsr_transition_targets(
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
    for slice in [point_vdd, point_vss, point_vinp, point_vinn, point_voutp, point_voutn] {
        if slice.len() != point_count {
            return Err(-375);
        }
    }
    for slice in [sample_vdd, sample_vss, sample_vinp, sample_vinn, sample_voutp, sample_voutn] {
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
            let vin_due = vin_idx < vin_event_times.len()
                && vin_event_times[vin_idx] <= time + 1.0e-18;
            let lfsr_due = lfsr_idx < lfsr_event_times.len()
                && lfsr_event_times[lfsr_idx] <= time + 1.0e-18;
            if !vin_due && !lfsr_due {
                break;
            }

            if vin_due
                && (!lfsr_due
                    || vin_event_times[vin_idx] <= lfsr_event_times[lfsr_idx] + 1.0e-18)
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
fn threshold_crossed(prev: f64, cur: f64, direction: i32, eps: f64) -> bool {
    if direction > 0 {
        return prev < -eps && cur >= -eps;
    }
    if direction < 0 {
        return prev > eps && cur <= eps;
    }
    (prev < -eps && cur >= -eps) || (prev > eps && cur <= eps)
}

fn sar_current_code(bits: &[u8], width: usize) -> u64 {
    let mut code = 0_u64;
    for idx in 0..width {
        if bits[idx] != 0 {
            code += 1_u64 << (width - 1 - idx);
        }
    }
    code
}

#[allow(clippy::too_many_arguments)]
fn transition_drive_index(
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
fn sar_drive_outputs(
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
                    cmp_decision_v = if vsampled >= trial_vdac_state { vdd } else { 0.0 };
                    bit_index_v = bit_idx as f64 / width as f64 * vdd;
                    trial_code_v = trial_code_int as f64 / total_code * vdd;
                    conv_done_v = 0.0;
                } else {
                    let final_code = sar_current_code(&dout_bits, width);
                    trial_vdac_state = final_code as f64 / total_code * vdd;
                    trial_code_v = trial_vdac_state;
                    bit_index_v = 0.0;
                    cmp_decision_v = if vsampled >= trial_vdac_state { vdd } else { 0.0 };
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
                cmp_decision_v = if vsampled >= trial_vdac_state { vdd } else { 0.0 };
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
        let clk_val = point_clk[row_idx];
        let vinn_val = point_vinn[row_idx];
        let vinp_val = point_vinp[row_idx];
        let vdd_val = point_vdd[row_idx];
        let clk_expr = clk_val - edge_vth;

        if threshold_crossed(prev_clk_expr, clk_expr, 1, 1.0e-12) {
            t_start = time;
            armed = true;
            let vdiff = vinp_val - vinn_val - voffset;
            let vdiff_eff = vdiff.abs().max(1.0e-9);
            let mut td = td0 + tau * (vdd_val / vdiff_eff).ln();
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
                    time,
                );
            }
            let (outp_target, outn_target) = if vdiff > 0.0 {
                (vdd_val, 0.0)
            } else {
                (0.0, vdd_val)
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
                time,
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
                time,
                outn_target,
                td,
                tedge,
                tedge,
            );
            clock_events += 1;
        } else if threshold_crossed(prev_clk_expr, clk_expr, -1, 1.0e-12) {
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
        if armed && threshold_crossed(prev_outp_expr, outp_expr, 1, 1.0e-12) {
            let delay_ps = (time - t_start) * 1.0e12;
            transition_evaluate_one(
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
            transition_set_target_one(
                &mut current[2],
                &mut target_values[2],
                &mut start_times[2],
                &mut start_values[2],
                &mut delays[2],
                &mut rises[2],
                &mut falls[2],
                &mut active[2],
                time,
                delay_ps,
                0.0,
                10.0e-12,
                10.0e-12,
            );
            armed = false;
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
    let mut vctrl_current = vctrl_event_values.first().copied().unwrap_or(0.5 * (vh + vl));

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
        while vctrl_idx < vctrl_event_times.len() && vctrl_event_times[vctrl_idx] <= time + 1.0e-18 {
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

fn to_veriloga_integer(value: f64) -> i64 {
    if !value.is_finite() {
        return 0;
    }
    if value >= 0.0 {
        (value + 0.5).floor() as i64
    } else {
        (value - 0.5).ceil() as i64
    }
}

fn truthy(value: f64) -> bool {
    value != 0.0
}

fn bool_value(value: bool) -> f64 {
    if value {
        1.0
    } else {
        0.0
    }
}

fn pop1(stack: &mut Vec<f64>) -> Result<f64, i32> {
    stack.pop().ok_or(-2240)
}

fn pop2(stack: &mut Vec<f64>) -> Result<(f64, f64), i32> {
    let right = stack.pop().ok_or(-2241)?;
    let left = stack.pop().ok_or(-2242)?;
    Ok((left, right))
}

fn evaluate_body_expr_segment(
    ops: &[EvasRustBodyExprOp],
    node_values: &[f64],
    state_values: &[f64],
    param_values: &[f64],
    stack: &mut Vec<f64>,
) -> Result<(), i32> {
    for op in ops {
        match op.op_kind {
            BODY_EXPR_CONST => stack.push(op.value),
            BODY_EXPR_READ_NODE => {
                if op.index >= node_values.len() {
                    return Err(-2250);
                }
                stack.push(node_values[op.index]);
            }
            BODY_EXPR_READ_STATE => {
                if op.index >= state_values.len() {
                    return Err(-2251);
                }
                stack.push(state_values[op.index]);
            }
            BODY_EXPR_READ_PARAM => {
                if op.index >= param_values.len() {
                    return Err(-2252);
                }
                stack.push(param_values[op.index]);
            }
            BODY_EXPR_NEG => {
                let value = pop1(stack)?;
                stack.push(-value);
            }
            BODY_EXPR_NOT => {
                let value = pop1(stack)?;
                stack.push(bool_value(!truthy(value)));
            }
            BODY_EXPR_ADD => {
                let (left, right) = pop2(stack)?;
                stack.push(left + right);
            }
            BODY_EXPR_SUB => {
                let (left, right) = pop2(stack)?;
                stack.push(left - right);
            }
            BODY_EXPR_MUL => {
                let (left, right) = pop2(stack)?;
                stack.push(left * right);
            }
            BODY_EXPR_DIV => {
                let (left, right) = pop2(stack)?;
                stack.push(left / right);
            }
            BODY_EXPR_MOD => {
                let (left, right) = pop2(stack)?;
                stack.push(left % right);
            }
            BODY_EXPR_GT => {
                let (left, right) = pop2(stack)?;
                stack.push(bool_value(left > right));
            }
            BODY_EXPR_LT => {
                let (left, right) = pop2(stack)?;
                stack.push(bool_value(left < right));
            }
            BODY_EXPR_GE => {
                let (left, right) = pop2(stack)?;
                stack.push(bool_value(left >= right));
            }
            BODY_EXPR_LE => {
                let (left, right) = pop2(stack)?;
                stack.push(bool_value(left <= right));
            }
            BODY_EXPR_EQ => {
                let (left, right) = pop2(stack)?;
                stack.push(bool_value(left == right));
            }
            BODY_EXPR_NE => {
                let (left, right) = pop2(stack)?;
                stack.push(bool_value(left != right));
            }
            BODY_EXPR_LAND => {
                let (left, right) = pop2(stack)?;
                stack.push(bool_value(truthy(left) && truthy(right)));
            }
            BODY_EXPR_LOR => {
                let (left, right) = pop2(stack)?;
                stack.push(bool_value(truthy(left) || truthy(right)));
            }
            BODY_EXPR_BITAND => {
                let (left, right) = pop2(stack)?;
                stack.push(((left as i64) & (right as i64)) as f64);
            }
            BODY_EXPR_BITOR => {
                let (left, right) = pop2(stack)?;
                stack.push(((left as i64) | (right as i64)) as f64);
            }
            BODY_EXPR_BITXOR => {
                let (left, right) = pop2(stack)?;
                stack.push(((left as i64) ^ (right as i64)) as f64);
            }
            BODY_EXPR_SELECT => {
                let false_value = stack.pop().ok_or(-2253)?;
                let true_value = stack.pop().ok_or(-2254)?;
                let condition = stack.pop().ok_or(-2255)?;
                stack.push(if truthy(condition) { true_value } else { false_value });
            }
            BODY_EXPR_ABS => {
                let value = pop1(stack)?;
                stack.push(value.abs());
            }
            BODY_EXPR_SQRT => {
                let value = pop1(stack)?;
                stack.push(value.sqrt());
            }
            BODY_EXPR_EXP => {
                let value = pop1(stack)?;
                stack.push(value.exp());
            }
            BODY_EXPR_LN => {
                let value = pop1(stack)?;
                stack.push(value.ln());
            }
            BODY_EXPR_LOG10 => {
                let value = pop1(stack)?;
                stack.push(value.log10());
            }
            BODY_EXPR_SIN => {
                let value = pop1(stack)?;
                stack.push(value.sin());
            }
            BODY_EXPR_COS => {
                let value = pop1(stack)?;
                stack.push(value.cos());
            }
            BODY_EXPR_FLOOR => {
                let value = pop1(stack)?;
                stack.push(value.floor());
            }
            BODY_EXPR_CEIL => {
                let value = pop1(stack)?;
                stack.push(value.ceil());
            }
            BODY_EXPR_MIN => {
                let (left, right) = pop2(stack)?;
                stack.push(left.min(right));
            }
            BODY_EXPR_MAX => {
                let (left, right) = pop2(stack)?;
                stack.push(left.max(right));
            }
            BODY_EXPR_POW => {
                let (left, right) = pop2(stack)?;
                stack.push(left.powf(right));
            }
            _ => return Err(-2256),
        }
    }
    Ok(())
}

fn evaluate_linear_value(
    bias: f64,
    terms: &[EvasRustLinearTerm],
    node_values: &[f64],
    state_values: &[f64],
) -> Result<f64, i32> {
    let mut value = bias;
    for term in terms {
        let source = match term.source_kind {
            SOURCE_NODE => {
                if term.source_id >= node_values.len() {
                    return Err(-8);
                }
                node_values[term.source_id]
            }
            SOURCE_STATE => {
                if term.source_id >= state_values.len() {
                    return Err(-9);
                }
                state_values[term.source_id]
            }
            _ => return Err(-10),
        };
        value += term.gain * source;
    }
    Ok(value)
}

fn evaluate_condition(
    condition: &EvasRustLinearCondition,
    terms: &[EvasRustLinearTerm],
    node_values: &[f64],
    state_values: &[f64],
) -> Result<bool, i32> {
    let left_term_end = condition
        .left_term_start
        .checked_add(condition.left_term_count)
        .ok_or(-17)?;
    let right_term_end = condition
        .right_term_start
        .checked_add(condition.right_term_count)
        .ok_or(-18)?;
    if left_term_end > terms.len() || right_term_end > terms.len() {
        return Err(-19);
    }
    let left = evaluate_linear_value(
        condition.left_bias,
        &terms[condition.left_term_start..left_term_end],
        node_values,
        state_values,
    )?;
    let right = evaluate_linear_value(
        condition.right_bias,
        &terms[condition.right_term_start..right_term_end],
        node_values,
        state_values,
    )?;
    match condition.op_kind {
        COND_GT => Ok(left > right),
        COND_LT => Ok(left < right),
        COND_GE => Ok(left >= right),
        COND_LE => Ok(left <= right),
        COND_EQ => Ok(left == right),
        COND_NE => Ok(left != right),
        _ => Err(-20),
    }
}

pub fn copy_f64_values(source: &[f64], target: &mut [f64]) -> Result<(), i32> {
    if source.len() != target.len() {
        return Err(-31);
    }
    target.copy_from_slice(source);
    Ok(())
}

pub fn max_err_ratio_for_nodes(
    values: &[f64],
    previous: &[f64],
    node_ids: &[usize],
    reltol: f64,
    vabstol: f64,
) -> Result<f64, i32> {
    if values.len() != previous.len() {
        return Err(-41);
    }
    let mut max_ratio = 0.0;
    for node_id in node_ids {
        let idx = *node_id;
        if idx >= values.len() {
            return Err(-42);
        }
        let vnew = values[idx];
        let vold = previous[idx];
        let dv = (vnew - vold).abs();
        let vref = vnew.abs().max(vold.abs());
        let tol = reltol * vref + vabstol;
        if tol > 0.0 {
            let ratio = dv / tol;
            if ratio > max_ratio {
                max_ratio = ratio;
            }
        }
    }
    Ok(max_ratio)
}

pub fn interpolate_event_values_for_arrays(
    previous_values: &[f64],
    current_values: &[f64],
    output_values: &mut [f64],
    previous_time: f64,
    current_time: f64,
    event_time: f64,
) -> Result<(), i32> {
    let count = previous_values.len();
    if current_values.len() != count || output_values.len() != count {
        return Err(-181);
    }
    if !previous_time.is_finite() || !current_time.is_finite() || !event_time.is_finite() {
        return Err(-182);
    }
    let frac = if current_time > previous_time + 1.0e-30 {
        ((event_time - previous_time) / (current_time - previous_time)).clamp(0.0, 1.0)
    } else {
        1.0
    };
    for idx in 0..count {
        let previous = previous_values[idx];
        let current = current_values[idx];
        if !previous.is_finite() || !current.is_finite() {
            return Err(-183);
        }
        output_values[idx] = previous + frac * (current - previous);
    }
    Ok(())
}

pub fn record_values_for_node_ids(
    values: &[f64],
    node_ids: &[usize],
    default_value: f64,
    out_values: &mut [f64],
) -> Result<(), i32> {
    if node_ids.len() != out_values.len() {
        return Err(-51);
    }
    for (idx, node_id) in node_ids.iter().enumerate() {
        out_values[idx] = if *node_id < values.len() {
            values[*node_id]
        } else {
            default_value
        };
    }
    Ok(())
}

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
        let t_begin = start_times[idx] + delays[idx];
        let going_up = target_values[idx] > start_values[idx];
        let ramp_time = if going_up {
            rise_times[idx]
        } else {
            fall_times[idx]
        };
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

fn transition_evaluate_one(
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

    let t_begin = start_time + delay;
    let going_up = target_value > start_value;
    let ramp_time = if going_up { rise_time } else { fall_time };

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

fn transition_set_target_one(
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
        let t_begin = *start_time + *delay_value;
        let going_up = *target_value > *start_value;
        let ramp_time = if going_up { *rise_time } else { *fall_time };
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
                *start_time = time - (vi - basis) / slope;
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

fn interpolate_cross_time(prev_time: f64, prev_value: f64, time: f64, value: f64) -> f64 {
    let dv = value - prev_value;
    let frac = if dv.abs() > 1.0e-30 {
        (-prev_value / dv).clamp(0.0, 1.0)
    } else {
        0.0
    };
    prev_time + frac * (time - prev_time)
}

pub fn dynamic_bus_offsets_for_arrays(
    base_offsets: &[usize],
    outer_lengths: &[usize],
    inner_strides: &[usize],
    inner_lengths: &[usize],
    first_indices: &[i64],
    second_indices: &[i64],
    has_second_index_flags: &[u8],
    out_node_ids: &mut [usize],
) -> Result<(), i32> {
    let count = base_offsets.len();
    if outer_lengths.len() != count
        || inner_strides.len() != count
        || inner_lengths.len() != count
        || first_indices.len() != count
        || second_indices.len() != count
        || has_second_index_flags.len() != count
        || out_node_ids.len() != count
    {
        return Err(-201);
    }

    for idx in 0..count {
        let first = first_indices[idx];
        if first < 0 || outer_lengths[idx] == 0 || first as usize >= outer_lengths[idx] {
            return Err(-202);
        }
        let mut offset = (first as usize)
            .checked_mul(inner_strides[idx])
            .ok_or(-203)?;
        if has_second_index_flags[idx] != 0 {
            let second = second_indices[idx];
            if second < 0 || inner_lengths[idx] == 0 || second as usize >= inner_lengths[idx] {
                return Err(-204);
            }
            offset = offset.checked_add(second as usize).ok_or(-205)?;
        }
        out_node_ids[idx] = base_offsets[idx].checked_add(offset).ok_or(-206)?;
    }
    Ok(())
}

#[no_mangle]
pub unsafe extern "C" fn evas_rust_evaluate_static_affine(
    ops: *const EvasRustStaticAffineOp,
    op_count: usize,
    values: *mut f64,
    value_count: usize,
) -> i32 {
    if op_count > 0 && ops.is_null() {
        return -1;
    }
    if value_count > 0 && values.is_null() {
        return -2;
    }

    let op_slice = if op_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(ops, op_count)
    };
    let value_slice = if value_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(values, value_count)
    };

    match evaluate_static_affine_ops(op_slice, value_slice) {
        Ok(()) => 0,
        Err(code) => code,
    }
}

#[no_mangle]
pub unsafe extern "C" fn evas_rust_evaluate_static_linear(
    ops: *const EvasRustLinearOp,
    op_count: usize,
    terms: *const EvasRustLinearTerm,
    term_count: usize,
    conditions: *const EvasRustLinearCondition,
    condition_count: usize,
    node_values: *mut f64,
    node_value_count: usize,
    state_values: *mut f64,
    state_value_count: usize,
) -> i32 {
    if op_count > 0 && ops.is_null() {
        return -1;
    }
    if term_count > 0 && terms.is_null() {
        return -2;
    }
    if condition_count > 0 && conditions.is_null() {
        return -21;
    }
    if node_value_count > 0 && node_values.is_null() {
        return -3;
    }
    if state_value_count > 0 && state_values.is_null() {
        return -4;
    }

    let op_slice = if op_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(ops, op_count)
    };
    let term_slice = if term_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(terms, term_count)
    };
    let condition_slice = if condition_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(conditions, condition_count)
    };
    let node_value_slice = if node_value_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(node_values, node_value_count)
    };
    let state_value_slice = if state_value_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(state_values, state_value_count)
    };

    match evaluate_static_linear_ops(
        op_slice,
        term_slice,
        condition_slice,
        node_value_slice,
        state_value_slice,
    ) {
        Ok(()) => 0,
        Err(code) => code,
    }
}

#[no_mangle]
pub unsafe extern "C" fn evas_rust_evaluate_body_ir(
    stmt_ops: *const EvasRustBodyStmtOp,
    stmt_count: usize,
    expr_ops: *const EvasRustBodyExprOp,
    expr_count: usize,
    node_values: *mut f64,
    node_value_count: usize,
    state_values: *mut f64,
    state_value_count: usize,
    param_values: *const f64,
    param_value_count: usize,
) -> i32 {
    if stmt_count > 0 && stmt_ops.is_null() {
        return -2301;
    }
    if expr_count > 0 && expr_ops.is_null() {
        return -2302;
    }
    if node_value_count > 0 && node_values.is_null() {
        return -2303;
    }
    if state_value_count > 0 && state_values.is_null() {
        return -2304;
    }
    if param_value_count > 0 && param_values.is_null() {
        return -2305;
    }

    let stmt_slice = if stmt_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(stmt_ops, stmt_count)
    };
    let expr_slice = if expr_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(expr_ops, expr_count)
    };
    let node_value_slice = if node_value_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(node_values, node_value_count)
    };
    let state_value_slice = if state_value_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(state_values, state_value_count)
    };
    let param_value_slice = if param_value_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(param_values, param_value_count)
    };

    match evaluate_body_ir_ops(
        stmt_slice,
        expr_slice,
        node_value_slice,
        state_value_slice,
        param_value_slice,
    ) {
        Ok(()) => 0,
        Err(code) => code,
    }
}

#[no_mangle]
pub unsafe extern "C" fn evas_rust_evaluate_body_expr(
    expr_ops: *const EvasRustBodyExprOp,
    expr_count: usize,
    node_values: *const f64,
    node_value_count: usize,
    state_values: *const f64,
    state_value_count: usize,
    param_values: *const f64,
    param_value_count: usize,
    out_value: *mut f64,
) -> i32 {
    if expr_count > 0 && expr_ops.is_null() {
        return -2321;
    }
    if node_value_count > 0 && node_values.is_null() {
        return -2322;
    }
    if state_value_count > 0 && state_values.is_null() {
        return -2323;
    }
    if param_value_count > 0 && param_values.is_null() {
        return -2324;
    }
    if out_value.is_null() {
        return -2325;
    }

    let expr_slice = if expr_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(expr_ops, expr_count)
    };
    let node_value_slice = if node_value_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(node_values, node_value_count)
    };
    let state_value_slice = if state_value_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(state_values, state_value_count)
    };
    let param_value_slice = if param_value_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(param_values, param_value_count)
    };

    match evaluate_body_expr_ops(
        expr_slice,
        node_value_slice,
        state_value_slice,
        param_value_slice,
    ) {
        Ok(value) => {
            *out_value = value;
            0
        }
        Err(code) => code,
    }
}

#[no_mangle]
pub unsafe extern "C" fn evas_rust_evaluate_body_expr_batch(
    expr_ops: *const EvasRustBodyExprOp,
    expr_count: usize,
    expr_starts: *const usize,
    expr_segment_count: usize,
    expr_counts: *const usize,
    expr_count_count: usize,
    node_values: *const f64,
    node_value_count: usize,
    state_values: *const f64,
    state_value_count: usize,
    param_values: *const f64,
    param_value_count: usize,
    output_values: *mut f64,
    output_value_count: usize,
) -> i32 {
    if expr_count > 0 && expr_ops.is_null() {
        return -2331;
    }
    if expr_segment_count > 0 && expr_starts.is_null() {
        return -2332;
    }
    if expr_count_count > 0 && expr_counts.is_null() {
        return -2333;
    }
    if node_value_count > 0 && node_values.is_null() {
        return -2334;
    }
    if state_value_count > 0 && state_values.is_null() {
        return -2335;
    }
    if param_value_count > 0 && param_values.is_null() {
        return -2336;
    }
    if output_value_count > 0 && output_values.is_null() {
        return -2337;
    }
    if expr_segment_count != expr_count_count || expr_segment_count != output_value_count {
        return -2338;
    }

    let expr_slice = if expr_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(expr_ops, expr_count)
    };
    let expr_start_slice = if expr_segment_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(expr_starts, expr_segment_count)
    };
    let expr_count_slice = if expr_count_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(expr_counts, expr_count_count)
    };
    let node_value_slice = if node_value_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(node_values, node_value_count)
    };
    let state_value_slice = if state_value_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(state_values, state_value_count)
    };
    let param_value_slice = if param_value_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(param_values, param_value_count)
    };
    let output_value_slice = if output_value_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(output_values, output_value_count)
    };

    match evaluate_body_expr_segments(
        expr_slice,
        expr_start_slice,
        expr_count_slice,
        node_value_slice,
        state_value_slice,
        param_value_slice,
        output_value_slice,
    ) {
        Ok(()) => 0,
        Err(code) => code,
    }
}

#[no_mangle]
pub unsafe extern "C" fn evas_rust_timer_static_linear_trace(
    times: *const f64,
    point_count: usize,
    source_node_ids: *const usize,
    source_count: usize,
    source_values: *const f64,
    node_values: *mut f64,
    node_value_count: usize,
    state_values: *mut f64,
    state_value_count: usize,
    event_ops: *const EvasRustLinearOp,
    event_op_count: usize,
    event_terms: *const EvasRustLinearTerm,
    event_term_count: usize,
    event_conditions: *const EvasRustLinearCondition,
    event_condition_count: usize,
    evaluate_ops: *const EvasRustLinearOp,
    evaluate_op_count: usize,
    evaluate_terms: *const EvasRustLinearTerm,
    evaluate_term_count: usize,
    evaluate_conditions: *const EvasRustLinearCondition,
    evaluate_condition_count: usize,
    record_node_ids: *const usize,
    record_count: usize,
    out_values: *mut f64,
    out_value_count: usize,
    timer_start: f64,
    timer_period: f64,
    has_start: u8,
    eps: f64,
    out_event_count: *mut usize,
) -> i32 {
    if point_count > 0 && times.is_null() {
        return -721;
    }
    if source_count > 0 && source_node_ids.is_null() {
        return -722;
    }
    let expected_source_values = match point_count.checked_mul(source_count) {
        Some(value) => value,
        None => return -723,
    };
    if expected_source_values > 0 && source_values.is_null() {
        return -724;
    }
    if node_value_count > 0 && node_values.is_null() {
        return -725;
    }
    if state_value_count > 0 && state_values.is_null() {
        return -726;
    }
    if event_op_count > 0 && event_ops.is_null() {
        return -727;
    }
    if event_term_count > 0 && event_terms.is_null() {
        return -728;
    }
    if event_condition_count > 0 && event_conditions.is_null() {
        return -729;
    }
    if evaluate_op_count > 0 && evaluate_ops.is_null() {
        return -730;
    }
    if evaluate_term_count > 0 && evaluate_terms.is_null() {
        return -731;
    }
    if evaluate_condition_count > 0 && evaluate_conditions.is_null() {
        return -732;
    }
    if record_count > 0 && record_node_ids.is_null() {
        return -733;
    }
    if out_value_count > 0 && out_values.is_null() {
        return -734;
    }
    if out_event_count.is_null() {
        return -735;
    }

    let times_slice = if point_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(times, point_count)
    };
    let source_node_slice = if source_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(source_node_ids, source_count)
    };
    let source_value_slice = if expected_source_values == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(source_values, expected_source_values)
    };
    let node_value_slice = if node_value_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(node_values, node_value_count)
    };
    let state_value_slice = if state_value_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(state_values, state_value_count)
    };
    let event_op_slice = if event_op_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(event_ops, event_op_count)
    };
    let event_term_slice = if event_term_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(event_terms, event_term_count)
    };
    let event_condition_slice = if event_condition_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(event_conditions, event_condition_count)
    };
    let evaluate_op_slice = if evaluate_op_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(evaluate_ops, evaluate_op_count)
    };
    let evaluate_term_slice = if evaluate_term_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(evaluate_terms, evaluate_term_count)
    };
    let evaluate_condition_slice = if evaluate_condition_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(evaluate_conditions, evaluate_condition_count)
    };
    let record_node_slice = if record_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(record_node_ids, record_count)
    };
    let out_value_slice = if out_value_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(out_values, out_value_count)
    };

    match timer_static_linear_trace_for_arrays(
        times_slice,
        source_node_slice,
        source_value_slice,
        node_value_slice,
        state_value_slice,
        event_op_slice,
        event_term_slice,
        event_condition_slice,
        evaluate_op_slice,
        evaluate_term_slice,
        evaluate_condition_slice,
        record_node_slice,
        out_value_slice,
        timer_start,
        timer_period,
        has_start != 0,
        eps,
    ) {
        Ok(event_count) => {
            *out_event_count = event_count;
            0
        }
        Err(code) => code,
    }
}

#[no_mangle]
pub unsafe extern "C" fn evas_rust_timer_static_linear_queue_trace(
    times: *const f64,
    point_count: usize,
    source_node_ids: *const usize,
    source_count: usize,
    source_values: *const f64,
    node_values: *mut f64,
    node_value_count: usize,
    state_values: *mut f64,
    state_value_count: usize,
    timer_starts: *const f64,
    timer_periods: *const f64,
    event_op_starts: *const usize,
    event_op_counts: *const usize,
    timer_count: usize,
    event_ops: *const EvasRustLinearOp,
    event_op_count: usize,
    event_terms: *const EvasRustLinearTerm,
    event_term_count: usize,
    event_conditions: *const EvasRustLinearCondition,
    event_condition_count: usize,
    evaluate_ops: *const EvasRustLinearOp,
    evaluate_op_count: usize,
    evaluate_terms: *const EvasRustLinearTerm,
    evaluate_term_count: usize,
    evaluate_conditions: *const EvasRustLinearCondition,
    evaluate_condition_count: usize,
    record_node_ids: *const usize,
    record_count: usize,
    out_values: *mut f64,
    out_value_count: usize,
    eps: f64,
    out_event_count: *mut usize,
) -> i32 {
    if point_count > 0 && times.is_null() {
        return -761;
    }
    if source_count > 0 && source_node_ids.is_null() {
        return -762;
    }
    let expected_source_values = match point_count.checked_mul(source_count) {
        Some(value) => value,
        None => return -763,
    };
    if expected_source_values > 0 && source_values.is_null() {
        return -764;
    }
    if node_value_count > 0 && node_values.is_null() {
        return -765;
    }
    if state_value_count > 0 && state_values.is_null() {
        return -766;
    }
    if timer_count > 0
        && (timer_starts.is_null()
            || timer_periods.is_null()
            || event_op_starts.is_null()
            || event_op_counts.is_null())
    {
        return -767;
    }
    if event_op_count > 0 && event_ops.is_null() {
        return -768;
    }
    if event_term_count > 0 && event_terms.is_null() {
        return -769;
    }
    if event_condition_count > 0 && event_conditions.is_null() {
        return -770;
    }
    if evaluate_op_count > 0 && evaluate_ops.is_null() {
        return -771;
    }
    if evaluate_term_count > 0 && evaluate_terms.is_null() {
        return -772;
    }
    if evaluate_condition_count > 0 && evaluate_conditions.is_null() {
        return -773;
    }
    if record_count > 0 && record_node_ids.is_null() {
        return -774;
    }
    if out_value_count > 0 && out_values.is_null() {
        return -775;
    }
    if out_event_count.is_null() {
        return -776;
    }

    let times_slice = if point_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(times, point_count)
    };
    let source_node_slice = if source_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(source_node_ids, source_count)
    };
    let source_value_slice = if expected_source_values == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(source_values, expected_source_values)
    };
    let node_value_slice = if node_value_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(node_values, node_value_count)
    };
    let state_value_slice = if state_value_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(state_values, state_value_count)
    };
    let timer_start_slice = if timer_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(timer_starts, timer_count)
    };
    let timer_period_slice = if timer_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(timer_periods, timer_count)
    };
    let event_op_start_slice = if timer_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(event_op_starts, timer_count)
    };
    let event_op_count_slice = if timer_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(event_op_counts, timer_count)
    };
    let event_op_slice = if event_op_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(event_ops, event_op_count)
    };
    let event_term_slice = if event_term_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(event_terms, event_term_count)
    };
    let event_condition_slice = if event_condition_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(event_conditions, event_condition_count)
    };
    let evaluate_op_slice = if evaluate_op_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(evaluate_ops, evaluate_op_count)
    };
    let evaluate_term_slice = if evaluate_term_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(evaluate_terms, evaluate_term_count)
    };
    let evaluate_condition_slice = if evaluate_condition_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(evaluate_conditions, evaluate_condition_count)
    };
    let record_node_slice = if record_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(record_node_ids, record_count)
    };
    let out_value_slice = if out_value_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(out_values, out_value_count)
    };

    match timer_static_linear_queue_trace_for_arrays(
        times_slice,
        source_node_slice,
        source_value_slice,
        node_value_slice,
        state_value_slice,
        timer_start_slice,
        timer_period_slice,
        event_op_start_slice,
        event_op_count_slice,
        event_op_slice,
        event_term_slice,
        event_condition_slice,
        evaluate_op_slice,
        evaluate_term_slice,
        evaluate_condition_slice,
        record_node_slice,
        out_value_slice,
        eps,
    ) {
        Ok(event_count) => {
            *out_event_count = event_count;
            0
        }
        Err(code) => code,
    }
}

#[no_mangle]
pub unsafe extern "C" fn evas_rust_event_lfsr_shift_xor_step(
    state_values: *mut f64,
    state_value_count: usize,
    node_values: *const f64,
    node_value_count: usize,
    lfsr_slots: *const usize,
    lfsr_slot_count: usize,
    tmp_slots: *const usize,
    tmp_slot_count: usize,
    tap_slots: *const usize,
    tap_slot_count: usize,
    gate_node_id: usize,
    gate_threshold: f64,
    high_node_id: usize,
    low_node_id: usize,
    output_state_id: usize,
    loop_state_id: usize,
    loop_final_value: f64,
    executed: *mut u8,
) -> i32 {
    if state_value_count > 0 && state_values.is_null() {
        return -211;
    }
    if node_value_count > 0 && node_values.is_null() {
        return -212;
    }
    if lfsr_slot_count > 0 && lfsr_slots.is_null() {
        return -213;
    }
    if tmp_slot_count > 0 && tmp_slots.is_null() {
        return -214;
    }
    if tap_slot_count > 0 && tap_slots.is_null() {
        return -215;
    }
    if executed.is_null() {
        return -216;
    }

    let state_slice = if state_value_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(state_values, state_value_count)
    };
    let node_slice = if node_value_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(node_values, node_value_count)
    };
    let lfsr_slice = if lfsr_slot_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(lfsr_slots, lfsr_slot_count)
    };
    let tmp_slice = if tmp_slot_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(tmp_slots, tmp_slot_count)
    };
    let tap_slice = if tap_slot_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(tap_slots, tap_slot_count)
    };

    match event_lfsr_shift_xor_step(
        state_slice,
        node_slice,
        lfsr_slice,
        tmp_slice,
        tap_slice,
        gate_node_id,
        gate_threshold,
        high_node_id,
        low_node_id,
        output_state_id,
        loop_state_id,
        loop_final_value,
    ) {
        Ok(true) => {
            *executed = 1;
            0
        }
        Ok(false) => {
            *executed = 0;
            0
        }
        Err(code) => code,
    }
}

#[no_mangle]
pub unsafe extern "C" fn evas_rust_timer_lfsr_output_step(
    state_values: *mut f64,
    state_value_count: usize,
    node_values: *mut f64,
    node_value_count: usize,
    next_fire_times: *mut f64,
    has_state_flags: *mut u8,
    period: f64,
    start: f64,
    has_start: u8,
    time: f64,
    eps: f64,
    lfsr_slots: *const usize,
    lfsr_slot_count: usize,
    tmp_slots: *const usize,
    tmp_slot_count: usize,
    tap_slots: *const usize,
    tap_slot_count: usize,
    gate_node_id: usize,
    gate_threshold: f64,
    high_node_id: usize,
    low_node_id: usize,
    output_state_id: usize,
    output_node_id: usize,
    loop_state_id: usize,
    loop_final_value: f64,
    due: *mut u8,
    skipped: *mut u8,
    executed: *mut u8,
    output_written: *mut u8,
) -> i32 {
    if state_value_count > 0 && state_values.is_null() {
        return -261;
    }
    if node_value_count > 0 && node_values.is_null() {
        return -262;
    }
    if next_fire_times.is_null() {
        return -263;
    }
    if has_state_flags.is_null() {
        return -264;
    }
    if lfsr_slot_count > 0 && lfsr_slots.is_null() {
        return -265;
    }
    if tmp_slot_count > 0 && tmp_slots.is_null() {
        return -266;
    }
    if tap_slot_count > 0 && tap_slots.is_null() {
        return -267;
    }
    if due.is_null() || skipped.is_null() || executed.is_null() || output_written.is_null() {
        return -268;
    }

    let state_slice = if state_value_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(state_values, state_value_count)
    };
    let node_slice = if node_value_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(node_values, node_value_count)
    };
    let next_fire_slice = std::slice::from_raw_parts_mut(next_fire_times, 1);
    let has_state_slice = std::slice::from_raw_parts_mut(has_state_flags, 1);
    let lfsr_slice = if lfsr_slot_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(lfsr_slots, lfsr_slot_count)
    };
    let tmp_slice = if tmp_slot_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(tmp_slots, tmp_slot_count)
    };
    let tap_slice = if tap_slot_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(tap_slots, tap_slot_count)
    };

    match timer_lfsr_output_step_for_arrays(
        state_slice,
        node_slice,
        next_fire_slice,
        has_state_slice,
        period,
        start,
        has_start,
        time,
        eps,
        lfsr_slice,
        tmp_slice,
        tap_slice,
        gate_node_id,
        gate_threshold,
        high_node_id,
        low_node_id,
        output_state_id,
        output_node_id,
        loop_state_id,
        loop_final_value,
        &mut *due,
        &mut *skipped,
        &mut *executed,
        &mut *output_written,
    ) {
        Ok(()) => 0,
        Err(code) => code,
    }
}

#[no_mangle]
pub unsafe extern "C" fn evas_rust_prbs7_trace(
    times: *const f64,
    point_count: usize,
    values: *mut f64,
    value_count: usize,
    signal_count: usize,
    clk_vlo: f64,
    clk_vhi: f64,
    clk_period: f64,
    clk_duty: f64,
    clk_rise: f64,
    clk_fall: f64,
    clk_delay: f64,
    clk_width: f64,
    clk_has_width: u8,
    rst_vlo: f64,
    rst_vhi: f64,
    rst_period: f64,
    rst_duty: f64,
    rst_rise: f64,
    rst_fall: f64,
    rst_delay: f64,
    rst_width: f64,
    rst_has_width: u8,
    en_voltage: f64,
    vdd: f64,
    vth: f64,
    trf: f64,
    td: f64,
    seed: i64,
    event_count: *mut usize,
) -> i32 {
    if point_count > 0 && times.is_null() {
        return -331;
    }
    if value_count > 0 && values.is_null() {
        return -332;
    }
    if event_count.is_null() {
        return -333;
    }
    let times_slice = if point_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(times, point_count)
    };
    let value_slice = if value_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(values, value_count)
    };
    match prbs7_trace_for_times(
        times_slice,
        value_slice,
        signal_count,
        clk_vlo,
        clk_vhi,
        clk_period,
        clk_duty,
        clk_rise,
        clk_fall,
        clk_delay,
        clk_width,
        clk_has_width != 0,
        rst_vlo,
        rst_vhi,
        rst_period,
        rst_duty,
        rst_rise,
        rst_fall,
        rst_delay,
        rst_width,
        rst_has_width != 0,
        en_voltage,
        vdd,
        vth,
        trf,
        td,
        seed,
    ) {
        Ok(count) => {
            *event_count = count;
            0
        }
        Err(code) => code,
    }
}

#[no_mangle]
pub unsafe extern "C" fn evas_rust_lfsr_transition_trace(
    times: *const f64,
    point_count: usize,
    values: *mut f64,
    value_count: usize,
    signal_count: usize,
    clk_vlo: f64,
    clk_vhi: f64,
    clk_period: f64,
    clk_duty: f64,
    clk_rise: f64,
    clk_fall: f64,
    clk_delay: f64,
    clk_width: f64,
    clk_has_width: u8,
    rst_vlo: f64,
    rst_vhi: f64,
    rst_period: f64,
    rst_duty: f64,
    rst_rise: f64,
    rst_fall: f64,
    rst_delay: f64,
    rst_width: f64,
    rst_has_width: u8,
    en_voltage: f64,
    vdd: f64,
    vth: f64,
    trf: f64,
    td: f64,
    seed: i64,
    width: usize,
    taps: *const usize,
    tap_count: usize,
    shift_sources: *const i32,
    shift_source_count: usize,
    output_bits: *const usize,
    output_count: usize,
    zero_guard_index: i32,
    event_count: *mut usize,
) -> i32 {
    if point_count > 0 && times.is_null() {
        return -361;
    }
    if value_count > 0 && values.is_null() {
        return -362;
    }
    if tap_count > 0 && taps.is_null() {
        return -363;
    }
    if shift_source_count > 0 && shift_sources.is_null() {
        return -364;
    }
    if output_count > 0 && output_bits.is_null() {
        return -365;
    }
    if event_count.is_null() {
        return -366;
    }
    let times_slice = if point_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(times, point_count)
    };
    let value_slice = if value_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(values, value_count)
    };
    let tap_slice = if tap_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(taps, tap_count)
    };
    let shift_slice = if shift_source_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(shift_sources, shift_source_count)
    };
    let output_slice = if output_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(output_bits, output_count)
    };
    match lfsr_transition_trace_for_times(
        times_slice,
        value_slice,
        signal_count,
        clk_vlo,
        clk_vhi,
        clk_period,
        clk_duty,
        clk_rise,
        clk_fall,
        clk_delay,
        clk_width,
        clk_has_width != 0,
        rst_vlo,
        rst_vhi,
        rst_period,
        rst_duty,
        rst_rise,
        rst_fall,
        rst_delay,
        rst_width,
        rst_has_width != 0,
        en_voltage,
        vdd,
        vth,
        trf,
        td,
        seed,
        width,
        tap_slice,
        shift_slice,
        output_slice,
        zero_guard_index,
    ) {
        Ok(count) => {
            *event_count = count;
            0
        }
        Err(code) => code,
    }
}

#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn evas_rust_gain_timer_reduction_trace(
    times: *const f64,
    point_count: usize,
    sample_times: *const f64,
    sample_count: usize,
    values: *mut f64,
    value_count: usize,
    signal_count: usize,
    point_vdd: *const f64,
    point_vss: *const f64,
    point_vinp: *const f64,
    point_vinn: *const f64,
    point_voutp: *const f64,
    point_voutn: *const f64,
    sample_vdd: *const f64,
    sample_vss: *const f64,
    sample_vinp: *const f64,
    sample_vinn: *const f64,
    sample_voutp: *const f64,
    sample_voutn: *const f64,
    start_time: f64,
    gain_scale: f64,
    min_input_span: f64,
    tedge: f64,
    sample_event_count: *mut usize,
) -> i32 {
    if point_count > 0 && times.is_null() {
        return -381;
    }
    if sample_count > 0 && sample_times.is_null() {
        return -382;
    }
    if value_count > 0 && values.is_null() {
        return -383;
    }
    for (idx, ptr) in [
        point_vdd,
        point_vss,
        point_vinp,
        point_vinn,
        point_voutp,
        point_voutn,
    ]
    .iter()
    .enumerate()
    {
        if point_count > 0 && ptr.is_null() {
            return -384 - idx as i32;
        }
    }
    for (idx, ptr) in [
        sample_vdd,
        sample_vss,
        sample_vinp,
        sample_vinn,
        sample_voutp,
        sample_voutn,
    ]
    .iter()
    .enumerate()
    {
        if sample_count > 0 && ptr.is_null() {
            return -391 - idx as i32;
        }
    }
    if sample_event_count.is_null() {
        return -397;
    }

    let times_slice = if point_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(times, point_count)
    };
    let sample_time_slice = if sample_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(sample_times, sample_count)
    };
    let value_slice = if value_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(values, value_count)
    };
    let point_vdd_slice = if point_count == 0 { &[] } else { std::slice::from_raw_parts(point_vdd, point_count) };
    let point_vss_slice = if point_count == 0 { &[] } else { std::slice::from_raw_parts(point_vss, point_count) };
    let point_vinp_slice = if point_count == 0 { &[] } else { std::slice::from_raw_parts(point_vinp, point_count) };
    let point_vinn_slice = if point_count == 0 { &[] } else { std::slice::from_raw_parts(point_vinn, point_count) };
    let point_voutp_slice = if point_count == 0 { &[] } else { std::slice::from_raw_parts(point_voutp, point_count) };
    let point_voutn_slice = if point_count == 0 { &[] } else { std::slice::from_raw_parts(point_voutn, point_count) };
    let sample_vdd_slice = if sample_count == 0 { &[] } else { std::slice::from_raw_parts(sample_vdd, sample_count) };
    let sample_vss_slice = if sample_count == 0 { &[] } else { std::slice::from_raw_parts(sample_vss, sample_count) };
    let sample_vinp_slice = if sample_count == 0 { &[] } else { std::slice::from_raw_parts(sample_vinp, sample_count) };
    let sample_vinn_slice = if sample_count == 0 { &[] } else { std::slice::from_raw_parts(sample_vinn, sample_count) };
    let sample_voutp_slice = if sample_count == 0 { &[] } else { std::slice::from_raw_parts(sample_voutp, sample_count) };
    let sample_voutn_slice = if sample_count == 0 { &[] } else { std::slice::from_raw_parts(sample_voutn, sample_count) };

    match gain_timer_reduction_trace_for_arrays(
        times_slice,
        sample_time_slice,
        value_slice,
        signal_count,
        point_vdd_slice,
        point_vss_slice,
        point_vinp_slice,
        point_vinn_slice,
        point_voutp_slice,
        point_voutn_slice,
        sample_vdd_slice,
        sample_vss_slice,
        sample_vinp_slice,
        sample_vinn_slice,
        sample_voutp_slice,
        sample_voutn_slice,
        start_time,
        gain_scale,
        min_input_span,
        tedge,
    ) {
        Ok(count) => {
            *sample_event_count = count;
            0
        }
        Err(code) => code,
    }
}

#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn evas_rust_gain_measurement_flow_trace(
    times: *const f64,
    point_count: usize,
    values: *mut f64,
    value_count: usize,
    signal_count: usize,
    vin_event_times: *const f64,
    vin_event_count: usize,
    vin_event_vinp: *const f64,
    vin_event_vinn: *const f64,
    lfsr_event_times: *const f64,
    lfsr_event_count: usize,
    vcm: f64,
    vth: f64,
    dither_amp: f64,
    actual_gain: f64,
    vin_transition: f64,
    lfsr_transition: f64,
    vdd: f64,
    vss: f64,
    lfsr_seed: i64,
    vin_event_out_count: *mut usize,
    lfsr_event_out_count: *mut usize,
) -> i32 {
    if point_count > 0 && times.is_null() {
        return -531;
    }
    if value_count > 0 && values.is_null() {
        return -532;
    }
    if vin_event_count > 0
        && (vin_event_times.is_null() || vin_event_vinp.is_null() || vin_event_vinn.is_null())
    {
        return -533;
    }
    if lfsr_event_count > 0 && lfsr_event_times.is_null() {
        return -534;
    }
    if vin_event_out_count.is_null() || lfsr_event_out_count.is_null() {
        return -535;
    }

    let times_slice = if point_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(times, point_count)
    };
    let value_slice = if value_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(values, value_count)
    };
    let vin_time_slice = if vin_event_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(vin_event_times, vin_event_count)
    };
    let vin_vinp_slice = if vin_event_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(vin_event_vinp, vin_event_count)
    };
    let vin_vinn_slice = if vin_event_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(vin_event_vinn, vin_event_count)
    };
    let lfsr_time_slice = if lfsr_event_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(lfsr_event_times, lfsr_event_count)
    };

    match gain_measurement_flow_trace_for_arrays(
        times_slice,
        value_slice,
        signal_count,
        vin_time_slice,
        vin_vinp_slice,
        vin_vinn_slice,
        lfsr_time_slice,
        vcm,
        vth,
        dither_amp,
        actual_gain,
        vin_transition,
        lfsr_transition,
        vdd,
        vss,
        lfsr_seed,
    ) {
        Ok((vin_events, lfsr_events)) => {
            *vin_event_out_count = vin_events;
            *lfsr_event_out_count = lfsr_events;
            0
        }
        Err(code) => code,
    }
}

#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn evas_rust_cmp_delay_trace(
    times: *const f64,
    point_count: usize,
    values: *mut f64,
    value_count: usize,
    signal_count: usize,
    point_clk: *const f64,
    point_vinn: *const f64,
    point_vinp: *const f64,
    point_vdd: *const f64,
    voffset: f64,
    tau: f64,
    td0: f64,
    td_min: f64,
    td_max: f64,
    tedge: f64,
    edge_vth: f64,
    clock_event_count: *mut usize,
) -> i32 {
    if point_count > 0 && times.is_null() {
        return -431;
    }
    if value_count > 0 && values.is_null() {
        return -432;
    }
    for (idx, ptr) in [point_clk, point_vinn, point_vinp, point_vdd]
        .iter()
        .enumerate()
    {
        if point_count > 0 && ptr.is_null() {
            return -433 - idx as i32;
        }
    }
    if clock_event_count.is_null() {
        return -437;
    }

    let times_slice = if point_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(times, point_count)
    };
    let value_slice = if value_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(values, value_count)
    };
    let point_clk_slice = if point_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(point_clk, point_count)
    };
    let point_vinn_slice = if point_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(point_vinn, point_count)
    };
    let point_vinp_slice = if point_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(point_vinp, point_count)
    };
    let point_vdd_slice = if point_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(point_vdd, point_count)
    };

    match cmp_delay_trace_for_arrays(
        times_slice,
        value_slice,
        signal_count,
        point_clk_slice,
        point_vinn_slice,
        point_vinp_slice,
        point_vdd_slice,
        voffset,
        tau,
        td0,
        td_min,
        td_max,
        tedge,
        edge_vth,
    ) {
        Ok(count) => {
            *clock_event_count = count;
            0
        }
        Err(code) => code,
    }
}

#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn evas_rust_sar_loop_trace(
    times: *const f64,
    point_count: usize,
    values: *mut f64,
    value_count: usize,
    signal_count: usize,
    point_vin: *const f64,
    point_clk: *const f64,
    point_rst: *const f64,
    vdd: f64,
    vth: f64,
    sh_tr: f64,
    default_tr: f64,
    width: usize,
    clock_event_count: *mut usize,
) -> i32 {
    if point_count > 0 && times.is_null() {
        return -471;
    }
    if value_count > 0 && values.is_null() {
        return -472;
    }
    for (idx, ptr) in [point_vin, point_clk, point_rst].iter().enumerate() {
        if point_count > 0 && ptr.is_null() {
            return -473 - idx as i32;
        }
    }
    if clock_event_count.is_null() {
        return -476;
    }

    let times_slice = if point_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(times, point_count)
    };
    let value_slice = if value_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(values, value_count)
    };
    let point_vin_slice = if point_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(point_vin, point_count)
    };
    let point_clk_slice = if point_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(point_clk, point_count)
    };
    let point_rst_slice = if point_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(point_rst, point_count)
    };

    match sar_loop_trace_for_arrays(
        times_slice,
        value_slice,
        signal_count,
        point_vin_slice,
        point_clk_slice,
        point_rst_slice,
        vdd,
        vth,
        sh_tr,
        default_tr,
        width,
    ) {
        Ok(count) => {
            *clock_event_count = count;
            0
        }
        Err(code) => code,
    }
}

#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn evas_rust_cppll_reacquire_trace(
    times: *const f64,
    point_count: usize,
    values: *mut f64,
    value_count: usize,
    signal_count: usize,
    ref_event_times: *const f64,
    ref_event_values: *const f64,
    ref_event_count: usize,
    dco_event_times: *const f64,
    dco_event_values: *const f64,
    dco_event_count: usize,
    fb_event_times: *const f64,
    fb_event_values: *const f64,
    fb_event_count: usize,
    lock_event_times: *const f64,
    lock_event_values: *const f64,
    lock_event_count: usize,
    vctrl_event_times: *const f64,
    vctrl_event_values: *const f64,
    vctrl_event_count: usize,
    vh: f64,
    vl: f64,
    ref_tedge: f64,
    pll_tedge: f64,
) -> i32 {
    if point_count > 0 && times.is_null() {
        return -501;
    }
    if value_count > 0 && values.is_null() {
        return -502;
    }
    for (idx, (time_ptr, value_ptr, count)) in [
        (ref_event_times, ref_event_values, ref_event_count),
        (dco_event_times, dco_event_values, dco_event_count),
        (fb_event_times, fb_event_values, fb_event_count),
        (lock_event_times, lock_event_values, lock_event_count),
        (vctrl_event_times, vctrl_event_values, vctrl_event_count),
    ]
    .iter()
    .enumerate()
    {
        if *count > 0 && (time_ptr.is_null() || value_ptr.is_null()) {
            return -503 - idx as i32;
        }
    }

    let times_slice = if point_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(times, point_count)
    };
    let value_slice = if value_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(values, value_count)
    };
    let ref_time_slice = if ref_event_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(ref_event_times, ref_event_count)
    };
    let ref_value_slice = if ref_event_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(ref_event_values, ref_event_count)
    };
    let dco_time_slice = if dco_event_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(dco_event_times, dco_event_count)
    };
    let dco_value_slice = if dco_event_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(dco_event_values, dco_event_count)
    };
    let fb_time_slice = if fb_event_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(fb_event_times, fb_event_count)
    };
    let fb_value_slice = if fb_event_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(fb_event_values, fb_event_count)
    };
    let lock_time_slice = if lock_event_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(lock_event_times, lock_event_count)
    };
    let lock_value_slice = if lock_event_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(lock_event_values, lock_event_count)
    };
    let vctrl_time_slice = if vctrl_event_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(vctrl_event_times, vctrl_event_count)
    };
    let vctrl_value_slice = if vctrl_event_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(vctrl_event_values, vctrl_event_count)
    };

    match cppll_reacquire_trace_for_arrays(
        times_slice,
        value_slice,
        signal_count,
        ref_time_slice,
        ref_value_slice,
        dco_time_slice,
        dco_value_slice,
        fb_time_slice,
        fb_value_slice,
        lock_time_slice,
        lock_value_slice,
        vctrl_time_slice,
        vctrl_value_slice,
        vh,
        vl,
        ref_tedge,
        pll_tedge,
    ) {
        Ok(()) => 0,
        Err(code) => code,
    }
}

#[no_mangle]
pub unsafe extern "C" fn evas_rust_evaluate_transition_targets(
    ops: *const EvasRustTransitionTargetOp,
    op_count: usize,
    terms: *const EvasRustLinearTerm,
    term_count: usize,
    conditions: *const EvasRustLinearCondition,
    condition_count: usize,
    node_values: *const f64,
    node_value_count: usize,
    state_values: *const f64,
    state_value_count: usize,
    target_values: *mut f64,
    target_value_count: usize,
    delay_values: *mut f64,
    delay_value_count: usize,
    rise_values: *mut f64,
    rise_value_count: usize,
    fall_values: *mut f64,
    fall_value_count: usize,
) -> i32 {
    if op_count > 0 && ops.is_null() {
        return -67;
    }
    if term_count > 0 && terms.is_null() {
        return -68;
    }
    if condition_count > 0 && conditions.is_null() {
        return -69;
    }
    if node_value_count > 0 && node_values.is_null() {
        return -70;
    }
    if state_value_count > 0 && state_values.is_null() {
        return -71;
    }
    if target_value_count > 0 && target_values.is_null() {
        return -72;
    }
    if delay_value_count > 0 && delay_values.is_null() {
        return -73;
    }
    if rise_value_count > 0 && rise_values.is_null() {
        return -74;
    }
    if fall_value_count > 0 && fall_values.is_null() {
        return -75;
    }

    let op_slice = if op_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(ops, op_count)
    };
    let term_slice = if term_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(terms, term_count)
    };
    let condition_slice = if condition_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(conditions, condition_count)
    };
    let node_value_slice = if node_value_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(node_values, node_value_count)
    };
    let state_value_slice = if state_value_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(state_values, state_value_count)
    };
    let target_value_slice = if target_value_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(target_values, target_value_count)
    };
    let delay_value_slice = if delay_value_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(delay_values, delay_value_count)
    };
    let rise_value_slice = if rise_value_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(rise_values, rise_value_count)
    };
    let fall_value_slice = if fall_value_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(fall_values, fall_value_count)
    };

    match evaluate_transition_target_ops(
        op_slice,
        term_slice,
        condition_slice,
        node_value_slice,
        state_value_slice,
        target_value_slice,
        delay_value_slice,
        rise_value_slice,
        fall_value_slice,
    ) {
        Ok(()) => 0,
        Err(code) => code,
    }
}

#[no_mangle]
pub unsafe extern "C" fn evas_rust_evaluate_ordered_transition_segment(
    linear_ops: *const EvasRustLinearOp,
    linear_op_count: usize,
    linear_terms: *const EvasRustLinearTerm,
    linear_term_count: usize,
    linear_conditions: *const EvasRustLinearCondition,
    linear_condition_count: usize,
    transition_ops: *const EvasRustTransitionTargetOp,
    transition_op_count: usize,
    transition_terms: *const EvasRustLinearTerm,
    transition_term_count: usize,
    transition_conditions: *const EvasRustLinearCondition,
    transition_condition_count: usize,
    node_values: *mut f64,
    node_value_count: usize,
    state_values: *mut f64,
    state_value_count: usize,
    target_values: *mut f64,
    target_value_count: usize,
    delay_values: *mut f64,
    delay_value_count: usize,
    rise_values: *mut f64,
    rise_value_count: usize,
    fall_values: *mut f64,
    fall_value_count: usize,
) -> i32 {
    if linear_op_count > 0 && linear_ops.is_null() {
        return -90;
    }
    if linear_term_count > 0 && linear_terms.is_null() {
        return -91;
    }
    if linear_condition_count > 0 && linear_conditions.is_null() {
        return -92;
    }
    if transition_op_count > 0 && transition_ops.is_null() {
        return -93;
    }
    if transition_term_count > 0 && transition_terms.is_null() {
        return -94;
    }
    if transition_condition_count > 0 && transition_conditions.is_null() {
        return -95;
    }
    if node_value_count > 0 && node_values.is_null() {
        return -96;
    }
    if state_value_count > 0 && state_values.is_null() {
        return -97;
    }
    if target_value_count > 0 && target_values.is_null() {
        return -98;
    }
    if delay_value_count > 0 && delay_values.is_null() {
        return -99;
    }
    if rise_value_count > 0 && rise_values.is_null() {
        return -100;
    }
    if fall_value_count > 0 && fall_values.is_null() {
        return -101;
    }

    let linear_op_slice = if linear_op_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(linear_ops, linear_op_count)
    };
    let linear_term_slice = if linear_term_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(linear_terms, linear_term_count)
    };
    let linear_condition_slice = if linear_condition_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(linear_conditions, linear_condition_count)
    };
    let transition_op_slice = if transition_op_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(transition_ops, transition_op_count)
    };
    let transition_term_slice = if transition_term_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(transition_terms, transition_term_count)
    };
    let transition_condition_slice = if transition_condition_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(transition_conditions, transition_condition_count)
    };
    let node_value_slice = if node_value_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(node_values, node_value_count)
    };
    let state_value_slice = if state_value_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(state_values, state_value_count)
    };
    let target_value_slice = if target_value_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(target_values, target_value_count)
    };
    let delay_value_slice = if delay_value_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(delay_values, delay_value_count)
    };
    let rise_value_slice = if rise_value_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(rise_values, rise_value_count)
    };
    let fall_value_slice = if fall_value_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(fall_values, fall_value_count)
    };

    match evaluate_ordered_transition_segment(
        linear_op_slice,
        linear_term_slice,
        linear_condition_slice,
        transition_op_slice,
        transition_term_slice,
        transition_condition_slice,
        node_value_slice,
        state_value_slice,
        target_value_slice,
        delay_value_slice,
        rise_value_slice,
        fall_value_slice,
    ) {
        Ok(()) => 0,
        Err(code) => code,
    }
}

#[no_mangle]
pub unsafe extern "C" fn evas_rust_next_transition_breakpoint(
    start_times: *const f64,
    start_value_count: usize,
    start_values: *const f64,
    target_values: *const f64,
    delays: *const f64,
    rise_times: *const f64,
    fall_times: *const f64,
    active_flags: *const u8,
    time: f64,
    min_ramp_time: f64,
    out_found: *mut u8,
    out_time: *mut f64,
) -> i32 {
    if start_value_count > 0 && start_times.is_null() {
        return -52;
    }
    if start_value_count > 0 && start_values.is_null() {
        return -53;
    }
    if start_value_count > 0 && target_values.is_null() {
        return -54;
    }
    if start_value_count > 0 && delays.is_null() {
        return -55;
    }
    if start_value_count > 0 && rise_times.is_null() {
        return -56;
    }
    if start_value_count > 0 && fall_times.is_null() {
        return -57;
    }
    if start_value_count > 0 && active_flags.is_null() {
        return -58;
    }
    if out_found.is_null() {
        return -59;
    }
    if out_time.is_null() {
        return -60;
    }

    let start_time_slice = if start_value_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(start_times, start_value_count)
    };
    let start_value_slice = if start_value_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(start_values, start_value_count)
    };
    let target_value_slice = if start_value_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(target_values, start_value_count)
    };
    let delay_slice = if start_value_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(delays, start_value_count)
    };
    let rise_time_slice = if start_value_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(rise_times, start_value_count)
    };
    let fall_time_slice = if start_value_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(fall_times, start_value_count)
    };
    let active_flag_slice = if start_value_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(active_flags, start_value_count)
    };

    match next_transition_breakpoint_for_arrays(
        start_time_slice,
        start_value_slice,
        target_value_slice,
        delay_slice,
        rise_time_slice,
        fall_time_slice,
        active_flag_slice,
        time,
        min_ramp_time,
    ) {
        Ok(Some(bp)) => {
            *out_found = 1;
            *out_time = bp;
            0
        }
        Ok(None) => {
            *out_found = 0;
            *out_time = 0.0;
            0
        }
        Err(code) => code,
    }
}

#[no_mangle]
pub unsafe extern "C" fn evas_rust_timer_periodic_step(
    next_fire_times: *mut f64,
    timer_count: usize,
    has_state_flags: *mut u8,
    periods: *const f64,
    starts: *const f64,
    has_start_flags: *const u8,
    due_flags: *mut u8,
    skipped_flags: *mut u8,
    time: f64,
    reschedule_on_due: u8,
    eps: f64,
) -> i32 {
    if timer_count > 0 && next_fire_times.is_null() {
        return -132;
    }
    if timer_count > 0 && has_state_flags.is_null() {
        return -133;
    }
    if timer_count > 0 && periods.is_null() {
        return -134;
    }
    if timer_count > 0 && starts.is_null() {
        return -135;
    }
    if timer_count > 0 && has_start_flags.is_null() {
        return -136;
    }
    if timer_count > 0 && due_flags.is_null() {
        return -137;
    }
    if timer_count > 0 && skipped_flags.is_null() {
        return -138;
    }

    let next_fire_slice = if timer_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(next_fire_times, timer_count)
    };
    let has_state_slice = if timer_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(has_state_flags, timer_count)
    };
    let period_slice = if timer_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(periods, timer_count)
    };
    let start_slice = if timer_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(starts, timer_count)
    };
    let has_start_slice = if timer_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(has_start_flags, timer_count)
    };
    let due_slice = if timer_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(due_flags, timer_count)
    };
    let skipped_slice = if timer_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(skipped_flags, timer_count)
    };

    match timer_periodic_step_for_arrays(
        next_fire_slice,
        has_state_slice,
        period_slice,
        start_slice,
        has_start_slice,
        due_slice,
        skipped_slice,
        time,
        reschedule_on_due != 0,
        eps,
    ) {
        Ok(()) => 0,
        Err(code) => code,
    }
}

#[no_mangle]
pub unsafe extern "C" fn evas_rust_timer_absolute_step(
    next_fire_times: *mut f64,
    timer_count: usize,
    has_state_flags: *mut u8,
    last_fired_times: *mut f64,
    has_last_fired_flags: *mut u8,
    targets: *const f64,
    due_flags: *mut u8,
    expired_flags: *mut u8,
    time: f64,
    eps: f64,
) -> i32 {
    if timer_count > 0 && next_fire_times.is_null() {
        return -142;
    }
    if timer_count > 0 && has_state_flags.is_null() {
        return -143;
    }
    if timer_count > 0 && last_fired_times.is_null() {
        return -144;
    }
    if timer_count > 0 && has_last_fired_flags.is_null() {
        return -145;
    }
    if timer_count > 0 && targets.is_null() {
        return -146;
    }
    if timer_count > 0 && due_flags.is_null() {
        return -147;
    }
    if timer_count > 0 && expired_flags.is_null() {
        return -148;
    }

    let next_fire_slice = if timer_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(next_fire_times, timer_count)
    };
    let has_state_slice = if timer_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(has_state_flags, timer_count)
    };
    let last_fired_slice = if timer_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(last_fired_times, timer_count)
    };
    let has_last_fired_slice = if timer_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(has_last_fired_flags, timer_count)
    };
    let target_slice = if timer_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(targets, timer_count)
    };
    let due_slice = if timer_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(due_flags, timer_count)
    };
    let expired_slice = if timer_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(expired_flags, timer_count)
    };

    match timer_absolute_step_for_arrays(
        next_fire_slice,
        has_state_slice,
        last_fired_slice,
        has_last_fired_slice,
        target_slice,
        due_slice,
        expired_slice,
        time,
        eps,
    ) {
        Ok(()) => 0,
        Err(code) => code,
    }
}

#[no_mangle]
pub unsafe extern "C" fn evas_rust_cross_detector_step(
    prev_values: *mut f64,
    detector_count: usize,
    prev_times: *mut f64,
    pprev_values: *mut f64,
    pprev_times: *mut f64,
    initialized_flags: *mut u8,
    directions: *const i32,
    last_cross_times: *mut f64,
    current_values: *const f64,
    triggered_flags: *mut u8,
    cross_times: *mut f64,
    trigger_directions: *mut i32,
    went_beyond_flags: *mut u8,
    time: f64,
    time_tol: f64,
    expr_tol: f64,
) -> i32 {
    if detector_count > 0 && prev_values.is_null() {
        return -162;
    }
    if detector_count > 0 && prev_times.is_null() {
        return -163;
    }
    if detector_count > 0 && pprev_values.is_null() {
        return -164;
    }
    if detector_count > 0 && pprev_times.is_null() {
        return -165;
    }
    if detector_count > 0 && initialized_flags.is_null() {
        return -166;
    }
    if detector_count > 0 && directions.is_null() {
        return -167;
    }
    if detector_count > 0 && last_cross_times.is_null() {
        return -168;
    }
    if detector_count > 0 && current_values.is_null() {
        return -169;
    }
    if detector_count > 0 && triggered_flags.is_null() {
        return -170;
    }
    if detector_count > 0 && cross_times.is_null() {
        return -171;
    }
    if detector_count > 0 && trigger_directions.is_null() {
        return -172;
    }
    if detector_count > 0 && went_beyond_flags.is_null() {
        return -173;
    }

    let prev_value_slice = if detector_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(prev_values, detector_count)
    };
    let prev_time_slice = if detector_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(prev_times, detector_count)
    };
    let pprev_value_slice = if detector_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(pprev_values, detector_count)
    };
    let pprev_time_slice = if detector_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(pprev_times, detector_count)
    };
    let initialized_slice = if detector_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(initialized_flags, detector_count)
    };
    let direction_slice = if detector_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(directions, detector_count)
    };
    let last_cross_slice = if detector_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(last_cross_times, detector_count)
    };
    let current_value_slice = if detector_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(current_values, detector_count)
    };
    let triggered_slice = if detector_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(triggered_flags, detector_count)
    };
    let cross_time_slice = if detector_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(cross_times, detector_count)
    };
    let trigger_direction_slice = if detector_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(trigger_directions, detector_count)
    };
    let went_beyond_slice = if detector_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(went_beyond_flags, detector_count)
    };

    match cross_detector_step_for_arrays(
        prev_value_slice,
        prev_time_slice,
        pprev_value_slice,
        pprev_time_slice,
        initialized_slice,
        direction_slice,
        last_cross_slice,
        current_value_slice,
        triggered_slice,
        cross_time_slice,
        trigger_direction_slice,
        went_beyond_slice,
        time,
        time_tol,
        expr_tol,
    ) {
        Ok(()) => 0,
        Err(code) => code,
    }
}

#[no_mangle]
pub unsafe extern "C" fn evas_rust_above_detector_step(
    prev_values: *mut f64,
    detector_count: usize,
    prev_times: *mut f64,
    pprev_values: *mut f64,
    pprev_times: *mut f64,
    initialized_flags: *mut u8,
    directions: *const i32,
    current_values: *const f64,
    triggered_flags: *mut u8,
    cross_times: *mut f64,
    time: f64,
) -> i32 {
    if detector_count > 0 && prev_values.is_null() {
        return -182;
    }
    if detector_count > 0 && prev_times.is_null() {
        return -183;
    }
    if detector_count > 0 && pprev_values.is_null() {
        return -184;
    }
    if detector_count > 0 && pprev_times.is_null() {
        return -185;
    }
    if detector_count > 0 && initialized_flags.is_null() {
        return -186;
    }
    if detector_count > 0 && directions.is_null() {
        return -187;
    }
    if detector_count > 0 && current_values.is_null() {
        return -188;
    }
    if detector_count > 0 && triggered_flags.is_null() {
        return -189;
    }
    if detector_count > 0 && cross_times.is_null() {
        return -190;
    }

    let prev_value_slice = if detector_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(prev_values, detector_count)
    };
    let prev_time_slice = if detector_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(prev_times, detector_count)
    };
    let pprev_value_slice = if detector_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(pprev_values, detector_count)
    };
    let pprev_time_slice = if detector_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(pprev_times, detector_count)
    };
    let initialized_slice = if detector_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(initialized_flags, detector_count)
    };
    let direction_slice = if detector_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(directions, detector_count)
    };
    let current_value_slice = if detector_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(current_values, detector_count)
    };
    let triggered_slice = if detector_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(triggered_flags, detector_count)
    };
    let cross_time_slice = if detector_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(cross_times, detector_count)
    };

    match above_detector_step_for_arrays(
        prev_value_slice,
        prev_time_slice,
        pprev_value_slice,
        pprev_time_slice,
        initialized_slice,
        direction_slice,
        current_value_slice,
        triggered_slice,
        cross_time_slice,
        time,
    ) {
        Ok(()) => 0,
        Err(code) => code,
    }
}

#[no_mangle]
pub unsafe extern "C" fn evas_rust_dynamic_bus_offsets(
    base_offsets: *const usize,
    access_count: usize,
    outer_lengths: *const usize,
    inner_strides: *const usize,
    inner_lengths: *const usize,
    first_indices: *const i64,
    second_indices: *const i64,
    has_second_index_flags: *const u8,
    out_node_ids: *mut usize,
) -> i32 {
    if access_count > 0 && base_offsets.is_null() {
        return -212;
    }
    if access_count > 0 && outer_lengths.is_null() {
        return -213;
    }
    if access_count > 0 && inner_strides.is_null() {
        return -214;
    }
    if access_count > 0 && inner_lengths.is_null() {
        return -215;
    }
    if access_count > 0 && first_indices.is_null() {
        return -216;
    }
    if access_count > 0 && second_indices.is_null() {
        return -217;
    }
    if access_count > 0 && has_second_index_flags.is_null() {
        return -218;
    }
    if access_count > 0 && out_node_ids.is_null() {
        return -219;
    }

    let base_slice = if access_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(base_offsets, access_count)
    };
    let outer_slice = if access_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(outer_lengths, access_count)
    };
    let stride_slice = if access_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(inner_strides, access_count)
    };
    let inner_slice = if access_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(inner_lengths, access_count)
    };
    let first_slice = if access_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(first_indices, access_count)
    };
    let second_slice = if access_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(second_indices, access_count)
    };
    let has_second_slice = if access_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(has_second_index_flags, access_count)
    };
    let out_slice = if access_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(out_node_ids, access_count)
    };

    match dynamic_bus_offsets_for_arrays(
        base_slice,
        outer_slice,
        stride_slice,
        inner_slice,
        first_slice,
        second_slice,
        has_second_slice,
        out_slice,
    ) {
        Ok(()) => 0,
        Err(code) => code,
    }
}

#[no_mangle]
pub unsafe extern "C" fn evas_rust_transition_state_step(
    current_values: *mut f64,
    state_count: usize,
    target_values: *mut f64,
    start_times: *mut f64,
    start_values: *mut f64,
    delays: *mut f64,
    rise_times: *mut f64,
    fall_times: *mut f64,
    active_flags: *mut u8,
    initialized_flags: *mut u8,
    input_targets: *const f64,
    input_delays: *const f64,
    input_rises: *const f64,
    input_falls: *const f64,
    output_values: *mut f64,
    time: f64,
    default_transition: f64,
    initial_condition_mode: u8,
) -> i32 {
    if state_count > 0 && current_values.is_null() {
        return -112;
    }
    if state_count > 0 && target_values.is_null() {
        return -113;
    }
    if state_count > 0 && start_times.is_null() {
        return -114;
    }
    if state_count > 0 && start_values.is_null() {
        return -115;
    }
    if state_count > 0 && delays.is_null() {
        return -116;
    }
    if state_count > 0 && rise_times.is_null() {
        return -117;
    }
    if state_count > 0 && fall_times.is_null() {
        return -118;
    }
    if state_count > 0 && active_flags.is_null() {
        return -119;
    }
    if state_count > 0 && initialized_flags.is_null() {
        return -120;
    }
    if state_count > 0 && input_targets.is_null() {
        return -121;
    }
    if state_count > 0 && input_delays.is_null() {
        return -122;
    }
    if state_count > 0 && input_rises.is_null() {
        return -123;
    }
    if state_count > 0 && input_falls.is_null() {
        return -124;
    }
    if state_count > 0 && output_values.is_null() {
        return -125;
    }

    let current_slice = if state_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(current_values, state_count)
    };
    let target_slice = if state_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(target_values, state_count)
    };
    let start_time_slice = if state_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(start_times, state_count)
    };
    let start_value_slice = if state_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(start_values, state_count)
    };
    let delay_slice = if state_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(delays, state_count)
    };
    let rise_time_slice = if state_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(rise_times, state_count)
    };
    let fall_time_slice = if state_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(fall_times, state_count)
    };
    let active_slice = if state_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(active_flags, state_count)
    };
    let initialized_slice = if state_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(initialized_flags, state_count)
    };
    let input_target_slice = if state_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(input_targets, state_count)
    };
    let input_delay_slice = if state_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(input_delays, state_count)
    };
    let input_rise_slice = if state_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(input_rises, state_count)
    };
    let input_fall_slice = if state_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(input_falls, state_count)
    };
    let output_slice = if state_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(output_values, state_count)
    };

    match transition_state_step_for_arrays(
        current_slice,
        target_slice,
        start_time_slice,
        start_value_slice,
        delay_slice,
        rise_time_slice,
        fall_time_slice,
        active_slice,
        initialized_slice,
        input_target_slice,
        input_delay_slice,
        input_rise_slice,
        input_fall_slice,
        output_slice,
        time,
        default_transition,
        initial_condition_mode != 0,
    ) {
        Ok(()) => 0,
        Err(code) => code,
    }
}

#[no_mangle]
pub unsafe extern "C" fn evas_rust_next_timer_breakpoint(
    next_fire_times: *const f64,
    timer_count: usize,
    last_fired_times: *const f64,
    has_last_fired_flags: *const u8,
    time: f64,
    out_found: *mut u8,
    out_time: *mut f64,
) -> i32 {
    if timer_count > 0 && next_fire_times.is_null() {
        return -82;
    }
    if timer_count > 0 && last_fired_times.is_null() {
        return -83;
    }
    if timer_count > 0 && has_last_fired_flags.is_null() {
        return -84;
    }
    if out_found.is_null() {
        return -85;
    }
    if out_time.is_null() {
        return -86;
    }

    let next_fire_slice = if timer_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(next_fire_times, timer_count)
    };
    let last_fired_slice = if timer_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(last_fired_times, timer_count)
    };
    let has_last_fired_slice = if timer_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(has_last_fired_flags, timer_count)
    };

    match next_timer_breakpoint_for_arrays(
        next_fire_slice,
        last_fired_slice,
        has_last_fired_slice,
        time,
    ) {
        Ok(Some(bp)) => {
            *out_found = 1;
            *out_time = bp;
            0
        }
        Ok(None) => {
            *out_found = 0;
            *out_time = 0.0;
            0
        }
        Err(code) => code,
    }
}

#[no_mangle]
pub unsafe extern "C" fn evas_rust_copy_f64(
    source: *const f64,
    source_count: usize,
    target: *mut f64,
    target_count: usize,
) -> i32 {
    if source_count > 0 && source.is_null() {
        return -32;
    }
    if target_count > 0 && target.is_null() {
        return -33;
    }

    let source_slice = if source_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(source, source_count)
    };
    let target_slice = if target_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(target, target_count)
    };

    match copy_f64_values(source_slice, target_slice) {
        Ok(()) => 0,
        Err(code) => code,
    }
}

#[no_mangle]
pub unsafe extern "C" fn evas_rust_max_err_ratio(
    values: *const f64,
    value_count: usize,
    previous: *const f64,
    previous_count: usize,
    node_ids: *const usize,
    node_count: usize,
    reltol: f64,
    vabstol: f64,
    out_ratio: *mut f64,
) -> i32 {
    if value_count > 0 && values.is_null() {
        return -43;
    }
    if previous_count > 0 && previous.is_null() {
        return -44;
    }
    if node_count > 0 && node_ids.is_null() {
        return -45;
    }
    if out_ratio.is_null() {
        return -46;
    }

    let value_slice = if value_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(values, value_count)
    };
    let previous_slice = if previous_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(previous, previous_count)
    };
    let node_slice = if node_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(node_ids, node_count)
    };

    match max_err_ratio_for_nodes(
        value_slice,
        previous_slice,
        node_slice,
        reltol,
        vabstol,
    ) {
        Ok(ratio) => {
            *out_ratio = ratio;
            0
        }
        Err(code) => code,
    }
}

#[no_mangle]
pub unsafe extern "C" fn evas_rust_interpolate_event_values(
    previous_values: *const f64,
    previous_count: usize,
    current_values: *const f64,
    current_count: usize,
    out_values: *mut f64,
    out_count: usize,
    previous_time: f64,
    current_time: f64,
    event_time: f64,
) -> i32 {
    if previous_count > 0 && previous_values.is_null() {
        return -184;
    }
    if current_count > 0 && current_values.is_null() {
        return -185;
    }
    if out_count > 0 && out_values.is_null() {
        return -186;
    }

    let previous_slice = if previous_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(previous_values, previous_count)
    };
    let current_slice = if current_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(current_values, current_count)
    };
    let out_slice = if out_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(out_values, out_count)
    };

    match interpolate_event_values_for_arrays(
        previous_slice,
        current_slice,
        out_slice,
        previous_time,
        current_time,
        event_time,
    ) {
        Ok(()) => 0,
        Err(code) => code,
    }
}

#[no_mangle]
pub unsafe extern "C" fn evas_rust_record_values_for_node_ids(
    values: *const f64,
    value_count: usize,
    node_ids: *const usize,
    node_count: usize,
    default_value: f64,
    out_values: *mut f64,
    out_count: usize,
) -> i32 {
    if value_count > 0 && values.is_null() {
        return -52;
    }
    if node_count > 0 && node_ids.is_null() {
        return -53;
    }
    if out_count > 0 && out_values.is_null() {
        return -54;
    }

    let value_slice = if value_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(values, value_count)
    };
    let node_slice = if node_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(node_ids, node_count)
    };
    let out_slice = if out_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(out_values, out_count)
    };

    match record_values_for_node_ids(value_slice, node_slice, default_value, out_slice) {
        Ok(()) => 0,
        Err(code) => code,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evaluates_static_affine_batch_in_order() {
        let ops = [
            EvasRustStaticAffineOp {
                read_node_id: 0,
                write_node_id: 1,
                gain: 2.0,
                bias: 0.25,
            },
            EvasRustStaticAffineOp {
                read_node_id: 1,
                write_node_id: 2,
                gain: -1.0,
                bias: 1.0,
            },
        ];
        let mut values = [0.5, 0.0, 0.0];

        evaluate_static_affine_ops(&ops, &mut values).unwrap();

        assert_eq!(values, [0.5, 1.25, -0.25]);
    }

    #[test]
    fn copies_f64_values() {
        let source = [0.25, 0.5, -1.0];
        let mut target = [0.0, 0.0, 0.0];

        copy_f64_values(&source, &mut target).unwrap();

        assert_eq!(target, source);
    }

    #[test]
    fn prbs7_trace_generates_clocked_state_bus() {
        let times = [
            0.0,
            0.1e-9,
            0.11e-9,
            0.12e-9,
            1.11e-9,
            1.12e-9,
        ];
        let mut values = vec![0.0; times.len() * 11];

        let events = prbs7_trace_for_times(
            &times,
            &mut values,
            11,
            0.0,
            0.9,
            1.0e-9,
            0.5,
            20.0e-12,
            20.0e-12,
            0.1e-9,
            0.5e-9,
            true,
            0.0,
            0.9,
            300.0e-9,
            298.0 / 300.0,
            20.0e-12,
            20.0e-12,
            2.0e-9,
            298.0e-9,
            true,
            0.9,
            0.9,
            0.45,
            10.0e-12,
            0.0,
            127,
        )
        .unwrap();

        assert_eq!(events, 2);
        assert!((values[2 * 11] - 0.45).abs() < 1.0e-12);
        assert!((values[5 * 11 + 3] - 0.9).abs() < 1.0e-12);
        for idx in 4..11 {
            assert!(values[5 * 11 + idx] >= -1.0e-12);
            assert!(values[5 * 11 + idx] <= 0.9 + 1.0e-12);
        }
    }

    #[test]
    fn transition_state_step_initializes_missing_state() {
        let mut current = [0.0];
        let mut target = [0.0];
        let mut start_time = [0.0];
        let mut start_value = [0.0];
        let mut delay = [0.0];
        let mut rise = [0.0];
        let mut fall = [0.0];
        let mut active = [0_u8];
        let mut initialized = [0_u8];
        let input_target = [1.2];
        let input_delay = [2.0e-9];
        let input_rise = [0.0];
        let input_fall = [3.0e-9];
        let mut output = [0.0];

        transition_state_step_for_arrays(
            &mut current,
            &mut target,
            &mut start_time,
            &mut start_value,
            &mut delay,
            &mut rise,
            &mut fall,
            &mut active,
            &mut initialized,
            &input_target,
            &input_delay,
            &input_rise,
            &input_fall,
            &mut output,
            5.0e-9,
            1.0e-12,
            false,
        )
        .unwrap();

        assert_eq!(initialized[0], 1);
        assert_eq!(active[0], 0);
        assert_eq!(current[0], 1.2);
        assert_eq!(target[0], 1.2);
        assert_eq!(start_value[0], 1.2);
        assert_eq!(start_time[0], 5.0e-9);
        assert_eq!(delay[0], 2.0e-9);
        assert_eq!(rise[0], 1.0e-12);
        assert_eq!(fall[0], 3.0e-9);
        assert_eq!(output[0], 1.2);
    }

    #[test]
    fn transition_state_step_matches_interrupted_transition_shape() {
        let mut current = [0.0];
        let mut target = [0.0];
        let mut start_time = [0.0];
        let mut start_value = [0.0];
        let mut delay = [0.0];
        let mut rise = [0.0];
        let mut fall = [0.0];
        let mut active = [0_u8];
        let mut initialized = [0_u8];
        let mut output = [0.0];

        transition_state_step_for_arrays(
            &mut current,
            &mut target,
            &mut start_time,
            &mut start_value,
            &mut delay,
            &mut rise,
            &mut fall,
            &mut active,
            &mut initialized,
            &[0.0],
            &[0.0],
            &[10.0e-9],
            &[10.0e-9],
            &mut output,
            0.0,
            1.0e-12,
            false,
        )
        .unwrap();
        transition_state_step_for_arrays(
            &mut current,
            &mut target,
            &mut start_time,
            &mut start_value,
            &mut delay,
            &mut rise,
            &mut fall,
            &mut active,
            &mut initialized,
            &[1.0],
            &[0.0],
            &[10.0e-9],
            &[10.0e-9],
            &mut output,
            0.0,
            1.0e-12,
            false,
        )
        .unwrap();
        assert_eq!(active[0], 1);

        transition_state_step_for_arrays(
            &mut current,
            &mut target,
            &mut start_time,
            &mut start_value,
            &mut delay,
            &mut rise,
            &mut fall,
            &mut active,
            &mut initialized,
            &[1.0],
            &[0.0],
            &[10.0e-9],
            &[10.0e-9],
            &mut output,
            5.0e-9,
            1.0e-12,
            false,
        )
        .unwrap();
        assert!((output[0] - 0.5).abs() < 1.0e-12);

        transition_state_step_for_arrays(
            &mut current,
            &mut target,
            &mut start_time,
            &mut start_value,
            &mut delay,
            &mut rise,
            &mut fall,
            &mut active,
            &mut initialized,
            &[0.0],
            &[0.0],
            &[10.0e-9],
            &[10.0e-9],
            &mut output,
            5.0e-9,
            1.0e-12,
            false,
        )
        .unwrap();
        assert!((output[0] - 0.5).abs() < 1.0e-12);
        assert_eq!(active[0], 1);
        assert!((start_value[0] - 1.0).abs() < 1.0e-12);
        assert!(start_time[0].abs() < 1.0e-18);

        transition_state_step_for_arrays(
            &mut current,
            &mut target,
            &mut start_time,
            &mut start_value,
            &mut delay,
            &mut rise,
            &mut fall,
            &mut active,
            &mut initialized,
            &[0.0],
            &[0.0],
            &[10.0e-9],
            &[10.0e-9],
            &mut output,
            7.5e-9,
            1.0e-12,
            false,
        )
        .unwrap();
        assert!((output[0] - 0.25).abs() < 1.0e-12);
    }

    #[test]
    fn computes_max_err_ratio_for_node_ids() {
        let values = [1.0, 2.1, 0.5];
        let previous = [1.0, 2.0, 0.0];
        let node_ids = [1_usize, 2_usize];

        let ratio = max_err_ratio_for_nodes(&values, &previous, &node_ids, 1.0e-3, 1.0e-6)
            .unwrap();

        assert!((ratio - (0.5 / (1.0e-3 * 0.5 + 1.0e-6))).abs() < 1.0e-9);
    }

    #[test]
    fn records_values_for_node_ids_with_default_for_missing_nodes() {
        let values = [1.0, 2.0, 3.0];
        let node_ids = [2_usize, 0_usize, 9_usize];
        let mut out = [0.0; 3];

        record_values_for_node_ids(&values, &node_ids, -1.0, &mut out).unwrap();

        assert_eq!(out, [3.0, 1.0, -1.0]);
    }

    #[test]
    fn interpolates_event_values_with_clamped_time_fraction() {
        let previous = [0.0, 2.0];
        let current = [10.0, 4.0];
        let mut out = [0.0; 2];

        interpolate_event_values_for_arrays(
            &previous,
            &current,
            &mut out,
            0.0,
            10.0,
            2.5,
        )
        .unwrap();
        assert!((out[0] - 2.5).abs() < 1.0e-15);
        assert!((out[1] - 2.5).abs() < 1.0e-15);

        interpolate_event_values_for_arrays(
            &previous,
            &current,
            &mut out,
            0.0,
            10.0,
            20.0,
        )
        .unwrap();
        assert_eq!(out, current);
    }

    #[test]
    fn scans_transition_breakpoints_from_arrays() {
        let start_times = [0.0, 1.0e-9, 2.0e-9];
        let start_values = [0.0, 1.0, 0.0];
        let target_values = [1.0, 0.0, 1.0];
        let delays = [0.0, 0.0, 0.0];
        let rise_times = [4.0e-9, 5.0e-9, 1.0e-9];
        let fall_times = [4.0e-9, 5.0e-9, 1.0e-9];
        let active_flags = [1_u8, 1_u8, 0_u8];

        let bp = next_transition_breakpoint_for_arrays(
            &start_times,
            &start_values,
            &target_values,
            &delays,
            &rise_times,
            &fall_times,
            &active_flags,
            1.5e-9,
            0.0,
        )
        .unwrap();

        assert_eq!(bp, Some(2.0e-9));
    }

    #[test]
    fn c_abi_scans_transition_breakpoints() {
        let start_times = [0.0, 1.0e-9];
        let start_values = [0.0, 1.0];
        let target_values = [1.0, 0.0];
        let delays = [0.0, 0.0];
        let rise_times = [4.0e-9, 5.0e-9];
        let fall_times = [4.0e-9, 5.0e-9];
        let active_flags = [1_u8, 1_u8];
        let mut found = 0_u8;
        let mut out_time = 0.0;

        let rc = unsafe {
            evas_rust_next_transition_breakpoint(
                start_times.as_ptr(),
                start_times.len(),
                start_values.as_ptr(),
                target_values.as_ptr(),
                delays.as_ptr(),
                rise_times.as_ptr(),
                fall_times.as_ptr(),
                active_flags.as_ptr(),
                1.5e-9,
                0.0,
                &mut found,
                &mut out_time,
            )
        };

        assert_eq!(rc, 0);
        assert_eq!(found, 1);
        assert_eq!(out_time, 2.0e-9);
    }

    #[test]
    fn scans_timer_breakpoints_from_arrays() {
        let next_fire_times = [10.0e-9, 4.0e-9, 7.0e-9];
        let last_fired_times = [0.0, 4.0e-9, 0.0];
        let has_last_fired_flags = [0_u8, 1_u8, 0_u8];

        let bp = next_timer_breakpoint_for_arrays(
            &next_fire_times,
            &last_fired_times,
            &has_last_fired_flags,
            1.0e-9,
        )
        .unwrap();

        assert_eq!(bp, Some(7.0e-9));
    }

    #[test]
    fn c_abi_scans_timer_breakpoints() {
        let next_fire_times = [10.0e-9, 4.0e-9, 7.0e-9];
        let last_fired_times = [0.0, 4.0e-9, 0.0];
        let has_last_fired_flags = [0_u8, 1_u8, 0_u8];
        let mut found = 0_u8;
        let mut out_time = 0.0;

        let rc = unsafe {
            evas_rust_next_timer_breakpoint(
                next_fire_times.as_ptr(),
                next_fire_times.len(),
                last_fired_times.as_ptr(),
                has_last_fired_flags.as_ptr(),
                1.0e-9,
                &mut found,
                &mut out_time,
            )
        };

        assert_eq!(rc, 0);
        assert_eq!(found, 1);
        assert_eq!(out_time, 7.0e-9);
    }

    #[test]
    fn steps_periodic_timers_from_arrays() {
        let mut next_fire_times = [0.0, 10.0e-9, 10.0e-9, 0.0];
        let mut has_state_flags = [0_u8, 1_u8, 1_u8, 0_u8];
        let periods = [10.0e-9, 10.0e-9, 10.0e-9, f64::NAN];
        let starts = [5.0e-9, 0.0, 0.0, 0.0];
        let has_start_flags = [1_u8, 0_u8, 0_u8, 0_u8];
        let mut due_flags = [0_u8; 4];
        let mut skipped_flags = [0_u8; 4];

        timer_periodic_step_for_arrays(
            &mut next_fire_times,
            &mut has_state_flags,
            &periods,
            &starts,
            &has_start_flags,
            &mut due_flags,
            &mut skipped_flags,
            10.0e-9,
            true,
            1.0e-18,
        )
        .unwrap();

        assert_eq!(has_state_flags, [1, 1, 1, 0]);
        assert_eq!(due_flags, [0, 1, 1, 0]);
        assert_eq!(skipped_flags, [1, 0, 0, 0]);
        assert!((next_fire_times[0] - 15.0e-9).abs() < 1.0e-21);
        assert!((next_fire_times[1] - 20.0e-9).abs() < 1.0e-21);
        assert!((next_fire_times[2] - 20.0e-9).abs() < 1.0e-21);
    }

    #[test]
    fn timer_static_linear_queue_preserves_same_time_source_order() {
        let times = [0.0, 1.0e-9, 2.0e-9];
        let mut node_values = [0.0];
        let mut state_values = [1.0];
        let terms = [
            EvasRustLinearTerm {
                source_kind: SOURCE_STATE,
                source_id: 0,
                gain: 1.0,
            },
            EvasRustLinearTerm {
                source_kind: SOURCE_STATE,
                source_id: 0,
                gain: 2.0,
            },
            EvasRustLinearTerm {
                source_kind: SOURCE_STATE,
                source_id: 0,
                gain: 1.0,
            },
        ];
        let event_ops = [
            EvasRustLinearOp {
                target_kind: TARGET_STATE,
                target_integer: 0,
                target_id: 0,
                term_start: 0,
                term_count: 1,
                bias: 1.0,
                condition_id: CONDITION_NONE,
                false_term_start: 0,
                false_term_count: 0,
                false_bias: 0.0,
            },
            EvasRustLinearOp {
                target_kind: TARGET_STATE,
                target_integer: 0,
                target_id: 0,
                term_start: 1,
                term_count: 1,
                bias: 0.0,
                condition_id: CONDITION_NONE,
                false_term_start: 0,
                false_term_count: 0,
                false_bias: 0.0,
            },
        ];
        let evaluate_ops = [EvasRustLinearOp {
            target_kind: TARGET_NODE,
            target_integer: 0,
            target_id: 0,
            term_start: 2,
            term_count: 1,
            bias: 0.0,
            condition_id: CONDITION_NONE,
            false_term_start: 0,
            false_term_count: 0,
            false_bias: 0.0,
        }];
        let mut out = [0.0; 3];

        let event_count = timer_static_linear_queue_trace_for_arrays(
            &times,
            &[],
            &[],
            &mut node_values,
            &mut state_values,
            &[0.0, 0.0],
            &[1.0e-9, 1.0e-9],
            &[0, 1],
            &[1, 1],
            &event_ops,
            &terms,
            &[],
            &evaluate_ops,
            &terms,
            &[],
            &[0],
            &mut out,
            1.0e-18,
        )
        .unwrap();

        assert_eq!(event_count, 6);
        assert_eq!(out, [4.0, 10.0, 22.0]);
    }

    #[test]
    fn steps_absolute_timers_from_arrays() {
        let mut next_fire_times = [0.0, 10.0e-9, 10.0e-9, 0.0];
        let mut has_state_flags = [0_u8, 1_u8, 1_u8, 0_u8];
        let mut last_fired_times = [0.0, 10.0e-9, 0.0, 0.0];
        let mut has_last_fired_flags = [0_u8, 1_u8, 0_u8, 0_u8];
        let targets = [5.0e-9, 10.0e-9, 10.0e-9, f64::NAN];
        let mut due_flags = [0_u8; 4];
        let mut expired_flags = [0_u8; 4];

        timer_absolute_step_for_arrays(
            &mut next_fire_times,
            &mut has_state_flags,
            &mut last_fired_times,
            &mut has_last_fired_flags,
            &targets,
            &mut due_flags,
            &mut expired_flags,
            10.0e-9,
            1.0e-18,
        )
        .unwrap();

        assert_eq!(has_state_flags, [1, 1, 1, 0]);
        assert_eq!(due_flags, [0, 0, 1, 0]);
        assert_eq!(expired_flags, [1, 0, 0, 0]);
        assert_eq!(has_last_fired_flags, [1, 1, 1, 0]);
        assert!((last_fired_times[0] - 5.0e-9).abs() < 1.0e-21);
        assert!((last_fired_times[2] - 10.0e-9).abs() < 1.0e-21);
    }

    #[test]
    fn steps_cross_detectors_from_arrays() {
        let mut prev_values = [0.0, -1.0, 1.0, -1.0];
        let mut prev_times = [0.0, 0.0, 0.0, 0.0];
        let mut pprev_values = [0.0; 4];
        let mut pprev_times = [0.0; 4];
        let mut initialized_flags = [0_u8, 1_u8, 1_u8, 1_u8];
        let directions = [1, 1, -1, 1];
        let mut last_cross_times = [-1.0, -1.0, -1.0, 0.5e-9];
        let current_values = [0.25, 1.0, -1.0, 1.0];
        let mut triggered_flags = [0_u8; 4];
        let mut cross_times = [0.0; 4];
        let mut trigger_directions = [0_i32; 4];
        let mut went_beyond_flags = [0_u8; 4];

        cross_detector_step_for_arrays(
            &mut prev_values,
            &mut prev_times,
            &mut pprev_values,
            &mut pprev_times,
            &mut initialized_flags,
            &directions,
            &mut last_cross_times,
            &current_values,
            &mut triggered_flags,
            &mut cross_times,
            &mut trigger_directions,
            &mut went_beyond_flags,
            1.0e-9,
            1.0e-18,
            1.0e-12,
        )
        .unwrap();

        assert_eq!(initialized_flags, [1, 1, 1, 1]);
        assert_eq!(triggered_flags, [0, 1, 1, 0]);
        assert_eq!(trigger_directions, [0, 1, -1, 0]);
        assert_eq!(went_beyond_flags, [0, 1, 1, 0]);
        assert!((cross_times[1] - 0.5e-9).abs() < 1.0e-21);
        assert!((cross_times[2] - 0.5e-9).abs() < 1.0e-21);
        assert!((prev_values[1] - 1.0).abs() < 1.0e-21);
        assert!((prev_values[2] + 1.0).abs() < 1.0e-21);
    }

    #[test]
    fn steps_above_detectors_from_arrays() {
        let mut prev_values = [0.0, -1.0, 1.0];
        let mut prev_times = [0.0, 0.0, 0.0];
        let mut pprev_values = [0.0; 3];
        let mut pprev_times = [0.0; 3];
        let mut initialized_flags = [0_u8, 1_u8, 1_u8];
        let directions = [1, 1, 1];
        let current_values = [0.0, 1.0, 0.5];
        let mut triggered_flags = [0_u8; 3];
        let mut cross_times = [0.0; 3];

        above_detector_step_for_arrays(
            &mut prev_values,
            &mut prev_times,
            &mut pprev_values,
            &mut pprev_times,
            &mut initialized_flags,
            &directions,
            &current_values,
            &mut triggered_flags,
            &mut cross_times,
            1.0e-9,
        )
        .unwrap();

        assert_eq!(initialized_flags, [1, 1, 1]);
        assert_eq!(triggered_flags, [1, 1, 0]);
        assert!((cross_times[0] - 1.0e-9).abs() < 1.0e-21);
        assert!((cross_times[1] - 0.5e-9).abs() < 1.0e-21);
        assert!((prev_values[0] - 0.0).abs() < 1.0e-21);
        assert!((prev_values[1] - 1.0).abs() < 1.0e-21);
    }

    #[test]
    fn computes_dynamic_bus_offsets_from_arrays() {
        let base_offsets = [10_usize, 100_usize, 200_usize];
        let outer_lengths = [4_usize, 8_usize, 3_usize];
        let inner_strides = [1_usize, 1_usize, 2_usize];
        let inner_lengths = [1_usize, 1_usize, 2_usize];
        let first_indices = [2_i64, 7_i64, 1_i64];
        let second_indices = [0_i64, 0_i64, 1_i64];
        let has_second = [0_u8, 0_u8, 1_u8];
        let mut out = [0_usize; 3];

        dynamic_bus_offsets_for_arrays(
            &base_offsets,
            &outer_lengths,
            &inner_strides,
            &inner_lengths,
            &first_indices,
            &second_indices,
            &has_second,
            &mut out,
        )
        .unwrap();

        assert_eq!(out, [12, 107, 203]);
    }

    #[test]
    fn rejects_dynamic_bus_offsets_out_of_bounds() {
        let mut out = [0_usize];

        let err = dynamic_bus_offsets_for_arrays(
            &[10],
            &[4],
            &[1],
            &[1],
            &[4],
            &[0],
            &[0],
            &mut out,
        )
        .unwrap_err();

        assert_eq!(err, -202);
    }

    #[test]
    fn evaluates_ordered_transition_segment_after_static_state_write() {
        let linear_ops = [EvasRustLinearOp {
            target_kind: TARGET_STATE,
            target_integer: 1,
            target_id: 0,
            term_start: 0,
            term_count: 0,
            bias: 1.0,
            condition_id: CONDITION_NONE,
            false_term_start: 0,
            false_term_count: 0,
            false_bias: 0.0,
        }];
        let transition_terms = [EvasRustLinearTerm {
            source_kind: SOURCE_STATE,
            source_id: 0,
            gain: 1.0,
        }];
        let transition_conditions = [EvasRustLinearCondition {
            op_kind: COND_NE,
            left_term_start: 0,
            left_term_count: 1,
            left_bias: 0.0,
            right_term_start: 0,
            right_term_count: 0,
            right_bias: 0.0,
        }];
        let transition_ops = [EvasRustTransitionTargetOp {
            target_id: 0,
            term_start: 0,
            term_count: 0,
            bias: 1.0,
            condition_id: 0,
            false_term_start: 0,
            false_term_count: 0,
            false_bias: 0.0,
            delay: 0.0,
            rise: 1.0e-9,
            fall: 2.0e-9,
        }];
        let mut node_values = [0.0];
        let mut state_values = [0.0];
        let mut targets = [0.0];
        let mut delays = [0.0];
        let mut rises = [0.0];
        let mut falls = [0.0];

        evaluate_ordered_transition_segment(
            &linear_ops,
            &[],
            &[],
            &transition_ops,
            &transition_terms,
            &transition_conditions,
            &mut node_values,
            &mut state_values,
            &mut targets,
            &mut delays,
            &mut rises,
            &mut falls,
        )
        .unwrap();

        assert_eq!(state_values, [1.0]);
        assert_eq!(targets, [1.0]);
        assert_eq!(rises, [1.0e-9]);
        assert_eq!(falls, [2.0e-9]);
    }

    #[test]
    fn c_abi_evaluates_ordered_transition_segment() {
        let linear_ops = [EvasRustLinearOp {
            target_kind: TARGET_STATE,
            target_integer: 1,
            target_id: 0,
            term_start: 0,
            term_count: 0,
            bias: 1.0,
            condition_id: CONDITION_NONE,
            false_term_start: 0,
            false_term_count: 0,
            false_bias: 0.0,
        }];
        let transition_terms = [EvasRustLinearTerm {
            source_kind: SOURCE_STATE,
            source_id: 0,
            gain: 1.0,
        }];
        let transition_conditions = [EvasRustLinearCondition {
            op_kind: COND_NE,
            left_term_start: 0,
            left_term_count: 1,
            left_bias: 0.0,
            right_term_start: 0,
            right_term_count: 0,
            right_bias: 0.0,
        }];
        let transition_ops = [EvasRustTransitionTargetOp {
            target_id: 0,
            term_start: 0,
            term_count: 0,
            bias: 1.0,
            condition_id: 0,
            false_term_start: 0,
            false_term_count: 0,
            false_bias: 0.0,
            delay: 0.0,
            rise: 1.0e-9,
            fall: 2.0e-9,
        }];
        let mut node_values = [0.0];
        let mut state_values = [0.0];
        let mut targets = [0.0];
        let mut delays = [0.0];
        let mut rises = [0.0];
        let mut falls = [0.0];

        let rc = unsafe {
            evas_rust_evaluate_ordered_transition_segment(
                linear_ops.as_ptr(),
                linear_ops.len(),
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                transition_ops.as_ptr(),
                transition_ops.len(),
                transition_terms.as_ptr(),
                transition_terms.len(),
                transition_conditions.as_ptr(),
                transition_conditions.len(),
                node_values.as_mut_ptr(),
                node_values.len(),
                state_values.as_mut_ptr(),
                state_values.len(),
                targets.as_mut_ptr(),
                targets.len(),
                delays.as_mut_ptr(),
                delays.len(),
                rises.as_mut_ptr(),
                rises.len(),
                falls.as_mut_ptr(),
                falls.len(),
            )
        };

        assert_eq!(rc, 0);
        assert_eq!(state_values, [1.0]);
        assert_eq!(targets, [1.0]);
        assert_eq!(rises, [1.0e-9]);
        assert_eq!(falls, [2.0e-9]);
    }

    #[test]
    fn evaluates_transition_targets_from_state_condition() {
        let terms = [EvasRustLinearTerm {
            source_kind: SOURCE_STATE,
            source_id: 0,
            gain: 1.0,
        }];
        let conditions = [EvasRustLinearCondition {
            op_kind: COND_NE,
            left_term_start: 0,
            left_term_count: 1,
            left_bias: 0.0,
            right_term_start: 0,
            right_term_count: 0,
            right_bias: 0.0,
        }];
        let ops = [EvasRustTransitionTargetOp {
            target_id: 0,
            term_start: 0,
            term_count: 0,
            bias: 1.0,
            condition_id: 0,
            false_term_start: 0,
            false_term_count: 0,
            false_bias: 0.0,
            delay: 0.0,
            rise: 1.0e-9,
            fall: 2.0e-9,
        }];
        let node_values = [0.0];
        let mut state_values = [1.0];
        let mut targets = [0.0];
        let mut delays = [0.0];
        let mut rises = [0.0];
        let mut falls = [0.0];

        evaluate_transition_target_ops(
            &ops,
            &terms,
            &conditions,
            &node_values,
            &state_values,
            &mut targets,
            &mut delays,
            &mut rises,
            &mut falls,
        )
        .unwrap();
        assert_eq!(targets, [1.0]);
        assert_eq!(rises, [1.0e-9]);
        assert_eq!(falls, [2.0e-9]);

        state_values[0] = 0.0;
        evaluate_transition_target_ops(
            &ops,
            &terms,
            &conditions,
            &node_values,
            &state_values,
            &mut targets,
            &mut delays,
            &mut rises,
            &mut falls,
        )
        .unwrap();
        assert_eq!(targets, [0.0]);
    }

    #[test]
    fn c_abi_evaluates_transition_targets() {
        let terms = [EvasRustLinearTerm {
            source_kind: SOURCE_STATE,
            source_id: 0,
            gain: 1.0,
        }];
        let conditions = [EvasRustLinearCondition {
            op_kind: COND_NE,
            left_term_start: 0,
            left_term_count: 1,
            left_bias: 0.0,
            right_term_start: 0,
            right_term_count: 0,
            right_bias: 0.0,
        }];
        let ops = [EvasRustTransitionTargetOp {
            target_id: 0,
            term_start: 0,
            term_count: 0,
            bias: 1.0,
            condition_id: 0,
            false_term_start: 0,
            false_term_count: 0,
            false_bias: 0.0,
            delay: 0.0,
            rise: 1.0e-9,
            fall: 2.0e-9,
        }];
        let node_values = [0.0];
        let state_values = [1.0];
        let mut targets = [0.0];
        let mut delays = [0.0];
        let mut rises = [0.0];
        let mut falls = [0.0];

        let rc = unsafe {
            evas_rust_evaluate_transition_targets(
                ops.as_ptr(),
                ops.len(),
                terms.as_ptr(),
                terms.len(),
                conditions.as_ptr(),
                conditions.len(),
                node_values.as_ptr(),
                node_values.len(),
                state_values.as_ptr(),
                state_values.len(),
                targets.as_mut_ptr(),
                targets.len(),
                delays.as_mut_ptr(),
                delays.len(),
                rises.as_mut_ptr(),
                rises.len(),
                falls.as_mut_ptr(),
                falls.len(),
            )
        };

        assert_eq!(rc, 0);
        assert_eq!(targets, [1.0]);
        assert_eq!(rises, [1.0e-9]);
        assert_eq!(falls, [2.0e-9]);
    }

    #[test]
    fn rejects_out_of_bounds_node_ids() {
        let ops = [EvasRustStaticAffineOp {
            read_node_id: 0,
            write_node_id: 4,
            gain: 1.0,
            bias: 0.0,
        }];
        let mut values = [0.5, 0.0];

        assert_eq!(
            evaluate_static_affine_ops(&ops, &mut values),
            Err(-3)
        );
        assert_eq!(values, [0.5, 0.0]);
    }

    #[test]
    fn c_abi_evaluates_batch() {
        let ops = [EvasRustStaticAffineOp {
            read_node_id: 0,
            write_node_id: 1,
            gain: 3.0,
            bias: -0.5,
        }];
        let mut values = [0.75, 0.0];

        let rc = unsafe {
            evas_rust_evaluate_static_affine(
                ops.as_ptr(),
                ops.len(),
                values.as_mut_ptr(),
                values.len(),
            )
        };

        assert_eq!(rc, 0);
        assert_eq!(values, [0.75, 1.75]);
    }

    #[test]
    fn evaluates_static_linear_node_and_state_ops_in_order() {
        let terms = [
            EvasRustLinearTerm {
                source_kind: SOURCE_NODE,
                source_id: 0,
                gain: 2.0,
            },
            EvasRustLinearTerm {
                source_kind: SOURCE_STATE,
                source_id: 0,
                gain: 0.5,
            },
            EvasRustLinearTerm {
                source_kind: SOURCE_STATE,
                source_id: 1,
                gain: 1.0,
            },
        ];
        let ops = [
            EvasRustLinearOp {
                target_kind: TARGET_STATE,
                target_integer: 0,
                target_id: 1,
                term_start: 0,
                term_count: 2,
                bias: 0.25,
                condition_id: CONDITION_NONE,
                false_term_start: 0,
                false_term_count: 0,
                false_bias: 0.0,
            },
            EvasRustLinearOp {
                target_kind: TARGET_NODE,
                target_integer: 0,
                target_id: 1,
                term_start: 2,
                term_count: 1,
                bias: -0.25,
                condition_id: CONDITION_NONE,
                false_term_start: 0,
                false_term_count: 0,
                false_bias: 0.0,
            },
        ];
        let mut node_values = [0.75, 0.0];
        let mut state_values = [4.0, 0.0];

        evaluate_static_linear_ops(
            &ops,
            &terms,
            &[],
            &mut node_values,
            &mut state_values,
        )
        .unwrap();

        assert_eq!(state_values, [4.0, 3.75]);
        assert_eq!(node_values, [0.75, 3.5]);
    }

    #[test]
    fn rejects_static_linear_out_of_bounds_state_sources() {
        let terms = [EvasRustLinearTerm {
            source_kind: SOURCE_STATE,
            source_id: 3,
            gain: 1.0,
        }];
        let ops = [EvasRustLinearOp {
            target_kind: TARGET_NODE,
            target_integer: 0,
            target_id: 0,
            term_start: 0,
            term_count: 1,
            bias: 0.0,
            condition_id: CONDITION_NONE,
            false_term_start: 0,
            false_term_count: 0,
            false_bias: 0.0,
        }];
        let mut node_values = [0.0];
        let mut state_values = [1.0];

        assert_eq!(
            evaluate_static_linear_ops(
                &ops,
                &terms,
                &[],
                &mut node_values,
                &mut state_values,
            ),
            Err(-9)
        );
    }

    #[test]
    fn evaluates_static_linear_conditional_select() {
        let terms = [
            EvasRustLinearTerm {
                source_kind: SOURCE_NODE,
                source_id: 0,
                gain: 1.0,
            },
            EvasRustLinearTerm {
                source_kind: SOURCE_NODE,
                source_id: 0,
                gain: 1.0,
            },
        ];
        let conditions = [EvasRustLinearCondition {
            op_kind: COND_GT,
            left_term_start: 1,
            left_term_count: 1,
            left_bias: 0.0,
            right_term_start: 0,
            right_term_count: 0,
            right_bias: 0.5,
        }];
        let ops = [EvasRustLinearOp {
            target_kind: TARGET_STATE,
            target_integer: 0,
            target_id: 0,
            term_start: 0,
            term_count: 0,
            bias: 2.0,
            condition_id: 0,
            false_term_start: 0,
            false_term_count: 0,
            false_bias: -2.0,
        }];
        let mut node_values = [0.75];
        let mut state_values = [0.0];

        evaluate_static_linear_ops(
            &ops,
            &terms,
            &conditions,
            &mut node_values,
            &mut state_values,
        )
        .unwrap();
        assert_eq!(state_values, [2.0]);

        node_values[0] = 0.25;
        evaluate_static_linear_ops(
            &ops,
            &terms,
            &conditions,
            &mut node_values,
            &mut state_values,
        )
        .unwrap();
        assert_eq!(state_values, [-2.0]);
    }

    #[test]
    fn evaluates_static_linear_integer_state_target_before_later_read() {
        let terms = [EvasRustLinearTerm {
            source_kind: SOURCE_STATE,
            source_id: 0,
            gain: 1.0,
        }];
        let ops = [
            EvasRustLinearOp {
                target_kind: TARGET_STATE,
                target_integer: 1,
                target_id: 0,
                term_start: 0,
                term_count: 0,
                bias: 1.6,
                condition_id: CONDITION_NONE,
                false_term_start: 0,
                false_term_count: 0,
                false_bias: 0.0,
            },
            EvasRustLinearOp {
                target_kind: TARGET_NODE,
                target_integer: 0,
                target_id: 0,
                term_start: 0,
                term_count: 1,
                bias: 0.0,
                condition_id: CONDITION_NONE,
                false_term_start: 0,
                false_term_count: 0,
                false_bias: 0.0,
            },
        ];
        let mut node_values = [0.0];
        let mut state_values = [0.0];

        evaluate_static_linear_ops(&ops, &terms, &[], &mut node_values, &mut state_values)
            .unwrap();

        assert_eq!(state_values, [2.0]);
        assert_eq!(node_values, [2.0]);
    }

    #[test]
    fn shifts_lfsr_event_body_and_updates_output_state() {
        let mut state_values = vec![0.0; 12];
        // lfsr slots 0..4 hold old [1, 0, 1, 1].
        state_values[0] = 1.0;
        state_values[1] = 0.0;
        state_values[2] = 1.0;
        state_values[3] = 1.0;
        let node_values = [0.2, 0.9, 0.7];
        let lfsr_slots = [0, 1, 2, 3];
        let tmp_slots = [4, 5, 6, 7, 8];
        let tap_slots = [3, 1, 0];

        let executed = event_lfsr_shift_xor_step(
            &mut state_values,
            &node_values,
            &lfsr_slots,
            &tmp_slots,
            &tap_slots,
            2,
            0.5,
            1,
            0,
            9,
            10,
            4.0,
        )
        .unwrap();

        assert!(executed);
        // feedback = old[3] ^ old[1] ^ old[0] = 1 ^ 0 ^ 1 = 0.
        assert_eq!(&state_values[0..4], &[0.0, 1.0, 0.0, 1.0]);
        assert_eq!(&state_values[4..9], &[0.0, 1.0, 0.0, 1.0, 1.0]);
        assert_eq!(state_values[9], 0.9);
        assert_eq!(state_values[10], 4.0);
    }

    #[test]
    fn lfsr_event_body_gate_can_skip_without_writes() {
        let mut state_values = vec![1.0, 0.0, 0.0, 0.0];
        let before = state_values.clone();
        let node_values = [0.4];
        let executed = event_lfsr_shift_xor_step(
            &mut state_values,
            &node_values,
            &[0, 1],
            &[2, 3, 1],
            &[0],
            0,
            0.5,
            CONDITION_NONE,
            CONDITION_NONE,
            CONDITION_NONE,
            CONDITION_NONE,
            0.0,
        )
        .unwrap();

        assert!(!executed);
        assert_eq!(state_values, before);
    }

    #[test]
    fn timer_lfsr_output_batch_updates_timer_state_and_output_node() {
        let mut state_values = vec![0.0; 12];
        state_values[0] = 1.0;
        state_values[1] = 0.0;
        state_values[2] = 1.0;
        state_values[3] = 1.0;
        let mut node_values = vec![0.0, 0.2, 0.9, 0.7];
        let mut next_fire_times = [1.0e-9];
        let mut has_state_flags = [1_u8];
        let mut due = 0_u8;
        let mut skipped = 0_u8;
        let mut executed = 0_u8;
        let mut output_written = 0_u8;

        timer_lfsr_output_step_for_arrays(
            &mut state_values,
            &mut node_values,
            &mut next_fire_times,
            &mut has_state_flags,
            1.0e-9,
            0.0,
            1,
            1.0e-9,
            1.0e-18,
            &[0, 1, 2, 3],
            &[4, 5, 6, 7, 8],
            &[3, 1, 0],
            3,
            0.5,
            2,
            1,
            9,
            0,
            10,
            4.0,
            &mut due,
            &mut skipped,
            &mut executed,
            &mut output_written,
        )
        .unwrap();

        assert_eq!(due, 1);
        assert_eq!(skipped, 0);
        assert_eq!(executed, 1);
        assert_eq!(output_written, 1);
        assert_eq!(has_state_flags[0], 1);
        assert!((next_fire_times[0] - 2.0e-9).abs() < 1.0e-21);
        assert_eq!(&state_values[0..4], &[0.0, 1.0, 0.0, 1.0]);
        assert_eq!(state_values[9], 0.9);
        assert_eq!(node_values[0], 0.9);
    }

    #[test]
    fn evaluates_body_ir_expression_stack_and_writes() {
        let mut node_values = vec![0.25, 0.0];
        let mut state_values = vec![2.0, 0.0];
        let param_values = vec![3.0];
        let expr_ops = vec![
            EvasRustBodyExprOp {
                op_kind: BODY_EXPR_READ_NODE,
                index: 0,
                value: 0.0,
            },
            EvasRustBodyExprOp {
                op_kind: BODY_EXPR_READ_PARAM,
                index: 0,
                value: 0.0,
            },
            EvasRustBodyExprOp {
                op_kind: BODY_EXPR_MUL,
                index: 0,
                value: 0.0,
            },
            EvasRustBodyExprOp {
                op_kind: BODY_EXPR_READ_STATE,
                index: 0,
                value: 0.0,
            },
            EvasRustBodyExprOp {
                op_kind: BODY_EXPR_ADD,
                index: 0,
                value: 0.0,
            },
        ];
        let stmt_ops = vec![EvasRustBodyStmtOp {
            target_kind: TARGET_NODE,
            target_integer: 0,
            target_id: 1,
            expr_start: 0,
            expr_count: expr_ops.len(),
        }];

        evaluate_body_ir_ops(
            &stmt_ops,
            &expr_ops,
            &mut node_values,
            &mut state_values,
            &param_values,
        )
        .unwrap();
        assert!((node_values[1] - 2.75).abs() < 1.0e-12);
    }

    #[test]
    fn evaluates_body_expr_stack_without_writes() {
        let node_values = vec![0.25];
        let state_values = vec![2.0];
        let param_values = vec![3.0];
        let expr_ops = vec![
            EvasRustBodyExprOp {
                op_kind: BODY_EXPR_READ_NODE,
                index: 0,
                value: 0.0,
            },
            EvasRustBodyExprOp {
                op_kind: BODY_EXPR_READ_PARAM,
                index: 0,
                value: 0.0,
            },
            EvasRustBodyExprOp {
                op_kind: BODY_EXPR_MUL,
                index: 0,
                value: 0.0,
            },
            EvasRustBodyExprOp {
                op_kind: BODY_EXPR_READ_STATE,
                index: 0,
                value: 0.0,
            },
            EvasRustBodyExprOp {
                op_kind: BODY_EXPR_ADD,
                index: 0,
                value: 0.0,
            },
        ];

        let value =
            evaluate_body_expr_ops(&expr_ops, &node_values, &state_values, &param_values).unwrap();
        assert!((value - 2.75).abs() < 1.0e-12);
    }

    #[test]
    fn evaluates_body_expr_segments_in_one_batch() {
        let node_values = vec![0.7, 0.2];
        let state_values = vec![2.0];
        let param_values = vec![0.5];
        let expr_ops = vec![
            EvasRustBodyExprOp {
                op_kind: BODY_EXPR_READ_NODE,
                index: 0,
                value: 0.0,
            },
            EvasRustBodyExprOp {
                op_kind: BODY_EXPR_READ_PARAM,
                index: 0,
                value: 0.0,
            },
            EvasRustBodyExprOp {
                op_kind: BODY_EXPR_SUB,
                index: 0,
                value: 0.0,
            },
            EvasRustBodyExprOp {
                op_kind: BODY_EXPR_READ_NODE,
                index: 1,
                value: 0.0,
            },
            EvasRustBodyExprOp {
                op_kind: BODY_EXPR_READ_STATE,
                index: 0,
                value: 0.0,
            },
            EvasRustBodyExprOp {
                op_kind: BODY_EXPR_MUL,
                index: 0,
                value: 0.0,
            },
        ];
        let mut output_values = vec![0.0, 0.0];

        evaluate_body_expr_segments(
            &expr_ops,
            &[0, 3],
            &[3, 3],
            &node_values,
            &state_values,
            &param_values,
            &mut output_values,
        )
        .unwrap();

        assert!((output_values[0] - 0.2).abs() < 1.0e-12);
        assert!((output_values[1] - 0.4).abs() < 1.0e-12);
    }

    #[test]
    fn evaluates_body_ir_select_and_integer_state_write() {
        let mut node_values = vec![0.0];
        let mut state_values = vec![0.0];
        let param_values = vec![];
        let expr_ops = vec![
            EvasRustBodyExprOp {
                op_kind: BODY_EXPR_CONST,
                index: 0,
                value: 1.0,
            },
            EvasRustBodyExprOp {
                op_kind: BODY_EXPR_CONST,
                index: 0,
                value: 2.6,
            },
            EvasRustBodyExprOp {
                op_kind: BODY_EXPR_CONST,
                index: 0,
                value: -4.0,
            },
            EvasRustBodyExprOp {
                op_kind: BODY_EXPR_SELECT,
                index: 0,
                value: 0.0,
            },
        ];
        let stmt_ops = vec![EvasRustBodyStmtOp {
            target_kind: TARGET_STATE,
            target_integer: 1,
            target_id: 0,
            expr_start: 0,
            expr_count: expr_ops.len(),
        }];

        evaluate_body_ir_ops(
            &stmt_ops,
            &expr_ops,
            &mut node_values,
            &mut state_values,
            &param_values,
        )
        .unwrap();
        assert_eq!(state_values[0], 3.0);
    }
}
