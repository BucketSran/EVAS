use crate::abi::*;

// Generic utility kernels used by specialized paths and RustSimProgram.

pub(crate) fn to_veriloga_integer(value: f64) -> i64 {
    if !value.is_finite() {
        return 0;
    }
    if value >= 0.0 {
        (value + 0.5).floor() as i64
    } else {
        (value - 0.5).ceil() as i64
    }
}

pub(crate) fn truthy(value: f64) -> bool {
    value != 0.0
}

pub(crate) fn bool_value(value: bool) -> f64 {
    if value {
        1.0
    } else {
        0.0
    }
}

pub(crate) fn pop1(stack: &mut Vec<f64>) -> Result<f64, i32> {
    stack.pop().ok_or(-2240)
}

pub(crate) fn pop2(stack: &mut Vec<f64>) -> Result<(f64, f64), i32> {
    let right = stack.pop().ok_or(-2241)?;
    let left = stack.pop().ok_or(-2242)?;
    Ok((left, right))
}

pub(crate) fn pop3(stack: &mut Vec<f64>) -> Result<(f64, f64, f64), i32> {
    let third = stack.pop().ok_or(-2243)?;
    let second = stack.pop().ok_or(-2244)?;
    let first = stack.pop().ok_or(-2245)?;
    Ok((first, second, third))
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut mixed = value;
    mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    mixed ^ (mixed >> 31)
}

fn uniform01_from_u64(value: u64) -> f64 {
    let mantissa = value >> 11;
    let scale = 1.0 / ((1_u64 << 53) as f64);
    ((mantissa as f64) * scale).clamp(1.0e-12, 1.0 - 1.0e-12)
}

thread_local! {
    // Per-seed draw counters for $rdist_normal. Hashing the draw index
    // instead of wall-clock time keeps the sequence identical to the python
    // engine even when event times differ at sub-ps level. Reset at the start
    // of every full-program simulation so runs are reproducible.
    static RDIST_DRAW_INDICES: std::cell::RefCell<Vec<(u64, u64)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

pub(crate) fn reset_rdist_draw_indices() {
    RDIST_DRAW_INDICES.with(|cell| cell.borrow_mut().clear());
}

fn next_rdist_draw_index(seed_bits: u64) -> u64 {
    RDIST_DRAW_INDICES.with(|cell| {
        let mut counts = cell.borrow_mut();
        for entry in counts.iter_mut() {
            if entry.0 == seed_bits {
                let index = entry.1;
                entry.1 = entry.1.wrapping_add(1);
                return index;
            }
        }
        counts.push((seed_bits, 1));
        0
    })
}

fn deterministic_normal(seed: f64, mean: f64, std: f64) -> f64 {
    let seed_bits = seed.to_bits();
    let index = next_rdist_draw_index(seed_bits);
    let stream = seed_bits ^ index.rotate_left(17) ^ 0xd1b5_4a32_d192_ed03_u64;
    let u1 = uniform01_from_u64(splitmix64(stream));
    let u2 = uniform01_from_u64(splitmix64(stream ^ 0xa076_1d64_78bd_642f_u64));
    let radius = (-2.0 * u1.ln()).sqrt();
    let angle = 2.0 * std::f64::consts::PI * u2;
    mean + std * radius * angle.cos()
}

pub(crate) fn evaluate_body_expr_segment(
    ops: &[EvasRustBodyExprOp],
    node_values: &[f64],
    state_values: &[f64],
    param_values: &[f64],
    time: f64,
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
            BODY_EXPR_READ_TIME => {
                stack.push(time);
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
            BODY_EXPR_SHL => {
                let (left, right) = pop2(stack)?;
                let shift = right as i64;
                if !(0..64).contains(&shift) {
                    return Err(-2257);
                }
                stack.push(((left as i64) << (shift as u32)) as f64);
            }
            BODY_EXPR_SHR => {
                let (left, right) = pop2(stack)?;
                let shift = right as i64;
                if shift < 0 {
                    return Err(-2258);
                }
                let value = left as i64;
                let shifted = if shift >= 64 {
                    if value < 0 {
                        -1_i64
                    } else {
                        0_i64
                    }
                } else {
                    value >> (shift as u32)
                };
                stack.push(shifted as f64);
            }
            BODY_EXPR_BITNOT => {
                let value = pop1(stack)?;
                stack.push((!(value as i64)) as f64);
            }
            BODY_EXPR_SELECT => {
                let false_value = stack.pop().ok_or(-2253)?;
                let true_value = stack.pop().ok_or(-2254)?;
                let condition = stack.pop().ok_or(-2255)?;
                stack.push(if truthy(condition) {
                    true_value
                } else {
                    false_value
                });
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
            BODY_EXPR_TAN => {
                let value = pop1(stack)?;
                stack.push(value.tan());
            }
            BODY_EXPR_TANH => {
                let value = pop1(stack)?;
                stack.push(value.tanh());
            }
            BODY_EXPR_RDIST_NORMAL => {
                let (seed, mean, std) = pop3(stack)?;
                stack.push(deterministic_normal(seed, mean, std));
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

pub(crate) fn evaluate_linear_value(
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

pub(crate) fn evaluate_condition(
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
