use crate::abi::*;
use crate::event::*;
use crate::expr::*;
use crate::program::*;
use crate::specialized::*;
use crate::transition::*;
use crate::util::*;

// C ABI entry points consumed by the Python rust_backend bridge.

#[no_mangle]
pub unsafe extern "C" fn evas_rust_run_source_record_program(
    sources: *const EvasRustSimSourceSpec,
    source_count: usize,
    source_data: *const f64,
    source_data_len: usize,
    node_values: *mut f64,
    node_count: usize,
    record_node_ids: *const usize,
    record_count: usize,
    time_values: *mut f64,
    signal_values: *mut f64,
    step_values: *mut f64,
    capacity: usize,
    out_count: *mut usize,
    tstop: f64,
    tstep: f64,
    max_step: f64,
    record_step: f64,
    use_record_step: u8,
    out_source_breakpoints: *mut usize,
) -> i32 {
    if source_count > 0 && sources.is_null() {
        return -861;
    }
    if source_data_len > 0 && source_data.is_null() {
        return -862;
    }
    if node_count > 0 && node_values.is_null() {
        return -863;
    }
    if record_count > 0 && record_node_ids.is_null() {
        return -864;
    }
    if capacity > 0 && time_values.is_null() {
        return -865;
    }
    if capacity > 0 && step_values.is_null() {
        return -866;
    }
    let signal_capacity = match capacity.checked_mul(record_count) {
        Some(value) => value,
        None => return -867,
    };
    if signal_capacity > 0 && signal_values.is_null() {
        return -868;
    }
    if out_count.is_null() {
        return -869;
    }
    if out_source_breakpoints.is_null() {
        return -870;
    }

    let source_slice = if source_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(sources, source_count)
    };
    let source_data_slice = if source_data_len == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(source_data, source_data_len)
    };
    let node_slice = if node_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(node_values, node_count)
    };
    let record_slice = if record_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(record_node_ids, record_count)
    };
    let time_slice = if capacity == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(time_values, capacity)
    };
    let signal_slice = if signal_capacity == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(signal_values, signal_capacity)
    };
    let step_slice = if capacity == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(step_values, capacity)
    };

    match rust_sim_source_record_trace(
        source_slice,
        source_data_slice,
        node_slice,
        record_slice,
        time_slice,
        signal_slice,
        step_slice,
        tstop,
        tstep,
        max_step,
        record_step,
        use_record_step != 0,
    ) {
        Ok((count, source_breakpoints)) => {
            *out_count = count;
            *out_source_breakpoints = source_breakpoints;
            0
        }
        Err(code) => code,
    }
}

#[no_mangle]
pub unsafe extern "C" fn evas_rust_run_source_linear_record_program(
    sources: *const EvasRustSimSourceSpec,
    source_count: usize,
    source_data: *const f64,
    source_data_len: usize,
    linear_ops: *const EvasRustLinearOp,
    linear_count: usize,
    linear_terms: *const EvasRustLinearTerm,
    linear_term_count: usize,
    linear_conditions: *const EvasRustLinearCondition,
    linear_condition_count: usize,
    node_values: *mut f64,
    node_count: usize,
    state_values: *mut f64,
    state_count: usize,
    record_node_ids: *const usize,
    record_count: usize,
    time_values: *mut f64,
    signal_values: *mut f64,
    step_values: *mut f64,
    capacity: usize,
    out_count: *mut usize,
    tstop: f64,
    tstep: f64,
    max_step: f64,
    record_step: f64,
    use_record_step: u8,
    out_source_breakpoints: *mut usize,
) -> i32 {
    if source_count > 0 && sources.is_null() {
        return -901;
    }
    if source_data_len > 0 && source_data.is_null() {
        return -902;
    }
    if linear_count > 0 && linear_ops.is_null() {
        return -903;
    }
    if linear_term_count > 0 && linear_terms.is_null() {
        return -904;
    }
    if linear_condition_count > 0 && linear_conditions.is_null() {
        return -905;
    }
    if node_count > 0 && node_values.is_null() {
        return -906;
    }
    if state_count > 0 && state_values.is_null() {
        return -907;
    }
    if record_count > 0 && record_node_ids.is_null() {
        return -908;
    }
    if capacity > 0 && time_values.is_null() {
        return -909;
    }
    if capacity > 0 && step_values.is_null() {
        return -910;
    }
    let signal_capacity = match capacity.checked_mul(record_count) {
        Some(value) => value,
        None => return -911,
    };
    if signal_capacity > 0 && signal_values.is_null() {
        return -912;
    }
    if out_count.is_null() {
        return -913;
    }
    if out_source_breakpoints.is_null() {
        return -914;
    }

    let source_slice = if source_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(sources, source_count)
    };
    let source_data_slice = if source_data_len == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(source_data, source_data_len)
    };
    let linear_op_slice = if linear_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(linear_ops, linear_count)
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
    let node_slice = if node_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(node_values, node_count)
    };
    let state_slice = if state_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(state_values, state_count)
    };
    let record_slice = if record_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(record_node_ids, record_count)
    };
    let time_slice = if capacity == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(time_values, capacity)
    };
    let signal_slice = if signal_capacity == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(signal_values, signal_capacity)
    };
    let step_slice = if capacity == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(step_values, capacity)
    };

    match rust_sim_source_linear_record_trace(
        source_slice,
        source_data_slice,
        linear_op_slice,
        linear_term_slice,
        linear_condition_slice,
        node_slice,
        state_slice,
        record_slice,
        time_slice,
        signal_slice,
        step_slice,
        tstop,
        tstep,
        max_step,
        record_step,
        use_record_step != 0,
    ) {
        Ok((count, source_breakpoints)) => {
            *out_count = count;
            *out_source_breakpoints = source_breakpoints;
            0
        }
        Err(code) => code,
    }
}

#[no_mangle]
pub unsafe extern "C" fn evas_rust_run_event_transition_record_program(
    sources: *const EvasRustSimSourceSpec,
    source_count: usize,
    source_data: *const f64,
    source_data_len: usize,
    linear_ops: *const EvasRustLinearOp,
    linear_count: usize,
    linear_terms: *const EvasRustLinearTerm,
    linear_term_count: usize,
    linear_conditions: *const EvasRustLinearCondition,
    linear_condition_count: usize,
    body_stmt_ops: *const EvasRustBodyStmtOp,
    body_stmt_count: usize,
    body_expr_ops: *const EvasRustBodyExprOp,
    body_expr_count: usize,
    events: *const EvasRustSimEventSpec,
    event_count: usize,
    transitions: *const EvasRustSimTransitionSpec,
    transition_count: usize,
    side_effect_kinds: *mut u8,
    side_effect_spec_ids: *mut usize,
    side_effect_arg_starts: *mut usize,
    side_effect_arg_counts: *mut usize,
    side_effect_times: *mut f64,
    side_effect_capacity: usize,
    out_side_effect_count: *mut usize,
    side_effect_values: *mut f64,
    side_effect_value_capacity: usize,
    out_side_effect_value_count: *mut usize,
    param_values: *const f64,
    param_count: usize,
    node_values: *mut f64,
    node_count: usize,
    state_values: *mut f64,
    state_count: usize,
    record_node_ids: *const usize,
    record_count: usize,
    time_values: *mut f64,
    signal_values: *mut f64,
    step_values: *mut f64,
    capacity: usize,
    out_count: *mut usize,
    tstop: f64,
    tstep: f64,
    max_step: f64,
    record_step: f64,
    use_record_step: u8,
    min_ramp_time: f64,
    cross_acceptance_slack_factor: f64,
    out_source_breakpoints: *mut usize,
    out_event_fires: *mut usize,
    out_transition_breakpoints: *mut usize,
) -> i32 {
    if source_count > 0 && sources.is_null() {
        return -1001;
    }
    if source_data_len > 0 && source_data.is_null() {
        return -1002;
    }
    if linear_count > 0 && linear_ops.is_null() {
        return -1003;
    }
    if linear_term_count > 0 && linear_terms.is_null() {
        return -1004;
    }
    if linear_condition_count > 0 && linear_conditions.is_null() {
        return -1005;
    }
    if body_stmt_count > 0 && body_stmt_ops.is_null() {
        return -1006;
    }
    if body_expr_count > 0 && body_expr_ops.is_null() {
        return -1007;
    }
    if event_count > 0 && events.is_null() {
        return -1008;
    }
    if transition_count > 0 && transitions.is_null() {
        return -1009;
    }
    if side_effect_capacity > 0
        && (side_effect_kinds.is_null()
            || side_effect_spec_ids.is_null()
            || side_effect_arg_starts.is_null()
            || side_effect_arg_counts.is_null()
            || side_effect_times.is_null())
    {
        return -1021;
    }
    if side_effect_value_capacity > 0 && side_effect_values.is_null() {
        return -1022;
    }
    if out_side_effect_count.is_null() || out_side_effect_value_count.is_null() {
        return -1023;
    }
    if param_count > 0 && param_values.is_null() {
        return -1010;
    }
    if node_count > 0 && node_values.is_null() {
        return -1011;
    }
    if state_count > 0 && state_values.is_null() {
        return -1012;
    }
    if record_count > 0 && record_node_ids.is_null() {
        return -1013;
    }
    if capacity > 0 && time_values.is_null() {
        return -1014;
    }
    if capacity > 0 && step_values.is_null() {
        return -1015;
    }
    let signal_capacity = match capacity.checked_mul(record_count) {
        Some(value) => value,
        None => return -1016,
    };
    if signal_capacity > 0 && signal_values.is_null() {
        return -1017;
    }
    if out_count.is_null()
        || out_source_breakpoints.is_null()
        || out_event_fires.is_null()
        || out_transition_breakpoints.is_null()
    {
        return -1018;
    }

    let source_slice = if source_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(sources, source_count)
    };
    let source_data_slice = if source_data_len == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(source_data, source_data_len)
    };
    let linear_op_slice = if linear_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(linear_ops, linear_count)
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
    let body_stmt_slice = if body_stmt_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(body_stmt_ops, body_stmt_count)
    };
    let body_expr_slice = if body_expr_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(body_expr_ops, body_expr_count)
    };
    let event_slice = if event_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(events, event_count)
    };
    let transition_slice = if transition_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(transitions, transition_count)
    };
    let side_effect_kind_slice = if side_effect_capacity == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(side_effect_kinds, side_effect_capacity)
    };
    let side_effect_spec_slice = if side_effect_capacity == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(side_effect_spec_ids, side_effect_capacity)
    };
    let side_effect_arg_start_slice = if side_effect_capacity == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(side_effect_arg_starts, side_effect_capacity)
    };
    let side_effect_arg_count_slice = if side_effect_capacity == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(side_effect_arg_counts, side_effect_capacity)
    };
    let side_effect_time_slice = if side_effect_capacity == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(side_effect_times, side_effect_capacity)
    };
    let side_effect_value_slice = if side_effect_value_capacity == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(side_effect_values, side_effect_value_capacity)
    };
    let param_slice = if param_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(param_values, param_count)
    };
    let node_slice = if node_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(node_values, node_count)
    };
    let state_slice = if state_count == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(state_values, state_count)
    };
    let record_slice = if record_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(record_node_ids, record_count)
    };
    let time_slice = if capacity == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(time_values, capacity)
    };
    let signal_slice = if signal_capacity == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(signal_values, signal_capacity)
    };
    let step_slice = if capacity == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(step_values, capacity)
    };

    match rust_sim_event_transition_record_trace(
        source_slice,
        source_data_slice,
        linear_op_slice,
        linear_term_slice,
        linear_condition_slice,
        body_stmt_slice,
        body_expr_slice,
        event_slice,
        transition_slice,
        side_effect_kind_slice,
        side_effect_spec_slice,
        side_effect_arg_start_slice,
        side_effect_arg_count_slice,
        side_effect_time_slice,
        &mut *out_side_effect_count,
        side_effect_value_slice,
        &mut *out_side_effect_value_count,
        param_slice,
        node_slice,
        state_slice,
        record_slice,
        time_slice,
        signal_slice,
        step_slice,
        tstop,
        tstep,
        max_step,
        record_step,
        use_record_step != 0,
        min_ramp_time,
        1.0e-12,
        cross_acceptance_slack_factor,
    ) {
        Ok((count, source_breakpoints, event_fires, transition_breakpoints)) => {
            *out_count = count;
            *out_source_breakpoints = source_breakpoints;
            *out_event_fires = event_fires;
            *out_transition_breakpoints = transition_breakpoints;
            0
        }
        Err(code) => code,
    }
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
    let point_vdd_slice = if point_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(point_vdd, point_count)
    };
    let point_vss_slice = if point_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(point_vss, point_count)
    };
    let point_vinp_slice = if point_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(point_vinp, point_count)
    };
    let point_vinn_slice = if point_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(point_vinn, point_count)
    };
    let point_voutp_slice = if point_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(point_voutp, point_count)
    };
    let point_voutn_slice = if point_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(point_voutn, point_count)
    };
    let sample_vdd_slice = if sample_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(sample_vdd, sample_count)
    };
    let sample_vss_slice = if sample_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(sample_vss, sample_count)
    };
    let sample_vinp_slice = if sample_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(sample_vinp, sample_count)
    };
    let sample_vinn_slice = if sample_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(sample_vinn, sample_count)
    };
    let sample_voutp_slice = if sample_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(sample_voutp, sample_count)
    };
    let sample_voutn_slice = if sample_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(sample_voutn, sample_count)
    };

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
pub unsafe extern "C" fn evas_rust_event_transition_core_trace_pulse(
    time_values: *mut f64,
    out_values: *mut f64,
    capacity: usize,
    out_count: *mut usize,
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
    pulse_has_width: u8,
    edge_threshold: f64,
    edge_direction: i32,
    initial_state_value: f64,
    event_state_value: f64,
    transition_delay: f64,
    transition_rise: f64,
    transition_fall: f64,
    default_transition: f64,
    final_state: *mut f64,
    fired_events: *mut usize,
    transition_breakpoints: *mut usize,
    source_breakpoints: *mut usize,
) -> i32 {
    if capacity > 0 && time_values.is_null() {
        return -721;
    }
    if capacity > 0 && out_values.is_null() {
        return -722;
    }
    if out_count.is_null() {
        return -723;
    }
    if final_state.is_null() {
        return -724;
    }
    if fired_events.is_null() {
        return -725;
    }
    if transition_breakpoints.is_null() {
        return -726;
    }
    if source_breakpoints.is_null() {
        return -727;
    }

    let time_slice = if capacity == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(time_values, capacity)
    };
    let out_slice = if capacity == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(out_values, capacity)
    };

    match event_transition_core_trace_pulse(
        time_slice,
        out_slice,
        tstop,
        sample_step,
        tstep,
        v_lo,
        v_hi,
        period,
        duty,
        pulse_rise,
        pulse_fall,
        pulse_delay,
        pulse_width,
        pulse_has_width != 0,
        edge_threshold,
        edge_direction,
        initial_state_value,
        event_state_value,
        transition_delay,
        transition_rise,
        transition_fall,
        default_transition,
    ) {
        Ok((count, state, fired, transition_bp, source_bp)) => {
            *out_count = count;
            *final_state = state;
            *fired_events = fired;
            *transition_breakpoints = transition_bp;
            *source_breakpoints = source_bp;
            0
        }
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

    match max_err_ratio_for_nodes(value_slice, previous_slice, node_slice, reltol, vabstol) {
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
