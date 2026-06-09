use crate::{
    abi::*, event::*, expr::*, ffi::*, program::*, specialized::*, transition::*, util::*,
};

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
fn pulse_source_next_breakpoint_uses_ordered_knees() {
    let source = EvasRustSimSourceSpec {
        kind: RUST_SIM_SOURCE_PULSE,
        flags: RUST_SIM_SOURCE_FLAG_HAS_WIDTH,
        node_id: 0,
        data_start: 0,
        data_count: 0,
        p0: 0.0,
        p1: 0.9,
        p2: 1.0e-9,
        p3: 0.5,
        p4: 10.0e-12,
        p5: 10.0e-12,
        p6: 100.0e-12,
        p7: 500.0e-12,
    };

    let cases = [
        (0.0, Some(100.0e-12)),
        (100.0e-12, Some(105.0e-12)),
        (105.0e-12, Some(110.0e-12)),
        (614.0e-12, Some(615.0e-12)),
        (619.0e-12, Some(620.0e-12)),
        (620.0e-12, Some(1.1e-9)),
    ];
    for (time, expected) in cases {
        let breakpoint = rust_sim_source_next_breakpoint(&source, &[], time).unwrap();
        match (breakpoint, expected) {
            (Some(value), Some(target)) => assert!((value - target).abs() < 1.0e-24),
            (None, None) => {}
            other => panic!("unexpected breakpoint for {time}: {other:?}"),
        }
    }
}

#[test]
fn prbs7_trace_generates_clocked_state_bus() {
    let times = [0.0, 0.1e-9, 0.11e-9, 0.12e-9, 1.11e-9, 1.12e-9];
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

    let ratio = max_err_ratio_for_nodes(&values, &previous, &node_ids, 1.0e-3, 1.0e-6).unwrap();

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

    interpolate_event_values_for_arrays(&previous, &current, &mut out, 0.0, 10.0, 2.5).unwrap();
    assert!((out[0] - 2.5).abs() < 1.0e-15);
    assert!((out[1] - 2.5).abs() < 1.0e-15);

    interpolate_event_values_for_arrays(&previous, &current, &mut out, 0.0, 10.0, 20.0).unwrap();
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
fn detects_transition_target_change_after_pre_always_state_update() {
    let body_stmt_ops = [
        EvasRustBodyStmtOp {
            target_kind: BODY_STMT_BOUND_STEP,
            target_integer: 0,
            target_id: 0,
            expr_start: 9,
            expr_count: 3,
        },
        EvasRustBodyStmtOp {
            target_kind: TARGET_STATE,
            target_integer: 0,
            target_id: 4,
            expr_start: 12,
            expr_count: 3,
        },
        EvasRustBodyStmtOp {
            target_kind: BODY_STMT_IF,
            target_integer: 0,
            target_id: 0,
            expr_start: 15,
            expr_count: 3,
        },
        EvasRustBodyStmtOp {
            target_kind: TARGET_STATE,
            target_integer: 0,
            target_id: 4,
            expr_start: 18,
            expr_count: 1,
        },
        EvasRustBodyStmtOp {
            target_kind: BODY_STMT_ENDIF,
            target_integer: 0,
            target_id: 0,
            expr_start: 0,
            expr_count: 0,
        },
        EvasRustBodyStmtOp {
            target_kind: BODY_STMT_IF,
            target_integer: 0,
            target_id: 0,
            expr_start: 19,
            expr_count: 3,
        },
        EvasRustBodyStmtOp {
            target_kind: TARGET_STATE,
            target_integer: 0,
            target_id: 4,
            expr_start: 22,
            expr_count: 1,
        },
        EvasRustBodyStmtOp {
            target_kind: BODY_STMT_ENDIF,
            target_integer: 0,
            target_id: 0,
            expr_start: 0,
            expr_count: 0,
        },
        EvasRustBodyStmtOp {
            target_kind: TARGET_STATE,
            target_integer: 0,
            target_id: 5,
            expr_start: 23,
            expr_count: 6,
        },
        EvasRustBodyStmtOp {
            target_kind: TARGET_STATE,
            target_integer: 0,
            target_id: 6,
            expr_start: 29,
            expr_count: 3,
        },
    ];
    let body_expr_ops = [
        EvasRustBodyExprOp {
            op_kind: BODY_EXPR_READ_NODE,
            index: 0,
            value: 0.0,
        },
        EvasRustBodyExprOp {
            op_kind: BODY_EXPR_READ_NODE,
            index: 1,
            value: 0.0,
        },
        EvasRustBodyExprOp {
            op_kind: BODY_EXPR_CONST,
            index: 0,
            value: 0.0,
        },
        EvasRustBodyExprOp {
            op_kind: BODY_EXPR_READ_PARAM,
            index: 0,
            value: 0.0,
        },
        EvasRustBodyExprOp {
            op_kind: BODY_EXPR_READ_STATE,
            index: 3,
            value: 0.0,
        },
        EvasRustBodyExprOp {
            op_kind: BODY_EXPR_READ_STATE,
            index: 3,
            value: 0.0,
        },
        EvasRustBodyExprOp {
            op_kind: BODY_EXPR_READ_PARAM,
            index: 0,
            value: 0.0,
        },
        EvasRustBodyExprOp {
            op_kind: BODY_EXPR_ADD,
            index: 0,
            value: 0.0,
        },
        EvasRustBodyExprOp {
            op_kind: BODY_EXPR_READ_STATE,
            index: 3,
            value: 0.0,
        },
        EvasRustBodyExprOp {
            op_kind: BODY_EXPR_READ_PARAM,
            index: 0,
            value: 0.0,
        },
        EvasRustBodyExprOp {
            op_kind: BODY_EXPR_READ_PARAM,
            index: 2,
            value: 0.0,
        },
        EvasRustBodyExprOp {
            op_kind: BODY_EXPR_DIV,
            index: 0,
            value: 0.0,
        },
        EvasRustBodyExprOp {
            op_kind: BODY_EXPR_READ_TIME,
            index: 0,
            value: 0.0,
        },
        EvasRustBodyExprOp {
            op_kind: BODY_EXPR_READ_STATE,
            index: 2,
            value: 0.0,
        },
        EvasRustBodyExprOp {
            op_kind: BODY_EXPR_SUB,
            index: 0,
            value: 0.0,
        },
        EvasRustBodyExprOp {
            op_kind: BODY_EXPR_READ_STATE,
            index: 4,
            value: 0.0,
        },
        EvasRustBodyExprOp {
            op_kind: BODY_EXPR_CONST,
            index: 0,
            value: 0.0,
        },
        EvasRustBodyExprOp {
            op_kind: BODY_EXPR_LT,
            index: 0,
            value: 0.0,
        },
        EvasRustBodyExprOp {
            op_kind: BODY_EXPR_CONST,
            index: 0,
            value: 0.0,
        },
        EvasRustBodyExprOp {
            op_kind: BODY_EXPR_READ_STATE,
            index: 4,
            value: 0.0,
        },
        EvasRustBodyExprOp {
            op_kind: BODY_EXPR_READ_PARAM,
            index: 0,
            value: 0.0,
        },
        EvasRustBodyExprOp {
            op_kind: BODY_EXPR_GT,
            index: 0,
            value: 0.0,
        },
        EvasRustBodyExprOp {
            op_kind: BODY_EXPR_READ_PARAM,
            index: 0,
            value: 0.0,
        },
        EvasRustBodyExprOp {
            op_kind: BODY_EXPR_READ_STATE,
            index: 4,
            value: 0.0,
        },
        EvasRustBodyExprOp {
            op_kind: BODY_EXPR_READ_PARAM,
            index: 1,
            value: 0.0,
        },
        EvasRustBodyExprOp {
            op_kind: BODY_EXPR_LE,
            index: 0,
            value: 0.0,
        },
        EvasRustBodyExprOp {
            op_kind: BODY_EXPR_CONST,
            index: 0,
            value: 1.0,
        },
        EvasRustBodyExprOp {
            op_kind: BODY_EXPR_CONST,
            index: 0,
            value: 0.0,
        },
        EvasRustBodyExprOp {
            op_kind: BODY_EXPR_SELECT,
            index: 0,
            value: 0.0,
        },
        EvasRustBodyExprOp {
            op_kind: BODY_EXPR_READ_STATE,
            index: 4,
            value: 0.0,
        },
        EvasRustBodyExprOp {
            op_kind: BODY_EXPR_READ_PARAM,
            index: 0,
            value: 0.0,
        },
        EvasRustBodyExprOp {
            op_kind: BODY_EXPR_DIV,
            index: 0,
            value: 0.0,
        },
        EvasRustBodyExprOp {
            op_kind: BODY_EXPR_READ_STATE,
            index: 5,
            value: 0.0,
        },
    ];
    let events = [EvasRustSimEventSpec {
        kind: RUST_SIM_EVENT_ALWAYS,
        phase: RUST_SIM_EVENT_PHASE_PRE,
        direction: 0,
        expr_start: 0,
        expr_count: 0,
        time_tol_start: 0,
        time_tol_count: 0,
        expr_tol_start: 0,
        expr_tol_count: 0,
        timer_start_expr_start: 0,
        timer_start_expr_count: 0,
        timer_period_expr_start: 0,
        timer_period_expr_count: 0,
        body_stmt_start: 0,
        body_stmt_count: body_stmt_ops.len(),
    }];
    let transitions = [EvasRustSimTransitionSpec {
        output_node_id: 2,
        reference_node_id: CONDITION_NONE,
        target_expr_start: 32,
        target_expr_count: 1,
        delay_expr_start: 0,
        delay_expr_count: 0,
        rise_expr_start: 0,
        rise_expr_count: 0,
        fall_expr_start: 0,
        fall_expr_count: 0,
        output_bias_expr_start: 0,
        output_bias_expr_count: 0,
        output_scale_expr_start: 0,
        output_scale_expr_count: 0,
        default_transition: 1.0e-12,
    }];
    let param_values = [8.0e-9, 1.5000000000000002e-9, 16.0, 40.0e-12];
    let node_values = [0.9, 0.0, 0.9];
    let state_values = [0.9, 0.0, 0.0, 8.0e-9, 1.5e-9, 1.0, 0.1875];

    let breakpoint = rust_sim_next_transition_target_change_breakpoint(
        &[],
        &[],
        &[],
        &[],
        &[],
        &body_stmt_ops,
        &body_expr_ops,
        &events,
        &transitions,
        &param_values,
        &node_values,
        &state_values,
        1.5e-9,
        0.5e-9,
    )
    .unwrap()
    .unwrap();

    assert!(breakpoint > 1.5e-9);
    assert!(breakpoint < 1.50000001e-9);
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
fn collects_same_step_cross_events_by_interpolated_time() {
    let expr_ops = [
        EvasRustBodyExprOp {
            op_kind: BODY_EXPR_READ_NODE,
            index: 0,
            value: 0.0,
        },
        EvasRustBodyExprOp {
            op_kind: BODY_EXPR_CONST,
            index: 0,
            value: 0.45,
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
            op_kind: BODY_EXPR_CONST,
            index: 0,
            value: 0.45,
        },
        EvasRustBodyExprOp {
            op_kind: BODY_EXPR_SUB,
            index: 0,
            value: 0.0,
        },
    ];
    let events = [
        EvasRustSimEventSpec {
            kind: RUST_SIM_EVENT_CROSS,
            phase: RUST_SIM_EVENT_PHASE_PRE,
            direction: 1,
            expr_start: 0,
            expr_count: 3,
            time_tol_start: 0,
            time_tol_count: 0,
            expr_tol_start: 0,
            expr_tol_count: 0,
            timer_start_expr_start: 0,
            timer_start_expr_count: 0,
            timer_period_expr_start: 0,
            timer_period_expr_count: 0,
            body_stmt_start: 0,
            body_stmt_count: 0,
        },
        EvasRustSimEventSpec {
            kind: RUST_SIM_EVENT_CROSS,
            phase: RUST_SIM_EVENT_PHASE_PRE,
            direction: 1,
            expr_start: 3,
            expr_count: 3,
            time_tol_start: 0,
            time_tol_count: 0,
            expr_tol_start: 0,
            expr_tol_count: 0,
            timer_start_expr_start: 0,
            timer_start_expr_count: 0,
            timer_period_expr_start: 0,
            timer_period_expr_count: 0,
            body_stmt_start: 0,
            body_stmt_count: 0,
        },
    ];
    let node_values = [0.45, 0.468];
    let mut cross_prev_values = [-0.45, -0.432];
    let mut cross_prev_times = [0.0, 0.0];
    let mut cross_pprev_values = [-0.45, -0.432];
    let mut cross_pprev_times = [0.0, 0.0];
    let mut cross_initialized = [1_u8, 1_u8];
    let mut cross_last_times = [-1.0, -1.0];

    let candidates = rust_sim_collect_cross_events(
        &events,
        &expr_ops,
        &node_values,
        &[],
        &[],
        &mut cross_prev_values,
        &mut cross_prev_times,
        &mut cross_pprev_values,
        &mut cross_pprev_times,
        &mut cross_initialized,
        &mut cross_last_times,
        5.0e-9,
        false,
        RUST_SIM_EVENT_PHASE_PRE,
    )
    .unwrap();

    assert_eq!(candidates.len(), 2);
    assert_eq!(candidates[0].event_idx, 1);
    assert_eq!(candidates[1].event_idx, 0);
    assert!((candidates[0].event_time - 4.8e-9).abs() < 1.0e-21);
    assert!((candidates[1].event_time - 5.0e-9).abs() < 1.0e-21);
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

    let err = dynamic_bus_offsets_for_arrays(&[10], &[4], &[1], &[1], &[4], &[0], &[0], &mut out)
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

    assert_eq!(evaluate_static_affine_ops(&ops, &mut values), Err(-3));
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
        evas_rust_evaluate_static_affine(ops.as_ptr(), ops.len(), values.as_mut_ptr(), values.len())
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

    evaluate_static_linear_ops(&ops, &terms, &[], &mut node_values, &mut state_values).unwrap();

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
        evaluate_static_linear_ops(&ops, &terms, &[], &mut node_values, &mut state_values,),
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

    evaluate_static_linear_ops(&ops, &terms, &[], &mut node_values, &mut state_values).unwrap();

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

#[test]
fn cmp_delay_trace_measures_interpolated_cross_times() {
    let signal_count = 6;
    let edge_vth = 0.45_f64;
    let vdd = 0.9_f64;
    let vdiff = 1.0e-3_f64;
    let tau = 4.34e-12_f64;
    let td0 = 20.5e-12_f64;
    let td_min = 20.0e-12_f64;
    let td_max = 200.0e-12_f64;
    let tedge = 30.0e-12_f64;
    let clk_cross = 60.0e-12_f64;
    let td = (td0 + tau * (vdd / vdiff).ln()).min(td_max).max(td_min);
    let out_cross = clk_cross + td + 0.5 * tedge;
    let times = vec![
        0.0,
        120.0e-12,
        out_cross,
        out_cross + 15.0e-12,
        out_cross + 35.0e-12,
    ];
    let point_clk = vec![0.0, vdd, vdd, vdd, vdd];
    let point_vinn = vec![0.450; times.len()];
    let point_vinp = vec![0.450 + vdiff; times.len()];
    let point_vdd = vec![vdd; times.len()];
    let mut values = vec![0.0; times.len() * signal_count];

    let events = cmp_delay_trace_for_arrays(
        &times,
        &mut values,
        signal_count,
        &point_clk,
        &point_vinn,
        &point_vinp,
        &point_vdd,
        0.0,
        tau,
        td0,
        td_min,
        td_max,
        tedge,
        edge_vth,
    )
    .unwrap();

    let expected_delay_ps = (td + 0.5 * tedge) * 1.0e12;
    let final_delay_ps = values[(times.len() - 1) * signal_count + 5];
    assert_eq!(events, 1);
    assert!((final_delay_ps - expected_delay_ps).abs() < 1.0e-6);
}
