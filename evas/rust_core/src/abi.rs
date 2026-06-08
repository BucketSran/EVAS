// Shared C ABI data structures and opcode constants.

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EvasRustStaticAffineOp {
    pub read_node_id: usize,
    pub write_node_id: usize,
    pub gain: f64,
    pub bias: f64,
}

pub(crate) const SOURCE_NODE: u8 = 0;
pub(crate) const SOURCE_STATE: u8 = 1;
pub(crate) const TARGET_NODE: u8 = 0;
pub(crate) const TARGET_STATE: u8 = 1;
pub(crate) const CONDITION_NONE: usize = usize::MAX;
pub(crate) const COND_GT: u8 = 1;
pub(crate) const COND_LT: u8 = 2;
pub(crate) const COND_GE: u8 = 3;
pub(crate) const COND_LE: u8 = 4;
pub(crate) const COND_EQ: u8 = 5;
pub(crate) const COND_NE: u8 = 6;

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

pub(crate) const BODY_EXPR_CONST: u8 = 0;
pub(crate) const BODY_EXPR_READ_NODE: u8 = 1;
pub(crate) const BODY_EXPR_READ_STATE: u8 = 2;
pub(crate) const BODY_EXPR_READ_PARAM: u8 = 3;
pub(crate) const BODY_EXPR_READ_TIME: u8 = 4;
pub(crate) const BODY_EXPR_NEG: u8 = 10;
pub(crate) const BODY_EXPR_NOT: u8 = 11;
pub(crate) const BODY_EXPR_ADD: u8 = 20;
pub(crate) const BODY_EXPR_SUB: u8 = 21;
pub(crate) const BODY_EXPR_MUL: u8 = 22;
pub(crate) const BODY_EXPR_DIV: u8 = 23;
pub(crate) const BODY_EXPR_MOD: u8 = 24;
pub(crate) const BODY_EXPR_GT: u8 = 30;
pub(crate) const BODY_EXPR_LT: u8 = 31;
pub(crate) const BODY_EXPR_GE: u8 = 32;
pub(crate) const BODY_EXPR_LE: u8 = 33;
pub(crate) const BODY_EXPR_EQ: u8 = 34;
pub(crate) const BODY_EXPR_NE: u8 = 35;
pub(crate) const BODY_EXPR_LAND: u8 = 36;
pub(crate) const BODY_EXPR_LOR: u8 = 37;
pub(crate) const BODY_EXPR_BITAND: u8 = 38;
pub(crate) const BODY_EXPR_BITOR: u8 = 39;
pub(crate) const BODY_EXPR_BITXOR: u8 = 40;
pub(crate) const BODY_EXPR_SHL: u8 = 41;
pub(crate) const BODY_EXPR_SHR: u8 = 42;
pub(crate) const BODY_EXPR_BITNOT: u8 = 43;
pub(crate) const BODY_EXPR_SELECT: u8 = 50;
pub(crate) const BODY_EXPR_ABS: u8 = 60;
pub(crate) const BODY_EXPR_SQRT: u8 = 61;
pub(crate) const BODY_EXPR_EXP: u8 = 62;
pub(crate) const BODY_EXPR_LN: u8 = 63;
pub(crate) const BODY_EXPR_LOG10: u8 = 64;
pub(crate) const BODY_EXPR_SIN: u8 = 65;
pub(crate) const BODY_EXPR_COS: u8 = 66;
pub(crate) const BODY_EXPR_FLOOR: u8 = 67;
pub(crate) const BODY_EXPR_CEIL: u8 = 68;
pub(crate) const BODY_EXPR_MIN: u8 = 69;
pub(crate) const BODY_EXPR_MAX: u8 = 70;
pub(crate) const BODY_EXPR_POW: u8 = 71;
pub(crate) const BODY_EXPR_TAN: u8 = 72;
pub(crate) const BODY_EXPR_TANH: u8 = 73;
pub(crate) const BODY_EXPR_RDIST_NORMAL: u8 = 80;
pub(crate) const BODY_STMT_WHILE: u8 = 245;
pub(crate) const BODY_STMT_ENDWHILE: u8 = 246;
pub(crate) const BODY_STMT_FILE_OPEN: u8 = 247;
pub(crate) const BODY_STMT_FILE_WRITE: u8 = 248;
pub(crate) const BODY_STMT_FILE_CLOSE: u8 = 249;
pub(crate) const BODY_STMT_IF: u8 = 250;
pub(crate) const BODY_STMT_ELSE: u8 = 251;
pub(crate) const BODY_STMT_ENDIF: u8 = 252;
pub(crate) const BODY_STMT_BOUND_STEP: u8 = 253;

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

pub(crate) const RUST_SIM_SOURCE_DC: u8 = 0;
pub(crate) const RUST_SIM_SOURCE_PULSE: u8 = 1;
pub(crate) const RUST_SIM_SOURCE_SINE: u8 = 2;
pub(crate) const RUST_SIM_SOURCE_PWL: u8 = 3;
pub(crate) const RUST_SIM_SOURCE_FLAG_HAS_WIDTH: u8 = 1;
pub(crate) const RUST_SIM_SOURCE_FLAG_ONE_SHOT: u8 = 2;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EvasRustSimSourceSpec {
    pub kind: u8,
    pub flags: u8,
    pub node_id: usize,
    pub data_start: usize,
    pub data_count: usize,
    pub p0: f64,
    pub p1: f64,
    pub p2: f64,
    pub p3: f64,
    pub p4: f64,
    pub p5: f64,
    pub p6: f64,
    pub p7: f64,
}

pub(crate) const RUST_SIM_EVENT_INITIAL_STEP: u8 = 0;
pub(crate) const RUST_SIM_EVENT_CROSS: u8 = 1;
pub(crate) const RUST_SIM_EVENT_ABOVE: u8 = 2;
pub(crate) const RUST_SIM_EVENT_TIMER: u8 = 3;
pub(crate) const RUST_SIM_EVENT_ALWAYS: u8 = 4;
pub(crate) const RUST_SIM_EVENT_FINAL_STEP: u8 = 5;
pub(crate) const RUST_SIM_EVENT_PHASE_PRE: u8 = 0;
pub(crate) const RUST_SIM_EVENT_PHASE_POST: u8 = 1;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EvasRustSimEventSpec {
    pub kind: u8,
    pub phase: u8,
    pub direction: i32,
    pub expr_start: usize,
    pub expr_count: usize,
    pub time_tol_start: usize,
    pub time_tol_count: usize,
    pub expr_tol_start: usize,
    pub expr_tol_count: usize,
    pub timer_start_expr_start: usize,
    pub timer_start_expr_count: usize,
    pub timer_period_expr_start: usize,
    pub timer_period_expr_count: usize,
    pub body_stmt_start: usize,
    pub body_stmt_count: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EvasRustSimTransitionSpec {
    pub output_node_id: usize,
    pub reference_node_id: usize,
    pub target_expr_start: usize,
    pub target_expr_count: usize,
    pub delay_expr_start: usize,
    pub delay_expr_count: usize,
    pub rise_expr_start: usize,
    pub rise_expr_count: usize,
    pub fall_expr_start: usize,
    pub fall_expr_count: usize,
    pub output_bias_expr_start: usize,
    pub output_bias_expr_count: usize,
    pub output_scale_expr_start: usize,
    pub output_scale_expr_count: usize,
    pub default_transition: f64,
}
