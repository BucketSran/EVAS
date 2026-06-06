//! Rust kernels and C ABI surface for EVAS.
//!
//! This crate is called from Python through `ctypes`, so the exported structs
//! and `extern "C"` functions intentionally use a flat C-compatible ABI.  The
//! main production path is the RustSimProgram source/event/transition/record
//! loop.  Several older specialized trace kernels are still kept here because
//! Python callers may probe them during speed experiments.
//!
//! Suggested module split once the current Rust migration stabilizes:
//! `abi`, `expr`, `event`, `transition`, `program`, `specialized`, and `ffi`.
mod abi;
mod event;
mod expr;
mod ffi;
mod program;
mod specialized;
mod transition;
mod util;

pub use abi::*;
pub use event::*;
pub use expr::*;
pub use ffi::*;
pub use program::*;
pub use specialized::*;
pub use transition::*;
pub use util::*;

#[cfg(test)]
mod tests;
