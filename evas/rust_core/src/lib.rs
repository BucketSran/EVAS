#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EvasRustStaticAffineOp {
    pub read_node_id: usize,
    pub write_node_id: usize,
    pub gain: f64,
    pub bias: f64,
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
}
