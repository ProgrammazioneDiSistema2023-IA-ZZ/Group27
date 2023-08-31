use std::sync::Arc;
use ndarray::Ix2;

use super::{Operation, OnnxError, onnx_error, Tensor, OperationResult};

impl Operation {

    /// *(From <https://onnx.ai/onnx/operators/onnx__MatMul.html>)*
    /// 
    /// Matrix product that behaves like numpy.matmul: <https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html>
    /// 
    /// # Inputs
    /// * **A** (heterogeneous) - `T`: N-dimensional matrix A
    /// * **B** (heterogeneous) - `T`: N-dimensional matrix B
    /// 
    /// # Outputs
    /// * **Y** (heterogeneous) - `T`: Matrix multiply results from A * B
    pub(super) fn execute_matmul(&self, inputs: Vec<&Tensor>) -> OperationResult {
        if inputs.len() != 2 {
            return Err(onnx_error!("MatMul operation requires 2 inputs."));
        }

        let a =
            inputs[0].view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| onnx_error!("Only two-dimensional product is supported. Matrix A was found to have {} dimension(s).", inputs[0].ndim()))?;
        let a_shape = a.shape();

        let b =
            inputs[1].view()
                .into_dimensionality::<Ix2>().map_err(|_| onnx_error!("Only two-dimensional product is supported. Matrix B was found to have {} dimension(s).", inputs[1].ndim()))?;
        let b_shape = b.shape();

        if a_shape[1] == b_shape[0] {
            Ok(Arc::new(a.dot(&b).into_dyn()))
        } else {
            Err(onnx_error!("Cannot perform dot product because the number of columns of the first matrix ({}) is different than the number of rows of the second matrix ({}).", a_shape[1], b_shape[0]))
        }
    }

}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use ndarray::{Array, array};
    use crate::operations::{Operation, OpType};

    #[test]
    fn valid_inputs() {
        let op = Operation::new(OpType::MatMul);

        let a =
            Array::from_iter((1..=6).map(|v| v as f32).cycle().take(3*2))
            .into_shape((3,2))
            .unwrap();

        let b =
            Array::from_iter((7..=12).map(|v| v as f32).cycle().take(2*3))
            .into_shape((2,3))
            .unwrap();
        
        let expected_result =
            array![
                [27., 30.,  33.],
                [61., 68.,  75.],
                [95., 106., 117.]
            ].into_dyn();

        let result = op.execute(vec![Arc::new(a.into_dyn()), Arc::new(b.into_dyn())]);
        assert!(result.is_ok(), "{:?}", result.unwrap_err());
        assert_eq!(result.unwrap(), Arc::new(expected_result));
    }

    #[test]
    fn incompatible_shapes() {
        let op = Operation::new(OpType::MatMul);

        let a =
            Array::from_iter((1..=12).map(|v| v as f32).cycle().take(6*2))
            .into_shape((6,2))
            .unwrap().into_dyn();

        let b =
            Array::from_iter((13..=24).map(|v| v as f32).cycle().take(4*3))
            .into_shape((4,3))
            .unwrap().into_dyn();

        let result = op.execute(vec![Arc::new(a), Arc::new(b)]);
        assert!(result.is_err());
    }

}