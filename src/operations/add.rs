use std::sync::Arc;
use super::{Operation, OnnxError, onnx_error, Tensor, OperationResult};

impl Operation {

    /// *(From <https://onnx.ai/onnx/operators/onnx__Add.html>)*
    /// 
    /// Performs element-wise binary addition (with Numpy-style broadcasting support).
    /// 
    /// This operator supports multidirectional (i.e., Numpy-style) broadcasting; for more details please check Broadcasting in ONNX.
    /// 
    /// (Opset 14 change): Extend supported types to include uint8, int8, uint16, and int16.
    /// 
    /// # Inputs
    /// * **A** (heterogeneous) - `T`: First operand.
    /// * **B** (heterogeneous) - `T`: Second operand.
    /// 
    /// # Outputs
    /// * **C** (heterogeneous) - `T`: Result, has same element type as two inputs
    pub(super) fn execute_add(&self, inputs: Vec<&Tensor>) -> OperationResult {
        // Inputs
        if inputs.len() != 2 {
            return Err(onnx_error!("[Add] Operation requires 2 inputs."));
        }
        
        let a = inputs[0];
        let b = inputs[1];

        Ok(Arc::new(a + b))
    }

}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use ndarray::Array;
    use crate::operations::{Operation, OpType};

    #[test]
    fn valid_inputs() {
        let op = Operation::new(OpType::Add);

        let a =
            Array::from_iter((1..=12).map(|v| v as f32).cycle().take(4*2))
            .into_shape((4,2))
            .unwrap().into_dyn();

        let b =
            Array::from_iter((13..=24).map(|v| v as f32).cycle().take(4*2))
            .into_shape((4,2))
            .unwrap().into_dyn();
        
        let expected_result =
            Array::from_iter((1..=12).zip(13..=24).map(|(v1, v2)| v1 as f32 + v2 as f32).cycle().take(4*2))
            .into_shape((4,2))
            .unwrap().into_dyn();

        let result = op.execute(vec![Arc::new(a), Arc::new(b)]);
        assert!(result.is_ok(), "{:?}", result.unwrap_err());
        assert_eq!(result.unwrap(), Arc::new(expected_result));
    }

}