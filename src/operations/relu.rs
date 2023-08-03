use std::sync::Arc;
use super::{Operation, OnnxError, onnx_error, Tensor, OperationResult};

impl Operation {

    /// *(From <https://onnx.ai/onnx/operators/onnx__Relu.html>)*
    /// 
    /// Relu takes one input data (Tensor) and produces one output data (Tensor) where the rectified linear function, `y = max(0, x)`, is applied to the tensor elementwise.
    /// 
    /// # Inputs
    /// * **X** (heterogeneous) - `T`: Input tensor
    /// 
    /// # Outputs
    /// * **Y** (heterogeneous) - `T`: Output tensor
    pub(super) fn execute_relu(&self, inputs: Vec<&Tensor>) -> OperationResult {
        if inputs.len() != 1 {
            return Err(onnx_error!("Relu operation must have 1 input, {} supplied.", inputs.len()));
        }
        
        let result = inputs[0].mapv(|el| f32::max(el, 0.));
        Ok(Arc::new(result))
    }

}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use ndarray::array;
    use crate::operations::{Operation, OpType};

    #[test]
    fn valid_input() {
        let op = Operation::new(OpType::Relu);

        let data = array![
            [ 
                [1.,2.,-3.,4.],
                [5.,-6.,7.,-8.],
                [-9.,10.,-11.,12.]
            ],
            [ 
                [1.,2.,3.,-4.],
                [-5.,-6.,7.,-8.],
                [9.,-10.,11.,12.]
            ]
        ].into_dyn();

        let expected_result = array![
            [ 
                [1.,2.,0.,4.],
                [5.,0.,7.,0.],
                [0.,10.,0.,12.]
            ],
            [ 
                [1.,2.,3.,0.],
                [0.,0.,7.,0.],
                [9.,0.,11.,12.]
            ]
        ].into_dyn();

        let result = op.execute(vec![Arc::new(data)]);

        assert!(result.is_ok(), "{:?}", result.unwrap_err());
        assert_eq!(result.unwrap(), Arc::new(expected_result));
    }

}