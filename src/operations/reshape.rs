use std::sync::Arc;
use super::{Operation, OnnxError, onnx_error, Tensor, OperationResult};

impl Operation {

    /// *(From <https://onnx.ai/onnx/operators/onnx__Reshape.html>)*
    /// 
    /// Reshape the input tensor similar to numpy.reshape. First input is the data tensor, second input is a shape tensor which
    /// specifies the output shape. It outputs the reshaped tensor. At most one dimension of the new shape can be -1. In this
    /// case, the value is inferred from the size of the tensor and the remaining dimensions. A dimension could also be 0, in
    /// which case the actual dimension value is unchanged (i.e. taken from the input tensor). If ‘allowzero’ is set, and the
    /// new shape includes 0, the dimension will be set explicitly to zero (i.e. not taken from input tensor). Shape (second
    /// input) could be an empty shape, which means converting to a scalar. The input tensor’s shape and the output tensor’s
    /// shape are required to have the same number of elements.
    /// 
    /// If the attribute `allowzero` is set, it is invalid for the specified shape to contain both a zero value and -1, as the
    /// value of the dimension corresponding to -1 cannot be determined uniquely.
    /// 
    /// # Attributes
    /// * **allowzero** - `INT` (default is `'0'`): (Optional) By default, when any value in the 'shape' input is equal to zero
    ///   the corresponding dimension value is copied from the input tensor dynamically. allowzero=1 indicates that if any value
    ///   in the 'shape' input is set to zero, the zero value is honored, similar to NumPy.
    /// 
    /// # Inputs
    /// * **data** (heterogeneous) - `T`: An input tensor.
    /// * **shape** (heterogeneous) - `tensor(int64)`: Specified shape for output.
    /// 
    /// # Outputs
    /// * **reshaped** (heterogeneous) - `T`: Reshaped data.
    pub(super) fn execute_reshape(&self, inputs: Vec<&Tensor>) -> OperationResult {        
        if inputs.len() != 2 {
            return Err(onnx_error!("Reshape operation must have 2 inputs."));
        }
       
        let (data, shape) = (inputs[0], inputs[1]);
        if shape.ndim() != 1 {
            return Err(onnx_error!("Shape (2nd) input must be one-dimensional"))
        }
        let shape_slice = shape.iter().map(|val| *val as usize).collect::<Vec<usize>>();

        let result =
            data.to_shape(shape_slice)
                .map_err(|e| onnx_error!("Could not change shape of tensor: {}", e.to_string()))?
                .to_owned();

        Ok(Arc::new(result))
    }

}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use ndarray::array;

    use crate::operations::{Operation, OpType};

    #[test]
    fn valid_inputs() {
        let op = Operation::new(OpType::Reshape);

        let data = array![
            [ 
                [1.,2.,3.,4.],
                [5.,6.,7.,8.],
                [9.,10.,11.,12.]
            ],
            [
                [1.,2.,3.,4.],
                [5.,6.,7.,8.],
                [9.,10.,11.,12.]
            ]
        ].into_dyn();
        let shape = array![4., 3., 2.].into_dyn();

        let expected_result = data.clone().into_shape((4,3,2)).unwrap().into_dyn(); 

        let result = op.execute(vec![Arc::new(data), Arc::new(shape)]);

        assert!(result.is_ok(), "{:?}", result.unwrap_err());
        assert_eq!(result.unwrap(), Arc::new(expected_result));
    }

    #[test]
    fn invalid_shape_input() {
        let op = Operation::new(OpType::Reshape);

        let data = array![
            [ 
                [1.,2.,3.,4.],
                [5.,6.,7.,8.],
                [9.,10.,11.,12.]
            ],
            [
                [1.,2.,3.,4.],
                [5.,6.,7.,8.],
                [9.,10.,11.,12.]
            ]
        ].into_dyn();
        let shape = array![[4., 3., 2.],[2., 3., 4.]].into_dyn(); 

        let result = op.execute(vec![Arc::new(data), Arc::new(shape)]);
        assert!(result.is_err());
    }

}