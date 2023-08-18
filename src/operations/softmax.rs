use std::{sync::Arc, f32::consts::E};
use ndarray::Axis;

use super::{Operation, OnnxError, onnx_error, Tensor, OperationResult, Attribute};

impl Operation {
    
    /// *(From <https://onnx.ai/onnx/operators/onnx__Softmax.html>)*
    /// 
    /// The operator computes the normalized exponential values for the given input:
    /// ```plain
    /// Softmax(input, axis) = Exp(input) / ReduceSum(Exp(input), axis=axis, keepdims=1)
    /// ```
    /// 
    /// The “axis” attribute indicates the dimension along which Softmax will be performed. The output tensor has the same shape
    /// and contains the Softmax values of the corresponding input.
    /// 
    /// # Attributes
    /// * **axis** - `INT` (default is `'-1'`): Describes the dimension Softmax will be performed on. Negative value means
    ///   counting dimensions from the back. Accepted range is `[-r, r-1]` where `r = rank(input)`.
    /// 
    /// # Inputs
    /// * **input** (heterogeneous) - `T`: The input tensor of `rank >= axis`.
    /// 
    /// # Outputs
    /// * **output** (heterogeneous) - `T`: The output values with the same shape as the input tensor.
    pub(super) fn execute_softmax(&self, inputs: Vec<&Tensor>) -> OperationResult {
        // Inputs
        let input = *inputs.get(0).ok_or(onnx_error!("Softmax must have 1 input."))?;

        // Attributes
        let axis: usize = match self.attributes.get("axis") {
            Some(Attribute::Int(val)) if *val >= 0 => (*val).try_into().unwrap(),
            Some(Attribute::Int(val)) if *val < 0 => {
                (inputs[0].shape().len() as isize - *val)
                    .try_into().map_err(|_| onnx_error!("Result axis is a negative number."))?
            },
            None => inputs[0].shape().len()-1,
            _ => return Err(onnx_error!("axis has an invalid attribute type."))
        };

        // input_exp = e^input
        let input_exp = input.mapv(|v| E.powf(v));
        
        // Sum values of input_exp over the axis (this generates a new array with one less dimension as the input), then
        // broadcast the result back to the same shape of the input.
        let axis_sums =
            input_exp
                .sum_axis(Axis(axis))
                .insert_axis(Axis(axis))
                .broadcast(input_exp.dim()).unwrap()
                .to_owned();

        let result = &input_exp / &axis_sums;
        Ok(Arc::new(result))
    }

}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, collections::HashMap};
    use ndarray::array;
    use crate::operations::{Operation, OpType, Attribute};

    #[test]
    fn one_dimension() {
        let op = Operation::new(OpType::Softmax);

        let input =
            array![[-1., 0., 1.]].into_dyn();

        let expected_result =
            array![[0.09003058, 0.24472848, 0.66524094]].into_dyn();

        let result = op.execute(vec![Arc::new(input)]);
        assert!(result.is_ok(), "{:?}", result.unwrap_err());
        assert_eq!(result.unwrap(), Arc::new(expected_result))
    }

    #[test]
    fn two_dimensions() {
        let op = Operation::new(OpType::Softmax);

        let input =
            array![
                [-1., 0., 1.],
                [2.,  3., 4.]
            ].into_dyn();

        let expected_result =
            array![
                [0.09003058, 0.24472848, 0.66524094],
                [0.09003057, 0.24472846, 0.66524094]
            ].into_dyn();

        let result = op.execute(vec![Arc::new(input)]);
        assert!(result.is_ok(), "{:?}", result.unwrap_err());
        assert_eq!(result.unwrap(), Arc::new(expected_result))
    }

    #[test]
    fn three_dimensions_axis() {
        let attributes = HashMap::from([
            ("axis".to_string(), Attribute::Int(1))
        ]);
        let op = Operation::with_attributes(OpType::Softmax, attributes);

        let input =
            array![
                [
                    [1.,2.],
                    [3.,4.]
                ],
                [
                    [5.,6.],
                    [7.,8.]
                ]
            ].into_dyn();

        let expected_result =
            array![
                [
                    [0.11920293, 0.11920293],
                    [0.880797,   0.880797  ]
                ],
                [
                    [0.11920293, 0.11920293],
                    [0.8807971,  0.8807971 ]
                ]
            ].into_dyn();

        let result = op.execute(vec![Arc::new(input)]);
        assert!(result.is_ok(), "{:?}", result.unwrap_err());
        assert_eq!(result.unwrap(), Arc::new(expected_result))
    }

}