use std::sync::Arc;
use ndarray::{concatenate, Axis};

use super::{Operation, OnnxError, onnx_error, Tensor, OperationResult, Attribute};

impl Operation {

    /// *(From <https://onnx.ai/onnx/operators/onnx__Concat.html>)*
    /// 
    /// Concatenate a list of tensors into a single tensor. All input tensors must have the same shape, except for the dimension
    /// size of the axis to concatenate on.
    /// 
    /// # Attributes
    /// * **axis** - `INT` (required) : Which axis to concat on. A negative value means counting dimensions from the back.
    ///   Accepted range is `[-r, r-1]` where `r = rank(inputs)`…
    /// 
    /// # Inputs
    /// Between 1 and 2147483647 inputs.
    /// 
    /// * **inputs** (variadic, heterogeneous) - `T`: List of tensors for concatenation
    /// 
    /// # Outputs
    /// * **concat_result** (heterogeneous) - `T`: Concatenated tensor
    pub(super) fn execute_concat(&self, inputs: Vec<&Tensor>) -> OperationResult {
        // Attributes
        let axis: usize = match self.attributes.get("axis") {
            Some(Attribute::Int(val)) if *val >= 0 => (*val).try_into().unwrap(),
            Some(Attribute::Int(val)) if *val < 0 => {
                (inputs[0].shape().len() as isize - *val)
                    .try_into().map_err(|_| onnx_error!("Result axis is a negative number."))?
            },
            None | Some(Attribute::Undefined) => return Err(onnx_error!("axis attribute not specified.")),
            _ => return Err(onnx_error!("axis has an invalid attribute type."))
        };

        let result = 
            concatenate(
                Axis(axis),
                inputs.iter().map(|i| i.view()).collect::<Vec<_>>().as_slice()
            ).map_err(|_| onnx_error!("Input shapes do not match."))?;

        Ok(Arc::new(result))
    }

}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, collections::HashMap};
    use ndarray::{Array, array};
    use crate::operations::{Operation, OpType, Attribute};

    #[test]
    fn simple() {
        let attributes = HashMap::from([
            ("axis".to_string(), Attribute::Int(1))
        ]);
        let op = Operation::with_attributes(OpType::Concat, attributes);

        let a =
            Array::from_iter((1..=4).map(|v| v as f32))
            .into_shape((2,2))
            .unwrap().into_dyn();

        let b =
            Array::from_iter((5..=10).map(|v| v as f32))
            .into_shape((2,3))
            .unwrap().into_dyn();

        let expected_result =
            array![
                [1., 2., 5., 6., 7. ],
                [3., 4., 8., 9., 10.]
            ].into_dyn();

        let result = op.execute(vec![Arc::new(a), Arc::new(b)]);
        assert!(result.is_ok(), "{:?}", result.unwrap_err());
        assert_eq!(result.unwrap(), Arc::new(expected_result))
    }

}