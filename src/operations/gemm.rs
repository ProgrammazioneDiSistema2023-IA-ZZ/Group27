use std::sync::Arc;
use ndarray::Ix2;

use super::{Operation, OnnxError, onnx_error, Tensor, OperationResult, Attribute};

impl Operation {

    /// *(From <https://onnx.ai/onnx/operators/onnx__Gemm.html>)
    /// 
    /// General Matrix multiplication: <https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3>
    /// * `A’ = transpose(A) if transA else A`
    /// * `B’ = transpose(B) if transB else B`
    /// 
    /// Compute `Y = alpha * A’ * B’ + beta * C`, where input tensor `A` has shape `(M, K)` or `(K, M)`, input tensor `B` has
    /// shape `(K, N)` or `(N, K)`, input tensor `C` is broadcastable to shape `(M, N)`, and output tensor `Y` has shape `(M,
    /// N)`. `A` will be transposed before doing the computation if attribute `transA` is non-zero, same for `B` and `transB`.
    /// This operator supports unidirectional broadcasting (tensor `C` should be unidirectional broadcastable to tensor `A *
    /// B`); for more details please check Broadcasting in ONNX. This operator has optional inputs/outputs. See ONNX IR for more
    /// details about the representation of optional arguments. An empty string may be used in the place of an actual argument’s
    /// name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may
    /// also be simply omitted.
    /// 
    /// # Attributes
    /// 
    /// * **alpha** - `FLOAT` (default is `'1.0'`): Scalar multiplier for the product of input tensors `A * B`.
    /// * **beta** - `FLOAT` (default is `'1.0'`): Scalar multiplier for input tensor `C`.
    /// * **transA** - `INT` (default is `'0'`): Whether `A` should be transposed
    /// * **transB** - `INT` (default is `'0'`): Whether `B` should be transposed
    /// 
    /// # Inputs
    /// * **A** (heterogeneous) - `T`: Input tensor `A`. The shape of `A` should be `(M, K)` if `transA` is 0, or `(K, M)` if `transA` is non-zero.
    /// * **B** (heterogeneous) - `T`: Input tensor `B`. The shape of `B` should be `(K, N)` if `transB` is 0, or `(N, K)` if `transB` is non-zero.
    /// * **C** (optional, heterogeneous) - `T`: Optional input tensor `C`. If not specified, the computation is done as if `C` is a scalar 0. The shape of `C` should be unidirectional broadcastable to `(M, N)`.
    /// 
    /// # Outputs
    /// * **Y** (heterogeneous) - `T`: Output tensor of shape `(M, N)`.
    pub(super) fn execute_gemm(&self, inputs: Vec<&Tensor>) -> OperationResult {
        // Inputs
        if ![2,3].contains(&inputs.len()) {
            return Err(onnx_error!("Gemm operation must contain 2 or 3 inputs."));
        }

        let a = inputs[0];
        let b = inputs[1];
        let c = inputs.get(2);

        // Attributes
        let alpha = match self.attributes.get("alpha") {
            Some(Attribute::Float(val)) => *val,
            None | Some(Attribute::Undefined) => 1.,
            _ => return Err(onnx_error!("alpha attribute has an invalid value type"))
        };

        let beta = match self.attributes.get("beta") {
            Some(Attribute::Float(val)) => *val,
            None | Some(Attribute::Undefined) => 1.,
            _ => return Err(onnx_error!("alpha attribute has an invalid value type"))
        };

        let trans_a = match self.attributes.get("transA") {
            Some(Attribute::Int(val)) => *val,
            None | Some(Attribute::Undefined) => 0,
            _ => return Err(onnx_error!("alpha attribute has an invalid value type"))
        };

        let trans_b = match self.attributes.get("transB") {
            Some(Attribute::Int(val)) => *val,
            None | Some(Attribute::Undefined) => 0,
            _ => return Err(onnx_error!("alpha attribute has an invalid value type"))
        };

        // Compute A' and B'
        let a_prime =
            if trans_a == 1 { a.t() } else { a.view() }
                .into_dimensionality::<Ix2>().map_err(|_| onnx_error!("Only two-dimensional products are supported. Matrix A was found to have {} dimension(s).", a.shape().len()))?;

        let b_prime =
            if trans_b == 1 { b.t() } else { b.view() }
                .into_dimensionality::<Ix2>().map_err(|_| onnx_error!("Only two-dimensional products are supported. Matrix B was found to have {} dimension(s).", b.shape().len()))?;

        // result = alpha * A' * B' (+ beta * C)
        let mut result = alpha * a_prime.dot(&b_prime);

        if let Some(c) = c {
            let c_shape = [a_prime.shape()[0], b_prime.shape()[1]];
            let c_2d =
                c.view()
                 .into_shape(c_shape)
                 .map_err(|_| onnx_error!("Could not convert matrix C to shape {}x{}.", c_shape[0], c_shape[1]))?;
            result = result + beta * &c_2d;
        }

        Ok(Arc::new(result.into_dyn()))
    }

}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, collections::HashMap};
    use ndarray::{Array, array, Array1};
    use crate::operations::{Operation, OpType, Attribute};

    #[test]
    fn without_c() {
        let attributes = HashMap::from([
            ("alpha".to_string(), Attribute::Float(2.)),
            ("beta".to_string(), Attribute::Float(1.)),
            ("transA".to_string(), Attribute::Int(0)),
            ("transB".to_string(), Attribute::Int(0))
        ]);
        let op = Operation::with_attributes(OpType::Gemm, attributes);

        let a =
            Array::from_iter((1..=6).map(|i| i as f32).cycle().take(3*2))
            .into_shape((3,2))
            .unwrap().into_dyn();

        let b =
            Array::from_iter((7..=12).map(|i| i as f32).cycle().take(2*3))
            .into_shape((2,3))
            .unwrap().into_dyn();

        let expected_result =
            2f32*array![
                [27., 30.,  33.],
                [61., 68.,  75.],
                [95., 106., 117.]
            ].into_dyn();
    
        let result = op.execute(vec![Arc::new(a), Arc::new(b)]);

        assert!(result.is_ok(), "{:?}", result.unwrap_err());
        assert_eq!(result.unwrap(), Arc::new(expected_result));
    }

    #[test]
    fn with_c() {
        let attributes = HashMap::from([
            ("alpha".to_string(), Attribute::Float(2.)),
            ("beta".to_string(), Attribute::Float(1.)),
            ("transA".to_string(), Attribute::Int(0)),
            ("transB".to_string(), Attribute::Int(0))
        ]);
        let op = Operation::with_attributes(OpType::Gemm, attributes);

        let a =
            Array::from_iter((1..=6).map(|i| i as f32).cycle().take(3*2))
            .into_shape((3,2))
            .unwrap().into_dyn();

        let b =
            Array::from_iter((7..=12).map(|i| i as f32).cycle().take(2*3))
            .into_shape((2,3))
            .unwrap().into_dyn();

        let c =
            Array1::from_iter((13..=21).map(|i| i as f32).cycle().take(3*3))
            .into_dyn();

        let expected_result =
            array![
                [67.,  74.,  81. ],
                [138., 153., 168.],
                [209., 232., 255.]
            ].into_dyn();
    
        let result = op.execute(vec![Arc::new(a), Arc::new(b), Arc::new(c)]);

        assert!(result.is_ok(), "{:?}", result.unwrap_err());
        assert_eq!(result.unwrap(), Arc::new(expected_result));
    }

    #[test]
    fn transposed() {
        let attributes = HashMap::from([
            ("alpha".to_string(), Attribute::Float(2.)),
            ("beta".to_string(), Attribute::Float(1.)),
            ("transA".to_string(), Attribute::Int(1)),
            ("transB".to_string(), Attribute::Int(1))
        ]);
        let op = Operation::with_attributes(OpType::Gemm, attributes);

        let a =
            Array::from_iter((1..=6).map(|i| i as f32).cycle().take(3*2))
            .into_shape((3,2))
            .unwrap().into_dyn();

        let b =
            Array::from_iter((7..=12).map(|i| i as f32).cycle().take(2*3))
            .into_shape((2,3))
            .unwrap().into_dyn();

        let expected_result =
            array![
                [152., 206.],
                [200., 272.]
            ].into_dyn();
    
        let result = op.execute(vec![Arc::new(a), Arc::new(b)]);

        assert!(result.is_ok(), "{:?}", result.unwrap_err());
        assert_eq!(result.unwrap(), Arc::new(expected_result));
    }

}