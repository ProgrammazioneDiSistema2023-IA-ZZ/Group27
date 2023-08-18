use std::sync::Arc;
use itertools::{iproduct, Itertools};
use ndarray::{s, Array};
use rayon::prelude::{ParallelBridge, ParallelIterator};

use super::{Operation, OnnxError, onnx_error, Tensor, OperationResult, Attribute};

impl Operation {

    /// (From <https://onnx.ai/onnx/operators/onnx__LRN.html>)
    /// 
    /// Local Response Normalization proposed in the [AlexNet
    /// paper][<https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf>]. It
    /// normalizes over local input regions. The local region is defined across the channels. For an element `X[n, c, d1, ..., dk]`
    /// in a tensor of shape `(N x C x D1 x D2, ..., Dk)`, its region is `{X[n, i, d1, ..., dk] | max(0, c - floor((size -
    /// 1) / 2)) <= i <= min(C - 1, c + ceil((size - 1) / 2))}`.
    /// 
    /// `square_sum[n, c, d1, ..., dk] = sum(X[n, i, d1, ..., dk] ^ 2)`, where `max(0, c - floor((size - 1) / 2)) <= i <= min(C - 1,
    /// c + ceil((size - 1) / 2))`.
    /// 
    /// `Y[n, c, d1, ..., dk] = X[n, c, d1, ..., dk] / (bias + alpha / size * square_sum[n, c, d1, ..., dk] ) ^ beta`
    /// 
    /// # Attributes
    /// * **alpha** - `FLOAT` (default is `'0.0001'`): Scaling parameter.
    /// * **beta** - `FLOAT` (default is `'0.75'`): The exponent.
    /// * **bias** - `FLOAT` (default is `'1.0'`)
    /// * **size** - `INT` (required) : The number of channels to sum over
    /// 
    /// # Inputs
    /// * **X** (heterogeneous) - `T`: Input data tensor from the previous operator; dimensions for image case are (`N x C x H x W`),
    ///   where `N` is the batch size, `C` is the number of channels, and `H` and `W` are the height and the width of the
    ///   data.
    /// 
    /// # Outputs
    /// * **Y** (heterogeneous) - `T`: Output tensor, which has the shape and type as input tensor
    pub(super) fn execute_lrn(&self, inputs: Vec<&Tensor>) -> OperationResult {
        // Inputs
        let data = *inputs.get(0).ok_or(onnx_error!("Missing input data from LRN operation."))?;
        
        // Per ora, solo gli input a 4 dimensioni sono supportati
        let data_shape: [usize; 4] = data.shape().try_into().map_err(|_| onnx_error!("Only 2D (image) LRN is supported at the moment."))?;
        let [ batches, channels, data_h, data_w ] = data_shape;
        
        // Attributes
        let alpha = match self.attributes.get("alpha") {
            Some(Attribute::Float(val)) => *val,
            None => 1e-4,
            _ => return Err(onnx_error!("alpha attribute has an invalid value type"))
        };

        let beta = match self.attributes.get("beta") {
            Some(Attribute::Float(val)) => *val,
            None => 0.75,
            _ => return Err(onnx_error!("beta attribute has an invalid value type"))
        };

        let bias = match self.attributes.get("bias") {
            Some(Attribute::Float(val)) => *val,
            None => 1.,
            _ => return Err(onnx_error!("bias attribute has an invalid value type"))
        };

        let size: usize = match self.attributes.get("size") {
            Some(Attribute::Int(val)) => (*val).try_into().map_err(|_| onnx_error!("Size attribute is a negative value."))?,
            None => return Err(onnx_error!("size attribute is required")),
            _ => return Err(onnx_error!("size attribute has an invalid value type"))
        };

        /*** LRN ***/

        // Calculate value of each cell of the result in parallel, then insert all values into the result array. The parallel
        // iterator collects values in a random fashion, so we also need to sort by the global index of each value
        let values =
            iproduct!(0..batches, 0..channels, 0..data_h, 0..data_w)
                .par_bridge()
                .map(|(n_batch, n_channel, i, j)| {
                    let ch_from = usize::checked_sub(n_channel, (size-1)/2).unwrap_or(0);
                    let ch_to = usize::min(channels-1, n_channel + f32::ceil((size-1) as f32/2.) as usize);

                    let square_sum = data.slice(s![n_batch, ch_from..=ch_to, i, j]).mapv(|v| v.powi(2)).sum();
                    let res = data[[n_batch, n_channel, i, j]] / (bias + (alpha / size as f32) * square_sum).powf(beta);
                    (
                        j + i * data_w + n_channel * data_w * data_h + n_batch * data_w * data_h * channels, // Global index
                        res // Result
                    )
                })
                .collect::<Vec<(usize, f32)>>()
                .into_iter()
                .sorted_by(|(i1, _), (|i2, _)| i1.cmp(i2))
                .map(|(_, v)| v)
                .collect::<Vec<f32>>();

        let result =
            Array::from_shape_vec((batches, channels, data_h, data_w), values)
            .unwrap().into_dyn();

        Ok(Arc::new(result))
    }

}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, collections::HashMap};
    use ndarray::array;
    
    use crate::operations::{Operation, OpType, Attribute};

    #[test]
    fn valid() {
        let attributes = HashMap::from([
            ("alpha".to_string(), Attribute::Float(1.)),
            ("beta".to_string(), Attribute::Float(1.)),
            ("bias".to_string(), Attribute::Float(0.)),
            ("size".to_string(), Attribute::Int(2))
        ]);
        let op = Operation::with_attributes(OpType::LRN, attributes);
    
        let data = array![
            [
                [1., 2., 3.],
                [4., 3., 6.],
                [7., 8., 9.],
            ],
            [
                [1., 2., 1.],
                [2., 3., 2.],
                [3., 4., 3.],
            ],
            [
                [2., 1., 2.],
                [3., 2., 3.],
                [4., 3., 4.]
            ],
            [
                [4., 2., 1.],
                [5., 2., 1.],
                [2., 2., 4.]
            ]
        ].into_shape((1, 4, 3, 3)).unwrap().into_dyn();

        let expected_result = array![
            [
                [ 1.00, 0.50, 0.60 ],
                [ 0.40, 0.33, 0.30 ],
                [ 0.24, 0.20, 0.2 ]
            ],
            [
                [ 0.40, 0.80, 0.40 ],
                [ 0.31, 0.46, 0.31 ],
                [ 0.24, 0.32, 0.24 ]
            ],
     
            [
                [ 0.20, 0.40, 0.80 ],
                [ 0.18, 0.50, 0.60 ],
                [ 0.40, 0.46, 0.25 ]
            ],
            [
                [ 0.50, 1.00, 2.00 ],
                [ 0.40, 1.00, 2.00 ],
                [ 1.00, 1.00, 0.50 ]
            ]
        ].into_shape((1, 4, 3, 3)).unwrap().into_dyn();

        let result = op.execute(vec![Arc::new(data)]);

        assert!(result.is_ok(), "{:?}", result.unwrap_err());
        assert_eq!(result.unwrap().mapv(|v| (v*100.).round() / 100.), expected_result);
    }

}