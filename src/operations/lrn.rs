use std::sync::Arc;
use ndarray::{Array1, s, Array4};

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
        // Input
        let data = *inputs.get(0).ok_or(onnx_error!("Missing input data from LRN operation."))?;
        
        // Per ora, solo gli input a 4 dimensioni sono supportati
        let data_shape: [usize; 4] = data.shape().try_into().map_err(|_| onnx_error!("Only 2D (image) LRN is supported at the moment."))?;
        let [ batches, channels, data_h, data_w ] = data_shape;
        
        // Attributi
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
        let mut result = Array4::<f32>::zeros((batches, channels, data_h, data_w));
        for (n_batch, batch) in data.outer_iter().enumerate() {
            for (n_channel, channel) in batch.outer_iter().enumerate() {
                let ch_from = usize::checked_sub(n_channel, size/2).unwrap_or(0);
                let ch_to = usize::min(channels-1, f32::ceil(n_channel as f32 + size as f32/2.) as usize);

                let normalized = 
                    channel
                        .indexed_iter()
                        .map(|(dim, val)| {
                            let quadratic_sum = batch.slice(s![ch_from..=ch_to, dim[0], dim[1]]).mapv(|v| v.powi(2)).sum();
                            *val / (bias + alpha * quadratic_sum).powf(beta) // NOTA: la formula potrebbe essere sbagliata
                        })
                        .collect::<Array1<f32>>()
                        .to_shape(channel.dim())
                        .unwrap()
                        .to_owned();
                
                result.slice_mut(s![n_batch, n_channel, .., ..]).assign(&normalized);
            }
        }

        Ok(Arc::new(result.into_dyn()))
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
                [0.50, 0.25, 0.30],
                [0.20, 0.17, 0.15],
                [0.12, 0.10, 0.10],
            ],
            [
                [0.17, 0.22, 0.07],
                [0.07, 0.14, 0.04],
                [0.04, 0.04, 0.03],
            ],
            [
                [0.10, 0.11, 0.33],
                [0.08, 0.12, 0.21],
                [0.14, 0.10, 0.10]
            ],
            [
                [0.20, 0.40, 0.20],
                [0.15, 0.25, 0.10],
                [0.10, 0.15, 0.13]
            ]
        ].into_shape((1, 4, 3, 3)).unwrap().into_dyn();

        let result = op.execute(vec![Arc::new(data)]);

        assert!(result.is_ok(), "{:?}", result.unwrap_err());
        assert_eq!(result.unwrap().mapv(|v| (v*100.).round() / 100.), expected_result);
    }

}