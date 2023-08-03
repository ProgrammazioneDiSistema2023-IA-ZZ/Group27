use std::sync::Arc;
use ndarray::{Array1, Array4, s};

use super::{Operation, OnnxError, onnx_error, Tensor, OperationResult, Attribute};

impl Operation {

    /// *(From <https://onnx.ai/onnx/operators/onnx__Conv.html>)*
    /// 
    /// The convolution operator consumes an input tensor and a filter, and computes the output.
    /// 
    /// # Attributes
    /// * **auto_pad** - `STRING` (default is `'NOTSET'`): auto_pad must be either `NOTSET`, `SAME_UPPER`, `SAME_LOWER` or
    ///   `VALID`. Where default value is `NOTSET`, which means explicit padding is used. `SAME_UPPER` or `SAME_LOWER` mean pad
    ///   the input so that `output_shape[i] = ceil(input_shape[i] / strides[i])` for each axis `i`. The padding is split
    ///   between the two sides equally or almost equally (depending on whether it is even or odd). In case the padding is an
    ///   odd number, the extra padding is added at the end for `SAME_UPPER` and at the beginning for `SAME_LOWER`.
    /// * **dilations** - `INTS` : dilation value along each spatial axis of the filter. If not present, the dilation defaults
    ///   is 1 along each spatial axis. (*Unhandled*)
    /// * **group** - `INT` (default is `'1'`): number of groups input channels and output channels are divided into.
    ///   (*Unhandled*)
    /// * **kernel_shape** - `INTS` : The shape of the convolution kernel. If not present, should be inferred from input `W`.
    /// * **pads** - `INTS` : Padding for the beginning and ending along each spatial axis, it can take any value greater than
    ///   or equal to 0. The value represent the number of pixels added to the beginning and end part of the corresponding axis.
    ///   pads format should be as follow `[x1_begin, x2_begin…x1_end, x2_end,…]`, where `xi_begin` the number of pixels added
    ///   at the beginning of axis `i` and `xi_end`, the number of pixels added at the end of axis `i`. This attribute cannot be
    ///   used simultaneously with auto_pad attribute. If not present, the padding defaults to 0 along start and end of each
    ///   spatial axis.
    /// * **strides** - `INTS` : Stride along each spatial axis. If not present, the stride defaults is 1 along each spatial
    ///   axis.
    /// 
    /// # Inputs
    /// * **X** (heterogeneous) - `T`: Input data tensor from previous layer; has size (`N x C x H x W`), where `N` is the batch
    ///   size, `C` is the number of channels, and `H` and `W` are the height and width. Note that this is for the 2D image.
    /// * **W** (heterogeneous) - `T`: The weight tensor that will be used in the convolutions; has size (`M x C/group x kH x kW`),
    ///   where `C` is the number of channels, and `kH` and `kW` are the height and width of the kernel, and `M` is the number of
    ///   feature maps.
    /// * **B** (optional, heterogeneous) - `T`: Optional 1D bias to be added to the convolution, has size of `M`.
    /// 
    /// # Outputs
    /// * **Y** (heterogeneous) - `T`: Output data tensor that contains the result of the convolution. The output dimensions are
    ///   functions of the kernel size, stride size, and pad lengths.
    pub(super) fn execute_conv(&self, inputs: Vec<&Tensor>) -> OperationResult {
        // Input
        let data = *inputs.get(0).ok_or(onnx_error!("Missing data input from Conv operation"))?;
        let weights = *inputs.get(1).ok_or(onnx_error!("Missing weights input from Conv operation"))?;
        
        // Per ora, solo gli input a 4 dimensioni sono supportati
        let data_shape: [usize; 4] = data.shape().try_into().map_err(|_| onnx_error!("Only 2D (image) convolution is supported at the moment."))?;
        let weights_shape: [usize; 4] = weights.shape().try_into().map_err(|_| onnx_error!("Only 2D (image) convolution is supported at the moment."))?;

        // Estrazione e controlli sulle dimensioni dei due input.
        let (
            [ batches, channels, data_h, data_w ],
            [ fmaps, channels_w, kernel_h, kernel_w ]
        ) = (data_shape, weights_shape);

        if channels != channels_w {
            return Err(onnx_error!("Input tensors must have the same number of channels, as groups are not supported ({channels} and {channels_w} supplied)."));
        }

        // Bias: valori da sommare, unici per ogni canale
        let bias =
            inputs.get(2).map_or_else(
                || Array1::<f32>::zeros(fmaps).into_dyn(),
                |b| (*b).clone()
            );

        // Attributi
        // Dilation: non gestita
        let [ dilation_h, dilation_w ] = match self.attributes.get("dilations") {
            Some(Attribute::Ints(val)) => val.as_slice().try_into().map_err(|_| onnx_error!("Dilation should contain two dimensions."))?,
            None => [1, 1],
            _ => return Err(onnx_error!("dilations attribute has an invalid value type"))
        };

        if dilation_h > 1 || dilation_w > 1 {
            return Err(onnx_error!("Dilated convolutions are not supported."))
        }
        
        // Dimensioni del kernel
        let [ kernel_h, kernel_w ] = match self.attributes.get("kernel_shape") {
            Some(Attribute::Ints(val)) => val.as_slice().try_into().map_err(|_| onnx_error!("Kernel size should contain two dimensions."))?,
            None => [kernel_h, kernel_w],
            _ => return Err(onnx_error!("kernel_shape attribute has an invalid value type"))
        };

        // Gruppi: non gestiti
        let _groups = match self.attributes.get("groups") {
            Some(Attribute::Int(1)) => 1usize,
            Some(Attribute::Int(_)) => return Err(onnx_error!("Grouped convolutions are not supported.")),
            None => 1usize,
            _ => return Err(onnx_error!("groups attribute has an invalid value type"))
        };

        // Strides: finestre saltate in lunghezza/altezza
        let [strides_h, strides_w] = match self.attributes.get("strides") {
            Some(Attribute::Ints(val)) => val.as_slice().try_into().map_err(|_| onnx_error!("Strides should contain two dimensions."))?,
            None => [1, 1],
            _ => return Err(onnx_error!("groups attribute has an invalid value type"))
        };
        
        // Padding: manuale o automatico, indica righe/colonne aggiuntive con valori costanti.
        let [ pad_n, pad_w, pad_s, pad_e ] = self.get_padding(fmaps, (data_h, data_w), (kernel_h, kernel_w), (strides_h, strides_w))?;

        /*** CONVOLUZIONE ***/

        // Clona l'input, con eventuale padding aggiunto.
        let (padded_h, padded_w) = (data_h + pad_n + pad_s, data_w + pad_e + pad_w);
        let mut padded_data = Array4::<f32>::zeros((batches, channels, padded_h, padded_w));
        padded_data.slice_mut(s![.., .., pad_n..pad_n+data_h, pad_w..pad_w+data_w]).assign(data);

        // Calcola le dimensioni dell'output
        let (out_h, out_w) = (0..fmaps).fold((padded_h, padded_w), |(p_h, p_w), _| ((p_h-kernel_h)/strides_h+1, (p_w-kernel_w)/strides_w+1));
        
        // Tensor risultato, inizializzato con tutti zeri
        let mut result = Array4::<f32>::zeros((batches, channels, out_h, out_w));

        for (n_batch, batch) in padded_data.outer_iter().enumerate() {
            for (n_channel, channel) in batch.outer_iter().enumerate() {
                // Performa tante convoluzioni quante sono le feature maps (valore M nella documentazione)
                let output =
                    (0..fmaps)
                        .into_iter()
                        .fold(Ok(channel.to_owned()), |output, fmap| {
                            let bias_val = bias[fmap];
                            let kernel = weights.slice(s![fmap, n_channel, .., ..]);
                            Self::map_2d_windows(
                                output?.view(),
                                kernel.dim(),
                                |window| (&window * &kernel).sum() + bias_val,
                                (strides_h, strides_w)
                            )
                        })?;
                // Assegna al risultato
                result.slice_mut(s![n_batch, n_channel, .., ..]).assign(&output);
            }
        }

        Ok(Arc::new(result.into_dyn()))
    }

}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, collections::HashMap};
    use ndarray::{Array, Array1};
    use crate::operations::{Operation, OpType, Attribute};

    #[test]
    fn simple() {
        let op = Operation::new(OpType::Conv);

        let data =
            Array::from_iter((1..=16).map(|i| i as f32).cycle().take(2*3*4*4))
            .into_shape((2,3,4,4))
            .unwrap().into_dyn();

        let weights =
            Array::from_iter((1..=4).map(|i| i as f32).cycle().take(1*3*2*2))
            .into_shape((1,3,2,2))
            .unwrap().into_dyn();

        let expected_result =
            Array::from_iter([
                44.,  54.,  64., 
                84.,  94.,  104.,
                124., 134., 144.
            ].into_iter().cycle().take(2*3*3*3))
            .into_shape((2,3,3,3))
            .unwrap().into_dyn();
    
        let result = op.execute(vec![Arc::new(data), Arc::new(weights)]);

        assert!(result.is_ok(), "{:?}", result.unwrap_err());
        assert_eq!(result.unwrap(), Arc::new(expected_result));
    }

    #[test]
    fn with_padding() {
        let attributes = HashMap::from([
            ("pads".to_string(), Attribute::Ints(vec![1,1,1,1]))
        ]);
        let op = Operation::with_attributes(OpType::Conv, attributes);

        let data =
            Array::from_iter((1..=4).map(|i| i as f32).cycle().take(2*3*2*2))
            .into_shape((2,3,2,2))
            .unwrap().into_dyn();

        let weights =
            Array::from_iter((1..=4).map(|i| i as f32).cycle().take(1*3*2*2))
            .into_shape((1,3,2,2))
            .unwrap().into_dyn();

        let expected_result =
            Array::from_iter([
                4.,  11., 6.,
                14., 30., 14.,
                6.,  11., 4.
            ].into_iter().cycle().take(2*3*3*3))
            .into_shape((2,3,3,3))
            .unwrap().into_dyn();
    
        let result = op.execute(vec![Arc::new(data), Arc::new(weights)]);

        assert!(result.is_ok(), "{:?}", result.unwrap_err());
        assert_eq!(result.unwrap(), Arc::new(expected_result));
    }

    #[test]
    fn with_strides() {
        let attributes = HashMap::from([
            ("strides".to_string(), Attribute::Ints(vec![2,2]))
        ]);
        let op = Operation::with_attributes(OpType::Conv, attributes);

        let data =
            Array::from_iter((1..=16).map(|i| i as f32).cycle().take(2*3*4*4))
            .into_shape((2,3,4,4))
            .unwrap().into_dyn();

        let weights =
            Array::from_iter((1..=4).map(|i| i as f32).cycle().take(1*3*2*2))
            .into_shape((1,3,2,2))
            .unwrap().into_dyn();

        let expected_result =
            Array::from_iter([44.,64.,124.,144.].into_iter().cycle().take(2*3*2*2))
            .into_shape((2,3,2,2))
            .unwrap().into_dyn();
    
        let result = op.execute(vec![Arc::new(data), Arc::new(weights)]);

        assert!(result.is_ok(), "{:?}", result.unwrap_err());
        assert_eq!(result.unwrap(), Arc::new(expected_result));
    }

    #[test]
    fn multiple_fmaps() {
        let op = Operation::new(OpType::Conv);

        let data =
            Array::from_iter((1..=16).map(|i| i as f32).cycle().take(2*3*4*4))
            .into_shape((2,3,4,4))
            .unwrap().into_dyn();

        let weights =
            Array::from_iter((1..=4).map(|i| i as f32).cycle().take(2*3*2*2))
            .into_shape((2,3,2,2))
            .unwrap().into_dyn();

        let expected_result =
            Array::from_iter([780., 880., 1180., 1280.].into_iter().cycle().take(2*3*2*2))
            .into_shape((2,3,2,2))
            .unwrap().into_dyn();
    
        let result = op.execute(vec![Arc::new(data), Arc::new(weights)]);

        assert!(result.is_ok(), "{:?}", result.unwrap_err());
        assert_eq!(result.unwrap(), Arc::new(expected_result));
    }

    #[test]
    fn with_bias() {
        let op = Operation::new(OpType::Conv);

        let data =
            Array::from_iter((1..=16).map(|i| i as f32).cycle().take(2*3*4*4))
            .into_shape((2,3,4,4))
            .unwrap().into_dyn();

        let weights =
            Array::from_iter((1..=4).map(|i| i as f32).cycle().take(2*3*2*2))
            .into_shape((2,3,2,2))
            .unwrap().into_dyn();

        let bias = Array1::from_iter([1., 2.].into_iter()).into_dyn();

        let expected_result =
            Array::from_iter([792., 892., 1192., 1292.].into_iter().cycle().take(2*3*2*2))
            .into_shape((2,3,2,2))
            .unwrap().into_dyn();
    
        let result = op.execute(vec![Arc::new(data), Arc::new(weights), Arc::new(bias)]);

        assert!(result.is_ok(), "{:?}", result.unwrap_err());
        assert_eq!(result.unwrap(), Arc::new(expected_result));
    }

    #[test]
    fn auto_pad() {
        let attributes = HashMap::from([
            ("auto_pad".to_string(), Attribute::String("SAME_LOWER".to_string()))
        ]);
        let op = Operation::with_attributes(OpType::Conv, attributes);

        let data =
            Array::from_iter((1..=16).map(|i| i as f32).cycle().take(2*3*4*4))
            .into_shape((2,3,4,4))
            .unwrap().into_dyn();

        let data_dim = data.dim();

        let weights =
            Array::from_iter((1..=4).map(|i| i as f32).cycle().take(2*3*3*3))
            .into_shape((2,3,3,3))
            .unwrap().into_dyn();
    
        let result = op.execute(vec![Arc::new(data), Arc::new(weights)]);

        assert!(result.is_ok(), "{:?}", result.unwrap_err());
        assert_eq!(result.unwrap().dim(), data_dim);
    }

}