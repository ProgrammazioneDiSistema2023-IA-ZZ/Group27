use std::sync::{Arc, Mutex};
use ndarray::{Array1, Array4, s, parallel::prelude::{IntoParallelIterator, IndexedParallelIterator, ParallelIterator}, Array2};

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
            [ filters, channels_w, kernel_h, kernel_w ]
        ) = (data_shape, weights_shape);

        if channels != channels_w {
            return Err(onnx_error!("Input tensors must have the same number of channels, as groups are not supported ({channels} and {channels_w} supplied)."));
        }

        // Bias: valori da sommare, unici per ogni canale
        let bias =
            inputs.get(2).map_or_else(
                || Array1::<f32>::zeros(filters).into_dyn(),
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
            // Converti Vec<&isize> a [usize; 2]
            Some(Attribute::Ints(val)) => {
                val.into_iter()
                   .map(|v| usize::try_from(*v))
                   .collect::<Result<Vec<_>, _>>()
                   .map_err(|_| onnx_error!("kernel_shape attribute contains a negative number"))?
                   .as_slice()
                   .try_into().map_err(|_| onnx_error!("Kernel size should contain two dimensions."))?
            },
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
            Some(Attribute::Ints(val)) => {
                // Converti Vec<&isize> a [usize; 2]
                val.into_iter()
                   .map(|v| usize::try_from(*v))
                   .collect::<Result<Vec<_>, _>>()
                   .map_err(|_| onnx_error!("Strides attribute contains a negative number"))?
                   .as_slice()
                   .try_into().map_err(|_| onnx_error!("Kernel size should contain two dimensions."))?
            },
            None => [1, 1],
            _ => return Err(onnx_error!("groups attribute has an invalid value type"))
        };
        
        // Padding: manuale o automatico, indica righe/colonne aggiuntive con valori costanti.
        let [ pad_n, pad_w, pad_s, pad_e ] =
            self.get_padding(
                (data_h, data_w),
                (kernel_h, kernel_w),
                (strides_h, strides_w)
            )?;

        /*** CONVOLUZIONE ***/

        // Clona l'input, con eventuale padding aggiunto.
        let (padded_h, padded_w) = (data_h + pad_n + pad_s, data_w + pad_e + pad_w);
        let mut padded_data = Array4::<f32>::zeros((batches, channels, padded_h, padded_w));
        padded_data.slice_mut(s![.., .., pad_n..pad_n+data_h, pad_w..pad_w+data_w]).assign(data);

        // Calcola le dimensioni dell'output
        let (out_h, out_w) = ((padded_h-kernel_h)/strides_h+1, (padded_w-kernel_w)/strides_w+1);
        
        // Tensor risultato, inizializzato con tutti zeri
        let result = Mutex::new(Array4::<f32>::zeros((batches, filters, out_h, out_w)));

        for (n_batch, batch) in padded_data.outer_iter().enumerate() {
            weights
                .outer_iter()
                .into_par_iter()
                .enumerate()
                .map(|(n_filter, kernel)| {
                    // Performa tante convoluzioni quanti sono i filtri (valore M nella documentazione)
                    let bias_val = bias[n_filter];
                    let output =
                        Self::map_windows(
                            batch.into_dyn(),
                            kernel.shape(),
                            |window| (&window * &kernel).sum() + bias_val,
                            &[1, strides_h, strides_w]
                        ).and_then(|res| Ok(res.into_shape((out_h, out_w)).unwrap()))?;

                    let mut result =
                        result.lock()
                            .map_err(|_| onnx_error!("A PoisonError occurred while convoluting"))?;
                    result.slice_mut(s![n_batch, n_filter, .., ..]).assign(&output);
                    
                    Ok(())
                })
                .collect::<Result<(), OnnxError>>()?;
        }

        // Estrai risultato dal mutex
        let result =
            result
                .into_inner()
                .map_err(|_| onnx_error!("A PoisonError occurred while extracting result from Mutex"))?
                .into_dyn();

        Ok(Arc::new(result))
    }

}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, collections::HashMap, iter};
    use ndarray::{Array, array};
    use crate::operations::{Operation, OpType, Attribute};

    #[test]
    fn simple() {
        let op = Operation::new(OpType::Conv);

        let data =
            Array::from_iter((0..=24).map(|i| i as f32).cycle().take(1*1*5*5))
            .into_shape((1,1,5,5))
            .unwrap().into_dyn();

        let weights =
            Array::from_iter(iter::repeat(1. as f32).take(1*1*3*3))
            .into_shape((1,1,3,3))
            .unwrap().into_dyn();

        let expected_result = array![
            54.,  63.,  72., 
            99.,  108., 117.,
            144., 153., 162.
        ].into_shape((1,1,3,3)).unwrap().into_dyn();
    
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
            Array::from_iter((0..=24).map(|i| i as f32).cycle().take(1*1*5*5))
            .into_shape((1,1,5,5))
            .unwrap().into_dyn();

        let weights =
            Array::from_iter(iter::repeat(1. as f32).take(1*1*3*3))
            .into_shape((1,1,3,3))
            .unwrap().into_dyn();

        let expected_result =
            array![
                12., 21.,  27.,  33.,  24.,
                33., 54.,  63.,  72.,  51.,
                63., 99.,  108., 117., 81.,
                93., 144., 153., 162., 111.,
                72., 111., 117., 123., 84.,
            ].into_shape((1,1,5,5)).unwrap().into_dyn();
    
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
            Array::from_iter((0..=34).map(|i| i as f32).cycle().take(1*1*7*5))
            .into_shape((1,1,7,5))
            .unwrap().into_dyn();

        let weights =
            Array::from_iter(iter::repeat(1. as f32).take(1*1*3*3))
            .into_shape((1,1,3,3))
            .unwrap().into_dyn();

        let expected_result = array![
            54.0, 72.0,
            144.0, 162.0,
            234.0, 252.0
        ].into_shape((1,1,3,2)).unwrap().into_dyn();
    
        let result = op.execute(vec![Arc::new(data), Arc::new(weights)]);

        assert!(result.is_ok(), "{:?}", result.unwrap_err());
        assert_eq!(result.unwrap(), Arc::new(expected_result));
    }

    #[test]
    fn multiple_channels() {
        let op = Operation::new(OpType::Conv);

        let data =
            Array::from_iter((0..=24).map(|i| i as f32).cycle().take(1*2*5*5))
            .into_shape((1,2,5,5))
            .unwrap().into_dyn();

        let weights =
            Array::from_iter(iter::repeat(1. as f32).take(1*2*3*3))
            .into_shape((1,2,3,3))
            .unwrap().into_dyn();

        let expected_result = array![
            108.,  126.,  144., 
            198.,  216., 234.,
            288., 306., 324.
        ].into_shape((1,1,3,3)).unwrap().into_dyn();
    
        let result = op.execute(vec![Arc::new(data), Arc::new(weights)]);

        assert!(result.is_ok(), "{:?}", result.unwrap_err());
        assert_eq!(result.unwrap(), Arc::new(expected_result));
    }

    #[test]
    fn multiple_filters() {
        let op = Operation::new(OpType::Conv);

        let data =
            Array::from_iter((0..=24).map(|i| i as f32).cycle().take(1*1*5*5))
            .into_shape((1,1,5,5))
            .unwrap().into_dyn();

        let weights =
            Array::from_iter(iter::repeat(1. as f32).take(1*1*3*3).chain(iter::repeat(2. as f32).take(1*1*3*3)))
            .into_shape((2,1,3,3))
            .unwrap().into_dyn();

        let expected_result = array![
            [
                54.,  63.,  72., 
                99.,  108., 117.,
                144., 153., 162.,
            ],
            [
                108.,  126.,  144., 
                198.,  216., 234.,
                288., 306., 324.
            ]
        ].into_shape((1,2,3,3)).unwrap().into_dyn();
    
        let result = op.execute(vec![Arc::new(data), Arc::new(weights)]);

        assert!(result.is_ok(), "{:?}", result.unwrap_err());
        assert_eq!(result.unwrap(), Arc::new(expected_result));
    }

    #[test]
    fn with_bias() {
        let op = Operation::new(OpType::Conv);

        let data =
            Array::from_iter((0..=24).map(|i| i as f32).cycle().take(1*1*5*5))
            .into_shape((1,1,5,5))
            .unwrap().into_dyn();

        let weights =
            Array::from_iter(iter::repeat(1. as f32).take(1*1*3*3))
            .into_shape((1,1,3,3))
            .unwrap().into_dyn();

        let bias = array![1.].into_dyn();

        let expected_result = array![
            55.,  64.,  73., 
            100.,  109., 118.,
            145., 154., 163.
        ].into_shape((1,1,3,3)).unwrap().into_dyn();
    
        let result = op.execute(vec![Arc::new(data), Arc::new(weights), Arc::new(bias)]);

        assert!(result.is_ok(), "{:?}", result.unwrap_err());
        assert_eq!(result.unwrap(), Arc::new(expected_result));
    }

    #[test]
    fn autopad() {
        let attributes = HashMap::from([
            ("auto_pad".to_string(), Attribute::String("SAME_LOWER".to_string()))
        ]);
        let op = Operation::with_attributes(OpType::Conv, attributes);

        let data =
            Array::from_iter((0..=24).map(|i| i as f32).cycle().take(1*1*5*5))
            .into_shape((1,1,5,5))
            .unwrap().into_dyn();

        let weights =
            Array::from_iter(iter::repeat(1. as f32).take(1*1*3*3))
            .into_shape((1,1,3,3))
            .unwrap().into_dyn();

        let expected_result = array![
            12., 21.,  27.,  33.,  24.,
            33., 54.,  63.,  72.,  51.,
            63., 99.,  108., 117., 81.,
            93., 144., 153., 162., 111.,
            72., 111., 117., 123., 84.,
        ].into_shape((1,1,5,5)).unwrap().into_dyn();
    
        let result = op.execute(vec![Arc::new(data), Arc::new(weights)]);

        assert!(result.is_ok(), "{:?}", result.unwrap_err());
        assert_eq!(result.unwrap(), Arc::new(expected_result));
    }

}