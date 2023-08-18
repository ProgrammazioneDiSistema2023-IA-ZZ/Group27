use std::sync::Arc;
use itertools::{iproduct, Itertools};
use ndarray::{Array4, s, Array};
use rayon::prelude::{ParallelBridge, ParallelIterator};

use super::{Operation, OnnxError, onnx_error, Tensor, OperationResult, Attribute};

impl Operation {

    /// *(From <https://onnx.ai/onnx/operators/onnx__MaxPool.html>)*
    /// 
    /// MaxPool consumes an input tensor X and applies max pooling across the tensor according to kernel sizes, stride sizes,
    /// and pad lengths. max pooling consisting of computing the max on all values of a subset of the input tensor according to
    /// the kernel size and downsampling the data into the output tensor Y for further processing. The output spatial shape will
    /// be following:
    /// ```plain
    /// output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1)) / strides_spatial_shape[i] + 1)
    /// ```
    /// or
    /// ```plain
    /// output_spatial_shape[i] = ceil((input_spatial_shape[i] + pad_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1)) / strides_spatial_shape[i] + 1)
    /// ```
    /// if ceil_mode is enabled `pad_shape[i]` is the sum of pads along axis `i`.
    /// 
    /// `auto_pad` is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following when
    /// ceil_mode is enabled:
    /// ```plain
    /// VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) + 1) / strides_spatial_shape[i])
    /// SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
    /// ```
    /// or when ceil_mode is disabled:
    /// ```plain
    /// VALID: output_spatial_shape[i] = floor((input_spatial_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) + 1) / strides_spatial_shape[i])
    /// SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = floor(input_spatial_shape[i] / strides_spatial_shape[i])
    /// ```
    /// And pad shape will be following if `SAME_UPPER` or `SAME_LOWER`:
    /// ```plain
    /// pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) - input_spatial_shape[i]
    /// ```
    /// The output of each pooling window is maximum number of elements exclude pad.
    /// 
    /// # Attributes
    /// * **auto_pad** - `STRING` (default is `'NOTSET'`): auto_pad must be either `NOTSET`, `SAME_UPPER`, `SAME_LOWER` or
    ///   `VALID`. Where default value is `NOTSET`, which means explicit padding is used. `SAME_UPPER` or `SAME_LOWER` mean pad
    ///   the input so that `output_shape[i] = ceil(input_shape[i] / strides[i])` for each axis `i`. The padding is split
    ///   between the two sides equally or almost equally (depending on whether it is even or odd). In case the padding is an
    ///   odd number, the extra padding is added at the end for `SAME_UPPER` and at the beginning for `SAME_LOWER`.
    /// * **ceil_mode** - `INT` (default is `'0'`): Whether to use ceil or floor (default) to compute the output shape.
    /// * **dilations** - `INTS` : dilation value along each spatial axis of the filter. If not present, the dilation defaults
    ///   is 1 along each spatial axis. (*Unhandled*)
    /// * **kernel_shape** - `INTS` : The shape of the convolution kernel. If not present, should be inferred from input `W`.
    /// * **pads** - `INTS` : Padding for the beginning and ending along each spatial axis, it can take any value greater than
    ///   or equal to 0. The value represent the number of pixels added to the beginning and end part of the corresponding axis.
    ///   pads format should be as follow `[x1_begin, x2_begin…x1_end, x2_end,…]`, where `xi_begin` the number of pixels added
    ///   at the beginning of axis `i` and `xi_end`, the number of pixels added at the end of axis `i`. This attribute cannot be
    ///   used simultaneously with auto_pad attribute. If not present, the padding defaults to 0 along start and end of each
    ///   spatial axis.
    /// * **storage_order** - `INT` (default is `'0'`): The storage order of the tensor. 0 is row major, and 1 is column major.
    ///   This attribute is used only to convert an n-tuple index value into a single integer value for producing the second
    ///   output. (*Unhandled*)
    /// * **strides** - `INTS` : Stride along each spatial axis. If not present, the stride defaults is 1 along each spatial
    ///   axis.
    /// 
    /// # Inputs
    /// * **X** (heterogeneous) - `T`: Input data tensor from the previous operator; dimensions for image case are (`N x C x H x W`),
    ///   where `N` is the batch size, `C` is the number of channels, and H and W are the height and the width of the data.
    /// 
    /// # Outputs
    /// * **Y** (heterogeneous) - `T`: Output data tensor from average or max pooling across the input tensor. Dimensions will
    ///   vary based on various kernel, stride, and pad sizes. Floor value of the dimension is used
    pub(super) fn execute_max_pool(&self, inputs: Vec<&Tensor>) -> OperationResult {
        // Inputs
        let data = *inputs.get(0).ok_or(onnx_error!("Missing X input from MatMul operation"))?;
        let data_shape: [usize; 4] = data.shape().try_into().map_err(|_| onnx_error!("Only 2D (image) convolution is supported at the moment."))?;
        let [ batches, channels, data_h, data_w ] = data_shape;

        // Attributes
        // Dilations: non gestite
        let [ dilation_h, dilation_w ] = match self.attributes.get("dilations") {
            Some(Attribute::Ints(val)) => val.as_slice().try_into().map_err(|_| onnx_error!("Dilation should contain two dimensions."))?,
            None | Some(Attribute::Undefined) => [1, 1],
            _ => return Err(onnx_error!("dilations attribute has an invalid value type"))
        };

        if dilation_h > 1 || dilation_w > 1 {
            return Err(onnx_error!("Dilations are not not supported."))
        }
        
        // Dimensioni kernel
        let [ kernel_h, kernel_w ]: [usize; 2] = match self.attributes.get("kernel_shape") {
            Some(Attribute::Ints(val)) => {
                // Converti Vec<&isize> a [usize; 2]
                val.into_iter()
                   .map(|v| usize::try_from(*v))
                   .collect::<Result<Vec<_>, _>>()
                   .map_err(|_| onnx_error!("kernel_shape attribute contains a negative number"))?
                   .as_slice()
                   .try_into().map_err(|_| onnx_error!("Kernel size should contain two dimensions."))?
            },
            None | Some(Attribute::Undefined) => return Err(onnx_error!("Kernel shape not given.")),
            _ => return Err(onnx_error!("kernel_shape attribute has an invalid value type"))
        };

        // Strides: windows skipped in length/height
        let [strides_h, strides_w] = match self.attributes.get("strides") {
            Some(Attribute::Ints(val)) => {
                // Converti Vec<&isize> a [usize; 2]
                val.into_iter()
                   .map(|v| usize::try_from(*v))
                   .collect::<Result<Vec<_>, _>>()
                   .map_err(|_| onnx_error!("Strides attribute contains a negative number"))?
                   .as_slice()
                   .try_into().map_err(|_| onnx_error!("Strides should contain two dimensions."))?
            },
            None | Some(Attribute::Undefined) => [1, 1],
            _ => return Err(onnx_error!("groups attribute has an invalid value type"))
        };
        
        // Padding: manual or automatic, describes amount of added rows/columns with constant values.
        let [ pad_n, pad_w, pad_s, pad_e ] =
            self.get_padding(
                (data_h, data_w),
                (kernel_h, kernel_w),
                (strides_h, strides_w)
            )?;

        // Ceil mode: use ceil or floor operation to round the output shape
        let ceil_mode = match self.attributes.get("ceil_mode") {
            Some(Attribute::Int(val)) => *val,
            None | Some(Attribute::Undefined) => 0,
            _ => return Err(onnx_error!("ceil_mode attribute has an invalid value type"))
        };

        // Storage order: non gestito
        let _storage_order = match self.attributes.get("storage_order") {
            Some(Attribute::Int(val)) => *val, 
            None | Some(Attribute::Undefined) => 0,
            _ => return Err(onnx_error!("storage_order attribute has an invalid value type"))
        };

        /*** MAXPOOL ***/
        
        // Clone the input, eventually with added padding.
        // Padding values are f32::MIN, so that they will never be chosen as the maximum.
        let (padded_h, padded_w) = (data_h + pad_n + pad_s, data_w + pad_e + pad_w);
        let mut padded_data = Array4::<f32>::from_elem((batches, channels, padded_h, padded_w), f32::MIN);
        padded_data.slice_mut(s![.., .., pad_n..pad_n+data_h, pad_w..pad_w+data_w]).assign(data);

        // Calculate output dimensions in base a ceil_mode
        let (out_h, out_w) = (
            if ceil_mode == 1 { f32::ceil((padded_h-kernel_h) as f32/strides_h as f32 + 1.) as usize } else { (padded_h-kernel_h)/strides_h+1 },
            if ceil_mode == 1 { f32::ceil((padded_w-kernel_w) as f32/strides_w as f32 + 1.) as usize } else { (padded_w-kernel_w)/strides_w+1 }
        );

        // Calculate value of each cell of the result in parallel, then insert all values into the result array. The parallel
        // iterator collects values in a random fashion, so we also need to sort by the global index of each value
        let values =
            iproduct!(0..batches, 0..channels)
                .par_bridge()
                .map(|(n_batch, n_channel)| {
                    let channel = padded_data.slice(s![n_batch, n_channel, .., ..]);
                    Self::map_windows(
                        channel.into_dyn(),
                        [kernel_h, kernel_w].as_slice(),
                        [strides_h, strides_w].as_slice(),
                        |window| *window.into_iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap(),
                    ).map(|res| (
                        n_channel + n_batch * channels, // Global index
                        res // Result (array)
                    ))
                })
                .collect::<Result<Vec<(usize, _)>, OnnxError>>()?
                .into_iter()
                .sorted_by(|(i1, _), (|i2, _)| i1.cmp(i2))
                .flat_map(|(_, v)| v)
                .collect::<Vec<f32>>();
        
        let result =
            Array::from_shape_vec((batches, channels, out_h, out_w), values)
            .unwrap().into_dyn();

        Ok(Arc::new(result))
    }

}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, collections::HashMap};
    use ndarray::Array;
    use crate::operations::{Operation, OpType, Attribute};

    #[test]
    fn simple() {
        let attributes = HashMap::from([
            ("kernel_shape".to_string(), Attribute::Ints(vec![2,2]))
        ]);
        let op = Operation::with_attributes(OpType::MaxPool, attributes);

        let data =
            Array::from_iter((1..=16).map(|i| i as f32).cycle().take(2*3*4*4))
            .into_shape((2,3,4,4))
            .unwrap().into_dyn();

        let expected_result =
            Array::from_iter([
                6.,  7.,  8.,
                10., 11., 12.,
                14., 15., 16.
            ].into_iter().cycle().take(2*3*3*3))
            .into_shape((2,3,3,3))
            .unwrap().into_dyn();
    
        let result = op.execute(vec![Arc::new(data) ]);

        assert!(result.is_ok(), "{:?}", result.unwrap_err());
        assert_eq!(result.unwrap(), Arc::new(expected_result));
    }

    #[test]
    fn with_strides() {
        let attributes = HashMap::from([
            ("kernel_shape".to_string(), Attribute::Ints(vec![2,2])),
            ("strides".to_string(), Attribute::Ints(vec![2,2]))
        ]);
        let op = Operation::with_attributes(OpType::MaxPool, attributes);

        let data =
            Array::from_iter((1..=16).map(|i| i as f32).cycle().take(2*3*4*4))
            .into_shape((2,3,4,4))
            .unwrap().into_dyn();

        let expected_result =
            Array::from_iter([6., 8., 14., 16.].into_iter().cycle().take(2*3*2*2))
            .into_shape((2,3,2,2))
            .unwrap().into_dyn();
    
        let result = op.execute(vec![Arc::new(data) ]);

        assert!(result.is_ok(), "{:?}", result.unwrap_err());
        assert_eq!(result.unwrap(), Arc::new(expected_result));
    }

    #[test]
    fn with_padding() {
        let attributes = HashMap::from([
            ("kernel_shape".to_string(), Attribute::Ints(vec![3,3])),
            ("pads".to_string(), Attribute::Ints(vec![1,1,1,1]))
        ]);
        let op = Operation::with_attributes(OpType::MaxPool, attributes);

        let data =
            Array::from_iter((-8..8).map(|i| i as f32).cycle().take(2*3*4*4))
            .into_shape((2,3,4,4))
            .unwrap().into_dyn();

        let expected_result =
            Array::from_iter([
                -3., -2., -1., -1.,
                1.,  2.,  3.,  3.,
                5.,  6.,  7.,  7.,
                5.,  6.,  7.,  7.    
            ].into_iter().map(|i| i as f32).cycle().take(2*3*4*4))
            .into_shape((2,3,4,4))
            .unwrap().into_dyn();
    
        let result = op.execute(vec![Arc::new(data) ]);

        assert!(result.is_ok(), "{:?}", result.unwrap_err());
        assert_eq!(result.unwrap(), Arc::new(expected_result));
    }

    #[test]
    fn auto_pad() {
        let attributes = HashMap::from([
            ("kernel_shape".to_string(), Attribute::Ints(vec![3,3])),
            ("auto_pad".to_string(), Attribute::String("SAME_LOWER".to_string()))
        ]);
        let op = Operation::with_attributes(OpType::MaxPool, attributes);

        let data =
            Array::from_iter((1..=16).map(|i| i as f32).cycle().take(2*3*4*4))
            .into_shape((2,3,4,4))
            .unwrap().into_dyn();
    
        let data_dim = data.dim();

        let result = op.execute(vec![Arc::new(data) ]);

        assert!(result.is_ok(), "{:?}", result.unwrap_err());
        assert_eq!(result.unwrap().dim(), data_dim);
    }

}