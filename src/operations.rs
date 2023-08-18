extern crate ndarray;

use std::{sync::Arc, collections::HashMap};
use ndarray::{ArrayD, Array1, ArrayViewD, Zip, IntoDimension, Dimension, IxDyn};
use crate::{error::OnnxError, onnx_error};

mod add;
mod averagepool;
mod concat;
mod constant;
mod conv;
mod div;
mod dropout;
mod gemm;
mod lrn;
mod matmul;
mod maxpool;
mod relu;
mod reshape;
mod softmax;

/// Generic multidimensional array containing floats ([`f32`]).
pub type Tensor = ArrayD<f32>;

/// Operation type
#[derive(Clone)]
pub enum OpType {
    /// Addition.
    Add,

    /// For each window, calculates the average.
    AveragePool,

    /// Concatenation on one axis.
    Concat,

    /// Output is a constant defined as an attribute.
    Constant,

    /// Convolution.
    Conv,

    // Division.
    Div,

    /// Operation used during training phase, does nothing in this context.
    Dropout,

    // General Matrix Multiplication.
    Gemm,

    /// Local Response Normalization.
    LRN,

    /// Matrix multiplication.
    MatMul,

    /// For each window, choose the maximum value.
    MaxPool,

    /// Y = max(0, X).
    Relu,

    // Change the shape of the input.
    Reshape,

    /// Softmax(input, axis) = Exp(input) / ReduceSum(Exp(input), axis=axis, keepdims=1)
    Softmax
}

impl TryFrom<&str> for OpType {
    type Error = OnnxError;
    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "Add" => Ok(Self::Add),
            "AveragePool" => Ok(Self::AveragePool),
            "Concat" => Ok(Self::Concat),
            "Constant" => Ok(Self::Constant),
            "Conv" => Ok(Self::Conv),
            "Div" => Ok(Self::Div),
            "Dropout" => Ok(Self::Dropout),
            "Gemm" => Ok(Self::Gemm),
            "LRN" => Ok(Self::LRN),
            "MatMul" => Ok(Self::MatMul),
            "MaxPool" => Ok(Self::MaxPool),
            "Relu" => Ok(Self::Relu),
            "Reshape" => Ok(Self::Reshape),
            "Softmax" => Ok(Self::Softmax),
            _ => Err(onnx_error!("Invalid operation name."))
        }
    }
}

/// Value of an attribute, which can vary in type.
#[derive(Clone,Debug)]
pub enum Attribute {
    Undefined,
    Float(f32),
    Int(isize),
    String(String),
    Tensor(Tensor),
    /// Unhandled
    Graph(()),
    Floats(Vec<f32>),
    Ints(Vec<isize>),
    Strings(Vec<String>),
    Tensors(Vec<Tensor>),
    /// Unhandled
    Graphs(()),
    /// Unhandled
    SparseTensor(()),
    /// Unhandled
    SparseTensors(()),
    /// Unhandled
    TypeProto(()),
    /// Unhandled
    TypeProtos(())
}

/// Operation with attributes.
pub struct Operation {
    /// Type of this operation.
    op_type: OpType,

    /// [`HashMap`] that maps the name of the attribute with the corresponding value.
    attributes: HashMap<String, Attribute>,

    /// Optional expected shape of the result (length of every dimension).
    /// 
    /// For instance, the shape `[1, 2, 3, 4]` relates to an array with 4 dimensions, where the dimensions of the axes are
    /// 1, 2, 3 and 4 respectively.
    /// 
    /// If [`None`], inputs of any shape will be accepted.
    expected_result_shape: Option<Box<[usize]>>,
}

/// Result of an operation. Since it can be shared between multiple threads, the value is already encapsulated in an Arc.
pub(crate) type OperationResult = Result<Arc<Tensor>, OnnxError>;

impl Operation {

    /// Creates a new operation to start.
    pub fn new(op_type: OpType) -> Self {
        Self::with_attributes(op_type, HashMap::new())
    }

    /// Creates a new operation to start, with attributes.
    pub fn with_attributes(op_type: OpType, mut attributes: HashMap<String, Attribute>) -> Self {
       
       /*LRN ha dei valori di default per alcuni parametri, nel file .onnx sono presenti i parametri alpha,beta,bias ma non sono valorizzati 
        i valori sono stati presi dalla documentazione ufficiale di LRN
       */
  
        if let OpType::LRN = op_type{
            if let Attribute::Undefined = attributes.get("alpha").unwrap(){
                attributes.insert("alpha".to_string(), Attribute::Float(0.0001));
            }
            if let Attribute::Undefined = attributes.get("beta").unwrap(){
                attributes.insert("beta".to_string(), Attribute::Float(0.75));
            }
            if let Attribute::Undefined = attributes.get("bias").unwrap(){
                attributes.insert("bias".to_string(), Attribute::Float(1.0));
            }
        }
        
        Self {
            op_type,
            attributes,
            expected_result_shape: None
        }
    }

    /// Executes the operation, given an input [`Tensor`].
    /// 
    /// In case of more inputs, the order is the same as specified in the [ONNX operators documentation][<https://onnx.ai/onnx/operators/>].
    pub(crate) fn execute(&self, inputs: Vec<Arc<Tensor>>) -> OperationResult {
        let inputs_pointers: Vec<&Tensor> = inputs.iter().map(|f| f.as_ref()).collect();
        
        let result = match self.op_type {
            OpType::Add => self.execute_add(inputs_pointers),
            OpType::AveragePool => self.execute_avg_pool(inputs_pointers),
            OpType::Concat => self.execute_concat(inputs_pointers),
            OpType::Constant => self.execute_constant(inputs_pointers),
            OpType::Conv => self.execute_conv(inputs_pointers),
            OpType::Div => self.execute_div(inputs_pointers),
            OpType::Dropout => self.execute_dropout(inputs),
            OpType::Gemm => self.execute_gemm(inputs_pointers),
            OpType::LRN => self.execute_lrn(inputs_pointers),
            OpType::MatMul => self.execute_matmul(inputs_pointers),
            OpType::MaxPool => self.execute_max_pool(inputs_pointers),
            OpType::Relu => self.execute_relu(inputs_pointers),
            OpType::Reshape => self.execute_reshape(inputs_pointers),
            OpType::Softmax => self.execute_softmax(inputs_pointers)
        };

        // Check shape of the result, if present.
        if let (Ok(res_val), Some(expected_shape)) = (&result, &self.expected_result_shape) {
            let res_shape = res_val.shape();
            if res_shape != expected_shape.as_ref() {
                return Err(onnx_error!("The result's shape ({res_shape:?}) is different than expected ({expected_shape:?})."))
            }
        }

        result
    }

    /// Calculate the length of the padding needed for each direction. This depends on:
    /// * Shape of the data (`data_w`, `data_w`)
    /// * Shape of the kernel (`kernel_h`, `kernel_w`)
    /// * Strides (`strides_h`, `strides_w`)
    /// 
    /// Padding may be manual or automatic, based on the operation attributes.
    fn get_padding(
        &self,
        (data_h, data_w): (usize, usize),
        (kernel_h, kernel_w): (usize, usize),
        (strides_h, strides_w): (usize, usize),
    ) -> Result<[usize; 4], OnnxError> {
        match self.attributes.get("auto_pad").or(Some(&Attribute::String("NOTSET".to_string()))) {
            Some(Attribute::String(auto_pad)) => match auto_pad.as_str() {
                // Manual padding
                "NOTSET" => match self.attributes.get("pads") {
                    Some(Attribute::Ints(val)) => Ok(
                        // Vec<&isize> -> [usize; 4]
                        val.into_iter()
                           .map(|v| usize::try_from(*v))
                           .collect::<Result<Vec<_>, _>>()
                           .map_err(|_| onnx_error!("pads attribute contains a negative number."))?
                           .as_slice()
                           .try_into().map_err(|_| onnx_error!("Padding should contain four values, {} found.", val.len()))?
                    ),
                    None => Ok([0,0,0,0]),
                    _ => return Err(onnx_error!("pads attribute has an invalid value type"))
                },

                // Given dimensions are supposed to be valid, without any padding.
                "VALID" => Ok([0,0,0,0]),

                // Automatic padding: the shape of the padding are such that the output has the same shape of the input without
                // the padding. The padding is split in half across both directions. If the padding shape is odd, the extra
                // padding is added at the beginning (UPPER) or at the end (LOWER) based on the attribute. 
                s @ ("SAME_UPPER" | "SAME_LOWER") => {
                    let (v_padding, h_padding) = (data_h*(strides_h-1)+kernel_h-strides_h, data_w*(strides_w-1)+kernel_w-strides_w);
                    Ok([
                        if v_padding % 2 == 0 || s.ends_with("UPPER") { v_padding/2 } else { v_padding/2+1 },
                        if h_padding % 2 == 0 || s.ends_with("UPPER") { h_padding/2 } else { h_padding/2+1 },
                        if v_padding % 2 == 0 || s.ends_with("LOWER") { v_padding/2 } else { v_padding/2+1 },
                        if h_padding % 2 == 0 || s.ends_with("LOWER") { h_padding/2 } else { h_padding/2+1 }
                    ])
                },
                _ => return Err(onnx_error!("auto_pad has an invalid value."))
            }
            _ => return Err(onnx_error!("auto_pad attribute has an invalid value type"))
        }
    }

    /// Obtains the windows of shape `window_dim` of the `data` array.
    /// 
    /// # Return
    /// Iterator over the windows.
    fn get_strided_windows<'a, I, D: IntoDimension<Dim = IxDyn>>(
        data: &'a ArrayViewD<I>,
        window_dim: D,
        strides: D
    ) -> impl Iterator<Item = ArrayViewD<'a, I>>
    {
        let window_dim = window_dim.into_dimension();
        let strides_dim = strides.into_dimension();

        // Amount of windows, for each dimension.
        let window_amounts =
            Zip::from(data.shape()).and(window_dim.as_array_view())
                .map_collect(|data_len, window_len| data_len - window_len + 1)
                .to_vec();

        data
            .windows(window_dim)
            .into_iter()
            .enumerate()
            .filter_map(move |(mut i, window)| {
                // Get current position
                let mut position = Vec::with_capacity(window_amounts.len());
                for w_i in 0..window_amounts.len() {
                    let product: usize = window_amounts.iter().skip(w_i+1).map(|v| *v).product();
                    position.push(i / product);
                    i %= product;
                }

                // Determine if current position is to skip based on the strides.
                if Zip::from(&position).and(strides_dim.as_array_view()).all(|&pos, &stride| pos % stride == 0) {
                    Some(window)
                } else {
                    None
                }
            })
    }

    /// Obtains the windows of shape `window_dim` of the `data` array, then applies the function `f` to each.
    /// 
    /// The result of each call to `f` is saved in a new multidimensional array, with shape depending on `data`, `window_dim`
    /// and `strides`.
    /// 
    /// # Error
    /// If the shape of the final array is not compatible with the number of results.
    fn map_windows<I, O, D: IntoDimension<Dim = IxDyn>>(
        data: ArrayViewD<I>,
        window_dim: D,
        strides: D,
        f: impl FnMut(ArrayViewD<I>) -> O
    ) -> Result<ArrayD<O>, OnnxError> {
        let window_dim = window_dim.into_dimension();
        let strides_dim = strides.into_dimension();

        // Compute output shapes
        let out_shape = 
            Zip::from(data.shape()).and(window_dim.as_array_view()).and(strides_dim.as_array_view())
                .map_collect(|data_len, window_len, strides_len| (data_len-window_len)/strides_len+1)
                .to_vec();

        Self::get_strided_windows(&data, window_dim, strides_dim)
            .map(f)
            .collect::<Array1<O>>()
            .into_shape(out_shape.clone())
            .map_err(|_| onnx_error!("Could not insert mapped values into {} matrix.", out_shape.into_iter().map(|v| v.to_string()).collect::<Vec<_>>().join("x")))
    }

}