use std::sync::Arc;

use crate::operations::Tensor;

/// Input node of a graph.
#[derive(Clone)]
pub struct OnnxGraphInput {
    /// Name of the node.
    pub(super) name: String,

    /// Optional default value for this input.
    /// 
    /// If, during inference, no values are provided for this node, this value will be used. In that case, if this value is
    /// [`None`], an error will occur.
    pub(super) default_value: Option<Arc<Tensor>>,

    /// Optional expected shape of the input (length of every dimension).
    /// 
    /// For instance, the shape `[1, 2, 3, 4]` relates to an array with 4 dimensions, where the dimensions of the axes are
    /// 1, 2, 3 and 4 respectively.
    /// 
    /// If [`None`], inputs of any shape will be accepted.
    expected_shape: Option<Box<[usize]>>,
}

impl OnnxGraphInput {
    /// Creates a new input node, without an expected shape.
    pub fn new(name: impl ToString) -> Self {
        Self {
            name: name.to_string(),
            expected_shape: None,
            default_value: None
        }
    }

    /// Creates a new input node with expected shape.
    pub fn with_expected_shape(name: impl ToString, shape: &[usize]) -> Self{
        Self {
            name: name.to_string(),
            expected_shape: Some(shape.to_vec().into_boxed_slice()),
            default_value: None
        }
    }

    /// Determines if `shape` and the expected shape for this node are the same.
    pub(super) fn valid_shape(&self, shape: &[usize]) -> bool {
        if let Some(expected_shape) = &self.expected_shape {
            shape == expected_shape.as_ref()
        } else {
            true
        }
    }
}