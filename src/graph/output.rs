use crate::{error::OnnxError, onnx_error};

use super::OnnxGraphNode;


/// Nodo output all'interno di un grafo.
pub struct OnnxGraphOutput {
    /// Name of the node.
    pub name: String,

    /// Optional expected shape of the output (length of every dimension).
    /// 
    /// For instance, the shape `[1, 2, 3, 4]` relates to an array with 4 dimensions, where the dimensions of the axes are
    /// 1, 2, 3 and 4 respectively.
    /// 
    /// If [`None`], outputs of any shape will be accepted.
    expected_shape: Option<Box<[usize]>>
}

impl OnnxGraphOutput {
    /// Creates a new output node, without an expected shape.
    pub fn new(name: impl ToString) -> Self {
        Self {
            name: name.to_string(),
            expected_shape: None
        }
    }

    /// Creates a new output node with expected shape.
    pub fn with_expected_shape(name: &str, shape: &[usize]) -> Self{
        Self {
            name: name.to_string(),
            expected_shape: Some(shape.to_vec().into_boxed_slice())
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

impl TryFrom<OnnxGraphNode> for OnnxGraphOutput {
    type Error = OnnxError;
    fn try_from(value: OnnxGraphNode) -> Result<Self, Self::Error> {
        match value {
            OnnxGraphNode::Output(out_node) => Ok(out_node),
            _ => Err(onnx_error!("Node {} is not an output.", value.name()))
        }
    }
}

impl<'a> TryFrom<&'a OnnxGraphNode> for &'a OnnxGraphOutput {
    type Error = OnnxError;
    fn try_from(value: &'a OnnxGraphNode) -> Result<Self, Self::Error> {
        match value {
            OnnxGraphNode::Output(out_node) => Ok(out_node),
            _ => Err(onnx_error!("Node {} is not an output.", value.name()))
        }
    }
}