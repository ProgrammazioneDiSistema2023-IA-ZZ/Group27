use std::sync::Arc;

use crate::{operations::Tensor, error::OnnxError, onnx_error};

use super::OnnxGraphNode;

/// Initializer node of a graph.

pub struct OnnxGraphInitializer {
    /// Name of the node.
    pub(super) name: String,

    /// Input value.
    pub(super) data: Arc<Tensor>
}

impl OnnxGraphInitializer {
    /// Creates a new initializer node with data.
    pub fn new(name: impl ToString, data: Tensor) -> Self {
        Self {
            name: name.to_string(),
            data: Arc::new(data)
        }
    }
}

impl<'a> TryFrom<&'a OnnxGraphNode> for &'a OnnxGraphInitializer {
    type Error = OnnxError;
    fn try_from(value: &'a OnnxGraphNode) -> Result<Self, Self::Error> {
        match value {
            OnnxGraphNode::Initializer(init_node) => Ok(init_node),
            _ => Err(onnx_error!("Node {} is not an initializer.", value.name()))
        }
    }
}