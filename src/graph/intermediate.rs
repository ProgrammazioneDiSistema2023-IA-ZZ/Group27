use std::collections::HashSet;

use crate::{onnx_error, error::OnnxError};

use super::OnnxGraphNode;

/// Intermediate node of a graph. Intermediate nodes are not expressly declared, but they're used as inputs/outputs to other nodes.

pub struct OnnxGraphIntermediate {
    /// Name of the node.
    pub name: String,

    /// Name of the node that is an input to this node.
    pub input: String,

    /// Names of the nodes that are outputs to this node.
    pub outputs: HashSet<String>
}

impl OnnxGraphIntermediate {
    /// Creates a new empty intermediate node.
    pub fn new<S, O>(name: impl ToString, input: impl ToString, outputs: O) -> Self
    where
        S: ToString,
        O: IntoIterator<Item = S>
    {
        Self {
            name: name.to_string(),
            input: input.to_string(),
            outputs: outputs.into_iter().map(|s| s.to_string()).collect()
        }
    }
}

impl TryFrom<OnnxGraphNode> for OnnxGraphIntermediate {
    type Error = OnnxError;
    fn try_from(value: OnnxGraphNode) -> Result<Self, Self::Error> {
        match value {
            OnnxGraphNode::Intermediate(interm_node) => Ok(interm_node),
            _ => Err(onnx_error!("Node {} is not an initializer.", value.name()))
        }
    }
}

impl<'a> TryFrom<&'a OnnxGraphNode> for &'a OnnxGraphIntermediate {
    type Error = OnnxError;
    fn try_from(value: &'a OnnxGraphNode) -> Result<Self, Self::Error> {
        match value {
            OnnxGraphNode::Intermediate(interm_node) => Ok(interm_node),
            _ => Err(onnx_error!("Node {} is not an initializer.", value.name()))
        }
    }
}