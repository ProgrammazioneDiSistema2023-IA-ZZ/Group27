use std::sync::Arc;

use crate::operations::Tensor;

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