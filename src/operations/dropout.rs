use std::sync::Arc;
use super::{Operation, OnnxError, onnx_error, Tensor, OperationResult};

impl Operation {

    /// Dropout is only useful for the training phase, which is not the point of this library. The operation limits to returning
    /// the first input as the only output.
    pub(super) fn execute_dropout(&self, inputs: Vec<Arc<Tensor>>) -> OperationResult {
        inputs
            .get(0)
            .ok_or(onnx_error!("Dropout requires 1 input."))
            .map(|v| v.clone())
    }

}