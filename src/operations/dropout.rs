use std::sync::Arc;
use super::{Operation, OnnxError, onnx_error, Tensor, OperationResult};

impl Operation {

    /// Dropout è un'operazione utile solo per la fase di training, che non è lo scopo di questa libreria. L'operazione si
    /// limita a restituire il primo input come output.
    pub(super) fn execute_dropout(&self, inputs: Vec<Arc<Tensor>>) -> OperationResult {
        inputs
            .get(0)
            .ok_or(onnx_error!("Dropout requires 1 input."))
            .map(|v| v.clone())
    }

}