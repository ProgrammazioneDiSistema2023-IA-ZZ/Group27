/// Errore generico durante una qualsiasi operazione in un grafo ONNX.
#[derive(Clone, Debug)]
pub struct OnnxError {
    pub msg: String
}

impl OnnxError {
    /// Creates a new error with the given message.
    pub fn new(msg: String) -> Self {
        Self {
            msg: msg.to_string()
        }
    }
}

/// Creates an [`OnnxError`] with a formatted message.
#[macro_export]
macro_rules! onnx_error {
    ($($args: tt)*) => {
        OnnxError::new(format!($($args)*))
    }
}