/// Errore generico durante una qualsiasi operazione in un grafo ONNX.
#[derive(Clone, Debug)]
pub struct OnnxError {
    pub msg: String
}

impl OnnxError {
    /// Crea un nuovo errore con un dato messaggio.
    pub fn new(msg: String) -> Self {
        Self {
            msg: msg.to_string()
        }
    }
}

/// Crea un [`OnnxError`] con un dato messaggio formattato.
#[macro_export]
macro_rules! onnx_error {
    ($($args: tt)*) => {
        OnnxError::new(format!($($args)*))
    }
}