use std::sync::Arc;

use crate::operations::Tensor;

/// Nodo "inizializzatore" (input costante) all'interno di un grafo.
pub struct OnnxGraphInitializer {
    /// Nome del nodo.
    pub(super) name: String,

    /// Valore dell'input.
    pub(super) data: Arc<Tensor>
}

impl OnnxGraphInitializer {
    /// Crea un nuovo nodo initializer.
    pub fn new(name: &str, data: Tensor) -> Self {
        Self {
            name: name.to_string(),
            data: Arc::new(data)
        }
    }
}