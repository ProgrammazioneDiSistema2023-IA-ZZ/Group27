use std::sync::Arc;

use crate::operations::Tensor;

/// Nodo input all'interno di un grafo.
#[derive(Clone)]
pub struct OnnxGraphInput {
    /// Nome del nodo.
    pub(super) name: String,

    /// Eventuale valore di default relativo all'input corrente.
    /// 
    /// Se durante l'inferenza non verranno forniti valori relativi a questo nodo, si ricadrà su questo valore. In tal caso, se
    /// anche il default vale [`None`] verrà restituito un errore.
    pub(super) default_value: Option<Arc<Tensor>>,

    /// Eventuale "forma" (dimensioni dell'array) dell'input attesa.
    /// 
    /// Ad esempio, la forma `[1, 2, 3, 4]` indica un array a 4 dimensioni, dove le dimensioni degli assi sono rispettivamente
    /// 1, 2, 3 e 4.
    /// 
    /// Se [`None`] verranno accettati input di qualsiasi dimensione.
    expected_shape: Option<Box<[usize]>>,
}

impl OnnxGraphInput {
    /// Crea un nuovo nodo input senza alcun valore opzionale.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            expected_shape: None,
            default_value: None
        }
    }

    /// Crea un nuovo nodo con forma attesa.
    pub fn with_expected_shape(name: &str, shape: &[usize]) -> Self{
        Self {
            name: name.to_string(),
            expected_shape: Some(shape.to_vec().into_boxed_slice()),
            default_value: None
        }
    }

    /// Determina se la forma attesa del nodo corrente e quella data come parametro coincidono.
    pub(super) fn valid_shape(&self, shape: &[usize]) -> bool {
        if let Some(expected_shape) = &self.expected_shape {
            shape == expected_shape.as_ref()
        } else {
            true
        }
    }
}