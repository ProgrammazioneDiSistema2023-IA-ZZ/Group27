/// Nodo output all'interno di un grafo.
pub struct OnnxGraphOutput {
    /// Nome del nodo.
    pub name: String,

    /// Eventuale "forma" (dimensioni dell'array) dell'output attesa.
    /// 
    /// Ad esempio, la forma `[1, 2, 3, 4]` indica un array a 4 dimensioni, dove le dimensioni degli assi sono rispettivamente
    /// 1, 2, 3 e 4.
    /// 
    /// Se [`None`] verranno accettati output di qualsiasi dimensione.
    expected_shape: Option<Box<[usize]>>
}

impl OnnxGraphOutput {
    /// Crea un nuovo nodo input senza alcun valore opzionale.
    pub fn new(name: impl ToString) -> Self {
        Self {
            name: name.to_string(),
            expected_shape: None
        }
    }

    /// Crea un nuovo nodo input con forma attesa.
    pub fn with_expected_shape(name: &str, shape: &[usize]) -> Self{
        Self {
            name: name.to_string(),
            expected_shape: Some(shape.to_vec().into_boxed_slice())
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