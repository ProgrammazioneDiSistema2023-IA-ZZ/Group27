/// Nodo intermedio all'interno di un grafo. Si riferisce ai nodi non espressamente dichiarati ma usati come input/output ad
/// altri nodi.
pub struct OnnxGraphIntermediate {
    /// Nome del nodo.
    pub(super) name: String,

    /// Nome del nodo entrante. Può essere impostato in un secondo momento dopo la creazione.
    pub(super) input: Option<String>,

    /// Nomi dei nodi uscenti.
    pub(super) outputs: Vec<String>
}

impl OnnxGraphIntermediate {
    /// Crea un nuovo nodo intemedio vuoto.
    pub(super) fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            input: None,
            outputs: Vec::new()
        }
    }
}