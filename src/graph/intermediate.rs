/// Nodo intermedio all'interno di un grafo. Si riferisce ai nodi non espressamente dichiarati ma usati come input/output ad
/// altri nodi.
pub struct OnnxGraphIntermediate {
    /// Nome del nodo.
    pub name: String,

    /// Nome del nodo entrante. Può essere impostato in un secondo momento dopo la creazione.
    pub input: String,

    /// Nomi dei nodi uscenti.
    pub outputs: Vec<String>
}

impl OnnxGraphIntermediate {
    /// Crea un nuovo nodo intemedio vuoto.
    pub fn new(name: &str, input: String, outputs: Vec<String>) -> Self {
        Self {
            name: name.to_string(),
            input,
            outputs
        }
    }
}