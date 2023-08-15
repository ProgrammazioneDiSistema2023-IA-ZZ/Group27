use std::collections::HashSet;

/// Nodo intermedio all'interno di un grafo. Si riferisce ai nodi non espressamente dichiarati ma usati come input/output ad
/// altri nodi.
pub struct OnnxGraphIntermediate {
    /// Nome del nodo.
    pub name: String,

    /// Nome del nodo entrante. Può essere impostato in un secondo momento dopo la creazione.
    pub input: String,

    /// Nomi dei nodi uscenti.
    pub outputs: HashSet<String>
}

impl OnnxGraphIntermediate {
    /// Crea un nuovo nodo intemedio vuoto.
    pub fn new<S, O>(name: impl ToString, input: impl ToString, outputs: O) -> Self
    where
        S: ToString,
        O: IntoIterator<Item = S>
    {
        Self {
            name: name.to_string(),
            input: input.to_string(),
            outputs: outputs.into_iter().map(|s| s.to_string()).collect()
        }
    }
}