use std::{str::FromStr, fs::read_to_string, collections::{HashMap, HashSet}, sync::{Arc, MutexGuard, Mutex, PoisonError, RwLock, RwLockReadGuard, atomic::{AtomicUsize, Ordering}}, thread::{JoinHandle, self}, hash::Hash};
use log;
use crate::{error::OnnxError, parser::OnnxParser, operations::Tensor, onnx_error};

pub use self::{operation::OnnxGraphOperation, input::OnnxGraphInput, output::OnnxGraphOutput, initializer::OnnxGraphInitializer, intermediate::OnnxGraphIntermediate};

pub mod operation;
pub mod input;
pub mod output;
pub mod initializer;
pub mod intermediate;

/// Rappresenta in modo generico il nodo di un grafo.
pub enum OnnxGraphNode {
    Initializer(OnnxGraphInitializer),
    Input(OnnxGraphInput),
    Output(OnnxGraphOutput),
    Operation(OnnxGraphOperation),
    Intermediate(OnnxGraphIntermediate)
}

impl OnnxGraphNode {
    /// Restituisce il nome del nodo, indipendentemente dal suo tipo.
    pub fn name(&self) -> &String {
        match self {
            OnnxGraphNode::Initializer(node) => &node.name,
            OnnxGraphNode::Input(node) => &node.name,
            OnnxGraphNode::Output(node) => &node.name,
            OnnxGraphNode::Operation(node) => &node.name,
            OnnxGraphNode::Intermediate(node) => &node.name
        }
    }

    /// Estrae un riferimento alla [`OnnxGraphOperation`] relativa al nodo corrente.
    /// 
    /// # Errore
    /// Se il nodo corrente non è un nodo operazione.
    pub fn ref_operation(&self) -> Result<&OnnxGraphOperation, OnnxError> {
        match self {
            OnnxGraphNode::Operation(node) => Ok(node),
            _ => Err(onnx_error!("Could not extract operation from {} because it's not an operation node.", self.name()))
        }
    }
}

impl TryInto<OnnxGraphIntermediate> for OnnxGraphNode {
    type Error = OnnxError;
    fn try_into(self) -> Result<OnnxGraphIntermediate, Self::Error> {
        match self {
            OnnxGraphNode::Intermediate(node) => Ok(node),
            _ => Err(onnx_error!("Could not extract intermediate node from {} because it's not an intermediate node.", self.name()))
        }
    }
}

impl PartialEq for OnnxGraphNode {
    fn eq(&self, other: &Self) -> bool {
        self.name().eq(other.name())
    }
}

impl Eq for OnnxGraphNode {}

impl Hash for OnnxGraphNode {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.name().hash(state)
    }
}

/// Identificativo dell'inferenza all'interno del processo corrente.
#[derive(Hash, PartialEq, Eq, Clone, Copy, Debug)]
struct InferenceID(usize);

/// Grafo che rappresenta una rete neurale.
pub struct OnnxGraph {
    /// Nome del grafo
    pub name: String,

    /// Nomi dei nodi initializer del grafo.
    initializers: HashSet<String>,

    /// Nomi dei nodi input del grafo.
    inputs: HashSet<String>,

    /// Nomi dei nodi output del grafo.
    outputs: HashSet<String>,

    /// Nomi dei nodi operazione del grafo.
    operations: HashSet<String>,

    /// [`HashMap`] che mappa il nome del nodo alla struttura [`OnnxGraphNode`] corrispondente.
    /// 
    /// Questo valore viene utilizzato da più thread contemporaneamente.
    /// 
    /// Le operazioni di scrittura (ad esempio l'aggiunta di un nodo) avviene solitamente prima di un'inferenza, mentre
    /// l'inferenza in sé richiede solo operazioni di lettura. Per questo motivo, è stato usato un [`RwLock`] anziché un
    /// [`Mutex`].
    nodes: RwLock<HashMap<String, OnnxGraphNode>>,

    /// Numero di inferenze attualmente in corso.
    inferences_in_progress: AtomicUsize,

    /// Memorizza l'ultimo ID di inferenza utilizzato. Viene incrementato ogni volta che viene avviata un'inferenza nello stesso
    /// processo.
    last_inference_id: Mutex<InferenceID>
}

/// Risultato della creazione del grafo, se questa prevede degli errori (ad esempio, a partire da un file).
pub type OnnxGraphResult = Result<OnnxGraph, OnnxError>;

/// Risultato dell'inferenza.
/// 
/// Se l'operazione ha successo, [`Ok`] contiene una [`HashMap`] che mappa il nome dell'output al valore corrispondente.
pub type OnnxInferenceResult = Result<HashMap<String, Tensor>, OnnxError>;

impl OnnxGraph {
    /// Crea un nuovo grafo vuoto.
    pub fn new() -> Self {
        Self {
            name: String::default(),
            initializers: HashSet::new(),
            inputs: HashSet::new(),
            outputs: HashSet::new(),
            operations: HashSet::new(),
            nodes: RwLock::new(HashMap::new()),
            inferences_in_progress: AtomicUsize::new(0),
            last_inference_id: Mutex::new(InferenceID(0))
        }
    }

    /// Crea un nuovo grafo a partire da un file (dato il suo percorso). 
    pub fn from_file(path: &str) -> OnnxGraphResult {
        // TODO: Lettura file più efficiente con BufReader (per ora immagazzina l'intero file in una stringa.)
        Self::from_str(
            read_to_string(path).map_err(|_| onnx_error!("Cannot read from file {path}."))?.as_str()
        )
    }

    /// Aggiunge un nodo esistente al grafo.
    pub fn add_node(&mut self, node: OnnxGraphNode) -> Result<(), OnnxError> {
        // Evita che vengano effettuate modifiche ai nodi mentre ci sono delle inferenze in corso.
        if *self.inferences_in_progress.get_mut() > 0 {
            return Err(onnx_error!("Cannot add nodes while there are inferences in progress."));
        }

        // Lock delle risorse
        let mut nodes =
            self.nodes.write()
                .map_err(|_| onnx_error!("PoisonError occurred while locking nodes for write (adding node {}).", node.name()))?;
        
        let name = node.name().clone();

        // Inserisci nome del nodo nel vettore adatto della struttura.
        match node {
            OnnxGraphNode::Initializer(_) => self.initializers.insert(name.clone()),
            OnnxGraphNode::Input(_) => self.inputs.insert(name.clone()),
            OnnxGraphNode::Output(_) => self.outputs.insert(name.clone()),
            OnnxGraphNode::Operation(_) => self.operations.insert(name.clone()),
            OnnxGraphNode::Intermediate(interm_node) => {
                // Dato il nodo entrante E al nodo intermedio I, aggiungi i nodi uscenti di I ai nodi uscenti di E e rimuovi I dalla
                // stessa collezione.
                let interm_input_node =
                    nodes.get_mut(&interm_node.input)
                         .ok_or_else(|| onnx_error!("[{}] Input node {} does not exist.", interm_node.name, interm_node.input))?;
                if let OnnxGraphNode::Operation(op_node) = interm_input_node {
                    op_node.outputs.remove(&interm_node.name);
                    op_node.outputs.extend(interm_node.outputs.clone());
                }

                // Per ogni nodo uscente U dal nodo intermedio I, sostituisci il nome di I dai nodi entranti di U con il nodo
                // entrante di I.
                for interm_output_name in interm_node.outputs {
                    let interm_output_node =
                        nodes.get_mut(&interm_output_name)
                            .ok_or_else(|| onnx_error!("[{}] Output node {} does not exist.", interm_node.name, interm_output_name))?;
                    if let OnnxGraphNode::Operation(op_node) = interm_output_node {
                        let opnode_interm_input =
                            op_node.inputs
                                .iter_mut()
                                .find(|input_name| **input_name == interm_node.name)
                                .ok_or_else(|| onnx_error!("[{}] ", interm_node.name))?;
                        *opnode_interm_input = interm_node.input.clone();
                    }
                }

                // Non seve aggiungere il nodo intermedio alla collezione
                return Ok(())
            }
        };

        // Aggiungi nodo alla collezione
        if let Some(old_node) = nodes.insert(name.clone(), node) {
            // Trovato nodo duplicato: rimuovi dall'HashMap
            let new_node = nodes.remove(&name).unwrap();

            match (old_node, new_node) {
                (OnnxGraphNode::Input(mut in_node), OnnxGraphNode::Initializer(init_node)) |
                (OnnxGraphNode::Initializer(init_node), OnnxGraphNode::Input(mut in_node)) => {
                    // I due nodi con lo stesso nome sono un input ed un initializer (in qualsiasi ordine):
                    // aggiorna il nodo di input in modo che abbia il valore dell'initializer come default
                    in_node.default_value = Some(init_node.data);
                    nodes.insert(name, OnnxGraphNode::Input(in_node));
                },
                (old_node, _) => {
                    // Re-inserisci il nodo di input nella HashMap e termina con un errore.
                    nodes.insert(name.clone(), old_node);
                    return Err(onnx_error!("Node {name} already exists."))
                }
            };
        }

        Ok(())
    }

    /// Estrae il valore di un nodo input da `inputs` o, se assente, il suo valore di default.
    /// 
    /// # Return
    /// [`Some(t)`] se il valore `t` è stato trovato ed è valido. [`None`] se questo non è stato trovato e non ha un valore di
    /// default, oppure se è stato trovato ma non ha una forma valida.
    fn get_input_value(
        &self,
        node: &OnnxGraphInput,
        inputs: &Arc<HashMap<String, Arc<Tensor>>>
    ) -> Option<Arc<Tensor>> {
        let input_data =
            inputs.get(&node.name)
                .or_else(|| node.default_value.as_ref())?
                .clone();

        if node.valid_shape(input_data.shape()) {
            Some(input_data)
        } else {
            None
        }
    }

    /// Esegue il procedimento generale di calcolo per un nodo operazione del grafo. Questo consiste nel
    /// 1. collezionare i valori di tutti i nodi in entrata, attendendo se necessario.
    /// 2. eseguire l'effettiva operazione.
    /// 3. al termine, propaga il risultato ai nodi in uscita.
    ///     * Se il nodo in uscita è un'operazione, questa viene avviata nel thread corrente o in un altro thread, in base alla
    ///       necessità (ad esempio, se l'unico nodo in uscita è un'operazione non serve avviare un thread apposta).
    /// 
    /// # Parametri
    /// * `infer_id`: ID dell'inferenza
    /// * `opnode_name`: Nome del nodo operazione di cui eseguire il procedimento
    /// * `graph_inputs`: [`HashMap`] che mappa il nome dell'input del grafo con il valore corrispondente
    /// 
    /// # Return
    /// In assenza di errori, la funzione ritorna un'[`HashMap`] che mappa il nome degli output al valore corrispondente.
    /// 
    /// Questo riguarda solo **una parte** degli output; il tutto verrà riunito nella funzione [`OnnxGraph::infer`].
    /// 
    /// # Errori
    /// * Se le operazioni di lock non vanno a buon fine.
    /// * Se uno qualsiasi dei nodi di input è assente o non valido.
    /// * Se il nodo operazione identificato da `opnode_name` non esiste.
    /// * Se il risultato dell'operazione del nodo corrente o di uno qualsiasi dei nodi entranti contiene un errore.
    /// * Se sono presenti anomalie con i nodi (ad esempio, un nodo input contiene dei nodi entranti.)
    /// 
    /// # Panico
    /// Se si verifica un errore e non è possibile notificare i thread in attesa di ciò. 
    fn compute_operation_node(
        self: Arc<Self>,
        infer_id: InferenceID,
        opnode_name: String,
        graph_inputs: Arc<HashMap<String, Arc<Tensor>>>
    ) -> Result<HashMap<String, Arc<Tensor>>, OnnxError> {
        // Lock delle risorse
        let nodes =
            self.nodes.read()
                .map_err(|_| onnx_error!("[{infer_id:?}, {opnode_name}] An error occurred while trying to lock nodes HashMap for read."))?;
        
        let this_opnode =
            nodes.get(&opnode_name)
                 .ok_or(onnx_error!("[{infer_id:?}, {opnode_name}] Node is not in the graph."))?
                 .ref_operation()?;
        
        // Colleziona valori in entrata (punto 1)
        let input_values: Vec<Arc<Tensor>> =
            this_opnode.inputs.iter()
            .map(|input_name| {
                // Estrai il valore in base al tipo di nodo di input.
                match nodes.get(input_name) {
                    Some(OnnxGraphNode::Operation(op_node)) => {
                        // Nodo operazione: vedi get_result.
                        log::debug!("[{infer_id:?}, {opnode_name}] Trying to get result from {}...", op_node.name);
                        let res = op_node.get_result(infer_id);
                        log::debug!("[{infer_id:?}, {opnode_name}] Got result from {}!", op_node.name);
                        res
                    },
                    Some(OnnxGraphNode::Input(in_node)) => {
                        // Nodo input: vedi get_input_value.
                        Ok(
                            self.get_input_value(in_node, &graph_inputs)
                                .ok_or(onnx_error!("[{infer_id:?}, {opnode_name}] Node {} is missing or invalid.", in_node.name))?
                        )
                    },
                    Some(OnnxGraphNode::Initializer(init_node)) => {
                        // Nodo initializer: il valore è costante.
                        Ok(init_node.data.clone())
                    },
                    Some(_) => return Err(onnx_error!("[{infer_id:?}, {opnode_name}] Node {input_name} cannot be used an an input to this node.")),
                    None => return Err(onnx_error!("[{infer_id:?}, {opnode_name}] Node {input_name} not found."))
                }
            })
            .collect::<Result<_, OnnxError>>()
            .map_err(|e| {
                // I nodi in attesa di questa operazione devono essere notificati dell'errore
                this_opnode.end_operation_with_error(infer_id, e.clone()).expect("Failed to force-end operation.");
                e
            })?;
        
        // Esegui operazione effettiva (punto 2)
        log::info!("[{infer_id:?}] Starting operation {opnode_name}...");
        let result =
            this_opnode.execute_operation(infer_id, input_values)
                .map_err(|e| onnx_error!("[{infer_id:?}, {opnode_name}] {}", e.msg))?;
        log::info!("[{infer_id:?}] Operation {opnode_name} finished!");

        // Propaga il risultato ai nodi in uscita (punto 3)
        let mut out_hashmap: HashMap<String, Arc<Tensor>> = HashMap::new();
        let mut output_opnodes: Vec<&OnnxGraphOperation> = Vec::new();
        for output_name in &this_opnode.outputs {
            // Tratta il risultato in modo diverso in base al tipo di nodo in uscita
            match nodes.get(output_name) {
                Some(OnnxGraphNode::Output(out_node)) => {
                    // Nodo output: inserisci il risultato nella HashMap da restituire
                    // Se l'output non ha una forma corretta, termina con un errore.
                    if out_node.valid_shape(result.shape()) {
                        out_hashmap.insert(out_node.name.clone(), result.clone());
                    } else {
                        // I nodi in attesa di questa operazione devono essere notificati dell'errore
                        let e = onnx_error!("[{infer_id:?}, {opnode_name}] Output {} has an invalid shape.", out_node.name);
                        this_opnode.end_operation_with_error(infer_id, e.clone()).expect("Failed to force-end operation.");
                        return Err(e);
                    }

                    // Il valore è stato passato al nodo output: serve incrementare il numero di notifiche al nodo corrente
                    this_opnode
                        .increment_notified_count(infer_id)
                        .map_err(|e| {
                            // I nodi in attesa di questa operazione devono essere notificati dell'errore
                            this_opnode.end_operation_with_error(infer_id, e.clone()).expect("Failed to force-end operation.");
                            e
                        })?;
                },
                Some(OnnxGraphNode::Operation(op_node)) => {
                    // Nodo operazione: marca l'operazione come avviata. Se l'operazione non è già stata avviata in precedenza
                    // (start_operation ritorna true), aggiungi nel vettore di operazioni da avviare (output_opnodes).
                    let start_operation_res = 
                        op_node
                            .start_operation(infer_id)
                            .map_err(|e| {
                                // I nodi in attesa di questa operazione devono essere notificati dell'errore
                                this_opnode.end_operation_with_error(infer_id, e.clone()).expect("Failed to force-end operation.");
                                e
                            })?;
                    if start_operation_res {
                        output_opnodes.push(op_node);
                    }
                },
                Some(_) => {
                    // Altri nodi: gli input/initializer non possono essere nodi in uscita.
                    // I nodi in attesa di questa operazione devono essere notificati dell'errore
                    let e = onnx_error!("[{infer_id:?}, {opnode_name}] Expected operation or output node, but {} is an input/initializer.", output_name);
                    this_opnode.end_operation_with_error(infer_id, e.clone()).expect("Failed to force-end operation.");
                    return Err(e);
                },
                None => {
                    // I nodi in attesa di questa operazione devono essere notificati dell'errore
                    let e = onnx_error!("[{infer_id:?}, {opnode_name}] Node {output_name} is not an output to this node.");
                    this_opnode.end_operation_with_error(infer_id, e.clone()).expect("Failed to force-end operation.");
                    return Err(e);
                }
            };
        }

        // Avvia le operazioni identificate in output_opnodes, se presenti (punto "3.1")
        out_hashmap.extend(self.clone().execute_node_operations(infer_id, output_opnodes, graph_inputs, opnode_name)?);

        Ok(out_hashmap)
    }

    /// Esegue [`compute_operation_node`][`OnnxGraph::compute_operation_node`] per tutti i nodi passati per parametro (contenuti in `op_nodes`).
    /// 
    /// L'esecuzione sarà diversa in base al numero di nodi operazione passati come parametro (`op_nodes.len()`):
    /// * Se == 1, ci sarà una sola operazione da eseguire, per cui non serve un thread apposta: l'operazione viene eseguita nel
    ///   thread corrente.
    /// * Se > 1, verranno creati tanti thread quanti sono le operazioni da eseguire. Il thread corrente si occuperà di eseguire
    ///   la join ed unire i vari risultati in uno unico.
    /// 
    /// # Parametri
    /// * `infer_id`: ID dell'inferenza interno al processo.
    /// * `op_nodes`: Vettore di nodi operazione da eseguire.
    /// * `graph_inputs`: [HashMap] che mappa il nome del nodo input al valore corrispondente.
    /// 
    /// # Return
    /// In assenza di errori, la funzione ritorna un'[`HashMap`] che mappa il nome degli output al valore corrispondente.
    /// 
    /// Questo riguarda solo **una parte** degli output; il tutto verrà riunito nella funzione [`OnnxGraph::infer`].
    /// 
    /// # Errore
    /// * Se non è possibile effettuare la [`join`][`JoinHandle::join`].
    /// * Se una qualsiasi delle operazioni eseguite termina con un errore.
    /// 
    /// # Panico
    /// Se si verifica un errore e non è possibile notificare i thread in attesa di ciò.
    fn execute_node_operations(
        self: Arc<Self>,
        infer_id: InferenceID,
        op_nodes: Vec<&OnnxGraphOperation>,
        graph_inputs: Arc<HashMap<String, Arc<Tensor>>>,
        parent_name: String
    ) -> Result<HashMap<String, Arc<Tensor>>, OnnxError> {
        if op_nodes.len() == 1 {
            // Una operazione: calcola il risultato nel thread corrente.
            let op_node = op_nodes[0];
            log::debug!("[{infer_id:?}, {parent_name}] Executing {} in the same thread...", op_node.name);
            self.compute_operation_node(infer_id, op_node.name.clone(), graph_inputs.clone())
        } else if op_nodes.len() > 1 {
            // Più operazioni: crea tanti thread quante sono le operazioni.
            let threads: Vec<(&OnnxGraphOperation, JoinHandle<_>)> = 
                op_nodes
                    .into_iter()
                    .map(|node| {
                        log::debug!("[{infer_id:?}, {parent_name}] Spawning thread for {}...", node.name);
                        let a_self = self.clone();
                        let c_input_name = node.name.clone();
                        let a_graph_inputs = graph_inputs.clone();
                        (node, thread::spawn(move || a_self.compute_operation_node(infer_id, c_input_name, a_graph_inputs)))
                    })
                    .collect();

            // Joina thread
            let out_hashmaps: Vec<HashMap<String, Arc<Tensor>>> =
                threads
                    .into_iter()
                    .map(|(node, t)| {
                        Ok(
                            t.join().map_err(|_| {
                                // I nodi in attesa di questa operazione devono essere notificati dell'errore
                                let e = onnx_error!("[{}] Failed to join thread for this node.", node.name);
                                node.end_operation_with_error(infer_id, e.clone()).expect("Failed to force-end operation.");
                                e
                            })??
                        )
                    })
                    .collect::<Result<_, OnnxError>>()?;

            // Unisci i risultati in un'unica HashMap.
            let out_hashmap: HashMap<String, Arc<Tensor>> =
                out_hashmaps
                    .into_iter()
                    .flat_map(|hm| hm.into_iter())
                    .collect();

            Ok(out_hashmap)
        } else {
            // Nessuna operazione: restituisci HashMap vuota.
            Ok(HashMap::new())
        }
    }

    /// Restituisce il nome dei nodi operazione di "primo livello", cioè quelli che contengono *solo* nodi input o initializer come
    /// nodi entranti. 
    fn get_first_layer_nodes<'a>(&self, nodes: &'a RwLockReadGuard<'a, HashMap<String, OnnxGraphNode>>) -> HashSet<&'a OnnxGraphNode> {
        nodes
            .values()
            .filter(|node| {
                if let OnnxGraphNode::Operation(op_node) = node {
                    op_node.inputs.iter().all(|input_name| self.inputs.contains(input_name) || self.initializers.contains(input_name))
                } else {
                    false
                }
            })
            .collect()
    }

    /// Restituisce l'ID dell'inferenza attuale e aggiorna l'ID più recente.
    fn generate_inference_id(&self) -> Result<InferenceID, PoisonError<MutexGuard<InferenceID>>> {
        let mut last_inference_id = self.last_inference_id.lock()?;
        let InferenceID(current)= *last_inference_id;

        *last_inference_id = InferenceID(current+1);
        return Ok(*last_inference_id);
    }

    /// Esegue l'inferenza sulla base del grafo attuale, seguendo un approccio top-down: a partire da ogni input, esegui le
    /// relative operazioni in modo parallelo e, al termine di ognuna, avvia le operazioni relative ai nodi uscenti finché tutti
    /// gli output non sono disponibili.
    /// 
    /// L'operazione richiede che il metodo possa essere chiamato da più thread contemporaneamente, quindi occorre che [`Self`]
    /// sia contenuto in un Arc.
    /// 
    /// # Parametri
    /// * `inputs`: [HashMap] che mappa il nome dei nodi input al valore corrispondente.
    /// 
    /// # Errori
    /// * Se le operazioni di lock non vanno a buon fine.
    /// * Se si verifica un errore durante qualsiasi delle operazioni parte del grafo.
    pub fn infer(self: Arc<Self>, inputs: HashMap<String, Tensor>) -> OnnxInferenceResult {
        // Circonda la mappa passata per parametro e ogni suo valore in un Arc, in modo da poter condividere i dati tra più
        // thread. Inoltre, Arc<Tensor> rende possibile salvare i dati del Tensor una sola volta in memoria, anziché doverli
        // clonare ogni volta che un thread ne ha bisogno (viene clonato solo il puntatore).
        let inputs = Arc::new(
            inputs
                .into_iter()
                .map(|(name, data)| (name, Arc::new(data)))
                .collect()
        );

        // Lock delle risorse
        let infer_id =
            self.generate_inference_id()
                .map_err(|_| onnx_error!("[Initial] PoisonError occurred while locking inference ID."))?;

        log::info!("Starting inference with {infer_id:?}...");

        let nodes =
            self.nodes.read()
                .map_err(|_| onnx_error!("[Initial] An error occurred while trying to lock nodes HashMap for read."))?;
                
        // Incrementa numero di inferenze in corso.
        self.inferences_in_progress.fetch_add(1, Ordering::Relaxed);

        // Ricava i nodi di "primo livello", i.e. quelli che hanno solo input/initializer tra i nodi entranti.
        let first_layer_nodes: Vec<&OnnxGraphOperation> =
            self.get_first_layer_nodes(&nodes)
                .into_iter()
                .map(|node| node.ref_operation())
                .collect::<Result<Vec<_>, OnnxError>>()?;

        // Marca l'operazione come avviata per ogni nodo trovato.
        for node in first_layer_nodes.iter() {
            assert!(node.start_operation(infer_id)?)
        }

        // Esegui in modo ricorsivo tutte le operazioni relative ai nodi trovati
        let out_hashmap: HashMap<String, Arc<Tensor>> =
            self.clone().execute_node_operations(
                infer_id, 
                first_layer_nodes, 
                inputs, 
                "Initial".to_string()
            )?;

        log::info!("Finished inference {infer_id:?}!");

        // Decrementa numero di inferenze in corso.
        self.inferences_in_progress.fetch_sub(1, Ordering::Relaxed);

        // A questo punto, tutti gli Arc dentro out_hashmap dovrebbero avere un solo riferimento, quindi si può estrarre ogni valore puntato tramite Arc::try_unwrap
        let final_hashmap: HashMap<String, Tensor> =
            out_hashmap
                .into_iter()
                .map(|(name, arc)| {
                    Ok(
                        (
                            name.clone(),
                            Arc::try_unwrap(arc)
                                .map_err(|_| onnx_error!("[{infer_id:?}, Initial] Failed to unwrap Arc for node {name}."))?
                        )
                    )
                })
                .collect::<Result<Vec<(String, Tensor)>, OnnxError>>()?
                .into_iter()
                .collect();

        Ok(final_hashmap)
    }

}

impl FromStr for OnnxGraph {
    type Err = OnnxError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let lines = s.lines().map(|s| s.trim());
        OnnxParser::new(lines)
            .parse()
            .map_err(|e| onnx_error!("An error occurred during parsing: {:?}.", e))
    }
}