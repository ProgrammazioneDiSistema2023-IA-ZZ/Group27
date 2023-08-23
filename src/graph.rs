use std::{collections::{HashMap, HashSet}, sync::{Arc, MutexGuard, Mutex, PoisonError, RwLock, RwLockReadGuard, atomic::{AtomicUsize, Ordering}}, thread::{JoinHandle, self}, hash::Hash};
use log;
use crate::{error::OnnxError, operations::Tensor, onnx_error, fileparser::fileparser::OnnxFileParser};

pub use self::{operation::OnnxGraphOperation, input::OnnxGraphInput, output::OnnxGraphOutput, initializer::OnnxGraphInitializer, intermediate::OnnxGraphIntermediate};
use pyo3::*;
pub mod operation;
pub mod input;
pub mod output;
pub mod initializer;
pub mod intermediate;

/// Generic node of a graph.

pub enum OnnxGraphNode {
    Initializer(OnnxGraphInitializer),
    Input(OnnxGraphInput),
    Output(OnnxGraphOutput),
    Operation(OnnxGraphOperation),
    Intermediate(OnnxGraphIntermediate)
}

impl OnnxGraphNode {
    /// Returns the name of this node, independently of its type.
    pub fn name(&self) -> &String {
        match self {
            OnnxGraphNode::Initializer(node) => &node.name,
            OnnxGraphNode::Input(node) => &node.name,
            OnnxGraphNode::Output(node) => &node.name,
            OnnxGraphNode::Operation(node) => &node.name,
            OnnxGraphNode::Intermediate(node) => &node.name
        }
    }

    /// Gets a reference to an [`OnnxGraphOperation`] with respect to the current node.
    /// 
    /// # Error
    /// If the current node is not an operation.
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

/// ID of an inference, internal to the current process.
#[derive(Hash, PartialEq, Eq, Clone, Copy, Debug)]
struct InferenceID(usize);
#[pyclass(frozen)]
/// Graph that represents a neural network.
pub struct OnnxGraph {
    /// Name of the graph.
    pub name: String,

    /// Names of initializer nodes inside the graph.
   pub initializers: HashSet<String>,

    /// Names of input nodes inside the graph.
   pub inputs: HashSet<String>,

    /// Names of output nodes inside the graph.
  pub  outputs: HashSet<String>,

    /// Names of operazione nodes inside the graph.
   pub operations: HashSet<String>,

    /// [`HashMap`] that maps the name of the node to the corresponding [`OnnxGraphNode`] enum.
    /// 
    /// This structure can be used by more threads concurrently.
    /// 
    /// Writing operations (such as adding a node) usually take place before an inference, while the inference itself requires
    /// only reading operations. For this reason, a [`RwLock`] is used instead of a [`Mutex`].
    pub  nodes: RwLock<HashMap<String, OnnxGraphNode>>,

    /// Number of inferences currently in progress.
   pub inferences_in_progress: AtomicUsize,

    /// Contains the last inference ID used. It's incremented everytime a new inference is started in the same process.
     last_inference_id: Mutex<InferenceID>
}

/// Result of the graph's creation, if errors may occur (for instance, starting from a file).
pub type OnnxGraphResult = Result<OnnxGraph, OnnxError>;

/// Result of the inference.
/// 
/// If the operation is successful, [`Ok`] contains a [`HashMap`] that maps the name of output to the corresponding value.
pub type OnnxInferenceResult = Result<HashMap<String, Tensor>, OnnxError>;

impl OnnxGraph {
    /// Creates a new empty graph.
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

    /// Creates a new graph starting from a file, given its path.
    pub fn from_file(path: &str) -> OnnxGraphResult {
        OnnxFileParser::parse_model(path)
            .map_err(|msg| OnnxError::new(msg))
    }

    /// Adds an existing node to the graph.
    pub fn add_node(&mut self, node: OnnxGraphNode) -> Result<(), OnnxError> {
        // Prevent adding nodes while there are inferences in progress.
        if *self.inferences_in_progress.get_mut() > 0 {
            return Err(onnx_error!("Cannot add nodes while there are inferences in progress."));
        }

        // Lock resources
        let mut nodes =
            self.nodes.write()
                .map_err(|_| onnx_error!("PoisonError occurred while locking nodes for write (adding node {}).", node.name()))?;
        
        let name = node.name().clone();

        // Add name of the node to the appropriate array of this structure.
        match node {
            OnnxGraphNode::Initializer(_) => self.initializers.insert(name.clone()),
            OnnxGraphNode::Input(_) => self.inputs.insert(name.clone()),
            OnnxGraphNode::Output(_) => self.outputs.insert(name.clone()),
            OnnxGraphNode::Operation(_) => self.operations.insert(name.clone()),
            OnnxGraphNode::Intermediate(interm_node) => {
                // For each entering node E to the intermediate node I, add the exiting nodes of I to the exiting nodes of E, and
                // remove I from that collection.
                let interm_input_node =
                    nodes.get_mut(&interm_node.input)
                         .ok_or_else(|| onnx_error!("[{}] Input node {} does not exist.", interm_node.name, interm_node.input))?;
                if let OnnxGraphNode::Operation(op_node) = interm_input_node {
                    op_node.outputs.remove(&interm_node.name);
                    op_node.outputs.extend(interm_node.outputs.clone());
                }

                // For each exiting node O from the intermediate node I, replace the name of I from the entering nodeodes of O with
                // the entering node of I.
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

                // No need to add the intermediate node to the collection.
                return Ok(())
            }
        };

        // Add node to the collection.
        if let Some(old_node) = nodes.insert(name.clone(), node) {
            // Found duplicate node: remove from HashMap
            let new_node = nodes.remove(&name).unwrap();

            match (old_node, new_node) {
                (OnnxGraphNode::Input(mut in_node), OnnxGraphNode::Initializer(init_node)) |
                (OnnxGraphNode::Initializer(init_node), OnnxGraphNode::Input(mut in_node)) => {
                    // The two nodes with the same name are an input and an initializer (in any order): update the input node so
                    // that it has the initializer value as its default.
                    in_node.default_value = Some(init_node.data);
                    nodes.insert(name, OnnxGraphNode::Input(in_node));
                },
                (old_node, _) => {
                    // Re-insert the input node in the HashMap and terminate with an error.
                    nodes.insert(name.clone(), old_node);
                    return Err(onnx_error!("Node {name} already exists."))
                }
            };
        }

        Ok(())
    }

    /// Gets the value of an input node from `input` or, if absent, its default value.
    /// 
    /// # Return
    /// [`Some`]`(t)` if a valid value `t` was found. [`None`] if it was not found and it doesn't have a default value, or if it
    /// was found but with an invalid shape.
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
 
    /// Runs through the general process of executing the operation of an operation node. This consists in:
    /// 1. collecting the value of all entering nodes, waiting if necessary.
    /// 2. executing the operation
    /// 3. when finished, propagating the result to its exiting nodes.
    ///     * If the exiting node is an operation, this function will be called again but for that node, either in this thread
    ///       or another based on the number of exiting nodes (see [`execute_node_operations`][Self::execute_node_operations]).
    /// 
    /// # Parameters
    /// * `infer_id`: inference ID
    /// * `opnode_name`: Name of the operation node of which to execute the procedure.
    /// * `graph_inputs`: [`HashMap`] which maps the name of the graph input with the corresponding value.
    /// 
    /// # Return 
    /// If no error occurs, a [`HashMap`] which maps the name of the outputs with the corresponding value.
    /// 
    /// Note that this only contains *part* of the outputs. The final value with all outputs will be available at the end of
    /// [`OnnxGraph::infer`].
    /// 
    /// # Errors
    /// * Is any lock operation fails.
    /// * If any input node is absent or invalid.
    /// * If the operation node identified by `opnode_name` is unavailable.
    /// * If the result of the current node or any entering node's operation contains an error.
    /// * If any anomalies with nodes (such as an input node having entering nodes) are detected.
    /// 
    /// # Panics
    /// If an error occurs and this thread fails to notify others of this. 
    fn compute_operation_node(
        self: Arc<Self>,
        infer_id: InferenceID,
        opnode_name: String,
        graph_inputs: Arc<HashMap<String, Arc<Tensor>>>
    ) -> Result<HashMap<String, Arc<Tensor>>, OnnxError> {
        // Lock resources
        let nodes =
            self.nodes.read()
                .map_err(|_| onnx_error!("[{infer_id:?}, {opnode_name}] An error occurred while trying to lock nodes HashMap for read."))?;
        
        let this_opnode =
            nodes.get(&opnode_name)
                 .ok_or(onnx_error!("[{infer_id:?}, {opnode_name}] Node is not in the graph."))?
                 .ref_operation()?;
        
        // Collect input values (point 1)
        let input_values: Vec<Arc<Tensor>> =
            this_opnode.inputs.iter()
            .map(|input_name| {
                // Get the value based on the type of entering node.
                match nodes.get(input_name) {
                    Some(OnnxGraphNode::Operation(op_node)) => {
                        // Operation node: see get_result.
                        log::debug!("[{infer_id:?}, {opnode_name}] Trying to get result from {}...", op_node.name);
                        let res = op_node.get_result(infer_id);
                        log::debug!("[{infer_id:?}, {opnode_name}] Got result from {}!", op_node.name);
                        res
                    },
                    Some(OnnxGraphNode::Input(in_node)) => {
                        // Input node: see get_input_value.
                        Ok(
                            self.get_input_value(in_node, &graph_inputs)
                                .ok_or(onnx_error!("[{infer_id:?}, {opnode_name}] Node {} is missing or invalid.", in_node.name))?
                        )
                    },
                    Some(OnnxGraphNode::Initializer(init_node)) => {
                        // Initializer node: the value is constant.
                        Ok(init_node.data.clone())
                    },
                    Some(_) => return Err(onnx_error!("[{infer_id:?}, {opnode_name}] Node {input_name} cannot be used an an input to this node.")),
                    None => return Err(onnx_error!("[{infer_id:?}, {opnode_name}] Node {input_name} not found."))
                }
            })
            .collect::<Result<_, OnnxError>>()
            .map_err(|e| {
                // Waiting nodes have to be notified of the error.
                this_opnode.end_operation_with_error(infer_id, e.clone()).expect("Failed to force-end operation.");
                e
            })?;
        
        // Execute operation (point 2)
        log::info!("[{infer_id:?}] Starting operation {opnode_name}...");
        let result =
            this_opnode.execute_operation(infer_id, input_values)
                .map_err(|e| onnx_error!("[{infer_id:?}, {opnode_name}] {}", e.msg))?;
        log::info!("[{infer_id:?}] Operation {opnode_name} finished!");

        // Propagate the result to exiting nodes (point 3)
        let mut out_hashmap: HashMap<String, Arc<Tensor>> = HashMap::new();
        let mut output_opnodes: Vec<&OnnxGraphOperation> = Vec::new();
        for output_name in &this_opnode.outputs {
            // Treat the result differently based on the type of exiting node
            match nodes.get(output_name) {
                Some(OnnxGraphNode::Output(out_node)) => {
                    // Output node: insert result inside the return HashMap.
                    // If the output has an invalid shape, terminate with an error.
                    if out_node.valid_shape(result.shape()) {
                        out_hashmap.insert(out_node.name.clone(), result.clone());
                    } else {
                        // Waiting nodes have to be notified of the error.
                        let e = onnx_error!("[{infer_id:?}, {opnode_name}] Output {} has an invalid shape.", out_node.name);
                        this_opnode.end_operation_with_error(infer_id, e.clone()).expect("Failed to force-end operation.");
                        return Err(e);
                    }

                    // The value has been passed to the output node: we need to increment the number of notified nodes
                    this_opnode
                        .increment_notified_count(infer_id)
                        .map_err(|e| {
                            // Waiting nodes have to be notified of the error.
                            this_opnode.end_operation_with_error(infer_id, e.clone()).expect("Failed to force-end operation.");
                            e
                        })?;
                },
                Some(OnnxGraphNode::Operation(op_node)) => {
                    // Operation node: mark the operation as started. If the operation hasn't been started before
                    // (start_operation returns true), add into the array of operations to start (output_opnodes).
                    let start_operation_res = 
                        op_node
                            .start_operation(infer_id)
                            .map_err(|e| {
                                // Waiting nodes have to be notified of the error.
                                this_opnode.end_operation_with_error(infer_id, e.clone()).expect("Failed to force-end operation.");
                                e
                            })?;
                    if start_operation_res {
                        output_opnodes.push(op_node);
                    }
                },
                Some(_) => {
                    // Other nodes: inputs/initializers can't be exiting nodes.
                    // Waiting nodes have to be notified of the error.
                    let e = onnx_error!("[{infer_id:?}, {opnode_name}] Expected operation or output node, but {} is an input/initializer.", output_name);
                    this_opnode.end_operation_with_error(infer_id, e.clone()).expect("Failed to force-end operation.");
                    return Err(e);
                },
                None => {
                    // Waiting nodes have to be notified of the error.
                    let e = onnx_error!("[{infer_id:?}, {opnode_name}] Node {output_name} is not an output to this node.");
                    this_opnode.end_operation_with_error(infer_id, e.clone()).expect("Failed to force-end operation.");
                    return Err(e);
                }
            };
        }

        // Start operations saved inside output_opnodes, if present (point "3.1")
        // See execute_node_operations
        out_hashmap.extend(self.clone().execute_node_operations(infer_id, output_opnodes, graph_inputs, opnode_name)?);

        Ok(out_hashmap)
    }
 
    /// Executes [`compute_operation_node`][`OnnxGraph::compute_operation_node`] for all given nodes (contained in `op_nodes`).
    /// 
    /// The execution will depend based on the number of given nodes (length of `op_nodes`):
    /// * If `==1`, there will only be one operation to execute, so there is no real need to start a new thread as the current
    ///   one would just be stuck waiting: the operation is instead executed in the same thread.
    /// * If `>1`, there are more operations to execute. This thread will be responsible of creating one thread per operation
    ///   and waiting for the result.
    /// * If `==0`, there is no operation to execute: the function immediately returns.
    /// 
    /// # Parameters
    /// * `infer_id`: inference ID
    /// * `opnode_name`: Name of the operation node of which to execute the procedure.
    /// * `graph_inputs`: [`HashMap`] which maps the name of the graph input with the corresponding value.
    /// * `parent_name`: Name of the node which called this function. Used for logging purposes only.
    /// 
    /// # Return 
    /// If no error occurs, a [`HashMap`] which maps the name of the outputs with the corresponding value.
    /// 
    /// Note that this only contains *part* of the outputs. The final value with all outputs will be available at the end of
    /// [`OnnxGraph::infer`].
    /// 
    /// # Error
    /// * If the [`join`][`JoinHandle::join`] operation fails.
    /// * If any "child" operation fails.
    /// 
    /// # Panics
    /// If an error occurs and this (or any child) thread fails to notify others of this. 
    fn execute_node_operations(
        self: Arc<Self>,
        infer_id: InferenceID,
        op_nodes: Vec<&OnnxGraphOperation>,
        graph_inputs: Arc<HashMap<String, Arc<Tensor>>>,
        parent_name: String
    ) -> Result<HashMap<String, Arc<Tensor>>, OnnxError> {
        if op_nodes.len() == 1 {
            // One operation: execute operation in this thread.
            let op_node = op_nodes[0];
            log::debug!("[{infer_id:?}, {parent_name}] Executing {} in the same thread...", op_node.name);
            self.compute_operation_node(infer_id, op_node.name.clone(), graph_inputs.clone())
        } else if op_nodes.len() > 1 {
            // More operations: create as many threads as operations to execute.
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

            // Join threads
            let out_hashmaps: Vec<HashMap<String, Arc<Tensor>>> =
                threads
                    .into_iter()
                    .map(|(node, t)| {
                        Ok(
                            t.join().map_err(|_| {
                                // Waiting nodes have to be notified of the error.
                                let e = onnx_error!("[{}] Failed to join thread for this node.", node.name);
                                node.end_operation_with_error(infer_id, e.clone()).expect("Failed to force-end operation.");
                                e
                            })??
                        )
                    })
                    .collect::<Result<_, OnnxError>>()?;

            // Merge all results in a HashMap.
            let out_hashmap: HashMap<String, Arc<Tensor>> =
                out_hashmaps
                    .into_iter()
                    .flat_map(|hm| hm.into_iter())
                    .collect();

            Ok(out_hashmap)
        } else {
            // No operation: return empty HashMap.
            Ok(HashMap::new())
        }
    }

    /// Returns the name of all "first layer nodes", i.e nodes that contain *only* input and initializer nodes as entering nodes.
    fn get_first_layer_nodes<'a>(&self, nodes: &'a RwLockReadGuard<'a, HashMap<String, OnnxGraphNode>>) -> HashSet<&'a OnnxGraphNode> {
        self.operations
            .iter()
            .map(|name| nodes.get(name).unwrap())
            .filter(|node|{
                node
                    .ref_operation().unwrap()
                    .inputs.iter()
                    .all(|input_name| self.inputs.contains(input_name) || self.initializers.contains(input_name))
            })
            .collect()
    }

    /// Generates the ID for a new inference, and updates the last used ID.
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
    /// Executes an inference with the given inputs based on the current graph's state, following a top-down approach: from each
    /// input node, execute all exiting operations (in parallel if possible) and, after each one finishes, execute their exiting
    /// operations and so on, until all outputs are available of if an error occurs at any point.
    /// 
    /// This operation requires that this structure be passed between threads, so [`Self`] must be contained in an [`Arc`].
    /// 
    /// # Parameters
    /// * `inputs`: [HashMap] that maps the name of each input with its corresponding value.
    /// 
    /// # Error
    /// * If any lock operation fails.
    /// * If any error occurs during the execution of any operation inside the graph.
    pub fn infer(self: Arc<Self>, inputs: HashMap<String, Tensor>) -> OnnxInferenceResult {

        // Surround each input value in an Arc, so that it can be shared between threads. Moreover, Arc allows saving each input
        // data in memory only once, without having to clone everytime it's needed by another thread (heavy operation).
        let inputs = Arc::new(
            inputs
                .into_iter()
                .map(|(name, data)| (name, Arc::new(data)))
                .collect()
        );

        // Lock resources
        let infer_id =
            self.generate_inference_id()
                .map_err(|_| onnx_error!("[Initial] PoisonError occurred while locking inference ID."))?;

        log::info!("Starting inference with {infer_id:?}...");

        let nodes =
            self.nodes.read()
                .map_err(|_| onnx_error!("[Initial] An error occurred while trying to lock nodes HashMap for read."))?;
                
        // Increment number of inferences in progress.
        self.inferences_in_progress.fetch_add(1, Ordering::Relaxed);

        // Get "first layer nodes", i.e those that only have inputs/initializers as entering nodes.
        let first_layer_nodes: Vec<&OnnxGraphOperation> =
            self.get_first_layer_nodes(&nodes)
                .into_iter()
                .map(|node| node.ref_operation())
                .collect::<Result<Vec<_>, OnnxError>>()?;

        // Mark operation as started for each node found.
        for node in first_layer_nodes.iter() {
            assert!(node.start_operation(infer_id)?)
        }

        // Execute operations of nodes found recursively.
        let out_hashmap: HashMap<String, Arc<Tensor>> =
            self.clone().execute_node_operations(
                infer_id, 
                first_layer_nodes, 
                inputs, 
                "Initial".to_string()
            )?;

        log::info!("Finished inference {infer_id:?}!");

        // Decrement number of inferences in progress.
        self.inferences_in_progress.fetch_sub(1, Ordering::Relaxed);

        // At this point, all Arcs inside out_hashmap should only have one reference, so we can consume the Arc and extract its
        // value by using Arc::try_unwrap
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