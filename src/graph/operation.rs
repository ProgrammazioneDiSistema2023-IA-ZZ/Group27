use std::{collections::{HashMap, HashSet}, sync::{Mutex, Condvar, Arc}};

use crate::{operations::{Operation, OperationResult, Tensor}, helper::InnerMutexGuard, error::OnnxError, onnx_error};

use super::InferenceID;

/// Status of execution of an operation.
pub(super) enum OpStatus {
    ToStart,

    Started,

    /// Finished, with the saved value of the result.
    Finished(OperationResult),

    /// Finished, value of the result is no longer owned by the operation as it's already been passed to all its outputs.
    Cleared
}

impl OpStatus {

    pub(super) fn is_finished(&self) -> bool {
        match self {
            Self::Finished(_) => true,
            _ => false
        }
    }

    pub(super) fn is_started(&self) -> bool {
        match self {
            Self::Started => true,
            _ => false
        }
    }

    pub(super) fn is_to_start(&self) -> bool {
        match self {
            Self::ToStart => true,
            _ => false
        }
    }
}
/// Operation node of a graph.
pub struct OnnxGraphOperation {
    /// Nome del nodo.
    pub(super) name: String,
    
    /// Names of the nodes that are inputs to this node.
    /// 
    /// It's a Vec instead of a HashSet because the input order is important.
    pub(super) inputs: Vec<String>,

    /// Names of the nodes that are outputs to this node.
    pub(super) outputs: HashSet<String>,

    /// Data related to the operation of this node.
    operation: Operation,

    /// [`HashMap`] that maps the inference ID with the respective status of the operation.
    statuses: Mutex<HashMap<InferenceID, OpStatus>>,

    /// Condition variable used to notify any inference that the operation has finished.
    /// 
    /// The others, thanks to `wait_while`, will keep waiting after being notified.
    cv: Condvar,

    /// [`HashMap`] that maps the inference ID with the number of times that the result has been passed to the outputs of this
    /// node.
    /// 
    /// This allows to free resources (mainly the result saved along with [`OpStatus::Finished`]) as soon as they're no longer
    /// needed.
    notified_count: Mutex<HashMap<InferenceID, usize>>
}

impl OnnxGraphOperation {
    /// Creates a new operation node with related data.
    pub fn new<SI, SO, I, O>(
        name: impl ToString,
        operation: Operation,
        inputs: I,
        outputs: O
    ) -> Self
    where
        SI: ToString,
        SO: ToString,
        I: IntoIterator<Item = SI>,
        O: IntoIterator<Item = SO>
    {
        Self {
            name: name.to_string(),
            inputs: inputs.into_iter().map(|s| s.to_string()).collect(),
            outputs: outputs.into_iter().map(|s| s.to_string()).collect(),
            operation,
            statuses: Mutex::new(HashMap::new()),
            cv: Condvar::new(),
            notified_count: Mutex::new(HashMap::new())
        }
    }

    /// Locks `self.statuses`and returns an [`InnerMutexGuard`] of the status related to the given inference.
    ///
    /// If there is no such status, a new one will be created with value [`OpStatus::ToStart`].
    /// 
    /// # Error
    /// If any lock operation fails.
    pub(super) fn get_status(&self, infer_id: InferenceID) -> Result<InnerMutexGuard<InferenceID, OpStatus>, OnnxError> {
        // Lock statuses map.
        let mut statuses =
            self.statuses.lock()
                .map_err(|_| onnx_error!("[{}] PoisonError occurred while locking statuses.", self.name))?;

        statuses
            .entry(infer_id)
            .or_insert(OpStatus::ToStart);

        Ok(InnerMutexGuard(statuses, infer_id))
    }
    
    /// Forcefully terminates the operation with the given error and notifies any [`Condvar`] subscriber of this.
    ///
    /// If the operation already appears to have terminated with an error, it will not be overwritten.
    /// 
    /// # Error
    /// If any lock operation fails.
    pub(super) fn end_operation_with_error(&self, infer_id: InferenceID, e: OnnxError) -> Result<(), OnnxError> {
        // Lock resources
        let mut status = self.get_status(infer_id)?;

        if let OpStatus::Finished(Err(_)) = *status {
            // Operation already finished with an error: don't do anything.
            Ok(())
        } else {
            // Operation still has to finish, or it has successfully finish: overwrite the status and notify the subscribers.
            *status = OpStatus::Finished(Err(e));
            self.cv.notify_all();
            Ok(())
        }
    }
 
    /// Marks the current operation as [`Started`][OpStatus::Started].
    /// 
    /// # Return
    /// `true` if the status was changed (i.e the previous status was [`OpStatus::ToStart`]),
    /// `false` otherwise.
    /// 
    /// # Error
    /// If any lock operation fails.
    pub(super) fn start_operation(&self, infer_id: InferenceID) -> Result<bool, OnnxError> {
        // Lock resources
        let mut status = self.get_status(infer_id)?;
    
        if status.is_to_start() {
            // Operation still has to start: change status and return true.
            *status = OpStatus::Started;
            Ok(true)
        } else {
            // Operation already started/finished: don't do anything and return false.
            Ok(false)
        }
    }

    /// Increments `notified_count` for the current inference (atomically) and returns the updated count.
    /// 
    /// Moreover, if the updated count is equal to the number of outputs to this node, this function will also update the status
    /// from [`Finished`][OpStatus::Finished] to [`Cleared`][OpStatus::Cleared], so that the saved result will be freed.
    /// 
    /// # Error
    /// If any lock operation fails.
    pub(super) fn increment_notified_count(&self, infer_id: InferenceID) -> Result<(), OnnxError> {
        // Lock resources
        let mut notified_count =
            self.notified_count.lock()
                .map_err(|_| onnx_error!("[{}] PoisonError occurred while locking notified count.", self.name))?;

        // Increment or insert 1 if missing
        notified_count
            .entry(infer_id)
            .and_modify(|count| *count += 1)
            .or_insert(1);

        // If the new count is equal to the number of outputs, then the result has been passed to all outputs: the intermediate
        // result saved with OpStatus::Finished can be cleared.
        let new_count = *notified_count.get(&infer_id).unwrap();
        drop(notified_count);
        if new_count == self.outputs.len() {
            self.clear_status_data(infer_id)?;
        }

        Ok(())
    }
    
    /// Returns the result of this operation, related to the given inference.
    ///
    /// If the operation has already been terminated, the function will immediately extract the result and return it or, if the
    /// operation is still in progress, it will wait for it to finish (in a blocking way!).
    /// 
    /// # Error
    /// If any lock operation fails.
    pub(super) fn get_result(&self, infer_id: InferenceID) -> OperationResult {
        // Lock resources
        let mut status = self.get_status(infer_id)?;

        // Estrai risultato in base allo stato dell'operazione
        let result: OperationResult =
            if let OpStatus::Finished(result) = &*status {
                // Operation already finished: return the result.
                result.clone()
            } else {
                // Operation still has to finish: wait via condition variable and return the result as soon as it finishes.
                let InnerMutexGuard(statuses, _) = status;
                status = InnerMutexGuard(
                    self.cv.wait_while(statuses, |statuses| !statuses.get(&infer_id).unwrap().is_finished())
                        .map_err(|_| onnx_error!("[{}] PoisonError occurred while waking up from Condvar.", self.name))?,
                    infer_id
                );

                let OpStatus::Finished(result) = &*status else { unreachable!() };

                result.clone()
            };

        drop(status);
        
        // Increment notified_count for current inference.
        self.increment_notified_count(infer_id)?;

        result
    }
    
    /// Executes the operation, given its inputs.
    ///
    /// When the operation finishes, the function will update the status to [`Finished`][OpStatus::Finished] and notify any
    /// [`Condvar`] subscriber of this.
    pub(super) fn execute_operation<'a>(&self, infer_id: InferenceID, inputs: Vec<Arc<Tensor>>) -> OperationResult {
        // Calculate the result.
        let result = self.operation.execute(inputs);

        // Update the status of this operation and notify the subscribers.
        let mut status = self.get_status(infer_id)?;

        assert!(status.is_started());
        *status = OpStatus::Finished(result.clone());
        self.cv.notify_all();

        result
    }

    /// Updates the status from [`Finished`][OpStatus::Finished] to [`Cleared`][OpStatus::Cleared], so that the saved result
    /// will be freed.
    pub(super) fn clear_status_data(&self, infer_id: InferenceID) -> Result<(), OnnxError> {
        // Lock statuses map.
        let mut statuses =
            self.statuses.lock()
                .map_err(|_| onnx_error!("[{}] Poison Error occurred while locking statuses.", self.name))?;

        statuses
            .entry(infer_id)
            .and_modify(|s| {
                assert!(s.is_finished());
                *s = OpStatus::Cleared
            });

        Ok(())
    }
}