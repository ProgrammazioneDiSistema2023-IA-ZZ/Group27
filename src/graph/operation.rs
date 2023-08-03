use std::{collections::HashMap, sync::{Mutex, Condvar, Arc}};

use crate::{operations::{Operation, OperationResult, Tensor}, helper::InnerMutexGuard, error::OnnxError, onnx_error};

use super::InferenceID;

/// Stato di esecuzione dell'operazione
pub(super) enum OpStatus {
    /// Da avviare.
    ToStart,

    /// Avviata.
    Started,

    /// Terminata, con dati sul risultato.
    Finished(OperationResult),

    /// Terminata, dati sul risultato liberati perché non più necessari.
    Cleared
}

impl OpStatus {

    /// Indica se lo stato corrente è operazione terminata.
    pub(super) fn is_finished(&self) -> bool {
        match self {
            Self::Finished(_) => true,
            _ => false
        }
    }

    /// Indica se lo stato corrente è operazione avviata.
    pub(super) fn is_started(&self) -> bool {
        match self {
            Self::Started => true,
            _ => false
        }
    }

    /// Indica se lo stato corrente è operazione da avviare.
    pub(super) fn is_to_start(&self) -> bool {
        match self {
            Self::ToStart => true,
            _ => false
        }
    }
}

/// Nodo operazione all'interno di un grafo.
pub struct OnnxGraphOperation {
    /// Nome del nodo.
    pub(super) name: String,
    
    /// Nomi dei nodi in entrata nel nodo operazione.
    pub(super) inputs: Vec<String>,

    /// Nomi dei nodi in uscita dal nodo operazione.
    pub(super) outputs: Vec<String>,

    /// Operazione da eseguire nel nodo.
    operation: Operation,

    /// [`HashMap`] che mappa l'identificativo dell'inferenza con i corrispondenti stati delle operazioni.
    statuses: Mutex<HashMap<InferenceID, OpStatus>>,

    /// Condition variable usata per notificare una delle inferenze che l'operazione è stata terminata.
    /// 
    /// Le altre, grazie alla wait_while, rimaranno in attesa dopo il risveglio.
    cv: Condvar,

    /// [`HashMap`] che mappa l'identificativo dell'inferenza con il numero di volte che il risultato è stato passato ai nodi
    /// uscenti.
    /// 
    /// Questo permette di liberare il prima possibile le risorse quando non sono più necessarie (principalmente il risultato
    /// intermedio memorizzato in [`OpStatus::Finished`] non appena questo è stato comunicato a tutti i nodi uscenti).
    notified_count: Mutex<HashMap<InferenceID, usize>>
}

impl OnnxGraphOperation {
    /// Crea un nuovo nodo operazione con i relativi dati.
    pub fn new(name: &str, operation: Operation, inputs: Vec<&str>, outputs: Vec<&str>) -> Self {
        Self {
            name: name.to_string(),
            inputs: inputs.into_iter().map(str::to_string).collect(),
            outputs: outputs.into_iter().map(str::to_string).collect(),
            operation,
            statuses: Mutex::new(HashMap::new()),
            cv: Condvar::new(),
            notified_count: Mutex::new(HashMap::new())
        }
    }

    /// Prende possesso di `self.statuses` e restituisce un [`InnerMutexGuard`] relativo all'operazione dell'inferenza data.
    /// 
    /// Se l'inferenza non ha alcuno status associato nella mappa, ne viene associato uno nuovo con valore [`OpStatus::ToStart`].
    /// 
    /// # Errore
    /// Se le operazioni di lock non vanno a buon fine.
    pub(super) fn get_status(&self, infer_id: InferenceID) -> Result<InnerMutexGuard<InferenceID, OpStatus>, OnnxError> {
        // Lock della mappa
        let mut statuses =
            self.statuses.lock()
                .map_err(|_| onnx_error!("[{}] PoisonError occurred while locking statuses.", self.name))?;

        statuses
            .entry(infer_id)
            .or_insert(OpStatus::ToStart);

        Ok(InnerMutexGuard(statuses, infer_id))
    }
    
    /// Termina forzatamente l'operazione con l'errore passato per parametro e notifica gli eventuali subscriber di ciò.
    /// 
    /// Se l'operazione risulta già terminata con un errore, questo non viene sovrascritto.
    /// 
    /// # Errore
    /// Se le operazioni di lock non hanno successo.
    pub(super) fn end_operation_with_error(&self, infer_id: InferenceID, e: OnnxError) -> Result<(), OnnxError> {
        // Lock delle risorse
        let mut status = self.get_status(infer_id)?;

        if let OpStatus::Finished(Err(_)) = *status {
            // L'operazione è già terminata con un errore: non fare nulla.
            Ok(())
        } else {
            // L'operazione deve ancora terminare o è già terminata con successo: sovrascrivi lo stato con l'errore e notifica i subscriber.
            *status = OpStatus::Finished(Err(e));
            self.cv.notify_all();
            Ok(())
        }
    }

    /// Marca l'operazione corrente con stato [`OpStatus::Started`], indicando se lo stato è stato effettivamente cambiato.
    /// 
    /// # Return
    /// `true` se lo status è stato effettivmente cambiato (l'operazione aveva come stato precedente [`OpStatus::ToStart`]),
    /// `false` altrimenti
    /// 
    /// # Errore
    /// Se le operazioni di lock non vanno a buon fine.
    pub(super) fn start_operation(&self, infer_id: InferenceID) -> Result<bool, OnnxError> {
        // Lock delle risorse
        let mut status = self.get_status(infer_id)?;
    
        if status.is_to_start() {
            // L'operazione doveva ancora cominciare: cambia stato e restituisci true.
            *status = OpStatus::Started;
            Ok(true)
        } else {
            // L'operazione è già cominciata/terminata: non fare nulla e restituisci false.
            Ok(false)
        }
    }

    /// Incrementa `notified_count` per l'inferenza corrente (in modo atomico) e restituisce il numero aggiornato.
    /// 
    /// Inoltre, se il conto aggiornato è pari al numero di nodi uscenti, effettua il cambio di stato da [`OpStatus::Finished`] a
    /// [`OpStatus::Cleared`] in modo da liberare le risorse occupate dal risultato memorizzato.
    /// 
    /// # Errore
    /// Se le operazioni di lock non vanno a buon fine.
    pub(super) fn increment_notified_count(&self, infer_id: InferenceID) -> Result<(), OnnxError> {
        // Lock delle risorse
        let mut notified_count =
            self.notified_count.lock()
                .map_err(|_| onnx_error!("[{}] PoisonError occurred while locking notified count.", self.name))?;

        // Incrementa o aggiungi 1 se assente
        notified_count
            .entry(infer_id)
            .and_modify(|count| *count += 1)
            .or_insert(1);

        // Se il conto incrementato è pari al numero di output, allora il risultato è stato passato a tutti: si può liberare il risultato intermedio.
        let new_count = *notified_count.get(&infer_id).unwrap();
        drop(notified_count);
        if new_count == self.outputs.len() {
            self.clear_status_data(infer_id)?;
        }

        Ok(())
    }
    
    /// Restituisce il risultato dell'operazione relativo all'inferenza corrente.
    /// 
    /// Se l'operazione è già terminata, estrae il riusultato e lo restituisce, mentre se deve terminare attende (in modo
    /// bloccante!) che l'operazione termini.
    /// 
    /// # Errore
    /// Se le operazioni di lock non vanno a buon fine.
    pub(super) fn get_result(&self, infer_id: InferenceID) -> OperationResult {
        // Lock delle risorse
        let mut status = self.get_status(infer_id)?;

        // Estrai risultato in base allo stato dell'operazione
        let result: OperationResult =
            if let OpStatus::Finished(result) = &*status {
                // Operazione già terminata: restituisci il risultato.
                result.clone()
            } else {
                // Operazione avviata ma ancora da terminare: attendi tramite condition variable e restituisci il risultato non appena l'operazione termina.
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
        
        // Incrementa notified_count per l'inferenza corrente
        self.increment_notified_count(infer_id)?;

        result
    }
    
    /// Esegue l'operazione dati i suoi input.
    /// 
    /// Al termine dell'operazione, aggiorna lo stato a [`OpStatus::Finished`] e notifica gli eventuali subscriber della
    /// [`Condvar`] del termine.
    pub(super) fn execute_operation<'a>(&self, infer_id: InferenceID, inputs: Vec<Arc<Tensor>>) -> OperationResult {
        // Calcola il risultato.
        let result = self.operation.execute(inputs);

        // Aggiorna lo stato dell'operazione e notifica i subscriber.
        let mut status = self.get_status(infer_id)?;

        assert!(status.is_started());
        *status = OpStatus::Finished(result.clone());
        self.cv.notify_all();

        result
    }

    /// Libera i dati dello status relativo all'inferenza corrente.
    pub(super) fn clear_status_data(&self, infer_id: InferenceID) -> Result<(), OnnxError> {
        // Prendi possesso di statuses.
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