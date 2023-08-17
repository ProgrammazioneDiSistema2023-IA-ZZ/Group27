extern crate ndarray;

use std::{sync::Arc, collections::HashMap};
use ndarray::{ArrayD, Array1, ArrayViewD, Zip};
use crate::{error::OnnxError, onnx_error};

mod add;
mod averagepool;
mod concat;
mod constant;
mod conv;
mod div;
mod dropout;
mod gemm;
mod lrn;
mod matmul;
mod maxpool;
mod relu;
mod reshape;
mod softmax;

/// Generico array multidimensionale contenente dei float ([`f32`]).
pub type Tensor = ArrayD<f32>;

/// Tipo di operazione
#[derive(Clone)]
pub enum OpType {
    /// Addizione tra due input.
    Add,

    /// Simile alla convoluzione, ma per ogni finestra calcola la media.
    AveragePool,

    /// Concatenazione su un asse.
    Concat,

    /// Costante definita negli attributi.
    Constant,

    /// Convoluzione.
    Conv,

    // Divisione tra due input.
    Div,

    /// Operazione usata durante il training, in questo contesto non fa nulla.
    Dropout,

    // General Matrix Multiplication.
    Gemm,

    /// Local Response Normalization.
    LRN,

    /// Moltiplicazione tra due matrici.
    MatMul,

    /// Simile alla convoluzione, ma per ogni finestra sceglie il massimo.
    MaxPool,

    /// Y = max(0, X).
    Relu,

    /// Cambio forma.
    Reshape,

    /// Softmax(input, axis) = Exp(input) / ReduceSum(Exp(input), axis=axis, keepdims=1)
    Softmax
}

impl TryFrom<&str> for OpType {
    type Error = OnnxError;
    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "Add" => Ok(Self::Add),
            "AveragePool" => Ok(Self::AveragePool),
            "Concat" => Ok(Self::Concat),
            "Constant" => Ok(Self::Constant),
            "Conv" => Ok(Self::Conv),
            "Div" => Ok(Self::Div),
            "Dropout" => Ok(Self::Dropout),
            "Gemm" => Ok(Self::Gemm),
            "LRN" => Ok(Self::LRN),
            "MatMul" => Ok(Self::MatMul),
            "MaxPool" => Ok(Self::MaxPool),
            "Relu" => Ok(Self::Relu),
            "Reshape" => Ok(Self::Reshape),
            "Softmax" => Ok(Self::Softmax),
            _ => Err(onnx_error!("Invalid operation name."))
        }
    }
}

/// Valore di un attributo, che può essere di un qualunque tipo.
#[derive(Clone,Debug)]
pub enum Attribute {
    Undefined,
    Float(f32),
    Int(isize),
    String(String),
    Tensor(Tensor),
    /// Non gestito
    Graph(()),
    Floats(Vec<f32>),
    Ints(Vec<isize>),
    Strings(Vec<String>),
    Tensors(Vec<Tensor>),
    /// Non gestito
    Graphs(()),
    /// Non gestito
    SparseTensor(()),
    /// Non gestito
    SparseTensors(()),
    /// Non gestito
    TypeProto(()),
    /// Non gestito
    TypeProtos(())
}

/// Operazione con stato e attributi.
pub struct Operation {
    /// Tipo dell'operazione.
    op_type: OpType,

    /// [`HashMap`] che mappa il nome dell'attributo con il valore corrispondente.
    attributes: HashMap<String, Attribute>,

    /// Eventuali dimensioni attese del risultato dell'operazione
    /// 
    /// Se [`None`], verranno accettati risultati con qualsiasi dimensione.
    expected_result_shape: Option<Box<[usize]>>,
}

pub(crate) type OperationResult = Result<Arc<Tensor>, OnnxError>;

impl Operation {

    /// Crea una nuova operazione da avviare.
    pub fn new(op_type: OpType) -> Self {
        Self::with_attributes(op_type, HashMap::new())
    }

    /// Crea una nuova operazione da avviare con attributi.
    pub fn with_attributes(op_type: OpType, attributes: HashMap<String, Attribute>) -> Self {
        Self {
            op_type,
            attributes,
            expected_result_shape: None
        }
    }

    /// Esegue l'operazione, dato un vettore di input.
    /// 
    /// Nel caso di più input, l'ordine è lo stesso specificato nella [documentazione delle operazioni ONNX][<https://onnx.ai/onnx/operators/>].
    pub(crate) fn execute(&self, inputs: Vec<Arc<Tensor>>) -> OperationResult {
        let inputs_pointers: Vec<&Tensor> = inputs.iter().map(|f| f.as_ref()).collect();
        
        let result = match self.op_type {
            OpType::Add => self.execute_add(inputs_pointers),
            OpType::AveragePool => self.execute_avg_pool(inputs_pointers),
            OpType::Concat => self.execute_concat(inputs_pointers),
            OpType::Constant => self.execute_constant(inputs_pointers),
            OpType::Conv => self.execute_conv(inputs_pointers),
            OpType::Div => self.execute_div(inputs_pointers),
            OpType::Dropout => self.execute_dropout(inputs),
            OpType::Gemm => self.execute_gemm(inputs_pointers),
            OpType::LRN => self.execute_lrn(inputs_pointers),
            OpType::MatMul => self.execute_matmul(inputs_pointers),
            OpType::MaxPool => self.execute_max_pool(inputs_pointers),
            OpType::Relu => self.execute_relu(inputs_pointers),
            OpType::Reshape => self.execute_reshape(inputs_pointers),
            OpType::Softmax => self.execute_softmax(inputs_pointers)
        };

        // Controllo forma del risultato
        if let (Ok(res_val), Some(expected_shape)) = (&result, &self.expected_result_shape) {
            let res_shape = res_val.shape();
            if res_shape != expected_shape.as_ref() {
                return Err(onnx_error!("The result's shape ({res_shape:?}) is different than expected ({expected_shape:?})."))
            }
        }

        result
    }

    /// Calcola le dimensioni del padding necessario. Questo dipende dal: 
    /// * Numero di feature maps (`fmaps`)
    /// * Dimensioni dei dati (`data_w`, `data_w`)
    /// * Dimensioni del kernel (`kernel_h`, `kernel_w`)
    /// * Strides (`strides_h`, `strides_w`)
    /// 
    /// Il padding può essere manuale o automatico, in base agli attributi dell'operazione.
    fn get_padding(
        &self,
        (data_h, data_w): (usize, usize),
        (kernel_h, kernel_w): (usize, usize),
        (strides_h, strides_w): (usize, usize),
    ) -> Result<[usize; 4], OnnxError> {
        match self.attributes.get("auto_pad").or(Some(&Attribute::String("NOTSET".to_string()))) {
            Some(Attribute::String(auto_pad)) => match auto_pad.as_str() {
                // Padding manuale
                "NOTSET" => match self.attributes.get("pads") {
                    Some(Attribute::Ints(val)) => Ok(
                        // Converti Vec<&isize> a [usize; 4]
                        val.into_iter()
                           .map(|v| usize::try_from(*v))
                           .collect::<Result<Vec<_>, _>>()
                           .map_err(|_| onnx_error!("pads attribute contains a negative number."))?
                           .as_slice()
                           .try_into().map_err(|_| onnx_error!("Padding should contain four values, {} found.", val.len()))?
                    ),
                    None => Ok([0,0,0,0]),
                    _ => return Err(onnx_error!("pads attribute has an invalid value type"))
                },

                // Dimensioni valide, senza padding.
                "VALID" => Ok([0,0,0,0]),

                // Padding automatico: le dimensioni del padding sono tali che l'output abbia dimensioni uguali all'input senza padding.
                // Il padding viene distribuito a metà in entrambe le dimensioni. Se le dimensioni sono dispari, il padding aggiuntivo
                // viene messo all'inizio (UPPER) o alla fine (LOWER) in base all'opzione
                s @ ("SAME_UPPER" | "SAME_LOWER") => {
                    let (v_padding, h_padding) = (data_h*(strides_h-1)+kernel_h-strides_h, data_w*(strides_w-1)+kernel_w-strides_w);
                    Ok([
                        if v_padding % 2 == 0 || s.ends_with("UPPER") { v_padding/2 } else { v_padding/2+1 },
                        if h_padding % 2 == 0 || s.ends_with("UPPER") { h_padding/2 } else { h_padding/2+1 },
                        if v_padding % 2 == 0 || s.ends_with("LOWER") { v_padding/2 } else { v_padding/2+1 },
                        if h_padding % 2 == 0 || s.ends_with("LOWER") { h_padding/2 } else { h_padding/2+1 }
                    ])
                },
                _ => return Err(onnx_error!("auto_pad has an invalid value."))
            }
            _ => return Err(onnx_error!("auto_pad attribute has an invalid value type"))
        }
    }

    /// Divide un [`Tensor`] di dimensioni generiche in finestre di dimensioni `window_dim` e applica la funzione `f` ad ognuna.
    /// 
    /// Il risultato di ogni chiamata di `f` viene salvato in un nuovo array multidimensionale generico, avente forma che
    /// dipende da `data`, `window_dim` e dalle `strides` (finestre saltate sia in lunghezza che in altezza).
    /// 
    /// # Errore
    /// Se le dimensioni della finestra non sono compatibili con il numero di risultati calcolati.
    fn map_windows<I, O>(
        data: ArrayViewD<I>,
        window_dim: &[usize],
        f: impl FnMut(ArrayViewD<I>) -> O,
        strides: &[usize]
    ) -> Result<ArrayD<O>, OnnxError> {
        // Numero totale delle finestre, per ogni dimensione
        let window_amounts =
            Zip::from(data.shape()).and(window_dim)
                .map_collect(|data_len, window_len| data_len - window_len + 1)
                .to_vec();

        // Dimensioni dell'output
        let out_shape = 
            Zip::from(data.shape()).and(window_dim).and(strides)
                .map_collect(|data_len, window_len, strides_len| (data_len-window_len)/strides_len+1)
                .to_vec();

        data
            .windows(window_dim)
            .into_iter()
            .enumerate()
            .filter_map(|(mut i, window)| {
                // Determina posizione corrente
                let mut position = Vec::with_capacity(window_amounts.len());
                for w_i in 0..window_amounts.len() {
                    let product: usize = window_amounts.iter().skip(w_i+1).map(|v| *v).product();
                    position.push(i / product);
                    i %= product;
                }

                // Determina se la posizione corrente è da saltare in base alle strides.
                if Zip::from(&position).and(strides).all(|&pos, &stride| pos % stride == 0) {
                    Some(window)
                } else {
                    None
                }
            })
            .map(f)
            .collect::<Array1<O>>()
            .into_shape(out_shape.clone())
            .map_err(|_| onnx_error!("Could not insert mapped values into {} matrix.", out_shape.into_iter().map(|v| v.to_string()).collect::<Vec<_>>().join("x")))
    }

}