use std::{
    collections::HashMap,
    fs::File,
    io::{BufReader, Read},
};

use ndarray::Array;

use crate::fileparser::protobufoperations::*;
use crate::fileparser::protobufstruct::*;
use crate::graph::*;
use crate::operations::*;
//use pyo3::prelude::*;


pub struct OnnxFileParser;

fn create_multidim_array(
    plain_vect: Vec<f32>,
    dims: &Vec<usize>,
) -> Option<Tensor> {
    Array::from_shape_vec(dims.clone(), plain_vect).ok()
}

impl OnnxFileParser {
    pub fn parse_model(path: &str) -> Result<OnnxGraph, String> {
        let mut building_graph = OnnxGraph::new(); //Empty graph
        let mut range = 0..; //used for generating unique node name
        let file = File::open(path);
        if file.is_err() {
            return Result::Err("Model file not found".to_string());
        }
        let mut br = BufReader::new(file.unwrap());
        let mut binary = Vec::new();

        if br.read_to_end(&mut binary).is_err() {
            //read all the .onnx file
            return Result::Err("Error during reading .onnx file".to_string());
        };

        let mut model: ProtoBufMessage = ProtoBufMessage::ModelProto(ModelProto::new()); //Every .onnx start with a model tag
        log::info!("Starting of parsing *.onnx file...");
        let res = proto_buffer_tag_reader(&mut model, &binary);
       
        if res.is_some() {
            return Result::Err("Error while parsing model - ".to_owned() + &res.unwrap());
        } 
     
        log::info!("End of parsing  *.onnx file");
        log::info!("Starting generation of graph...");
      
        let mut graph = ModelProto::try_from(model).unwrap().graph;
        building_graph.name=graph.name;
        //operation node
        for e in &mut graph.node {
            if e.inputs.len() != 0 && e.outputs.len() != 0 {
                /*Dropout ha 2 output, ma solo il primo è quello vero, il secondo è "mask" che viene ignorato */
                if e.op_type == "Dropout" {
                    e.outputs = vec![e.outputs[0].clone()];
                }
                /*
                   Alcuni modelli non prevedono un nome proprio per i nodi operazione.
                   ma il nome serve per individuare correttamente i nodi.
                   GENERO un nome fittizio UNIVOCO composto da [NOME ORIGINALE]__[TIPO_OP]__[INDEX], dove index è un indice incrementale
                */
                e.name = vec![
                    e.name.clone(),
                    e.op_type.clone(),
                    (range.next()).unwrap().to_string(),
                ]
                .join("__");
                log::info!("[Generation] Node Op.->[{:}]", &(e.name));
                log::debug!("[Generation] Node Op.->{:}   Attributes->-> {:?}", &(e.name), e.attributes);
                log::debug!("[Generation] Node Op.->{:}  Input-> {:?}", &(e.name), e.inputs);
                log::debug!("[Generation] Node Op.->{:}  Ouput-> {:?}", &(e.name), e.outputs);

                let node_op = OnnxGraphNode::Operation(OnnxGraphOperation::new(
                    &e.name,
                    Operation::with_attributes(
                        OpType::try_from(e.op_type.as_str()).unwrap(),
                        e.attributes.clone(),
                    ),
                    &e.inputs,
                    &e.outputs,
                ));
                let res = building_graph.add_node(node_op);
                if res.is_err() {
                    return Result::Err(
                        "Error while adding operation node - ".to_string()
                            + &res.err().unwrap().msg,
                    );
                }
            }
        }
        //initializer node
        let mut err: Option<String> = None;
        graph.tensor_initializer.iter_mut().for_each(|x| {
            log::info!("[Generation] Node Initial.->{:} ", x.name);
            log::debug!("[Generation] Node Initial.->{:}  Dims-> {:?}  ", x.name, x.dims);

            /*  Raw Data contiene dati che devono essere convertiti correttamente in base al data_type
               Per comodità poichè tutte le operazioni sono implementate per lavorare con f32 si fa il seguente passaggio
               raw -> data_type -> f32
            */
            if x.raw_data.len() > 0 {
                x.float_data = converter_raw(&x.raw_data, x.data_type);
                log::debug!(
                    "[Generation] Node Initial.->{:} Vect of {:?} elem.",
                    x.name,
                    x.float_data.len()
                );
                x.raw_data = Vec::new();
            }

            let val = create_multidim_array(x.float_data.clone(), &x.dims).unwrap();
            let node_init = OnnxGraphNode::Initializer(OnnxGraphInitializer::new(&x.name, val));
            let res = building_graph.add_node(node_init);
            if res.is_err() {
                err = Some(
                    "Error while adding initialer node - ".to_string() + &res.err().unwrap().msg,
                );
            }
        });
        if err.is_some() {
            return Err(err.unwrap());
        }

        for e in &graph.inputs_node {
            if graph
                .tensor_initializer
                .iter()
                .find(|x| x.name == e.name)
                .is_none()
            {
                log::info!("[Generation] Node Input [Of the Graph]->{}", e.name);
                let shape: Vec<usize> = e.tp.t.ts.dim.iter().map(|x| x.value as usize).collect();
                let node_in =
                    OnnxGraphNode::Input(OnnxGraphInput::with_expected_shape(&e.name, &shape));
                let res = building_graph.add_node(node_in);
                if res.is_err() {
                    return Result::Err(
                        "Error while adding input node - ".to_string() + &res.err().unwrap().msg,
                    );
                }
            }
        }
        for e in &graph.outputs_node {
            log::info!("[Generation] Node Output [Of the Graph]->{}", e.name);
            let shape: Vec<usize> = e.tp.t.ts.dim.iter().map(|x| x.value as usize).collect();
            //print!("{:?}",shape);
            let node_in =
                OnnxGraphNode::Output(OnnxGraphOutput::with_expected_shape(&e.name, &shape));
            if building_graph.add_node(node_in).is_err() {
                return Result::Err("Error while adding input node".to_string());
            }
        }

        //node intermediate
        for e in &graph.value_info_node {
            if !graph.tensor_initializer.iter().any(|z| z.name == *e) {
                let mut lista_input = Vec::new();
                graph.node.iter().for_each(|x| {
                    if x.outputs.iter().any(|x| x == e) {
                        lista_input.push(x.name.clone());
                    }
                });
                let mut lista_output = Vec::new();
                graph.node.iter().for_each(|x| {
                    if x.inputs.iter().any(|x| x == e) {
                        lista_output.push(x.name.clone());
                    }
                });
                log::info!("[Generation] Node Interm.->{} ", e);
                log::debug!("[Generation] Node Interm.->[{:}]  Input-> {:?}", e, lista_input);
                log::debug!("[Generation] Node Interm.->[{:}]  Ouput-> {:?}", e, lista_output);
                let node_inter = OnnxGraphNode::Intermediate(OnnxGraphIntermediate::new(
                    e,
                    &lista_input[0],
                    lista_output,
                ));

                let res = building_graph.add_node(node_inter);
                if res.is_err() {
                    return Result::Err(
                        "Error while adding intermediate node - ".to_string()
                            + &res.err().unwrap().msg,
                    );
                }
            }
        }
        log::info!("End of graph generation");
        /*FINE FUNZIONE DI PARSING , assegnare valore corretto*/
        return Ok(building_graph);
    }

    /// Open a file(using its `path`) and extract data, input or output data are stored in the same way
    /// This function work only with model that has just one input/one output
    /// The value returned consist in a HashMap with key = name of node and value = tensor
    pub fn parse_data(
        path: &str,
    ) -> Result<HashMap<String, Tensor>, String>
    {
        log::info!("Begin of reading data...");
        let file = File::open(path);
        let mut br = BufReader::new(file.unwrap());
        let mut binary = Vec::new();
        if br.read_to_end(&mut binary).is_err() {
            //read all the .onnx file
            return Result::Err("Error during reading .onnx file".to_string());
        };
       
        let mut input_data = ProtoBufMessage::TensorProto(TensorProto::new());
        let res = proto_buffer_tag_reader(&mut input_data, &binary);
        if res.is_some() {
            return Result::Err("Error while parsing model - ".to_owned() + &res.unwrap());
        }
        let mut data = TensorProto::try_from(input_data).unwrap();
        if data.raw_data.len() > 0 {
            data.float_data = converter_raw(&data.raw_data, data.data_type);
            log::info!("[Reading]Find a tensor of {:?} elem", data.float_data.len());
            data.raw_data = Vec::new();
        }

        let val = create_multidim_array(data.float_data, &data.dims).unwrap();

        let input_values = HashMap::from([
            ( data.name, val )
        ]);

        log::info!("End of reading data");
        return Ok(input_values);
    }
}
