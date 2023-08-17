use std::{
    array,
    collections::HashMap,
    fs::File,
    io::{BufReader, Read},
    sync::Arc,
   
};

use ndarray::{arr0, arr1, arr2, array, Array, Array2, ArrayBase, Axis, OwnedRepr};

use crate::fileParser::protobufoperations::*;
use crate::fileParser::protobufstruct::*;
use crate::graph::*;
use crate::operations::*;

/*
#[derive(Debug)]
pub struct OnnxParseFileError {
    msg: String,
}
pub type OnnxParseFileResult<'a> = Result<OnnxGraph, OnnxParseFileError>;

impl OnnxParseFileError {
    pub fn new(msg: String) -> Self {
        Self { msg }
    }
}*/
pub struct OnnxFileParser {
    pub result: Result<OnnxGraph, String>,
}

fn create_multidim_array(
    plain_vect: Vec<f32>,
    dims: &Vec<usize>,
) -> Option<ArrayBase<OwnedRepr<f32>, ndarray::Dim<ndarray::IxDynImpl>>> {
    
    match dims.len() {
        0 => Some(
            Array::from_iter(plain_vect.iter().map(|v| *v).cycle().take(0))
                .into_shape(0)
                .unwrap()
                .into_dyn(),
        ),
        1 => Some(
            Array::from_iter(plain_vect.iter().map(|v| *v).take(dims[0]))
                .into_shape(dims[0])
                .unwrap()
                .into_dyn(),
        ),
        2 => Some(
            Array::from_iter(plain_vect.iter().map(|v| *v).take(dims[0] * dims[1]))
                .into_shape((dims[0], dims[1]))
                .unwrap()
                .into_dyn(),
        ),
        3 => Some(
            Array::from_iter(
                plain_vect
                    .iter()
                    .map(|v| *v)
                    .take(dims[0] * dims[1] * dims[2]),
            )
            .into_shape((dims[0], dims[1], dims[2]))
            .unwrap()
            .into_dyn(),
        ),
        4 => Some(
            Array::from_iter(
                plain_vect
                    .iter()
                    .map(|v| *v)
                    .take(dims[0] * dims[1] * dims[2] * dims[3]),
            )
            .into_shape((dims[0], dims[1], dims[2], dims[3])).unwrap()
            .into_dyn(),
        ),
        _ => None,
    }
}

impl OnnxFileParser {
    pub fn new() -> Self {
        OnnxFileParser {
            result: Result::Err("Not yet parsed".to_string()),
        }
    }

    pub fn parse(&mut self, path: &str, path2: &str) {
        /*Creazione grafo vuoto */
        let mut building_graph = OnnxGraph::new();
       
        let file = File::open(path);
        if file.is_err() {
            self.result = Result::Err("model file not found".to_string());
            return;
        }
        let mut br = BufReader::new(file.unwrap());
        let mut binary = Vec::new();
        let mut index = 0;
        /*Leggo tutto il file .onnx*/
        if br.read_to_end(&mut binary).is_err() {
            self.result = Result::Err("Error during reading model file".to_string());
            return;
        };

        let mut model: ProtoBufMessage = ProtoBufMessage::ModelProto(ModelProto::new());

        let res = proto_buffer_tag_reader(&mut model, &binary);
        if res.is_some() {
            self.result = Result::Err("Error while parsing model - ".to_owned() + &res.unwrap());
            return;
        }
        println!("\n\n---------------CREAZIONE GRAFO------------");
        let graph = ModelProto::try_from(model).unwrap().graph;

        /*
           MNIST -> il nodo di input è solo 1 ed è taggato input ,nessun nodo taggato initializer. i nodi initializer vengono gestiti come nodi [nodes]
           GOOGLENET-> il nodo di input è 1 [data_0] ma anche i nodi di tipo initializer sono taggati come [input] e come [initializer] ,occorre separare input
                       vero, quindi con solo tag [input] da nodi initializer [input]+[initializer]
        */
        for e in &graph.node {
  
            if e.inputs.len() != 0 && e.outputs.len() != 0 {
                //NODO OPERATION 
                // non è detto che un nodo abbia un nome... googlenet ha i nomi ai nodi operazione -> è un problema di pytorch che quando esporta il modello non mette i nomi 
               /*  if e.name.is_empty(){
                    self.result = Result::Err("Error: node without a name".to_string());
                    return;
                }*/
                println!("+--------");
                println!("|OPERATION->{: <20}  ATTR-> {:?}  ", &(e.op_type),e.attributes);
                println!("|IN-> {:?}  OUT-> {:?}",e.inputs,e.outputs);
                println!("+--------\n");
                let node_op = OnnxGraphNode::Operation(OnnxGraphOperation::new(
                    &e.name,
                    Operation::with_attributes(OpType::try_from(e.op_type.as_str()).unwrap(),e.attributes.clone()),
                    &e.inputs,
                    &e.outputs,
                ));
             

                let res =building_graph.add_node(node_op);
                if res.is_err() {
                    self.result = Result::Err("Error while adding operation node - ".to_string()+ &res.err().unwrap().msg);
                }
                
            }
        }
       
            graph
            .tensor_initializer
            .iter()
            .for_each(|x| { 
                println!("+--------");
                println!("|INITIALIZER->{: <20}  DIMS-> {:?}  ", x.name,x.dims);
                println!("+--------\n");
                let mut val: ArrayBase<OwnedRepr<f32>, ndarray::Dim<ndarray::IxDynImpl>> =
                    create_multidim_array(
                        x.float_data.clone(),
                        &x.dims,
                    )
                    .unwrap();
                
                let node_init = OnnxGraphNode::Initializer(OnnxGraphInitializer::new(
                    &x.name,
                    val,
                ));
                let res =building_graph.add_node(node_init);
                if res.is_err() {
                    self.result = Result::Err("Error while adding initialer node - ".to_string()+ &res.err().unwrap().msg);
                }
            });
        
        let mut name_input_node = String::new();
        for e in &graph.inputs_node {
        
            if graph
                .tensor_initializer
                .iter()
                .find(|x| x.name == e.name)
                .is_none()
            {
                println!("INPUT->{}", e.name);
                name_input_node = e.name.clone();
                let shape: Vec<usize> = e.tp.t.ts.dim.iter().map(|x| x.value as usize).collect();
                let node_in =
                    OnnxGraphNode::Input(OnnxGraphInput::with_expected_shape(&e.name, &shape));
                let res =building_graph.add_node(node_in);
                if res.is_err() {
                    self.result = Result::Err("Error while adding input node - ".to_string()+ &res.err().unwrap().msg);
                }
            }
        }
        for e in &graph.outputs_node {
            println!("OUTPUT->{}", e.name);
            let shape: Vec<usize> = e.tp.t.ts.dim.iter().map(|x| x.value as usize).collect();
            //print!("{:?}",shape);
            let node_in =
                OnnxGraphNode::Output(OnnxGraphOutput::with_expected_shape(&e.name, &shape));
            if building_graph.add_node(node_in).is_err() {
                self.result = Result::Err("Error while adding input node".to_string());
            }
        }

        /*RICERCA NODI INTERMEDIATE */
        for e in &graph.value_info_node{
            if !graph.tensor_initializer.iter().any(|z|z.name==*e){
              
                let mut lista_input = Vec::new();
                graph.node.iter().for_each(|x|{
                    if x.outputs.iter().any(|x| x==e){
                        lista_input.push(x.name.clone());
                    }
                });
                let mut lista_output = Vec::new();
                graph.node.iter().for_each(|x|{
                    if x.inputs.iter().any(|x| x==e){
                        lista_output.push(x.name.clone());
                    }
                });
               
                    println!("INTERMEDIATE->{} IN ->{:?}, OUT->{:?}", e,lista_input,lista_output);
                    let node_inter =
                        OnnxGraphNode::Intermediate(OnnxGraphIntermediate::new(e, &lista_input[0], lista_output));
    
                    let res =building_graph.add_node(node_inter);
                    if res.is_err() {
                        self.result = Result::Err("Error while adding intermediate node - ".to_string()+ &res.err().unwrap().msg);
                    }
                
                
            }
           
        }

        /*FINE FUNZIONE DI PARSING , assegnare valore corretto*/
       // self.result= Result::Ok(building_graph);
          self.result= Result::Err("Parsing OK".to_string());  

        /*Lettura input file*/

        let file = File::open(path2);

        let mut br = BufReader::new(file.unwrap());
        let mut binary = Vec::new();
        br.read_to_end(&mut binary);
        println!("\nCONTENUTO FILE INPUT ->");
        let mut input_data = ProtoBufMessage::TensorProto(TensorProto::new());
        let res = proto_buffer_tag_reader(&mut input_data, &binary);
        if res.is_some() {
            self.result = Result::Err("Error while parsing model - ".to_owned() + &res.unwrap());
            return;
        }
        let data = TensorProto::try_from(input_data).unwrap();

        
        let val = create_multidim_array(
            data.float_data,
            &data.dims,
        )
        .unwrap();

        println!("{:?}",val);

    
        let mut input_values = HashMap::new();
        input_values.insert(name_input_node, val);
        
        let result = Arc::new(building_graph).infer(input_values);
     println!("\n\n\n{:?}", result);

        let r = result.unwrap().get("Plus214_Output_0").unwrap().clone();
        //r.iter().map(|x| x).for_each(|x|)

        return;
    }
   
}
