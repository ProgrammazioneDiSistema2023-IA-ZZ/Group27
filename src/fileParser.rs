use std::{
    array,
    fs::File,
    io::{BufReader, Read}, collections::HashMap, sync::Arc,
};

use crate::{
    graph::{OnnxGraph, OnnxGraphInitializer, OnnxGraphNode, OnnxGraphInput, OnnxGraphOperation, OnnxGraphOutput},
    protobufoperations::{leggifloats, readVarint, leggibytes},
    protobufstruct::{
        AttributeProto, Dimension, GraphProto, ModelProto, NodeProto, ProtoBufMessage, Tensor2,
        TensorProto, TensorShapeProto, TypeProto, ValueInfoProto,
    }, operations::{Operation, OpType},
};
use ndarray::{arr0, arr1, arr2, array, Array, Array2, ArrayBase, Axis, OwnedRepr};

#[derive(Debug)]
pub struct OnnxParseFileError {
    msg: String,
}
pub type OnnxParseFileResult<'a> = Result<OnnxGraph, OnnxParseFileError>;

impl OnnxParseFileError {
    pub fn new(msg: String) -> Self {
        Self { msg }
    }
}
pub struct OnnxFileParser<'a> {
    path: &'a str,
    pub error: Option<String>,
    pub graph: OnnxGraph,
}

impl<'a> OnnxFileParser<'a> {
    pub fn new(path: &'a str) -> Self {
        OnnxFileParser {
            path: path,
            error: Option::Some("Not yet parsed".to_string()),
            graph: OnnxGraph::new(),
        }
    }

    pub fn parse(&mut self) {

        let mut GRAFO = OnnxGraph::new();
        let file = File::open(self.path);
        if file.is_err() {
            self.error = Option::Some("File not found".to_string());
            return;
        }
        let mut br = BufReader::new(file.unwrap());
        let mut binary = Vec::new();
        let mut index = 0;
        if br.read_to_end(&mut binary).is_err() {
            self.error = Option::Some("Error during reading file".to_string());
            return;
        };

        let mut pbm = ProtoBufMessage::ModelProto(ModelProto::new());

        while index < binary.len() {
            let val = readVarint(&binary, &mut index);
            let wireType = val & 0x3;
            let fieldNumber = val >> 3;
            match wireType {
                0 => {
                    self.wireType_zero(&mut pbm, &fieldNumber, &binary, &mut index);
                }
                2 => {
                    self.wireType_two(&mut pbm, &fieldNumber, &binary, &mut index);
                }
                _ => {
                    self.error = Option::Some("Unsupported data type".to_string());
                }
            }
        }
        let graph = ModelProto::try_from(pbm).unwrap().graph;

        for e in &graph.node {
            //RICERCA NODI INITIALIZER
            if e.inputs.len() == 0 {
                let dims = e.attr.tp.dims.len();
                let float_data = e.attr.tp.float_data.clone();
                println!("INIT->{},{:?},{}", e.name, e.attr.tp.dims, float_data.len());
                let mut val;
                match dims {
                    0 => {
                        val =
                            Array::from_iter(float_data.iter().map(|v| *v as f32).cycle().take(0))
                                .into_shape((0))
                                .unwrap()
                                .into_dyn();
                    }
                    1 => {
                        val = Array::from_iter(
                            float_data.iter().map(|v| *v as f32).take(e.attr.tp.dims[0]),
                        )
                        .into_shape((e.attr.tp.dims[0]))
                        .unwrap()
                        .into_dyn();
                    }
                    2 => {
                        val = Array::from_iter(
                            float_data
                                .iter()
                                .map(|v| *v as f32)
                                .take(e.attr.tp.dims[0] * e.attr.tp.dims[1]),
                        )
                        .into_shape((e.attr.tp.dims[0], e.attr.tp.dims[1]))
                        .unwrap()
                        .into_dyn();
                    }
                    3 => {
                        val = Array::from_iter(
                            float_data
                                .iter()
                                .map(|v| *v as f32)
                                .take(e.attr.tp.dims[0] * e.attr.tp.dims[1] * e.attr.tp.dims[2]),
                        )
                        .into_shape((e.attr.tp.dims[0], e.attr.tp.dims[1], e.attr.tp.dims[2]))
                        .unwrap()
                        .into_dyn();
                    }
                    4 => {
                        val = Array::from_iter(float_data.iter().map(|v| *v as f32).take(
                            e.attr.tp.dims[0]
                                * e.attr.tp.dims[1]
                                * e.attr.tp.dims[2]
                                * e.attr.tp.dims[3],
                        ))
                        .into_shape((
                            e.attr.tp.dims[0],
                            e.attr.tp.dims[1],
                            e.attr.tp.dims[2],
                            e.attr.tp.dims[3],
                        ))
                        .unwrap()
                        .into_dyn();
                    }
                    _ => {
                        self.error = Some("Error: Tensor with an unsupported dims".to_string());
                        return;
                    }
                }

                let node_init =
                    OnnxGraphNode::Initializer(OnnxGraphInitializer::new(&e.name, val.into_dyn()));
                if GRAFO.add_node(node_init).is_err() {
                    self.error = Some("Error while adding init node".to_string());
                }
            }
            if e.inputs.len() != 0 && e.outputs.len() != 0 {
                //NODO OPERATION

                  let node_op = OnnxGraphNode::Operation(OnnxGraphOperation::new(
                    &e.name,
                    Operation::new(OpType::try_from(e.op_type.as_str()).unwrap()),
                    e.inputs.iter().map(|s| &s[..]).collect(),
                    e.outputs.iter().map(|s| &s[..]).collect()));
                     if self.graph.add_node(node_op).is_err(){
                        self.error=Some("Error while adding init node".to_string());
                }
               
            }
        }
        for e in &graph.inputs_node{
            let shape :Vec<usize>= e.tp.t.ts.dim.iter().map(|x| x.value as usize).collect();
            print!("{:?}",shape);
            let node_in =OnnxGraphNode::Input(OnnxGraphInput::with_expected_shape(&e.name,&shape));
            if GRAFO.add_node(node_in).is_err() {
                self.error = Some("Error while adding input node".to_string());
            }
        }
        for e in &graph.outputs_node{
            let shape :Vec<usize>= e.tp.t.ts.dim.iter().map(|x| x.value as usize).collect();
            print!("{:?}",shape);
            let node_in =OnnxGraphNode::Output(OnnxGraphOutput::with_expected_shape(&e.name,&shape));
            if GRAFO.add_node(node_in).is_err() {
                self.error = Some("Error while adding input node".to_string());
            }
        }
        
        /*Lettura input file*/

        let file = File::open("./onnxFile/input_0.pb");
        if file.is_err() {
            self.error = Option::Some("File not found".to_string());
            return;
        }
        
        let mut br = BufReader::new(file.unwrap());
        let mut binary = Vec::new();
        br.read_to_end(&mut binary);
         let mut input_data = ProtoBufMessage::TensorProto(TensorProto::new());
            index=0;
            println!("{} - {}",index,binary.len());
         while index < binary.len() {
             let val = readVarint(&binary, &mut index);
             let wireType = val & 0x3;
             let fieldNumber = val >> 3;
           
             match wireType {
                 0 => {
                     self.wireType_zero(&mut input_data, &fieldNumber, &binary, &mut index);
                 }
                 2 => {
                     self.wireType_two(&mut input_data, &fieldNumber, &binary, &mut index);
                 }
                 _ => {
                     self.error = Option::Some("Unsupported data type".to_string());
                 }
             }
         }
         let data = TensorProto::try_from(input_data).unwrap();

         println!("{:?}",data.dims);
        let  val = Array::from_iter(data.raw_data.iter().map(|v| *v as f32).take(
            data.dims[0]
                * data.dims[1]
                * data.dims[2]
                * data.dims[3],
        ))
        .into_shape((
            data.dims[0],
            data.dims[1],
            data.dims[2],
            data.dims[3],
        ))
        .unwrap()
        .into_dyn();
    //Input73
    let mut input_values = HashMap::new();
    input_values.insert("Input73".to_string(), val);


    let result = Arc::new(GRAFO).infer(input_values);
        println!("\n\n\n{:?}",result);
   // assert!(result.is_ok(), "{:?}", result.unwrap_err());
   // assert_eq!(result.unwrap(), output_values);

        return;
    }

    fn wireType_two(
        &mut self,
        pbm: &mut ProtoBufMessage,
        field_number: &usize,
        vettore: &Vec<u8>,
        index: &mut usize,
    ) {
        match (pbm) {
            ProtoBufMessage::ModelProto(p) => {
                print!("\"{}\":", *p.fieldNumber.get(field_number).unwrap());
                let val = readVarint(&vettore, index);
                if *field_number == 7 {
                    println!("GRAFOOOOOO!");
                    let mut graph = ProtoBufMessage::GraphProto(GraphProto::new());
                    self.tagReader(&(vettore[*index..(*index + val)]).to_vec(), &mut graph);
                    (*p).graph = GraphProto::try_from(graph).unwrap();
                    println!("FINE GRAFO");
                } else if *field_number != 8 {
                    let word =
                        String::from_utf8(vettore[*index..(*index + val)].to_owned()).unwrap();
                    println!("{:?}", word);
                } //index..(index + val)
                (*index) += val;
            }
            ProtoBufMessage::GraphProto(p) => {
                print!("\"{:}\":", *p.fieldNumber.get(field_number).unwrap());
                let val = readVarint(&vettore, index) as usize;
                match *field_number {
                    1 => {
                        println!("");
                        let mut node: ProtoBufMessage =
                            ProtoBufMessage::NodeProto(NodeProto::new());
                        self.tagReader(&(vettore[*index..(*index + val)]).to_vec(), &mut node);

                        //(*p).attr=AttributeProto::try_from(at).unwrap();

                        (*p).node.push(NodeProto::try_from(node).unwrap());
                    }
                    11 => {
                        println!("");
                        println!("INPUT!");
                        let mut node: ProtoBufMessage =
                            ProtoBufMessage::ValueInfoProto(ValueInfoProto::new());
                        self.tagReader(&(vettore[*index..(*index + val)]).to_vec(), &mut node);
                        (*p).inputs_node
                            .push(ValueInfoProto::try_from(node).unwrap());
                        println!("FINE INPUT!");
                    }
                    12 => {
                        println!("");
                        println!("OUTPUT!");
                        let mut node: ProtoBufMessage =
                            ProtoBufMessage::ValueInfoProto(ValueInfoProto::new());
                        self.tagReader(&(vettore[*index..(*index + val)]).to_vec(), &mut node);
                        (*p).outputs_node
                            .push(ValueInfoProto::try_from(node).unwrap());
                        println!("FINE OUTPUT!");
                    }
                    13 => {/*TAG VALUE INFO - NON NECESSARIO */}
                    _ => {
                        let word =
                            String::from_utf8(vettore[*index..(*index + val)].to_owned()).unwrap();
                        println!("{:?}", word);
                        if *field_number == 2 {
                            self.graph.name = word;
                        }
                    }
                }
                (*index) += val;
            }
            ProtoBufMessage::NodeProto(p) => {
                print!("\t\t\"{}\":", *p.fieldNumber.get(field_number).unwrap());
                let val = readVarint(&vettore, index) as usize;
                // print!("{}] : ", val);

                if *field_number == 5 {
                    println!("");
                    //attributo
                    let mut at = ProtoBufMessage::AttributeProto(AttributeProto::new());
                    self.tagReader(&vettore[*index..(*index + val)].to_vec(), &mut at);
                    (*p).attr = AttributeProto::try_from(at).unwrap();
                } else {
                    let word =
                        String::from_utf8(vettore[*index..(*index + val)].to_owned()).unwrap();
                    println!("{:?}", word);
                    if *field_number == 3 {
                        (*p).name = word;
                    } else if *field_number == 1 {
                        (*p).inputs.push(word);
                    } else if *field_number == 2 {
                        (*p).outputs.push(word);
                    } else if *field_number == 4 {
                        (*p).op_type = word;
                    }
                }

                (*index) += val;
            }
            ProtoBufMessage::AttributeProto(p) => {
                print!("\t\t\t\"{}\":", *p.fieldNumber.get(field_number).unwrap());
                let val = readVarint(&vettore, index) as usize;
                // print!("{}] : ", val);
                if *field_number == 5 {
                    println!("");
                    let mut tp = ProtoBufMessage::TensorProto(TensorProto::new());
                    self.tagReader(&vettore[*index..(*index + val)].to_vec(), &mut tp);

                    (*p).tp = TensorProto::try_from(tp).unwrap();
                    print!("Struttura ok [{:?}]", (*p).tp.dims);

                    //HO UN TENSORE COMPLETO
                } else {
                    let word =
                        String::from_utf8(vettore[*index..(*index + val)].to_owned()).unwrap();
                    println!("{:?}", word);
                }
                (*index) += val;
            }
            ProtoBufMessage::TensorProto(p) => {
                print!("\t\t\t\t\"{}\":", *p.fieldNumber.get(field_number).unwrap());
                let val = readVarint(&vettore, index) as usize;
                // print!("{}] : ", val);

                if *field_number == 4 {
                    let v = leggifloats(&vettore[*index..*index + val].to_vec());
                    println!("{:?}", v);
                    (*p).float_data = v;
                }else if *field_number == 9 { //RAW_DATA
                    print!("ffdfjdighdsghdi");
                    let v = leggibytes(&vettore[*index..*index + val].to_vec());
                    println!("{:?}", v);
                    (*p).raw_data = v;
                } else {
                    let word =
                        String::from_utf8(vettore[*index..(*index + val)].to_owned()).unwrap();
                    println!("{:?}", word);
                }
                (*index) += val;
            }
            ProtoBufMessage::ValueInfoProto(p) => {
                print!("\t\t\"{}\":", *p.fieldNumber.get(field_number).unwrap());
                let val = readVarint(&vettore, index);
                if *field_number == 2 {
                    //typeProto
                    println!("");
                    let mut tp = ProtoBufMessage::TypeProto(TypeProto::new());
                    self.tagReader(&vettore[*index..(*index + val)].to_vec(), &mut tp);
                    (*p).tp = TypeProto::try_from(tp).unwrap();
                } else {
                    let word =
                        String::from_utf8(vettore[*index..(*index + val)].to_owned()).unwrap();
                    println!("{:?}", word);
                    if *field_number==1{
                        (*p).name=word;
                    }
                } //index..(index + val)
                
                (*index) += val;
            }
            ProtoBufMessage::TypeProto(p) => {
                print!("\t\t\t\"{}\":", *p.fieldNumber.get(field_number).unwrap());
                let val = readVarint(&vettore, index);
                if *field_number == 1 {
                    //tensor
                    println!("");
                    let mut t = ProtoBufMessage::Tensor2(Tensor2::new());
                    self.tagReader(&vettore[*index..(*index + val)].to_vec(), &mut t);
                    (*p).t = Tensor2::try_from(t).unwrap();
                } else {
                    let word =
                        String::from_utf8(vettore[*index..(*index + val)].to_owned()).unwrap();
                    println!("{:?}", word);
                } //index..(index + val)
                (*index) += val;
            }
            ProtoBufMessage::Tensor2(p) => {
                print!("\t\t\t\t\"{}\":", *p.fieldNumber.get(field_number).unwrap());
                let val = readVarint(&vettore, index);
                if *field_number == 2 {
                    //tensorShape
                    println!("");
                    let mut t = ProtoBufMessage::TensorShapeProto(TensorShapeProto::new());
                    self.tagReader(&vettore[*index..(*index + val)].to_vec(), &mut t);
                    (*p).ts = TensorShapeProto::try_from(t).unwrap();
                } else {
                    let word =
                        String::from_utf8(vettore[*index..(*index + val)].to_owned()).unwrap();
                    println!("{:?}", word);
                } //index..(index + val)
                (*index) += val;
            }
            ProtoBufMessage::TensorShapeProto(p) => {
                print!(
                    "\t\t\t\t\t\"{}\":",
                    *p.fieldNumber.get(field_number).unwrap()
                );
                let val = readVarint(&vettore, index);
                if *field_number == 1 {
                    //dimension
                    println!("");
                    let mut t = ProtoBufMessage::Dimension(Dimension::new());
                    self.tagReader(&vettore[*index..(*index + val)].to_vec(), &mut t);
                    (*p).dim.push(Dimension::try_from(t).unwrap());
                }else{
                    let word =
                    String::from_utf8(vettore[*index..(*index + val)].to_owned()).unwrap();
                    println!("{:?}", word);
                }
                  
                //index..(index + val)
                (*index) += val;
            }
            ProtoBufMessage::Dimension(p) => {
                print!(
                    "\t\t\t\t\t\t\"{}\":",
                    *p.fieldNumber.get(field_number).unwrap()
                );
                let val = readVarint(&vettore, index);

                let word = String::from_utf8(vettore[*index..(*index + val)].to_owned()).unwrap();
                println!("{:?}", word);
                    
                    (*index) += val;
                
            }

            _ => {}
        }
    }

    pub fn wireType_zero(
        &mut self,
        pbm: &mut ProtoBufMessage,
        field_number: &usize,
        vettore: &Vec<u8>,
        index: &mut usize,
    ) {
        let mut fieldName;
        let val = readVarint(&vettore, index);
        match (pbm) {
            ProtoBufMessage::ModelProto(p) => {
                fieldName = (*p).fieldNumber.get(field_number);
                if fieldName.is_some() {
                    print!("\"{}\":S", *p.fieldNumber.get(field_number).unwrap());
                    println!("{}", val);
                } else {
                    println!("\t NON SUPPORTATO val = {}", field_number);
                }
            }
            ProtoBufMessage::GraphProto(p) => {
                fieldName = (*p).fieldNumber.get(field_number);
                if fieldName.is_some() {
                    print!("\t\"{}\":S", *p.fieldNumber.get(field_number).unwrap());
                    println!("{}", val);
                } else {
                    println!("\t NON SUPPORTATO val = {}", field_number);
                }
            }
            ProtoBufMessage::AttributeProto(p) => {
                fieldName = (*p).fieldNumber.get(field_number);
                if fieldName.is_some() {
                    print!(
                        "\t\t\"{}\":V",
                        (*p).fieldNumber.get((field_number)).unwrap()
                    );
                    println!("{}", val);
                } else {
                    print!("\t\t\tNON SUPPORTATO val = {}", field_number);
                }
            }
            ProtoBufMessage::TensorProto(p) => {
                fieldName = (*p).fieldNumber.get(field_number);
                if fieldName.is_some() {
                    print!(
                        "\t\t\t\t\"{}\":V",
                        (*p).fieldNumber.get(field_number).unwrap()
                    );
                    println!("{}", val);
                    if *field_number == 1 {
                        // Campo "dims", è gestito come varint
                        (*p).dims.push(val);
                    }
                    if *field_number == 9 { //RAW_DATA
                        print!("\n\n\n\nffdfjdighdsghdi");
                    }
                } else {
                    print!("\t\t\t\tNON SUPPORTATO val = {}", field_number);
                }
            }
            ProtoBufMessage::NodeProto(p) => {
                fieldName = (*p).fieldNumber.get(field_number);
                if fieldName.is_some() {
                    print!("\t\t\"{}\":S", (*p).fieldNumber.get(field_number).unwrap());
                    println!("{}", val);
                } else {
                    print!("\t\tNON SUPPORTATO val = {}", field_number);
                }
            }
            ProtoBufMessage::ValueInfoProto(p) => {
                fieldName = (*p).fieldNumber.get(field_number);
                if fieldName.is_some() {
                    print!(
                        "\t\t\t\"{}\":V",
                        (*p).fieldNumber.get((field_number)).unwrap()
                    );
                    println!("{}", val);
                } else {
                    print!("\t\t\tNON SUPPORTATO val = {}", field_number);
                }
            }
            ProtoBufMessage::TypeProto(p) => {
                fieldName = (*p).fieldNumber.get(field_number);
                if fieldName.is_some() {
                    print!(
                        "\t\t\t\"{}\":V",
                        (*p).fieldNumber.get((field_number)).unwrap()
                    );
                    println!("{}", val);
                } else {
                    print!("\t\t\tNON SUPPORTATO val = {}", field_number);
                }
            }

            ProtoBufMessage::Tensor2(p) => {
                fieldName = (*p).fieldNumber.get(field_number);
                if fieldName.is_some() {
                    print!(
                        "\t\t\t\t\"{}\":V",
                        (*p).fieldNumber.get((field_number)).unwrap()
                    );
                    println!("{}", val);
                } else {
                    print!("\t\t\tNON SUPPORTATO val = {}", field_number);
                }
            }
            ProtoBufMessage::Dimension(p) => {
                fieldName = p.fieldNumber.get(field_number);
                if fieldName.is_some() {
                    print!(
                        "\t\t\t\t\t\t\"{}\"[V]:",
                        p.fieldNumber.get(field_number).unwrap()
                    );
                    println!("{}", val);
                    if *field_number == 1 {
                        //dim_value
                        p.value = val;
                    } 
                } else {
                    print!("\t\tNON SUPPORTATO val = {}", field_number);
                }
            }
            _ => {}
        }
    }

    fn tagReader(&mut self, binary: &Vec<u8>, pbm: &mut ProtoBufMessage) {
        let mut index = 0;
        while index < binary.len() {
            let val = readVarint(&binary, &mut index);
            let wireType = val & 0x3;
            let fieldNumber = val >> 3;
            match wireType {
                0 => {
                    self.wireType_zero(pbm, &fieldNumber, &binary, &mut index);
                }
                2 => {
                    self.wireType_two(pbm, &fieldNumber, &binary, &mut index);
                }
                _ => {
                    self.error = Option::Some("Unsupported data type".to_string());
                }
            }
        }
        return;
    }
}




