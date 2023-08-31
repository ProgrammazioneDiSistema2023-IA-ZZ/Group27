use std::collections::HashMap;

use ndarray::array;

use crate::operations::{Tensor, Attribute};

#[derive(Debug)]
pub enum ProtoBufMessage {
    NodeProto(NodeProto),
    GraphProto(GraphProto),
    TensorProto(TensorProto),
    ModelProto(ModelProto),
    AttributeProto(AttributeProto),

    Dimension(Dimension),
    ValueInfoProto(ValueInfoProto),
    TensorShapeProto(TensorShapeProto),
    Tensor2(Tensor2),    //è un tensore ma poichè Tensor è gia definito devo utilizzare un altro nome
    TypeProto(TypeProto),
    
}
#[derive(Debug)]
pub struct ValueInfoProto{
    pub field_number : HashMap<usize,String>,
    pub tp : TypeProto,
    pub name:String,

}

#[derive(Debug)]
pub struct Tensor2{
    pub field_number : HashMap<usize,String>,
    pub ts: TensorShapeProto,
}

#[derive(Debug)]
pub  struct TensorShapeProto{
    pub field_number : HashMap<usize,String>,
    pub dim : Vec<Dimension>
}

#[derive(Debug)]
pub struct Dimension{
   pub field_number : HashMap<usize,String>,
   pub value:usize,
}

#[derive(Debug)]
pub struct TypeProto{
    pub  field_number : HashMap<usize,String>,
    pub t : Tensor2,
}


#[derive(Debug)]
pub struct NodeProto {
   pub field_number: HashMap<usize, String>,
   pub attr: AttributeProto,
   pub name: String,
   pub inputs : Vec<String>,
   pub outputs: Vec<String>,
   pub op_type: String,
   pub attributes: HashMap<String, Attribute>,
}
#[derive(Debug)]
pub struct GraphProto {
    pub field_number: HashMap<usize, String>,
    pub node: Vec<NodeProto>,
    pub inputs_node: Vec<ValueInfoProto>,
    pub outputs_node: Vec<ValueInfoProto>,
    pub tensor_initializer : Vec<TensorProto>,
    pub value_info_node: Vec<String>,
    pub name:String,

}

#[derive(Debug)]
pub struct TensorProto {
    pub field_number: HashMap<usize, String>,
    pub tensor : Tensor,
    pub dims: Vec<usize>,
    pub float_data : Vec<f32> ,
    pub raw_data : Vec<u8>,
    pub data_type : usize,
    pub name: String
}
#[derive(Debug)]
pub struct ModelProto {
    pub field_number: HashMap<usize, String>,
    pub graph : GraphProto,
}
#[derive(Debug)]
pub struct AttributeProto {
    pub field_number: HashMap<usize, String>,
    pub  tp : TensorProto, // per i nodi di input/ouput
    pub name :String,
    pub attr : Attribute
}

impl AttributeProto {

    pub fn new() -> Self {
        AttributeProto {
            name:String::new(),
            attr:Attribute::Undefined,
            tp:TensorProto::new(),
            field_number: HashMap::from([
                (0, "undefined".to_string()),
                (7, "floats".to_string()),
                (8, "ints".to_string()),
                (9, "strings".to_string()),
                (10, "tensors".to_string()),
                (2, "f".to_string()),
                (3, "i".to_string()),
                (4, "s".to_string()),
                (5, "t".to_string()),
                (1, "name".to_string()),
                (20, "type".to_string())
            ])
        }
    }
}
impl TensorProto {
    pub fn new() -> Self {
        TensorProto {
            data_type: 0,
            name:String::new(),
            tensor:array![[]].into_dyn(),
            dims:Vec::new(),
            float_data:Vec::new(),
            raw_data:Vec::new(),
            field_number: HashMap::from([
                (1, "dims".to_string()),
                (4, "float_data".to_string()),
                (2, "data_type".to_string()),
                (9, "raw_data".to_string()),
                (7, "int64_data".to_string()),
                (8, "name".to_string())
            ])
        }
    }
}
impl ModelProto {
    pub fn new() -> Self {
        ModelProto {
            graph:GraphProto::new(),
            field_number: HashMap::from([
                (2, "producer_name".to_string()),
                (5, "model_version".to_string()),
                (1, "ir_version".to_string()),
                (3, "producer_version".to_string()),
                (7, "graph".to_string()),
                (14, "metadata_props".to_string()),
                (20, "training_info".to_string()),
                (25, "functions".to_string()),
                (8, "opset_import".to_string()),
                (4, "domain".to_string())
            ])
        }
    }
}
impl GraphProto {
    pub fn new() -> Self {
        GraphProto {
            value_info_node: Vec::new(),
            tensor_initializer: Vec::new(),
            node:Vec::new(),
            inputs_node:Vec::new(),
            outputs_node:Vec::new(),
            name:String::new(),
            field_number: HashMap::from([
                (1, "node".to_string()),
                (2, "name".to_string()),
                (11, "input".to_string()),
                (12, "output".to_string()),
                (5, "initializer".to_string()),
                (15, "sparse_initializer".to_string()),
                (10, "doc_string".to_string()),
                (13, "value_info".to_string())
            ])
        }
    }
}
impl NodeProto {
    pub fn new() -> Self {
        NodeProto { 
            attr:AttributeProto::new(),
            name : "".to_string(),
            inputs:vec![],
            outputs:vec![],
            op_type:String::new(),
            attributes: HashMap::new(),
            field_number: HashMap::from([
                (1, "input".to_string()),
                (2, "output".to_string()),
                (3, "name".to_string()),
                (4, "op_type".to_string()),
                (5, "attribute".to_string()),
                (6, "doc_string".to_string()),
                (7, "domain".to_string())
            ])
        }
    }
}
impl Dimension {
    pub fn new() -> Self {
        Dimension {
            value:0,
            field_number: HashMap::from([
                (1, "dim_value".to_string())
            ])
        }
    }
}
impl Tensor2 {
    pub fn new() -> Self {
        Tensor2 {
            ts:TensorShapeProto::new(),
            field_number: HashMap::from([
                (1, "elem_type".to_string()),
                (2, "shape".to_string())
            ])
        }
    }
}
impl TypeProto {
    pub fn new() -> Self {
        TypeProto {
            t: Tensor2::new(),
            field_number: HashMap::from([
                (1, "tensor_type".to_string())
            ])
        }
    }
}
impl ValueInfoProto {
    pub fn new() -> Self {
        ValueInfoProto {
            name:String::new(),
            tp:TypeProto::new(),
            field_number: HashMap::from([
                (1, "name".to_string()),
                (2, "type".to_string()),
                (9, "elem_type".to_string())
            ]),
        }
    }
}
impl TensorShapeProto {
    pub fn new() -> Self {
        TensorShapeProto{
            dim: Vec::new(),
            field_number: HashMap::from([
                (1, "dimension".to_string())
            ])
        }
    }
}

impl TryFrom <ProtoBufMessage> for TensorProto{
    type Error=&'static str;

    fn try_from(value: ProtoBufMessage) -> Result<Self, Self::Error> {

        if let ProtoBufMessage::TensorProto(v)=value{
            return Ok(v)
        }else{
            return Err("Conversion Error")
        }
    }
}
impl TryFrom <ProtoBufMessage> for AttributeProto{
    type Error=&'static str;

    fn try_from(value: ProtoBufMessage) -> Result<Self, Self::Error> {

        if let ProtoBufMessage::AttributeProto(v)=value{
            return Ok(v)
        }else{
            return Err("Conversion Error")
        }
    }
}
impl TryFrom <ProtoBufMessage> for NodeProto{
    type Error=&'static str;

    fn try_from(value: ProtoBufMessage) -> Result<Self, Self::Error> {

        if let ProtoBufMessage::NodeProto(v)=value{
            return Ok(v)
        }else{
            return Err("Conversion Error")
        }
    }
}
impl TryFrom <ProtoBufMessage> for GraphProto{
    type Error=&'static str;

    fn try_from(value: ProtoBufMessage) -> Result<Self, Self::Error> {

        if let ProtoBufMessage::GraphProto(v)=value{
            return Ok(v)
        }else{
            return Err("Conversion Error")
        }
    }
}
impl TryFrom <ProtoBufMessage> for ModelProto{
    type Error=&'static str;

    fn try_from(value: ProtoBufMessage) -> Result<Self, Self::Error> {

        if let ProtoBufMessage::ModelProto(v)=value{
            return Ok(v)
        }else{
            return Err("Conversion Error")
        }
    }
}
impl TryFrom <ProtoBufMessage> for ValueInfoProto{
    type Error=&'static str;

    fn try_from(value: ProtoBufMessage) -> Result<Self, Self::Error> {

        if let ProtoBufMessage::ValueInfoProto(v)=value{
            return Ok(v)
        }else{
            return Err("Conversion Error")
        }
    }
}
impl TryFrom <ProtoBufMessage> for TypeProto{
    type Error=&'static str;

    fn try_from(value: ProtoBufMessage) -> Result<Self, Self::Error> {

        if let ProtoBufMessage::TypeProto(v)=value{
            return Ok(v)
        }else{
            return Err("Conversion Error")
        }
    }
}
impl TryFrom <ProtoBufMessage> for Tensor2{
    type Error=&'static str;

    fn try_from(value: ProtoBufMessage) -> Result<Self, Self::Error> {

        if let ProtoBufMessage::Tensor2(v)=value{
            return Ok(v)
        }else{
            return Err("Conversion Error")
        }
    }
}
impl TryFrom <ProtoBufMessage> for TensorShapeProto{
    type Error=&'static str;

    fn try_from(value: ProtoBufMessage) -> Result<Self, Self::Error> {

        if let ProtoBufMessage::TensorShapeProto(v)=value{
            return Ok(v)
        }else{
            return Err("Conversion Error")
        }
    }
}
impl TryFrom <ProtoBufMessage> for Dimension{
    type Error=&'static str;

    fn try_from(value: ProtoBufMessage) -> Result<Self, Self::Error> {

        if let ProtoBufMessage::Dimension(v)=value{
            return Ok(v)
        }else{
            return Err("Conversion Error")
        }
    }
}