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
    Tensor2(Tensor2),
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
    pub node:Vec<NodeProto>,
    pub inputs_node: Vec<ValueInfoProto>,
    pub outputs_node: Vec<ValueInfoProto>,
    pub tensor_initializer :Vec<TensorProto>,
    pub value_info_node:Vec<String>,
    pub name:String,

}

#[derive(Debug)]
pub struct TensorProto {
    pub field_number: HashMap<usize, String>,
    pub tensor : Tensor,
    pub dims:Vec<usize>,
    pub float_data :Vec<f32> ,
    pub raw_data :Vec<u8>,
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
    pub  field_number: HashMap<usize, String>,
    pub  tp : TensorProto, // per i nodi di input/ouput
    pub name :String,
    pub attr : Attribute,
    
   
}

impl AttributeProto {
    pub fn new() -> Self {
        let mut result = AttributeProto { field_number: HashMap::new(),
            name :String::new(),
            attr:Attribute::Undefined,
            tp:TensorProto::new()
        };
        result.field_number.insert(0, "undefined".to_string());
        result.field_number.insert(7, "floats".to_string());
        result.field_number.insert(8, "ints".to_string());
        result.field_number.insert(9, "strings".to_string());
        result.field_number.insert(10, "tensors".to_string());
        result.field_number.insert(2, "f".to_string());
        result.field_number.insert(3, "i".to_string());
        result.field_number.insert(4, "s".to_string());
        result.field_number.insert(5, "t".to_string());
      
        result.field_number.insert(1, "name".to_string());
        result.field_number.insert(20, "type".to_string());
   
        result
    }
}
impl TensorProto {
    pub fn new() -> Self {
        let mut result = TensorProto { data_type: 0,name:String::new(),field_number: HashMap::new(),tensor:array![ []  ].into_dyn() ,dims:Vec::new(),float_data:Vec::new(),raw_data:Vec::new()};
        result.field_number.insert(1, "dims".to_string());
        result.field_number.insert(4, "float_data".to_string());
        result.field_number.insert(2, "data_type".to_string());
        result.field_number.insert(9, "raw_data".to_string());
        result.field_number.insert(7, "int64_data".to_string());
        result.field_number.insert(8, "name".to_string());
        result
    }
}
impl ModelProto {
    pub fn new() -> Self {
        let mut result = ModelProto { field_number: HashMap::new(), graph:GraphProto::new() };
        result.field_number.insert(2, "producer_name".to_string());
        result.field_number.insert(5, "model_version".to_string());
        result.field_number.insert(1, "ir_version".to_string());
        result.field_number.insert(3, "producer_version".to_string());
        result.field_number.insert(7, "graph".to_string());
        result.field_number.insert(14, "metadata_props".to_string());
        result.field_number.insert(20, "training_info".to_string());
        result.field_number.insert(25, "functions".to_string());
        result.field_number.insert(8, "opset_import".to_string());
        result.field_number.insert(4, "domain".to_string());
        result
    }
}
impl GraphProto {
    pub fn new() -> Self {
        let mut result = GraphProto { value_info_node:Vec::new(),tensor_initializer: Vec::new(),field_number: HashMap::new(), node:Vec::new(),inputs_node:Vec::new(),outputs_node:Vec::new() ,name:String::new()};
        result.field_number.insert(1, "node".to_string());
        result.field_number.insert(2, "name".to_string());
        result.field_number.insert(11, "input".to_string());
        result.field_number.insert(12, "output".to_string());
        result.field_number.insert(5, "initializer".to_string());
        result.field_number.insert(15, "sparse_initializer".to_string());
        result.field_number.insert(10, "doc_string".to_string());
        result.field_number.insert(13, "value_info".to_string());
        
        result
    }
}
impl NodeProto {
    pub fn new() -> Self {
        let mut result = NodeProto { 
            field_number: HashMap::new(),
            attr:AttributeProto::new(),
            name : "".to_string(),
            inputs:vec![],outputs:vec![],
            op_type:String::new(),
            attributes:HashMap::new(),
        };
        result.field_number.insert(1, "input".to_string());
        result.field_number.insert(2, "output".to_string());
        result.field_number.insert(3, "name".to_string());
        result.field_number.insert(4, "op_type".to_string());
        result.field_number.insert(5, "attribute".to_string());
        result.field_number.insert(6, "doc_string".to_string());
        result.field_number.insert(7, "domain".to_string());
        result
    }
}
impl Dimension {
    pub fn new() -> Self {
        let mut result = Dimension{ field_number: HashMap::new(),value:0 };
        result.field_number.insert(1, "dim_value".to_string());
        result
    }
}
impl Tensor2 {
    pub fn new() -> Self {
        let mut result = Tensor2 { field_number: HashMap::new(),ts:TensorShapeProto::new() };
        result.field_number.insert(1, "elem_type".to_string());
        result.field_number.insert(2, "shape".to_string());
        result
    }
}
impl TypeProto {
    pub fn new() -> Self {
        let mut result = TypeProto { field_number: HashMap::new() ,t:Tensor2::new()};
        result.field_number.insert(1, "tensor_type".to_string());
        result
    }
}
impl ValueInfoProto {
    pub fn new() -> Self {
        let mut result = ValueInfoProto { name:String::new(),field_number: HashMap::new() ,tp:TypeProto::new()};
        result.field_number.insert(1, "name".to_string());
        result.field_number.insert(2, "type".to_string());
        result.field_number.insert(9, "elem_type".to_string());
        result
    }
}
impl TensorShapeProto {
    pub fn new() -> Self {
        let mut result = TensorShapeProto{ field_number: HashMap::new(),dim:Vec::new() };
        result.field_number.insert(1, "dimension".to_string());
        result
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