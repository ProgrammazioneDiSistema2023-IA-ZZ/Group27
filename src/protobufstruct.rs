use std::collections::HashMap;

use ndarray::{SliceArg, array};

use crate::operations::Tensor;

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
    pub fieldNumber : HashMap<usize,String>,
    pub tp : TypeProto,
    pub name:String,

}

#[derive(Debug)]
pub struct Tensor2{
    pub fieldNumber : HashMap<usize,String>,
    pub ts: TensorShapeProto,
}

#[derive(Debug)]
pub  struct TensorShapeProto{
    pub fieldNumber : HashMap<usize,String>,
    pub dim : Vec<Dimension>
}

#[derive(Debug)]
pub struct Dimension{
   pub fieldNumber : HashMap<usize,String>,
   pub value:usize,
}

#[derive(Debug)]
pub struct TypeProto{
    pub  fieldNumber : HashMap<usize,String>,
    pub t : Tensor2,
}


#[derive(Debug)]
pub struct NodeProto {
   pub fieldNumber: HashMap<usize, String>,
   pub attr: AttributeProto,
   pub name: String,
   pub inputs : Vec<String>,
   pub outputs: Vec<String>,
   pub op_type: String,
}
#[derive(Debug)]
pub struct GraphProto {
    pub fieldNumber: HashMap<usize, String>,
    pub node:Vec<NodeProto>,
    pub inputs_node: Vec<ValueInfoProto>,
    pub outputs_node: Vec<ValueInfoProto>,

}

#[derive(Debug)]
pub struct TensorProto {
    pub fieldNumber: HashMap<usize, String>,
    pub tensor : Tensor,
    pub dims:Vec<usize>,
    pub float_data :Vec<f32> 
}
#[derive(Debug)]
pub struct ModelProto {
    pub fieldNumber: HashMap<usize, String>,
    pub graph : GraphProto,
}
#[derive(Debug)]
pub struct AttributeProto {
    pub  fieldNumber: HashMap<usize, String>,
    pub  tp : TensorProto,
   
}

impl AttributeProto {
    pub fn new() -> Self {
        let mut result = AttributeProto { fieldNumber: HashMap::new(),tp:TensorProto::new()};
        result.fieldNumber.insert(0, "undefined".to_string());
        result.fieldNumber.insert(7, "floats".to_string());
        result.fieldNumber.insert(8, "ints".to_string());
        result.fieldNumber.insert(9, "strings".to_string());
        result.fieldNumber.insert(10, "tensors".to_string());
        result.fieldNumber.insert(2, "f".to_string());
        result.fieldNumber.insert(3, "i".to_string());
        result.fieldNumber.insert(4, "s".to_string());
        result.fieldNumber.insert(5, "t".to_string());
        result.fieldNumber.insert(1, "name".to_string());
        result.fieldNumber.insert(20, "type".to_string());
        result
    }
}
impl TensorProto {
    pub fn new() -> Self {
        let mut result = TensorProto { fieldNumber: HashMap::new(),tensor:array![ []  ].into_dyn() ,dims:Vec::new(),float_data:Vec::new()};
        result.fieldNumber.insert(1, "dims".to_string());
        result.fieldNumber.insert(4, "float_data".to_string());
        result.fieldNumber.insert(2, "data_type".to_string());

        result
    }
}
impl ModelProto {
    pub fn new() -> Self {
        let mut result = ModelProto { fieldNumber: HashMap::new(), graph:GraphProto::new() };
        result.fieldNumber.insert(2, "producer_name".to_string());
        result.fieldNumber.insert(5, "model_version".to_string());
        result.fieldNumber.insert(1, "ir_version".to_string());
        result.fieldNumber.insert(3, "producer_version".to_string());
        result.fieldNumber.insert(7, "graph".to_string());
        result.fieldNumber.insert(14, "metadata_props".to_string());
        result.fieldNumber.insert(20, "training_info".to_string());
        result.fieldNumber.insert(25, "functions".to_string());
        result.fieldNumber.insert(8, "opset_import".to_string());
        result
    }
}
impl GraphProto {
    pub fn new() -> Self {
        let mut result = GraphProto { fieldNumber: HashMap::new(), node:Vec::new(),inputs_node:Vec::new(),outputs_node:Vec::new() };
        result.fieldNumber.insert(1, "node".to_string());
        result.fieldNumber.insert(2, "name".to_string());
        result.fieldNumber.insert(11, "input".to_string());
        result.fieldNumber.insert(12, "ouput".to_string());
        result.fieldNumber.insert(5, "initializer".to_string());
        result.fieldNumber.insert(15, "sparse_initializer".to_string());
        result.fieldNumber.insert(10, "doc_string".to_string());
        result.fieldNumber.insert(13, "value_info".to_string());
        result
    }
}
impl NodeProto {
    pub fn new() -> Self {
        let mut result = NodeProto { 
            fieldNumber: HashMap::new(),
            attr:AttributeProto::new(),
            name : "".to_string(),
            inputs:vec![],outputs:vec![],
            op_type:String::new(),
        };
        result.fieldNumber.insert(1, "input".to_string());
        result.fieldNumber.insert(2, "output".to_string());
        result.fieldNumber.insert(3, "name".to_string());
        result.fieldNumber.insert(4, "op_type".to_string());
        result.fieldNumber.insert(5, "attribute".to_string());
        result.fieldNumber.insert(6, "doc_string".to_string());
        result.fieldNumber.insert(7, "domain".to_string());
        result
    }
}
impl Dimension {
    pub fn new() -> Self {
        let mut result = Dimension{ fieldNumber: HashMap::new(),value:0 };
        result.fieldNumber.insert(1, "dim_value".to_string());
        result
    }
}
impl Tensor2 {
    pub fn new() -> Self {
        let mut result = Tensor2 { fieldNumber: HashMap::new(),ts:TensorShapeProto::new() };
        result.fieldNumber.insert(1, "elem_type".to_string());
        result.fieldNumber.insert(2, "shape".to_string());
        result
    }
}
impl TypeProto {
    pub fn new() -> Self {
        let mut result = TypeProto { fieldNumber: HashMap::new() ,t:Tensor2::new()};
        result.fieldNumber.insert(1, "tensor_type".to_string());
        result
    }
}
impl ValueInfoProto {
    pub fn new() -> Self {
        let mut result = ValueInfoProto { name:String::new(),fieldNumber: HashMap::new() ,tp:TypeProto::new()};
        result.fieldNumber.insert(1, "name".to_string());
        result.fieldNumber.insert(2, "type".to_string());
        result.fieldNumber.insert(9, "elem_type".to_string());
        result
    }
}
impl TensorShapeProto {
    pub fn new() -> Self {
        let mut result = TensorShapeProto{ fieldNumber: HashMap::new(),dim:Vec::new() };
        result.fieldNumber.insert(1, "dimension".to_string());
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