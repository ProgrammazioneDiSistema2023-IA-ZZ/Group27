use crate::{fileparser::protobufstruct::*, operations::Attribute};

/// Partendo da un vettore di u8 `vect` e un indice di partenza `index` , scorre `vect` per ottenere il varint rappresentato su n bytes, dove n non è noto a priori
pub fn read_varint(vect: &Vec<u8>, index: &mut usize) -> usize {
    let mut varint: usize = 0;
    let mut shift: usize = 0;
    let mut fine = false;
    let mut value;
    while fine == false {
        let  byte = vect.get(index.clone()).unwrap();
        if (byte.clone() >> 7) == 0 {
            value = (byte & 0b01111111) as usize;
            varint = ((value << shift.clone()) | varint) as usize;
            fine = true;
        } else {
            value = (byte & 0b01111111) as usize;
            varint = ((value << shift.clone()) | varint) as usize;
            shift += 7;
        }
        (*index) += 1;
    }
    return varint;
}

/// Starting from a `vect`, the function cast all the value to a type given by `data_type`
/// For instance, if the `data_type` is a float the function take 4 bytes and then cast to a f32
#[allow(unused_parens)]
pub fn converter_raw(vect: &Vec<u8>, data_type: usize) -> Vec<f32> {
    let mut result = Vec::new();
    match data_type {
        1 => {
            //float -> 4 bytes
            let mut i = 0;
            while i < vect.len() {
                result.push(f32::from_le_bytes(vect[i..i + 4].try_into().unwrap()));
                i += 4;
            }
        }
        7 => {
            //int64 -> 8bytes
            let mut i = 0;
            while i < vect.len() {
                result.push((u64::from_le_bytes(vect[i..i + 8].try_into().unwrap()) as f32));
                i += 8;
            }
        }
        _ => {
            panic!("{} is not implemented!", data_type);
        }
    }
    result
}
/// Convert `Vec<u8>` to `Vec<f32>`
pub fn read_floats(v: &Vec<u8>) -> Vec<f32> {
    let mut result = Vec::new();
    let mut i = 0;
    while i < v.len() {
        result.push(f32::from_le_bytes(v[i..i + 4].try_into().unwrap()));
        i += 4;
    }
    result
}
/// è utile ????????
pub fn leggibytes(v: &Vec<u8>) -> Vec<u8> {
    let mut result = Vec::new();
    let mut i = 0;
    while i < v.len() {
        result.push(u8::from_le_bytes(v[i..i + 1].try_into().unwrap()));
        i += 1;
    }
    result
}
///  Convert `Vec<u8>` to `Vec<u64>`
/// WARNING: in this case the u64 number is read like a Varint number and then cast to u64
/// /// IT'S NOT READ TAKING 8 BYTES FOR EACH u64
pub fn read_int64(v: &Vec<u8>) -> Vec<u64> {
    let mut result = Vec::new();
    let mut i = 0;
    while i < v.len() {
        result.push(read_varint(v, &mut i) as u64);
    }
    result
}
/// Function that for each generic tag it parse by looking what kind of `ProtoBufMessage` is the tag
/// -Every file .onnx start with a `ModelProto`
/// -Data inside the tag is encoded using wire encoding
/// The first varint contains information on what wireType is and the number of field (linking the number to the field name is done by looking .proto file)
pub fn proto_buffer_tag_reader(proto: &mut ProtoBufMessage, binary: &Vec<u8>) -> Option<String> {
    let mut index = 0;
    while index < binary.len() {
        let val = read_varint(&binary, &mut index);
        //see documentation on why data is encoded in this way
        let wire_type = val & 0x3;
        let field_number = val >> 3;
        //for our purpose we don't need to handle other wiretype
        match wire_type {
            0 => {
                //field_number: numeric_value
                let error = wire_type_zero(proto, &field_number, &binary, &mut index);
                if error.is_some() {
                    //if there is some error
                    return error;
                }
            }
            2 => {
                //field_number: string
                let error = wire_type_two(proto, &field_number, &binary, &mut index);
                if error.is_some() {
                    //if there is some error
                    return error;
                }
            }
            _ => {
                //Like groups,ecc..
                return Some("Unsupported data type".to_string());
            }
        }
    }
    //if no error occur during parsing
    return None;
}

/// function for read field that are encoded using |field_number| dimension | data | where data size is given by dimension value
/// use `ProtoBufMessage` for translate field_number -> field_name and for save the data that is parsed
/// `field_number` is read in `proto_buffer_tag_reader` and is used for choose get the field_name
/// `vect` is where to read the next value
/// `index` is the index used for `vect`
fn wire_type_two(
    pbm: &mut ProtoBufMessage,
    field_number: &usize,
    vect: &Vec<u8>,
    index: &mut usize,
) -> Option<String> {
    match pbm {
        ProtoBufMessage::ModelProto(p) => {
            let  opt_tag_name = p.field_number.get(field_number);
            if opt_tag_name.is_none() {
                return Some(
                    vec![
                        "Field number ",
                        &(field_number.to_string()),
                        "is not implemented",
                    ]
                    .join(""),
                );
            }
            let tag_name = opt_tag_name.unwrap();
            print!("\"{}\":", tag_name);
            let val = read_varint(&vect, index);

            if tag_name == "graph" {
                let mut graph = ProtoBufMessage::GraphProto(GraphProto::new());
                proto_buffer_tag_reader(&mut graph, &(vect[*index..(*index + val)]).to_vec());
                p.graph = GraphProto::try_from(graph).unwrap();
            } else if tag_name != "opset_import" {
                let word = String::from_utf8(vect[*index..(*index + val)].to_owned()).unwrap();
                println!("{:?}", word);
            }
            (*index) += val;
        }
        ProtoBufMessage::GraphProto(p) => {
            let opt_tag_name = p.field_number.get(field_number);
            if opt_tag_name.is_none() {
                return Some(
                    vec![
                        "Field number ",
                        &(field_number.to_string()),
                        "is not implemented",
                    ]
                    .join(""),
                );
            }
            let tag_name = opt_tag_name.unwrap();
            print!("\t \"{}\":", tag_name);
            let val = read_varint(&vect, index);
            //inside tag Graph there is more tag nested
            match tag_name.as_str() {
                "node" => {
                    println!("");
                    let mut node: ProtoBufMessage = ProtoBufMessage::NodeProto(NodeProto::new());
                    proto_buffer_tag_reader(&mut node, &(vect[*index..(*index + val)]).to_vec());
                    p.node.push(NodeProto::try_from(node).unwrap());
                }
                "input" => {
                    println!("");
                    let mut node: ProtoBufMessage =
                        ProtoBufMessage::ValueInfoProto(ValueInfoProto::new());
                    proto_buffer_tag_reader(&mut node, &(vect[*index..(*index + val)]).to_vec());
                    p.inputs_node.push(ValueInfoProto::try_from(node).unwrap());
                }
                "output" => {
                    println!("");
                    let mut node: ProtoBufMessage =
                        ProtoBufMessage::ValueInfoProto(ValueInfoProto::new());
                    proto_buffer_tag_reader(&mut node, &(vect[*index..(*index + val)]).to_vec());
                    p.outputs_node.push(ValueInfoProto::try_from(node).unwrap());
                }
                "value_info" => {
                    println!("");
                    let mut node: ProtoBufMessage =
                        ProtoBufMessage::ValueInfoProto(ValueInfoProto::new());
                    proto_buffer_tag_reader(&mut node, &(vect[*index..(*index + val)]).to_vec());
                    p.value_info_node
                        .push(ValueInfoProto::try_from(node).unwrap().name);
                }
                "initializer" => {
                    let mut node: ProtoBufMessage =
                        ProtoBufMessage::TensorProto(TensorProto::new());
                    proto_buffer_tag_reader(&mut node, &(vect[*index..(*index + val)]).to_vec());
                    p.tensor_initializer
                        .push(TensorProto::try_from(node).unwrap());
                }
                _ => {
                    let word = String::from_utf8(vect[*index..(*index + val)].to_owned()).unwrap();
                    println!("{:?}", word);
                    if *field_number == 2 {
                        p.name = word;
                    }
                }
            }
            (*index) += val;
        }
        ProtoBufMessage::NodeProto(p) => {
            let opt_tag_name = p.field_number.get(field_number);
            if opt_tag_name.is_none() {
                return Some(
                    vec![
                        "Field number ",
                        &(field_number.to_string()),
                        "is not implemented",
                    ]
                    .join(""),
                );
            }
            let tag_name = opt_tag_name.unwrap();
            print!("\t\t \"{}\":", tag_name);
            let val = read_varint(&vect, index);
            if tag_name == "attribute" {
                println!("");
                let mut at = ProtoBufMessage::AttributeProto(AttributeProto::new());
                proto_buffer_tag_reader(&mut at, &(vect[*index..(*index + val)]).to_vec());
                let attr = AttributeProto::try_from(at).unwrap();
                p.attributes.insert(attr.name, attr.attr);
       
            } else {
                let word = String::from_utf8(vect[*index..(*index + val)].to_owned()).unwrap();
                println!("{:?}", word);
                match tag_name.as_str(){
                    "name"=>p.name = word,
                    "input"=> p.inputs.push(word),
                    "output" =>p.outputs.push(word),
                    "op_type"=>p.op_type = word,
                    _=>{ }//nothing to do, just print
                }
            }
            (*index) += val;
        }
        ProtoBufMessage::AttributeProto(p) => {
            let  opt_tag_name = p.field_number.get(field_number);
            if opt_tag_name.is_none() {
                return Some(
                    vec![
                        "Field number ",
                        &(field_number.to_string()),
                        "is not implemented",
                    ]
                    .join(""),
                );
            }
            let tag_name = opt_tag_name.unwrap();
            print!("\t\t \"{}\":", tag_name);
            let val = read_varint(&vect, index);
            if tag_name=="t" {
                println!("");
                let mut tp = ProtoBufMessage::TensorProto(TensorProto::new());
                proto_buffer_tag_reader(&mut tp, &(vect[*index..(*index + val)]).to_vec());

                p.tp = TensorProto::try_from(tp).unwrap();
                //HO UN TENSORE COMPLETO
            } else {
                
                let word = String::from_utf8(vect[*index..(*index + val)].to_owned()).unwrap();
                println!("{:?}", word);
                if tag_name=="name" {
                    //name
                    p.name = word.clone();
                }
                if tag_name=="s" {
                    //s
                    if let Attribute::Undefined = p.attr {
                        p.attr = Attribute::String(word.clone());
                    }
                }
            }
            (*index) += val;
        }
        ProtoBufMessage::TensorProto(p) => {
            if p.field_number.get(field_number).is_some() {
                print!("\t\t\t\t\"{:}\":", p.field_number.get(field_number).unwrap());
            } else {
                panic!("{}", field_number);
            }
            let val = read_varint(&vect, index) as usize;
            if *field_number == 4 {
                let v = read_floats(&vect[*index..*index + val].to_vec());
                println!(" Vett di {:?} elem", v.len());
                p.float_data = v;
            } else if *field_number == 9 {
                //RAW_DATA -> leggo bytes poi in base al campo data_type converto nel tipo corretto
                let v = leggibytes(&vect[*index..*index + val].to_vec());
                //let v = leggiraw(&vect[*index..*index + val].to_vec());
                println!(" Vett di {:?} elem RAW", v.len());
                p.raw_data = v;
            } else if *field_number == 7 {
                let v = read_int64(&vect[*index..*index + val].to_vec());
                println!(" Vett di {:?} elem", v.len());
                p.float_data = v.iter().map(|x| *x as f32).collect();
            } else {
                let word = String::from_utf8(vect[*index..(*index + val)].to_owned()).unwrap();
                println!("{:?}", word);
                if *field_number == 8 {
                    p.name = word;
                }
            }
            (*index) += val;
        }
        ProtoBufMessage::ValueInfoProto(p) => {
            print!("\t\t\"{}\":", *p.field_number.get(field_number).unwrap());
            let val = read_varint(&vect, index);
            if *field_number == 2 {
                //typeProto
                println!("");
                let mut tp = ProtoBufMessage::TypeProto(TypeProto::new());
                proto_buffer_tag_reader(&mut tp, &(vect[*index..(*index + val)]).to_vec());
                p.tp = TypeProto::try_from(tp).unwrap();
            } else {
                let word = String::from_utf8(vect[*index..(*index + val)].to_owned()).unwrap();
                println!("{:?}", word);
                if *field_number == 1 {
                    p.name = word;
                }
            } //index..(index + val)

            (*index) += val;
        }
        ProtoBufMessage::TypeProto(p) => {
            print!("\t\t\t\"{}\":", *p.field_number.get(field_number).unwrap());
            let val = read_varint(&vect, index);
            if *field_number == 1 {
                //tensor
                println!("");
                let mut t = ProtoBufMessage::Tensor2(Tensor2::new());
                proto_buffer_tag_reader(&mut t, &(vect[*index..(*index + val)]).to_vec());
                p.t = Tensor2::try_from(t).unwrap();
            } else {
                let word = String::from_utf8(vect[*index..(*index + val)].to_owned()).unwrap();
                println!("{:?}", word);
            } //index..(index + val)
            (*index) += val;
        }
        ProtoBufMessage::Tensor2(p) => {
            print!("\t\t\t\t\"{}\":", *p.field_number.get(field_number).unwrap());
            let val = read_varint(&vect, index);
            if *field_number == 2 {
                //tensorShape
                println!("");
                let mut t = ProtoBufMessage::TensorShapeProto(TensorShapeProto::new());
                proto_buffer_tag_reader(&mut t, &(vect[*index..(*index + val)]).to_vec());
                p.ts = TensorShapeProto::try_from(t).unwrap();
            } else {
                let word = String::from_utf8(vect[*index..(*index + val)].to_owned()).unwrap();
                println!("{:?}", word);
            } //index..(index + val)
            (*index) += val;
        }
        ProtoBufMessage::TensorShapeProto(p) => {
            print!(
                "\t\t\t\t\t\"{}\":",
                *p.field_number.get(field_number).unwrap()
            );
            let val = read_varint(&vect, index);
            if *field_number == 1 {
                //dimension
                println!("");
                let mut t = ProtoBufMessage::Dimension(Dimension::new());
                proto_buffer_tag_reader(&mut t, &(vect[*index..(*index + val)]).to_vec());
                p.dim.push(Dimension::try_from(t).unwrap());
            } else {
                let word = String::from_utf8(vect[*index..(*index + val)].to_owned()).unwrap();
                println!("{:?}", word);
            }

            //index..(index + val)
            (*index) += val;
        }
        ProtoBufMessage::Dimension(p) => {
            print!(
                "\t\t\t\t\t\t\"{}\":",
                *p.field_number.get(field_number).unwrap()
            );
            let val = read_varint(&vect, index);

            let word = String::from_utf8(vect[*index..(*index + val)].to_owned()).unwrap();
            println!("{:?}", word);

            (*index) += val;
        }

        
    }
    return None;
}

pub fn wire_type_zero(
    pbm: &mut ProtoBufMessage,
    field_number: &usize,
    vettore: &Vec<u8>,
    index: &mut usize,
) -> Option<String> {
    let fieldName;
    let val = read_varint(&vettore, index);
    match pbm {
        ProtoBufMessage::ModelProto(p) => {
            fieldName = p.field_number.get(field_number);
            if fieldName.is_some() {
                print!("\"{}\":S", p.field_number.get(field_number).unwrap());
                println!("{}", val);
            } else {
                println!("\t NON SUPPORTATO val = {}", field_number);
            }
        }
        ProtoBufMessage::GraphProto(p) => {
            fieldName = p.field_number.get(field_number);
            if fieldName.is_some() {
                print!("\t\"{}\":S", p.field_number.get(field_number).unwrap());
                println!("{}", val);
            } else {
                println!("\t NON SUPPORTATO val = {}", field_number);
            }
        }
        ProtoBufMessage::AttributeProto(p) => {
            fieldName = (p).field_number.get(field_number);
            if fieldName.is_some() {
                print!("\t\t\t\"{}\":", p.field_number.get(field_number).unwrap());
                println!("{}", val);
                match fieldName.unwrap().as_str() {
                    "f" => {
                        println!("f");
                        if let Attribute::Undefined = p.attr {
                            p.attr = Attribute::Float(val as f32);
                        }
                    }
                    "i" => {
                        if let Attribute::Undefined = p.attr {
                            p.attr = Attribute::Int(val as isize);
                        }

                        println!("i");
                    }
                    "ints" => {
                        if let Attribute::Undefined = p.attr {
                            p.attr = Attribute::Ints(Vec::new());
                        }
                        if let Attribute::Ints(vect) = &p.attr {
                            let mut temp = vect.clone();
                            temp.push(val as isize);
                            p.attr = Attribute::Ints(temp);
                        }
                        println!("ints");
                    }
                    "strings" => {
                        println!("strings");
                    }
                    "t" => {
                        println!("t");
                    }
                    "tensors" => {
                        println!("tensors");
                    }
                    "floats" => {
                        println!("floats");
                    }
                    _ => {
                        println!("tipo non gestito {:?}", fieldName.unwrap());
                    }
                }
            } else {
                print!("\t\t\tNON SUPPORTATO val = {}", field_number);
            }
        }
        ProtoBufMessage::TensorProto(p) => {
            fieldName = p.field_number.get(field_number);
            if fieldName.is_some() {
                print!("\t\t\t\t\"{}\":", p.field_number.get(field_number).unwrap());
                println!("{}", val);
                if *field_number == 1 {
                    // Campo "dims", è gestito come varint
                    p.dims.push(val);
                }
                if *field_number == 9 {
                    //RAW_DATA
                    print!("\n\n\n\nffdfjdighdsghdi");
                }
                if *field_number == 2 {
                    //data_type
                    p.data_type = val;
                }
            } else {
                print!("\t\t\t\tNON SUPPORTATO val = {}", field_number);
            }
        }
        ProtoBufMessage::NodeProto(p) => {
            fieldName = p.field_number.get(field_number);
            if fieldName.is_some() {
                print!("\t\t\"{}\":S", p.field_number.get(field_number).unwrap());
                println!("{}", val);
            } else {
                print!("\t\tNON SUPPORTATO val = {}", field_number);
            }
        }
        ProtoBufMessage::ValueInfoProto(p) => {
            fieldName = p.field_number.get(field_number);
            if fieldName.is_some() {
                print!("\t\t\t\"{}\":", p.field_number.get(field_number).unwrap());
                println!("{}", val);
            } else {
                print!("\t\t\tNON SUPPORTATO val = {}", field_number);
            }
        }
        ProtoBufMessage::TypeProto(p) => {
            fieldName = p.field_number.get(field_number);
            if fieldName.is_some() {
                print!("\t\t\t\"{}\":", p.field_number.get(field_number).unwrap());
                println!("{}", val);
            } else {
                print!("\t\t\tNON SUPPORTATO val = {}", field_number);
            }
        }

        ProtoBufMessage::Tensor2(p) => {
            fieldName = p.field_number.get(field_number);
            if fieldName.is_some() {
                print!("\t\t\t\t\"{}\":", p.field_number.get(field_number).unwrap());
                println!("{}", val);
            } else {
                print!("\t\t\tNON SUPPORTATO val = {}", field_number);
            }
        }
        ProtoBufMessage::Dimension(p) => {
            fieldName = p.field_number.get(field_number);
            if fieldName.is_some() {
                print!(
                    "\t\t\t\t\t\t\"{}\":",
                    p.field_number.get(field_number).unwrap()
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
        _ => {
            return Some("protoBuf message not implemented".to_string());
        }
    }
    return None;
}
