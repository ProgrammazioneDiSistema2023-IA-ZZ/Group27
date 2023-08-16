use crate::{fileParser::protobufstruct::*, operations::Attribute};

pub fn readVarint(vect: &Vec<u8>, index: &mut usize) -> usize {
    let mut varint: usize = 0;
    let mut shift: usize = 0;
    let mut fine = false;
    let mut value;
    while fine == false {
        let mut byte = vect.get(index.clone()).unwrap();
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
pub fn leggifloats(v: &Vec<u8>) -> Vec<f32> {
    let mut result = Vec::new();
    let mut i = 0;
    while i < v.len() {
        result.push(f32::from_le_bytes(v[i..i + 4].try_into().unwrap()));
        i += 4;
    }
    result
}

pub fn leggibytes(v: &Vec<u8>) -> Vec<u8> {
    let mut result = Vec::new();
    let mut i = 0;
    while i < v.len() {
        result.push(u8::from_le_bytes(v[i..i + 1].try_into().unwrap()));
        i += 1;
    }
    result
}

pub fn leggiint64(v: &Vec<u8>) -> Vec<u64> {
    let mut result = Vec::new();
    let mut i = 0;
    while i < v.len() {
        result.push(readVarint(v, &mut i) as u64);
        //  result.push(u64::from_le_bytes(v[i..i + 8].try_into().unwrap()));
        // i += 8;
    }
    result
}
pub fn leggiraw(v: &Vec<u8>) -> Vec<u32> {
    let mut result = Vec::new();
    let mut i = 0;
    while i < v.len() {
        result.push(u32::from_le_bytes(v[i..i + 4].try_into().unwrap()));
        i += 4;
    }
    result
}
pub fn proto_buffer_tag_reader(proto: &mut ProtoBufMessage, binary: &Vec<u8>) -> Option<String> {
    let mut index = 0;
    while index < binary.len() {
        let val = readVarint(&binary, &mut index);
        let wireType = val & 0x3;
        let fieldNumber = val >> 3;
        match wireType {
            0 => {
                let res = wireType_zero_2(proto, &fieldNumber, &binary, &mut index);
                if res.is_some() {
                    return res;
                }
            }
            2 => {
                let res = wireType_two_2(proto, &fieldNumber, &binary, &mut index);
                if res.is_some() {
                    return res;
                }
            }
            _ => {
                return Some("Unsupported data type".to_string());
            }
        }
    }
    return None;
}

fn wireType_two_2(
    pbm: &mut ProtoBufMessage,
    field_number: &usize,
    vettore: &Vec<u8>,
    index: &mut usize,
) -> Option<String> {
    match pbm {
        ProtoBufMessage::ModelProto(p) => {
            print!("\"{:?}\":", p.fieldNumber.get(field_number).unwrap());
            let val = readVarint(&vettore, index);
            if *field_number == 7 {
                println!("GRAFOOOOOO!");
                let mut graph = ProtoBufMessage::GraphProto(GraphProto::new());
                proto_buffer_tag_reader(&mut graph, &(vettore[*index..(*index + val)]).to_vec());
                p.graph = GraphProto::try_from(graph).unwrap();
                println!("FINE GRAFO");
            } else if *field_number != 8 {
                let word = String::from_utf8(vettore[*index..(*index + val)].to_owned()).unwrap();
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
                    let mut node: ProtoBufMessage = ProtoBufMessage::NodeProto(NodeProto::new());
                    proto_buffer_tag_reader(&mut node, &(vettore[*index..(*index + val)]).to_vec());
                    p.node.push(NodeProto::try_from(node).unwrap());
                }
                11 => {
                    println!("");
                    println!("INPUT!");
                    let mut node: ProtoBufMessage =
                        ProtoBufMessage::ValueInfoProto(ValueInfoProto::new());
                    proto_buffer_tag_reader(&mut node, &(vettore[*index..(*index + val)]).to_vec());
                    p.inputs_node.push(ValueInfoProto::try_from(node).unwrap());
                    println!("FINE INPUT!");
                }
                12 => {
                    println!("");
                    println!("OUTPUT!");
                    let mut node: ProtoBufMessage =
                        ProtoBufMessage::ValueInfoProto(ValueInfoProto::new());
                    proto_buffer_tag_reader(&mut node, &(vettore[*index..(*index + val)]).to_vec());
                    p.outputs_node.push(ValueInfoProto::try_from(node).unwrap());
                    println!("FINE OUTPUT!");
                }
                13 => {
                    /*TAG VALUE INFO - NON NECESSARIO */

                    println!("");
                    println!("VALUE INFO!");
                    let mut node: ProtoBufMessage =
                        ProtoBufMessage::ValueInfoProto(ValueInfoProto::new());
                    proto_buffer_tag_reader(&mut node, &(vettore[*index..(*index + val)]).to_vec());
                    p.value_info_node
                        .push(ValueInfoProto::try_from(node).unwrap().name);
                    println!("VALUE INFO!");
                }
                5 => {
                    println!(" INITIAL");
                    let mut node: ProtoBufMessage =
                        ProtoBufMessage::TensorProto(TensorProto::new());
                    proto_buffer_tag_reader(&mut node, &(vettore[*index..(*index + val)]).to_vec());
                    p.tensor_initializer
                        .push(TensorProto::try_from(node).unwrap());
                }
                _ => {
                    let word =
                        String::from_utf8(vettore[*index..(*index + val)].to_owned()).unwrap();
                    println!("{:?}", word);
                    if *field_number == 2 {
                        p.name = word;
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
                proto_buffer_tag_reader(&mut at, &(vettore[*index..(*index + val)]).to_vec());
               let attr= AttributeProto::try_from(at).unwrap();


               p.attributes.insert(attr.name, attr.attr);

                println!("{:?}",p.attributes);
            } else {
                let word = String::from_utf8(vettore[*index..(*index + val)].to_owned()).unwrap();
                println!("{:?}", word);
                if *field_number == 3 {
                    p.name = word;
                } else if *field_number == 1 {
                    p.inputs.push(word);
                } else if *field_number == 2 {
                    p.outputs.push(word);
                } else if *field_number == 4 {
                    p.op_type = word;
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
                proto_buffer_tag_reader(&mut tp, &(vettore[*index..(*index + val)]).to_vec());

                p.tp = TensorProto::try_from(tp).unwrap();
                print!("Struttura ok [{:?}]", p.tp.dims);

                //HO UN TENSORE COMPLETO
            } else {
                let word = String::from_utf8(vettore[*index..(*index + val)].to_owned()).unwrap();
                println!("{:?}", word);
                if *field_number==1 { //name
                    p.name=word.clone();
                }
                if *field_number==4 { //s
                    if let Attribute::Undefined = p.attr{
                       
                        p.attr=Attribute::String(word.clone());
                    }
                   
                }
            }
            (*index) += val;
        }
        ProtoBufMessage::TensorProto(p) => {
            if p.fieldNumber.get(field_number).is_some() {
                print!("\t\t\t\t\"{:}\":", p.fieldNumber.get(field_number).unwrap());
            } else {
                panic!("{}", field_number);
            }

            let val = readVarint(&vettore, index) as usize;
            //  print!("val =|{}| : ", val);

            if *field_number == 4 {
                let v = leggifloats(&vettore[*index..*index + val].to_vec());
                println!(" Vett di {:?} elem", v.len());
                p.float_data = v;
            } else if *field_number == 9 {
                //RAW_DATA
                let v = leggiraw(&vettore[*index..*index + val].to_vec());
                println!(" Vett di {:?} elem", v.len());
                p.float_data = v.iter().map(|x| *x as f32).collect();
            } else if *field_number == 7 {
                let v = leggiint64(&vettore[*index..*index + val].to_vec());
                println!(" Vett di {:?} elem", v.len());
                p.float_data = v.iter().map(|x| *x as f32).collect();
            } else {
                let word = String::from_utf8(vettore[*index..(*index + val)].to_owned()).unwrap();
                println!("{:?}", word);
                if *field_number == 8 {
                    p.name = word;
                }
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
                proto_buffer_tag_reader(&mut tp, &(vettore[*index..(*index + val)]).to_vec());
                p.tp = TypeProto::try_from(tp).unwrap();
            } else {
                let word = String::from_utf8(vettore[*index..(*index + val)].to_owned()).unwrap();
                println!("{:?}", word);
                if *field_number == 1 {
                    p.name = word;
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
                proto_buffer_tag_reader(&mut t, &(vettore[*index..(*index + val)]).to_vec());
                p.t = Tensor2::try_from(t).unwrap();
            } else {
                let word = String::from_utf8(vettore[*index..(*index + val)].to_owned()).unwrap();
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
                proto_buffer_tag_reader(&mut t, &(vettore[*index..(*index + val)]).to_vec());
                p.ts = TensorShapeProto::try_from(t).unwrap();
            } else {
                let word = String::from_utf8(vettore[*index..(*index + val)].to_owned()).unwrap();
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
                proto_buffer_tag_reader(&mut t, &(vettore[*index..(*index + val)]).to_vec());
                p.dim.push(Dimension::try_from(t).unwrap());
            } else {
                let word = String::from_utf8(vettore[*index..(*index + val)].to_owned()).unwrap();
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

        _ => {
            return Some("protoBuf message not implemented".to_string());
        }
    }
    return None;
}

pub fn wireType_zero_2(
    pbm: &mut ProtoBufMessage,
    field_number: &usize,
    vettore: &Vec<u8>,
    index: &mut usize,
) -> Option<String> {
    let mut fieldName;
    let val = readVarint(&vettore, index);
    match (pbm) {
        ProtoBufMessage::ModelProto(p) => {
            fieldName = p.fieldNumber.get(field_number);
            if fieldName.is_some() {
                print!("\"{}\":S", p.fieldNumber.get(field_number).unwrap());
                println!("{}", val);
            } else {
                println!("\t NON SUPPORTATO val = {}", field_number);
            }
        }
        ProtoBufMessage::GraphProto(p) => {
            fieldName = p.fieldNumber.get(field_number);
            if fieldName.is_some() {
                print!("\t\"{}\":S", p.fieldNumber.get(field_number).unwrap());
                println!("{}", val);
            } else {
                println!("\t NON SUPPORTATO val = {}", field_number);
            }
        }
        ProtoBufMessage::AttributeProto(p) => {
            fieldName = (p).fieldNumber.get(field_number);
            if fieldName.is_some() {
                print!("\t\t\t\"{}\":", p.fieldNumber.get(field_number).unwrap());
                println!("{}", val);
                match fieldName.unwrap().as_str() {

                    "f" => {
                        println!("f");
                    }
                    "i" => {
                        println!("i");
                        if let Attribute::Undefined = p.attr{
                            p.attr=Attribute::Int(val as isize);
                        }
                    
                        println!("i");
                    }
                    "ints" => {
                        if let Attribute::Undefined = p.attr{
                            p.attr=Attribute::Ints(Vec::new());
                        }
                        if let Attribute::Ints(vect) = &p.attr{
                            let mut temp = vect.clone();
                            temp.push(val as isize);
                            p.attr=Attribute::Ints(temp);
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
                    _ => {}
                }
            } else {
                print!("\t\t\tNON SUPPORTATO val = {}", field_number);
            }
        }
        ProtoBufMessage::TensorProto(p) => {
            fieldName = p.fieldNumber.get(field_number);
            if fieldName.is_some() {
                print!("\t\t\t\t\"{}\":", p.fieldNumber.get(field_number).unwrap());
                println!("{}", val);
                if *field_number == 1 {
                    // Campo "dims", è gestito come varint
                    p.dims.push(val);
                }
                if *field_number == 9 {
                    //RAW_DATA
                    print!("\n\n\n\nffdfjdighdsghdi");
                }
            } else {
                print!("\t\t\t\tNON SUPPORTATO val = {}", field_number);
            }
        }
        ProtoBufMessage::NodeProto(p) => {
            fieldName = p.fieldNumber.get(field_number);
            if fieldName.is_some() {
                print!("\t\t\"{}\":S", p.fieldNumber.get(field_number).unwrap());
                println!("{}", val);
            } else {
                print!("\t\tNON SUPPORTATO val = {}", field_number);
            }
        }
        ProtoBufMessage::ValueInfoProto(p) => {
            fieldName = p.fieldNumber.get(field_number);
            if fieldName.is_some() {
                print!("\t\t\t\"{}\":", p.fieldNumber.get(field_number).unwrap());
                println!("{}", val);
            } else {
                print!("\t\t\tNON SUPPORTATO val = {}", field_number);
            }
        }
        ProtoBufMessage::TypeProto(p) => {
            fieldName = p.fieldNumber.get(field_number);
            if fieldName.is_some() {
                print!("\t\t\t\"{}\":", p.fieldNumber.get(field_number).unwrap());
                println!("{}", val);
            } else {
                print!("\t\t\tNON SUPPORTATO val = {}", field_number);
            }
        }

        ProtoBufMessage::Tensor2(p) => {
            fieldName = p.fieldNumber.get(field_number);
            if fieldName.is_some() {
                print!("\t\t\t\t\"{}\":", p.fieldNumber.get(field_number).unwrap());
                println!("{}", val);
            } else {
                print!("\t\t\tNON SUPPORTATO val = {}", field_number);
            }
        }
        ProtoBufMessage::Dimension(p) => {
            fieldName = p.fieldNumber.get(field_number);
            if fieldName.is_some() {
                print!(
                    "\t\t\t\t\t\t\"{}\":",
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
        _ => {
            return Some("protoBuf message not implemented".to_string());
        }
    }
    return None;
}
