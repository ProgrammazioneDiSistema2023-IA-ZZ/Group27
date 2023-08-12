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
    while i < v.len()
    {
        result.push(f32::from_le_bytes(v[i..i + 4].try_into().unwrap()));
        i += 4;
    }
    result
}
