

use onnx_rust::fileParser::OnnxFileParser;

fn main() {
    

    let mut  parser = OnnxFileParser::new("./onnxFile/mnist-1.onnx");
    parser.parse();

    /*if parser.error.is_some(){
        println!("{}",parser.error.unwrap());
    }else{
       print!("Grafo generato");
       
    } println!("{:?}",parser.graph.name);*/
    
}