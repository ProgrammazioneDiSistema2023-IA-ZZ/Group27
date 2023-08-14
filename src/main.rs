

use onnx_rust::{ fileParser::fileParser::OnnxFileParser};

fn main() {
    

    let mut  parser = OnnxFileParser::new();
    
    /* GOOGLENET */
    
    //    parser.parse("./onnxFile/googlenet-12/googlenet-12.onnx","./onnxFile/googlenet-12/test_data_set_0/input_0.pb");

  
    /* MNIST */
    parser.parse("./onnxFile/mnist-12/mnist-12.onnx","./onnxFile/mnist-12/test_data_set_0/input_0.pb");
    
     if parser.result.is_err(){
        println!("{}",parser.result.err().unwrap());
     }
}