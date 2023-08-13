

use onnx_rust::{ fileParser::fileParser::OnnxFileParser};

fn main() {
    

    let mut  parser = OnnxFileParser::new();
    
    /* GOOGLENET */
    
 //  parser.parse("./onnxFile/googlenet/model.onnx","./onnxFile/googlenet/test_data_set_0/input_0.pb");

  
    /* MNIST */
    parser.parse("./onnxFile/mnist/mnist-1.onnx","./onnxFile/mnist/input_0.pb");
    
    /* mancano operazioni */
    // parser.parse("./onnxFile/shufflenet/model.onnx","./onnxFile/shufflenet/test_data_0.npz");
    //parser.parse("./onnxFile/mobilenetv2-7/mobilenetv2-7.onnx","./onnxFile/mobilenetv2-7/test_data_set_0/input_0.pb");
     
     if parser.result.is_err(){
        println!("{}",parser.result.err().unwrap());
     }
}