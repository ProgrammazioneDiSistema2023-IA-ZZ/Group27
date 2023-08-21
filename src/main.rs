use std::{sync::Arc, env};

use onnx_rust::fileparser::fileparser::OnnxFileParser;

fn main() {
  // env::set_var("RUST_LOG", "debug");
     env::set_var("RUST_LOG", "info");
    env_logger::init();

    /* GOOGLENET */

    //let graph= OnnxFileParser::parse_model("./onnxFile/googlenet-12/googlenet-12.onnx");
    //let input = OnnxFileParser::parse_data("./onnxFile/googlenet-12/test_data_set_0/input_0.pb");
    //let output = OnnxFileParser::parse_data("./onnxFile/googlenet-12/test_data_set_0/output_0.pb");
    /* MNIST */
    let graph: Result<onnx_rust::graph::OnnxGraph, String> =
       OnnxFileParser::parse_model("./onnxFile/mnist-12/mnist-12.onnx");
    let input = OnnxFileParser::parse_data("./onnxFile/mnist-12/test_data_set_0/input_0.pb");
    let output = OnnxFileParser::parse_data("./onnxFile/mnist-12/test_data_set_0/output_0.pb");
    if graph.is_err() {
        println!("{}", graph.err().unwrap());
        return;
    }
    if input.is_err() {
        println!("{}", input.err().unwrap());
        return;
    }
    let result = Arc::new(graph.unwrap()).infer(input.unwrap());
    


    //confronto risultati
    let key: Vec<String> = result.clone().unwrap().into_keys().collect();

    log::info!(
        "Result->    {:?}",
        result
            .unwrap()
            .get(&key.get(0).unwrap().clone())
            .unwrap()
            .clone()
            .into_raw_vec()
    );
    log::info!(
        "Expected->  {:?}",
        output
            .unwrap()
            .get(&key.get(0).unwrap().clone())
            .unwrap()
            .clone()
            .into_raw_vec()
    );
}
