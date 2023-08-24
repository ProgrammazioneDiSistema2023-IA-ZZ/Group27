use std::{sync::Arc, collections::HashMap};

use onnx_rust::{graph::OnnxGraph, fileparser::fileparser::OnnxFileParser, operations::Tensor};

fn round_data(input: HashMap<String, Tensor>) -> HashMap<String, Tensor> {
    input
        .into_iter()
        .map(|(name, values)| (
            name,
            values.mapv(|v| {
                // Round number to 3 significant digits
                let pow = 10f32.powf(v.abs().log10().ceil());
                let mant = v / pow;
                (mant * 10e2).round() / 10e2 * pow
            })
        ))
        .collect()
}

#[test]
fn mnist() {
    let graph_res = OnnxGraph::from_file("./onnxFile/mnist-12/mnist-12.onnx");
    assert!(graph_res.is_ok(), "{:?}", graph_res.err().unwrap());
    let graph = graph_res.unwrap();

    let inputs_res = OnnxFileParser::parse_data("./onnxFile/mnist-12/test_data_set_0/input_0.pb");
    assert!(inputs_res.is_ok(), "{:?}", inputs_res.unwrap_err());
    let inputs = inputs_res.unwrap();

    let outputs_res = Arc::new(graph).infer(inputs);
    assert!(outputs_res.is_ok(), "{:?}", outputs_res.unwrap_err());
    let outputs = outputs_res.unwrap();

    let expected_res = OnnxFileParser::parse_data("./onnxFile/mnist-12/test_data_set_0/output_0.pb");
    assert!(expected_res.is_ok(), "{:?}", expected_res.unwrap_err());
    let expected = expected_res.unwrap();

    assert_eq!(round_data(outputs), round_data(expected));
}

#[test]
fn googlenet() {
    let graph_res = OnnxGraph::from_file("./onnxFile/googlenet-12/googlenet-12.onnx");
    assert!(graph_res.is_ok(), "{:?}", graph_res.err().unwrap());
    let graph = graph_res.unwrap();

    let inputs_res = OnnxFileParser::parse_data("./onnxFile/googlenet-12/test_data_set_0/input_0.pb");
    assert!(inputs_res.is_ok(), "{:?}", inputs_res.unwrap_err());
    let inputs = inputs_res.unwrap();

    let outputs_res = Arc::new(graph).infer(inputs);
    assert!(outputs_res.is_ok(), "{:?}", outputs_res.unwrap_err());
    let outputs = outputs_res.unwrap();

    let expected_res = OnnxFileParser::parse_data("./onnxFile/googlenet-12/test_data_set_0/output_0.pb");
    assert!(expected_res.is_ok(), "{:?}", expected_res.unwrap_err());
    let expected = expected_res.unwrap();

    assert_eq!(round_data(outputs), round_data(expected));
}