use std::{sync::Arc, collections::HashMap, thread};
use ndarray::array;
use onnx_rust::{graph::{OnnxGraph, OnnxGraphNode, OnnxGraphInput, OnnxGraphOperation, OnnxGraphOutput, OnnxGraphInitializer, OnnxGraphIntermediate}, operations::{Operation, OpType}};

#[test]
fn basic_graph() {
    /* CREATE GRAPH */
    let mut graph = OnnxGraph::new();
    let x: OnnxGraphNode = OnnxGraphNode::Input(OnnxGraphInput::new("X"));
    let a = OnnxGraphNode::Input(OnnxGraphInput::new("A"));
    let b = OnnxGraphNode::Input(OnnxGraphInput::new("B"));

    let mut add_res = graph.add_node(x);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());
    add_res = graph.add_node(a);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());
    add_res = graph.add_node(b);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());

    let mat_mul =
        OnnxGraphNode::Operation(
            OnnxGraphOperation::new(
                "MatMul",
                Operation::new(OpType::MatMul),
                vec!["X", "A"],
                vec!["Add"]
            )
        );
    add_res = graph.add_node(mat_mul);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());

    let add =
        OnnxGraphNode::Operation(
            OnnxGraphOperation::new(
                "Add",
                Operation::new(OpType::Add),
                vec!["MatMul", "B"],
                vec!["Y"]
            )
        );
    add_res = graph.add_node(add);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());

    let y = OnnxGraphNode::Output(OnnxGraphOutput::new("Y"));
    add_res = graph.add_node(y);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());

    /* INPUT VALUES */
    let mut input_values = HashMap::new();

    let x_values = array![
        [1.,2.],
        [3.,4.],
        [5.,6.],
    ].into_dyn();
    assert!(input_values.insert("X".to_string(), x_values).is_none());

    let a_values = array![
        [1.,2.],
        [3.,4.]
    ].into_dyn();
    assert!(input_values.insert("A".to_string(), a_values).is_none());

    let b_values = array![
        [2.,3.],
        [4.,5.],
        [6.,7.],
    ].into_dyn();
    assert!(input_values.insert("B".to_string(), b_values).is_none());

    /* EXPECTED OUTPUT VALUES */
    let mut output_values = HashMap::new();

    let y = array![
            [9.,13.],
            [19.,27.],
            [29.,41.]
    ].into_dyn();
    output_values.insert("Y".to_string(), y);

    /* INFERENCE */
    let a_graph = Arc::new(graph);
    let result = a_graph.infer(input_values);

    assert!(result.is_ok(), "{:?}", result.unwrap_err());
    assert_eq!(result.unwrap(), output_values);
}

#[test]
fn correct_expected_shapes() {
    /* CRATE GRAPH */
    let mut graph = OnnxGraph::new();
    let x = OnnxGraphNode::Input(OnnxGraphInput::with_expected_shape("X", &[3,2]));
    let a = OnnxGraphNode::Input(OnnxGraphInput::with_expected_shape("A", &[2,2]));
    let b = OnnxGraphNode::Input(OnnxGraphInput::with_expected_shape("B", &[3,2]));

    let mut add_res = graph.add_node(x);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());
    add_res = graph.add_node(a);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());
    add_res = graph.add_node(b);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());

    let mat_mul =
        OnnxGraphNode::Operation(
            OnnxGraphOperation::new(
                "MatMul",
                Operation::new(OpType::MatMul),
                vec!["X", "A"],
                vec!["Add"]
            )
        );
    add_res = graph.add_node(mat_mul);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());

    let add =
        OnnxGraphNode::Operation(
            OnnxGraphOperation::new(
                "Add",
                Operation::new(OpType::Add),
                vec!["MatMul", "B"],
                vec!["Y"]
            )
        );
    add_res = graph.add_node(add);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());

    let y = OnnxGraphNode::Output(OnnxGraphOutput::with_expected_shape("Y", &[3,2]));
    add_res = graph.add_node(y);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());

    /* INPUT VALUES */
    let mut input_values = HashMap::new();

    let x_values = array![
        [1.,2.],
        [3.,4.],
        [5.,6.],
    ].into_dyn();
    assert!(input_values.insert("X".to_string(), x_values).is_none());

    let a_values = array![
        [1.,2.],
        [3.,4.]
    ].into_dyn();
    assert!(input_values.insert("A".to_string(), a_values).is_none());

    let b_values = array![
        [2.,3.],
        [4.,5.],
        [6.,7.],
    ].into_dyn();
    assert!(input_values.insert("B".to_string(), b_values).is_none());

    /* EXPECTED OUTPUT VALUES */
    let mut output_values = HashMap::new();

    let y = array![
        [9.,13.],
        [19.,27.],
        [29.,41.]
    ].into_dyn();
    output_values.insert("Y".to_string(), y);

    /* INFERENCE */
    let a_graph = Arc::new(graph);
    let result = a_graph.infer(input_values);

    assert!(result.is_ok(), "{:?}", result.unwrap_err());
    assert_eq!(result.unwrap(), output_values);
}

#[test]
fn wrong_input_shapes() {
    /* CREATE GRAPH */
    let mut graph = OnnxGraph::new();
    let x = OnnxGraphNode::Input(OnnxGraphInput::with_expected_shape("X", &[5,7]));
    let a = OnnxGraphNode::Input(OnnxGraphInput::with_expected_shape("A", &[6,3]));
    let b = OnnxGraphNode::Input(OnnxGraphInput::with_expected_shape("B", &[8,5]));

    let mut add_res = graph.add_node(x);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());
    add_res = graph.add_node(a);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());
    add_res = graph.add_node(b);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());

    let mat_mul =
        OnnxGraphNode::Operation(
            OnnxGraphOperation::new(
                "MatMul",
                Operation::new(OpType::MatMul),
                vec!["X", "A"],
                vec!["Add"]
            )
        );
    add_res = graph.add_node(mat_mul);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());

    let add =
        OnnxGraphNode::Operation(
            OnnxGraphOperation::new(
                "Add",
                Operation::new(OpType::Add),
                vec!["MatMul", "B"],
                vec!["Y"]
            )
        );
    add_res = graph.add_node(add);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());

    let y: OnnxGraphNode = OnnxGraphNode::Output(OnnxGraphOutput::with_expected_shape("Y", &[3,2]));
    add_res = graph.add_node(y);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());

    /* INPUT VALUES */
    let mut input_values = HashMap::new();

    let x_values = array![
        [1.,2.],
        [3.,4.],
        [5.,6.],
    ].into_dyn();
    assert!(input_values.insert("X".to_string(), x_values).is_none());

    let a_values = array![
        [1.,2.],
        [3.,4.]
    ].into_dyn();
    assert!(input_values.insert("A".to_string(), a_values).is_none());

    let b_values = array![
        [2.,3.],
        [4.,5.],
        [6.,7.],
    ].into_dyn();
    assert!(input_values.insert("B".to_string(), b_values).is_none());

    /* INFERENCE */
    let a_graph = Arc::new(graph);
    let result = a_graph.infer(input_values);

    assert!(result.is_err());
}

#[test]
fn wrong_output_shapes() {
    /* CREATE GRAPH */
    let mut graph = OnnxGraph::new();
    let x = OnnxGraphNode::Input(OnnxGraphInput::with_expected_shape("X", &[3,2]));
    let a = OnnxGraphNode::Input(OnnxGraphInput::with_expected_shape("A", &[2,2]));
    let b = OnnxGraphNode::Input(OnnxGraphInput::with_expected_shape("B", &[3,2]));

    let mut add_res = graph.add_node(x);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());
    add_res = graph.add_node(a);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());
    add_res = graph.add_node(b);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());

    let mat_mul =
        OnnxGraphNode::Operation(
            OnnxGraphOperation::new(
                "MatMul",
                Operation::new(OpType::MatMul),
                vec!["X", "A"],
                vec!["Add"]
            )
        );
    add_res = graph.add_node(mat_mul);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());

    let add =
        OnnxGraphNode::Operation(
            OnnxGraphOperation::new(
                "Add",
                Operation::new(OpType::Add),
                vec!["MatMul", "B"],
                vec!["Y"]
            )
        );
    add_res = graph.add_node(add);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());

    let y = OnnxGraphNode::Output(OnnxGraphOutput::with_expected_shape("Y", &[7,6]));
    add_res = graph.add_node(y);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());

    /* INPUT VALUES */
    let mut input_values = HashMap::new();

    let x_values = array![
        [1.,2.],
        [3.,4.],
        [5.,6.],
    ].into_dyn();
    assert!(input_values.insert("X".to_string(), x_values).is_none());

    let a_values = array![
        [1.,2.],
        [3.,4.]
    ].into_dyn();
    assert!(input_values.insert("A".to_string(), a_values).is_none());

    let b_values = array![
        [2.,3.],
        [4.,5.],
        [6.,7.],
    ].into_dyn();
    assert!(input_values.insert("B".to_string(), b_values).is_none());

    /* INFERENCE */
    let a_graph = Arc::new(graph);
    let result = a_graph.infer(input_values);

    assert!(result.is_err());
}

#[test]
fn common_input_node() {
    /* CREATE GRAPH */
    let mut graph = OnnxGraph::new();
    let x = OnnxGraphNode::Input(OnnxGraphInput::new("X"));
    let a = OnnxGraphNode::Input(OnnxGraphInput::new("A"));

    let mut add_res = graph.add_node(x);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());
    add_res = graph.add_node(a);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());

    let mat_mul =
        OnnxGraphNode::Operation(
            OnnxGraphOperation::new(
                "MatMul",
                Operation::new(OpType::MatMul),
                vec!["X", "A"],
                vec!["Add"]
            )
        );
    add_res = graph.add_node(mat_mul);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());

    let add =
        OnnxGraphNode::Operation(
            OnnxGraphOperation::new(
                "Add",
                Operation::new(OpType::Add),
                vec!["MatMul", "X"],
                vec!["Y"]
            )
        );
    add_res = graph.add_node(add);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());

    let y: OnnxGraphNode = OnnxGraphNode::Output(OnnxGraphOutput::new("Y"));
    add_res = graph.add_node(y);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());

    /* INPUT VALUES */
    let mut input_values = HashMap::new();

    let x_values = array![
        [1.,2.],
        [3.,4.],
        [5.,6.],
    ].into_dyn();
    assert!(input_values.insert("X".to_string(), x_values).is_none());

    let a_values = array![
        [1.,2.],
        [3.,4.]
    ].into_dyn();
    assert!(input_values.insert("A".to_string(), a_values).is_none());

    /* EXPECTED OUTPUT VALUES */
    let mut output_values = HashMap::new();

    let y = array![
        [8.,12.],
        [18.,26.],
        [28.,40.]
    ].into_dyn();
    output_values.insert("Y".to_string(), y);

    /* INFERENCE */
    let a_graph = Arc::new(graph);
    let result = a_graph.infer(input_values);

    assert!(result.is_ok(), "{:?}", result.unwrap_err());
    assert_eq!(result.unwrap(), output_values);
}

#[test]
fn use_input_default_value() {
    /* CREATE GRAPH */
    let mut graph = OnnxGraph::new();
    
    let x: OnnxGraphNode = OnnxGraphNode::Input(OnnxGraphInput::new("X"));
    let x_default = OnnxGraphNode::Initializer(
        OnnxGraphInitializer::new("X", array![
            [1.,2.],
            [3.,4.],
            [5.,6.],
        ].into_dyn())
    );
    let a = OnnxGraphNode::Input(OnnxGraphInput::new("A"));
    let b = OnnxGraphNode::Input(OnnxGraphInput::new("B"));

    let mut add_res = graph.add_node(x);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());
    add_res = graph.add_node(x_default);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());
    add_res = graph.add_node(a);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());
    add_res = graph.add_node(b);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());

    let mat_mul =
        OnnxGraphNode::Operation(
            OnnxGraphOperation::new(
                "MatMul",
                Operation::new(OpType::MatMul),
                vec!["X", "A"],
                vec!["Add"]
            )
        );
    add_res = graph.add_node(mat_mul);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());

    let add =
        OnnxGraphNode::Operation(
            OnnxGraphOperation::new(
                "Add",
                Operation::new(OpType::Add),
                vec!["MatMul", "B"],
                vec!["Y"]
            )
        );
    add_res = graph.add_node(add);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());

    let y = OnnxGraphNode::Output(OnnxGraphOutput::new("Y"));
    add_res = graph.add_node(y);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());

    /* INPUT VALUES */
    let mut input_values = HashMap::new();

    let a_values = array![
        [1.,2.],
        [3.,4.]
    ].into_dyn();
    assert!(input_values.insert("A".to_string(), a_values).is_none());

    let b_values = array![
        [2.,3.],
        [4.,5.],
        [6.,7.],
    ].into_dyn();
    assert!(input_values.insert("B".to_string(), b_values).is_none());

    /* EXPECTED OUTPUT VALUES */
    let mut output_values = HashMap::new();

    let y = array![
        [9.,13.],
        [19.,27.],
        [29.,41.]
    ].into_dyn();
    output_values.insert("Y".to_string(), y);

    /* INFERENCE */
    let a_graph = Arc::new(graph);
    let result = a_graph.infer(input_values);

    assert!(result.is_ok(), "{:?}", result.unwrap_err());
    assert_eq!(result.unwrap(), output_values);
}

#[test]
fn override_input_default_value() {
    /* CREATE GRAPH */
    let mut graph = OnnxGraph::new();
    
    let x: OnnxGraphNode = OnnxGraphNode::Input(OnnxGraphInput::new("X"));
    let x_default = OnnxGraphNode::Initializer(
        OnnxGraphInitializer::new("X", array![
            [1.,2.],
            [3.,4.],
            [5.,6.],
        ].into_dyn())
    );
    let a = OnnxGraphNode::Input(OnnxGraphInput::new("A"));
    let b = OnnxGraphNode::Input(OnnxGraphInput::new("B"));

    let mut add_res = graph.add_node(x);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());
    add_res = graph.add_node(x_default);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());
    add_res = graph.add_node(a);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());
    add_res = graph.add_node(b);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());

    let mat_mul =
        OnnxGraphNode::Operation(
            OnnxGraphOperation::new(
                "MatMul",
                Operation::new(OpType::MatMul),
                vec!["X", "A"],
                vec!["Add"]
            )
        );
    add_res = graph.add_node(mat_mul);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());

    let add =
        OnnxGraphNode::Operation(
            OnnxGraphOperation::new(
                "Add",
                Operation::new(OpType::Add),
                vec!["MatMul", "B"],
                vec!["Y"]
            )
        );
    add_res = graph.add_node(add);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());

    let y = OnnxGraphNode::Output(OnnxGraphOutput::new("Y"));
    add_res = graph.add_node(y);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());

    /* INPUT VALUES */
    let mut input_values = HashMap::new();

    let x_values = array![
        [2.,3.],
        [4.,5.],
        [6.,7.],
    ].into_dyn();
    assert!(input_values.insert("X".to_string(), x_values).is_none());

    let a_values = array![
        [1.,2.],
        [3.,4.]
    ].into_dyn();
    assert!(input_values.insert("A".to_string(), a_values).is_none());

    let b_values = array![
        [2.,3.],
        [4.,5.],
        [6.,7.],
    ].into_dyn();
    assert!(input_values.insert("B".to_string(), b_values).is_none());

    /* EXPECTED OUTPUT VALUES */
    let mut output_values = HashMap::new();

    let y = array![
        [13.,19.],
        [23.,33.],
        [33.,47.]
    ].into_dyn();
    output_values.insert("Y".to_string(), y);

    /* INFERENCE */
    let a_graph = Arc::new(graph);
    let result = a_graph.infer(input_values);

    assert!(result.is_ok(), "{:?}", result.unwrap_err());
    assert_eq!(result.unwrap(), output_values);
}

#[test]
fn missing_input_value() {
    /* CREATE GRAPH */
    let mut graph = OnnxGraph::new();

    let x: OnnxGraphNode = OnnxGraphNode::Input(OnnxGraphInput::new("X"));
    let a = OnnxGraphNode::Input(OnnxGraphInput::new("A"));
    let b = OnnxGraphNode::Input(OnnxGraphInput::new("B"));

    let mut add_res = graph.add_node(x);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());
    add_res = graph.add_node(a);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());
    add_res = graph.add_node(b);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());

    let mat_mul =
        OnnxGraphNode::Operation(
            OnnxGraphOperation::new(
                "MatMul",
                Operation::new(OpType::MatMul),
                vec!["X", "A"],
                vec!["Add"]
            )
        );
    add_res = graph.add_node(mat_mul);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());

    let add =
        OnnxGraphNode::Operation(
            OnnxGraphOperation::new(
                "Add",
                Operation::new(OpType::Add),
                vec!["MatMul", "B"],
                vec!["Y"]
            )
        );
    add_res = graph.add_node(add);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());

    let y = OnnxGraphNode::Output(OnnxGraphOutput::new("Y"));
    add_res = graph.add_node(y);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());

    /* INPUT VALUES */
    let mut input_values = HashMap::new();

    let a_values = array![
        [1.,2.],
        [3.,4.]
    ].into_dyn();
    assert!(input_values.insert("A".to_string(), a_values).is_none());

    let b_values = array![
        [2.,3.],
        [4.,5.],
        [6.,7.],
    ].into_dyn();
    assert!(input_values.insert("B".to_string(), b_values).is_none());

    /* INFERENCE */
    let a_graph = Arc::new(graph);
    let result = a_graph.infer(input_values);

    assert!(result.is_err());
}

#[test]
fn multiple_sequential_inferences() {
    /* CREATE GRAPH */
    let mut graph = OnnxGraph::new();
    let x = OnnxGraphNode::Input(OnnxGraphInput::new("X"));
    let a = OnnxGraphNode::Input(OnnxGraphInput::new("A"));
    let b = OnnxGraphNode::Input(OnnxGraphInput::new("B"));

    let mut add_res = graph.add_node(x);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());
    add_res = graph.add_node(a);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());
    add_res = graph.add_node(b);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());

    let mat_mul =
        OnnxGraphNode::Operation(
            OnnxGraphOperation::new(
                "MatMul",
                Operation::new(OpType::MatMul),
                vec!["X", "A"],
                vec!["Add"]
            )
        );
    add_res = graph.add_node(mat_mul);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());

    let add =
        OnnxGraphNode::Operation(
            OnnxGraphOperation::new(
                "Add",
                Operation::new(OpType::Add),
                vec!["MatMul", "B"],
                vec!["Y"]
            )
        );
    add_res = graph.add_node(add);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());

    let y = OnnxGraphNode::Output(OnnxGraphOutput::new("Y"));
    add_res = graph.add_node(y);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());

    /* INPUT VALUES #1 */
    let mut input_values_1 = HashMap::new();

    let x_values = array![
        [1.,2.],
        [3.,4.],
        [5.,6.],
    ].into_dyn();
    assert!(input_values_1.insert("X".to_string(), x_values).is_none());

    let a_values = array![
        [1.,2.],
        [3.,4.]
    ].into_dyn();
    assert!(input_values_1.insert("A".to_string(), a_values).is_none());

    let b_values = array![
        [2.,3.],
        [4.,5.],
        [6.,7.],
    ].into_dyn();
    assert!(input_values_1.insert("B".to_string(), b_values).is_none());

    /* INPUT VALUES #2 */
    let mut input_values_2 = HashMap::new();

    let x_values = array![
        [2.,3.],
        [4.,5.],
        [6.,7.],
    ].into_dyn();
    assert!(input_values_2.insert("X".to_string(), x_values).is_none());

    let a_values = array![
        [2.,3.],
        [4.,5.]
    ].into_dyn();
    assert!(input_values_2.insert("A".to_string(), a_values).is_none());

    let b_values = array![
        [3.,4.],
        [5.,6.],
        [7.,8.],
    ].into_dyn();
    assert!(input_values_2.insert("B".to_string(), b_values).is_none());

    /* EXPECTED OUTPUT VALUES */
    let mut output_values_1 = HashMap::new();
    let y1 = array![
            [9.,13.],
            [19.,27.],
            [29.,41.]
    ].into_dyn();
    output_values_1.insert("Y".to_string(), y1);

    let mut output_values_2 = HashMap::new();
    let y2 = array![
            [19.,25.],
            [33.,43.],
            [47.,61.]
    ].into_dyn();
    output_values_2.insert("Y".to_string(), y2);

    /* INFERENCES */
    let a_graph = Arc::new(graph);

    let result_1 = a_graph.clone().infer(input_values_1);
    assert!(result_1.is_ok(), "{:?}", result_1.unwrap_err());
    assert_eq!(result_1.unwrap(), output_values_1);
    
    let result_2 = a_graph.infer(input_values_2);
    assert!(result_2.is_ok(), "{:?}", result_2.unwrap_err());    
    assert_eq!(result_2.unwrap(), output_values_2);
}


#[test]
fn multiple_parallel_inferences() {
    /* CREATE GRAPH */
    let mut graph = OnnxGraph::new();
    let x = OnnxGraphNode::Input(OnnxGraphInput::new("X"));
    let a = OnnxGraphNode::Input(OnnxGraphInput::new("A"));
    let b = OnnxGraphNode::Input(OnnxGraphInput::new("B"));

    let mut add_res = graph.add_node(x);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());
    add_res = graph.add_node(a);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());
    add_res = graph.add_node(b);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());

    let mat_mul =
        OnnxGraphNode::Operation(
            OnnxGraphOperation::new(
                "MatMul",
                Operation::new(OpType::MatMul),
                vec!["X", "A"],
                vec!["Add"]
            )
        );
    add_res = graph.add_node(mat_mul);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());

    let add =
        OnnxGraphNode::Operation(
            OnnxGraphOperation::new(
                "Add",
                Operation::new(OpType::Add),
                vec!["MatMul", "B"],
                vec!["Y"]
            )
        );
    add_res = graph.add_node(add);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());

    let y = OnnxGraphNode::Output(OnnxGraphOutput::new("Y"));
    add_res = graph.add_node(y);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());

    /* INPUT VALUES #1 */
    let mut input_values_1 = HashMap::new();

    let x_values = array![
        [1.,2.],
        [3.,4.],
        [5.,6.],
    ].into_dyn();
    assert!(input_values_1.insert("X".to_string(), x_values).is_none());

    let a_values = array![
        [1.,2.],
        [3.,4.]
    ].into_dyn();
    assert!(input_values_1.insert("A".to_string(), a_values).is_none());

    let b_values = array![
        [2.,3.],
        [4.,5.],
        [6.,7.],
    ].into_dyn();
    assert!(input_values_1.insert("B".to_string(), b_values).is_none());

    /* INPUT VALUES #2 */
    let mut input_values_2 = HashMap::new();

    let x_values = array![
        [2.,3.],
        [4.,5.],
        [6.,7.],
    ].into_dyn();
    assert!(input_values_2.insert("X".to_string(), x_values).is_none());

    let a_values = array![
        [2.,3.],
        [4.,5.]
    ].into_dyn();
    assert!(input_values_2.insert("A".to_string(), a_values).is_none());

    let b_values = array![
        [3.,4.],
        [5.,6.],
        [7.,8.],
    ].into_dyn();
    assert!(input_values_2.insert("B".to_string(), b_values).is_none());

    /* EXPECTED OUTPUT VALUES */
    let mut output_values_1 = HashMap::new();
    let y1 = array![
            [9.,13.],
            [19.,27.],
            [29.,41.]
    ].into_dyn();
    output_values_1.insert("Y".to_string(), y1);

    let mut output_values_2 = HashMap::new();
    let y2 = array![
            [19.,25.],
            [33.,43.],
            [47.,61.]
    ].into_dyn();
    output_values_2.insert("Y".to_string(), y2);

    /* INFERENCES */
    let a_graph_1 = Arc::new(graph);
    let a_graph_2 = a_graph_1.clone();

    let jh1 = thread::spawn(move || a_graph_1.infer(input_values_1));
    let jh2 = thread::spawn(move || a_graph_2.infer(input_values_2));

    let result_1 = jh1.join().unwrap();
    let result_2 = jh2.join().unwrap();

    assert!(result_1.is_ok(), "{:?}", result_1.unwrap_err());
    assert!(result_2.is_ok(), "{:?}", result_2.unwrap_err());

    assert_eq!(result_1.unwrap(), output_values_1);
    assert_eq!(result_2.unwrap(), output_values_2);
}

#[test]
fn complex_graph() {
    /* CREATE GRAPH */
    let mut graph = OnnxGraph::new();
    let x = OnnxGraphNode::Input(OnnxGraphInput::new("X"));
    let a = OnnxGraphNode::Input(OnnxGraphInput::new("A"));
    let b = OnnxGraphNode::Input(OnnxGraphInput::new("B"));
    
    let mut add_res = graph.add_node(x);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());
    add_res = graph.add_node(a);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());
    add_res = graph.add_node(b);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());

    let mat_mul_1 =
        OnnxGraphNode::Operation(
            OnnxGraphOperation::new(
                "MatMul1",
                Operation::new(OpType::MatMul),
                vec!["A", "B"],
                vec!["Add2", "Y1"]
            )
        );
    add_res = graph.add_node(mat_mul_1);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());

    let add_1 =
        OnnxGraphNode::Operation(
            OnnxGraphOperation::new(
                "Add1",
                Operation::new(OpType::Add),
                vec!["A", "B"],
                vec!["MatMul2", "Add2"]
            )
        );
    add_res = graph.add_node(add_1);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());

    let mat_mul_2 =
        OnnxGraphNode::Operation(
            OnnxGraphOperation::new(
                "MatMul2",
                Operation::new(OpType::MatMul),
                vec!["X", "Add1"],
                vec!["MatMul3"]
            )
        );
    add_res = graph.add_node(mat_mul_2);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());

    let add_2 =
        OnnxGraphNode::Operation(
            OnnxGraphOperation::new(
                "Add2",
                Operation::new(OpType::Add),
                vec!["Add1", "MatMul1"],
                vec!["MatMul3", "Y2"]
            )
        );
    add_res = graph.add_node(add_2);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());

    let mat_mul_3 =
        OnnxGraphNode::Operation(
            OnnxGraphOperation::new(
                "MatMul3",
                Operation::new(OpType::MatMul),
                vec!["MatMul2", "Add2"],
                vec!["Y3"]
            )
        );
    add_res = graph.add_node(mat_mul_3);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());

    let y1 = OnnxGraphNode::Output(OnnxGraphOutput::new("Y1"));
    add_res = graph.add_node(y1);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());

    let y2 = OnnxGraphNode::Output(OnnxGraphOutput::new("Y2"));
    add_res = graph.add_node(y2);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());

    let y3 = OnnxGraphNode::Output(OnnxGraphOutput::new("Y3"));
    add_res = graph.add_node(y3);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());

    /* INPUT VALUES */
    let mut input_values = HashMap::new();

    let x_values = array![
        [1.,2.],
        [3.,4.],
    ].into_dyn();
    assert!(input_values.insert("X".to_string(), x_values).is_none());

    let a_values = array![
        [2.,3.],
        [4.,5.]
    ].into_dyn();
    assert!(input_values.insert("A".to_string(), a_values).is_none());

    let b_values = array![
        [3.,4.],
        [5.,6.],
    ].into_dyn();
    assert!(input_values.insert("B".to_string(), b_values).is_none());

    /* OUTPUT VALUES */
    let mut output_values = HashMap::new();

    let y1_values = array![
        [21.,26.],
        [37.,46.],
    ].into_dyn();
    assert!(output_values.insert("Y1".to_string(), y1_values).is_none());

    let y2_values = array![
        [26.,33.],
        [46.,57.],
    ].into_dyn();
    assert!(output_values.insert("Y2".to_string(), y2_values).is_none());

    let y3_values = array![
        [1932.,2412.],
        [4316.,5388.],
    ].into_dyn();
    assert!(output_values.insert("Y3".to_string(), y3_values).is_none());

    /* INFERENCE */
    let a_graph = Arc::new(graph);
    let result = a_graph.infer(input_values);

    assert!(result.is_ok(), "{:?}", result.unwrap_err());
    assert_eq!(result.unwrap(), output_values);
}

#[test]
fn graph_with_intermediate() {
    /* CREATE GRAPH */
    let mut graph = OnnxGraph::new();
    let x: OnnxGraphNode = OnnxGraphNode::Input(OnnxGraphInput::new("X"));
    let a = OnnxGraphNode::Input(OnnxGraphInput::new("A"));
    let b = OnnxGraphNode::Input(OnnxGraphInput::new("B"));

    let mut add_res = graph.add_node(x);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());
    add_res = graph.add_node(a);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());
    add_res = graph.add_node(b);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());

    let mat_mul =
        OnnxGraphNode::Operation(
            OnnxGraphOperation::new(
                "MatMul",
                Operation::new(OpType::MatMul),
                vec!["X", "A"],
                vec!["MatMul_out"]
            )
        );
    add_res = graph.add_node(mat_mul);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());

    let add =
        OnnxGraphNode::Operation(
            OnnxGraphOperation::new(
                "Add",
                Operation::new(OpType::Add),
                vec!["MatMul_out", "B"],
                vec!["Y"]
            )
        );
    add_res = graph.add_node(add);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());

    let mat_mul_out = OnnxGraphNode::Intermediate(OnnxGraphIntermediate::new("MatMul_out", "MatMul".to_string(), vec!["Add".to_string()]));
    add_res = graph.add_node(mat_mul_out);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());

    let y = OnnxGraphNode::Output(OnnxGraphOutput::new("Y"));
    add_res = graph.add_node(y);
    assert!(add_res.is_ok(), "{:?}", add_res.unwrap_err());

    /* INPUT VALUES */
    let mut input_values = HashMap::new();

    let x_values = array![
        [1.,2.],
        [3.,4.],
        [5.,6.],
    ].into_dyn();
    assert!(input_values.insert("X".to_string(), x_values).is_none());

    let a_values = array![
        [1.,2.],
        [3.,4.]
    ].into_dyn();
    assert!(input_values.insert("A".to_string(), a_values).is_none());

    let b_values = array![
        [2.,3.],
        [4.,5.],
        [6.,7.],
    ].into_dyn();
    assert!(input_values.insert("B".to_string(), b_values).is_none());

    /* EXPECTED OUTPUT VALUES */
    let mut output_values = HashMap::new();

    let y = array![
            [9.,13.],
            [19.,27.],
            [29.,41.]
    ].into_dyn();
    output_values.insert("Y".to_string(), y);

    /* INFERENCE */
    let a_graph = Arc::new(graph);
    let result = a_graph.infer(input_values);

    assert!(result.is_ok(), "{:?}", result.unwrap_err());
    assert_eq!(result.unwrap(), output_values);
}