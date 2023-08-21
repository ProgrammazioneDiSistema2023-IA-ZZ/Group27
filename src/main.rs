use std::sync::Arc;
use clap::{Parser, ArgAction::Count};
use itertools::Itertools;
use log::LevelFilter;
use onnx_rust::{fileparser::fileparser::OnnxFileParser, graph::OnnxGraph, helper::PrettyTensor};
use prettytable::ptable;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
#[command(next_line_help = true)]
struct Args {
    /// Path to file containing the model, saved using the ONNX format.
    model: String,

    /// Path to file containing input data for the model.
    #[arg(short, long)]
    input: String,

    /// Path to file containing expected output data, used for verification.
    #[arg(short, long)]
    output: Option<String>,

    #[arg(long, short, action = Count)]
    /// Show additional information while reading the model and performing the inference.
    /// 
    /// Use -vv to display even more information.
    verbose: u8
}

fn main() {
    // Parse arguments
    let args = Args::parse();

    // Initialize logger
    let log_level = match args.verbose {
        0 => LevelFilter::Error,
        1 => LevelFilter::Info,
        _ => LevelFilter::Debug
    };
    env_logger::builder()
        .filter_level(log_level)
        .format_timestamp(None)
        .init();

    // Parse input model
    log::info!("Attempting to read model from path `{}`...", args.model);
    let graph = match OnnxGraph::from_file(&args.model) {
        Ok(graph) => Arc::new(graph),
        Err(e) => {
            log::error!("Could not read graph: {}", e.msg);
            return;
        }
    };
    log::info!("Successfully read graph {}!", graph.name);

    // Read input data
    log::info!("Attempting to read input data from path `{}`...", args.input);
    let inputs = match OnnxFileParser::parse_data(args.input.as_str()) {
        Ok(data) => data,
        Err(e) => {
            log::error!("Could not read input data: {e}");
            return;
        }
    };
    log::info!("Successfully read input data!");
    inputs.values().for_each(|input| log::debug!("\n{}", PrettyTensor::from(input)));

    // Perform inference
    log::info!("Attempting to perform inference on graph {} with input data...", graph.name);
    let outputs = match graph.infer(inputs) {
        Ok(outputs) => outputs,
        Err(e) => {
            log::error!("An error occurred during inference: {}", e.msg);
            return;
        }
    };

    // Read outputs if included
    let expected_outputs = if args.output.is_some() {
        let output = args.output.unwrap();
        log::info!("Attempting to read output data from path `{}`...", output);
        let outputs = match OnnxFileParser::parse_data(output.as_str()) {
            Ok(data) => data,
            Err(e) => {
                log::error!("Could not read output data: {e}");
                return;
            }
        };
        log::info!("Successfully read output data!");
        Some(outputs)
    } else {
        None
    };
    
    // Print outputs
    for (name, value) in outputs {
        let shape = value.shape().iter().join("x");
        if let Some(expected_value) = expected_outputs.as_ref().and_then(|map| map.get(&name)) {
            // Expected output given: print both values side-by-side
            let expected_shape = expected_value.shape().iter().join("x");
            ptable!(
                [ bFrH2c->name ],
                [ format!("Expected [{expected_shape}]"), format!("Actual [{shape}]") ],
                [ format!("{}", PrettyTensor::from(expected_value)), format!("{}", PrettyTensor::from(&value)) ]
            );
        } else {
            // Expected output not given: print actual value only
            ptable!(
                [ bFrc->format!("{name}") ],
                [ format!("Shape: [{shape}]") ],
                [ format!("{}", PrettyTensor::from(&value)) ]
            );
        }
    }
}
