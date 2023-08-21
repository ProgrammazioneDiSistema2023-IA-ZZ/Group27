use std::{sync::Arc, io::{stdout, Write}};
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
    print!("Reading model from path `{}`...", args.model);
    if args.verbose > 0 { print!("\n"); } else { stdout().flush().unwrap(); }
    let graph = match OnnxGraph::from_file(&args.model) {
        Ok(graph) => Arc::new(graph),
        Err(e) => {
            log::error!("Could not read graph: {}", e.msg);
            return;
        }
    };
    if args.verbose > 0 {
        println!("Successfully read graph {}!", args.model);
    } else {
        println!("\rReading model from path `{}`... \x1b[1m\x1b[32m\u{2713}\x1b[0m", args.model);
    }

    // Read input data
    print!("Reading input data from path `{}`...", args.input);
    if args.verbose > 0 { print!("\n"); } else { stdout().flush().unwrap(); }
    let inputs = match OnnxFileParser::parse_data(args.input.as_str()) {
        Ok(data) => data,
        Err(e) => {
            log::error!("Could not read input data: {e}");
            return;
        }
    };
    if args.verbose > 0 {
        println!("Successfully read input data!");
    } else {
        println!("\rReading input data from path `{}`... \x1b[1m\x1b[32m\u{2713}\x1b[0m", args.input);
    }

    // Perform inference
    print!("Performing inference...");
    if args.verbose > 0 { print!("\n"); } else { stdout().flush().unwrap(); }
    let outputs = match graph.infer(inputs) {
        Ok(outputs) => outputs,
        Err(e) => {
            log::error!("An error occurred during inference: {}", e.msg);
            return;
        }
    };
    if args.verbose > 0 {
        println!("Inference finished!");
    } else {
        println!("\rPerforming inference... \x1b[1m\x1b[32m\u{2713}\x1b[0m");
    }

    // Read outputs if included
    let expected_outputs = if args.output.is_some() {
        let output = args.output.unwrap();
        print!("Reading output data from path `{}`...", output);
        if args.verbose > 0 { print!("\n"); } else { stdout().flush().unwrap(); }
        let outputs = match OnnxFileParser::parse_data(output.as_str()) {
            Ok(data) => data,
            Err(e) => {
                log::error!("Could not read output data: {e}");
                return;
            }
        };
        if args.verbose > 0 {
            println!("Successfully read output data!");
        } else {
            println!("\rReading output data from path `{}`... \x1b[1m\x1b[32m\u{2713}\x1b[0m", output);
        }
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
