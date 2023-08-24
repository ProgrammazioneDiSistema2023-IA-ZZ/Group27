pub mod graph;
pub mod error;
pub mod operations;
pub mod helper;
pub mod fileparser;
pub mod python_binding;



/* part of rust binding to python */
use pyo3::prelude::*;
use graph::OnnxGraph;
use pyo3_log::{Logger, Caching};
use python_binding::*;



///functions exported to python
#[pymodule]
fn onnx_rust(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
   let _ = Logger::new(_py, Caching::LoggersAndLevels)?
   .install();
    m.add_function(wrap_pyfunction!(read_data_file, m)?)?;
    m.add_function(wrap_pyfunction!(inference, m)?)?;
    m.add_class::<OnnxGraph>()?;
    Ok(())
}