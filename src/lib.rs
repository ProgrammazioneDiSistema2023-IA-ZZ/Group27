pub mod graph;
pub mod error;
pub mod operations;
pub mod helper;
pub mod fileparser;
pub mod pythonBinding;



use graph::OnnxGraph;
/* part of rust binding to python */
use pyo3::prelude::*;

use pyo3_log::{Logger, Caching};
use pythonBinding::*;



///functions exported to python
#[pymodule]
fn onnx_rust(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
   // pyo3_log::init();
   let _ = Logger::new(_py, Caching::LoggersAndLevels)?
   .install();
   // m.add_function(wrap_pyfunction!(leggi_file, m)?)?;
    m.add_function(wrap_pyfunction!(leggi_file_dati, m)?)?;
    m.add_function(wrap_pyfunction!(interferenza, m)?)?;
    m.add_class::<OnnxGraph>()?;
    Ok(())
}