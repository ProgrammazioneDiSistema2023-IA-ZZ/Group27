
use std::sync::Arc;
use pyo3::exceptions::PyTypeError;

use numpy::PyArrayDyn;
use pyo3::{pyfunction, Python, PyObject, PyErr, ToPyObject};
use pyo3::types::{PyDict, PyString};

use crate::graph::OnnxGraph;
use crate::fileparser::fileparser::OnnxFileParser;



#[pyfunction]
pub fn read_data_file(str : &PyString,_py: Python<'_>)->Result< PyObject,PyErr>{
    
    let res =OnnxFileParser::parse_data(str.to_string_lossy().to_mut());
    if res.is_err(){
       return Err(PyTypeError::new_err(res.err().unwrap()));
    }
    let dict = PyDict::new(_py);
    for key in res.clone().unwrap().keys(){
        let value = res.clone().unwrap().get(key).unwrap().clone();
        dict.set_item(key, PyArrayDyn::from_array(_py, &value))?;
       
    }
    let  obj = dict.to_object(_py);
    return Ok(obj);
}

#[pyfunction]
pub fn inference(model: &PyString,input_data :&PyString,_py: Python<'_>)->Result< PyObject,PyErr>{
 
    //Reading graph
    let graph = match OnnxGraph::from_file(model.to_string_lossy().to_string().as_str()) {
        Ok(graph) => Arc::new(graph),
        Err(e) => {
            println!("Could not read graph: {}", e.msg);
            return Err(PyTypeError::new_err(String::from("Could not read graph: ").push_str( &e.msg)));
        }
    };
    //reading inputfile
    let input =OnnxFileParser::parse_data(input_data.to_string_lossy().to_mut());
    if input.is_err(){
       return Err(PyTypeError::new_err(String::from("Could not read input data: ").push_str( &input.err().unwrap())));
    }
   
    //inference
  let result= _py.allow_threads(||{     //allow_threads need for handle correctly log print
    return  graph.infer(input.unwrap())
   });

    let outputs = match result {
        Ok(outputs) => outputs,
        Err(e) => {
            return Err(PyTypeError::new_err(String::from("An error occurred during inference:").push_str( &e.msg)));
        
        }
    };
   
    let dict = PyDict::new(_py);
    for key in outputs.clone().keys(){
        let value = outputs.clone().get(key).unwrap().clone();
        dict.set_item(key, PyArrayDyn::from_array(_py, &value))?;
       
    }
    return Ok(dict.to_object(_py)); 
}