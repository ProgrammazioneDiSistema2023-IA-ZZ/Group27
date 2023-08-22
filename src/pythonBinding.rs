
use std::collections::HashMap;
use std::ops::Deref;
use std::sync::Arc;
use std::sync::RwLock;

use crate::fileparser::fileparser::OnnxFileParser;
use log::info;
use ndarray::Array;
use ndarray::ArrayD;
use ndarray::OwnedRepr;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::*;


use crate::graph::OnnxGraph;
use pyo3::create_exception;
use pyo3_log::*;
use ndarray;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayDyn, PyReadonlyArrayDyn};


/* 
#[pyfunction]
pub fn leggi_file(str : &PyString,_py: Python<'_>)->Result<OnnxGraph,PyErr>{
    let res =OnnxGraph::from_file(str.to_string_lossy().to_mut());
    if res.is_err(){
       return Err(PyTypeError::new_err("err"));
      
    }
    return Ok(res.unwrap());
}*/


#[pyfunction]
pub fn leggi_file_dati(str : &PyString,_py: Python<'_>)->Result< PyObject,PyErr>{
    
    let res =OnnxFileParser::parse_data(str.to_string_lossy().to_mut());
    if res.is_err(){
       return Err(PyTypeError::new_err(res.err().unwrap()));
    }
    let dict = PyDict::new(_py);
    for key in res.clone().unwrap().keys(){
        let value = res.clone().unwrap().get(key).unwrap().clone();
        dict.set_item(key, PyArrayDyn::from_array(_py, &value));
       
    }
    let  obj = dict.to_object(_py);
    return Ok(obj);
}

#[pyfunction]
pub fn interferenza(model: &PyString,input_data :&PyString,_py: Python<'_>)->Result< PyObject,PyErr>{
 
 
    let graph = match OnnxGraph::from_file(model.to_string_lossy().to_string().as_str()) {
        Ok(graph) => Arc::new(graph),
        Err(e) => {
            println!("Could not read graph: {}", e.msg);
            return Err(PyTypeError::new_err(String::from("Could not read graph: ").push_str( &e.msg)));
        }
    };
    let input =OnnxFileParser::parse_data(input_data.to_string_lossy().to_mut());
    if input.is_err(){
       return Err(PyTypeError::new_err(String::from("Could not read input data: ").push_str( &input.err().unwrap())));
    }
   
  let result= _py.allow_threads(||{
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
        dict.set_item(key, PyArrayDyn::from_array(_py, &value));
       
    }
    return Ok(dict.to_object(_py)); 
}