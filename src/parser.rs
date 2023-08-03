use crate::graph::OnnxGraph;
use std::iter::Peekable;

#[derive(Debug)]
pub struct OnnxParseError {
    msg: String, 
    line: usize
}

impl OnnxParseError {
    pub fn new(msg: String, line: usize) -> Self {
        Self { msg, line }
    }
}

pub type OnnxParseResult<'a> = Result<OnnxGraph, OnnxParseError>;

pub struct OnnxParser<'a, I: Iterator<Item = &'a str>> {
    iter: Peekable<I>
}

impl<'a, I: Iterator<Item = &'a str>> OnnxParser<'a, I> {

    pub fn new(iter: I) -> Self {
        Self { iter: iter.peekable() }
    }

    pub fn parse(&mut self) -> OnnxParseResult {
        todo!("Costruisci grafo a partire da un iteratore sulle singole linee.");
    }

}