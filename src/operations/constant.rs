use std::sync::Arc;
use super::{Operation, OnnxError, onnx_error, Tensor, OperationResult, Attribute};

impl Operation {

    pub(super) fn execute_constant(&self, _inputs: Vec<&Tensor>) -> OperationResult {        
        // L'output è l'attributo "value"
        match self.attributes.get("value") {
            Some(Attribute::Tensor(val)) => Ok(Arc::new(val.clone())),
            None => Err(onnx_error!("Value attribute not specified.")),
            _ => Err(onnx_error!("Value has an invalid attribute type."))
        }
    }

}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, collections::HashMap};
    use ndarray::Array;
    use crate::operations::{Operation, OpType, Attribute};

    #[test]
    fn valid() {
        let val =
            Array::from_iter((1..=12).map(|v| v as f32).cycle().take(4*2))
            .into_shape((4,2))
            .unwrap().into_dyn();

        let attributes = HashMap::from([
            ("value".to_string(), Attribute::Tensor(val.clone()))
        ]);
        let op = Operation::with_attributes(OpType::Constant, attributes);

        let result = op.execute(vec![]);
        assert!(result.is_ok(), "{:?}", result.unwrap_err());
        assert_eq!(result.unwrap(), Arc::new(val))
    }

    #[test]
    fn invalid_attribute_type() {
        let attr_val = Attribute::Int(1);
        let op = Operation::with_attributes(OpType::Constant, HashMap::from([("value".to_string(), attr_val)]));
        let result = op.execute(vec![]);
        assert!(result.is_err());
    }

}