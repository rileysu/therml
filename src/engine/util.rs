use crate::helper::Shape;

use super::EngineError;

pub fn return_if_matched_shape<T>(a: &Shape, b: &Shape, out: T) -> Result<T, EngineError> {
    if a == b {
        Ok(out)
    } else {
        Err(EngineError::ShapeMismatch(a.clone(), b.clone()))
    }
}