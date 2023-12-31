use crate::helper::Shape;

use super::EngineError;

pub fn return_if_matched_shape<T>(a: &Shape, b: &Shape, out: T) -> Result<T, EngineError> {
    if a == b {
        Ok(out)
    } else {
        Err(EngineError::ShapeMismatch(a.clone(), b.clone()))
    }
}

pub fn err_if_incorrect_dimensions(a: &Shape, expected_dims: usize) -> Result<(), EngineError> {
    let a_dims = a.dims();

    if a_dims == expected_dims {
        Ok(())
    } else {
        Err(EngineError::IncorrectDimensions(a_dims, expected_dims))
    }
}

pub fn err_if_dimension_mismatch(provided_dim: usize, expected_dim: usize) -> Result<(), EngineError> {
    if provided_dim == expected_dim {
        Ok(())
    } else {
        Err(EngineError::DimensionMismatch(provided_dim, expected_dim))
    }
}