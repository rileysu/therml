use num::Signed;

use crate::engine::{Engine, basic::Basic, EngineError};

use super::{EngineTensor, factory::EngineTensorFactory};

impl<E: EngineTensorFactory> Engine for Basic<E> 
    where E::Unit: Signed {
    type Unit = E::Unit;

    fn abs(a: &EngineTensor<Self::Unit>) -> Result<EngineTensor<Self::Unit>, crate::engine::EngineError> {
        Ok(E::from_iter(&mut a.iter().map(|x| x.abs()), a.shape.clone()))
    }

    fn neg(a: &EngineTensor<Self::Unit>) -> Result<EngineTensor<Self::Unit>, crate::engine::EngineError> {
        Ok(E::from_iter(&mut a.iter().map(|x| -x), a.shape.clone()))
    }

    fn add(a: &EngineTensor<Self::Unit>, b: &EngineTensor<Self::Unit>) -> Result<EngineTensor<Self::Unit>, crate::engine::EngineError> {
        if a.shape() == b.shape() {
            Ok(E::from_iter(&mut a.iter().zip(b.iter()).map(|(x, y)| x + y), a.shape.clone()))
        } else {
            Err(EngineError::ShapeMismatch(a.shape().clone(), b.shape().clone()))
        }
    }

    fn sub(a: &EngineTensor<Self::Unit>, b: &EngineTensor<Self::Unit>) -> Result<EngineTensor<Self::Unit>, crate::engine::EngineError> {
        if a.shape() == b.shape() {
            Ok(E::from_iter(&mut a.iter().zip(b.iter()).map(|(x, y)| x - y), a.shape.clone()))
        } else {
            Err(EngineError::ShapeMismatch(a.shape().clone(), b.shape().clone()))
        }
    }

    fn mul(a: &EngineTensor<Self::Unit>, b: &EngineTensor<Self::Unit>) -> Result<EngineTensor<Self::Unit>, crate::engine::EngineError> {
        if a.shape() == b.shape() {
            Ok(E::from_iter(&mut a.iter().zip(b.iter()).map(|(x, y)| x * y), a.shape.clone()))
        } else {
            Err(EngineError::ShapeMismatch(a.shape().clone(), b.shape().clone()))
        }
    }

    fn div(a: &EngineTensor<Self::Unit>, b: &EngineTensor<Self::Unit>) -> Result<EngineTensor<Self::Unit>, crate::engine::EngineError> {
        if a.shape() == b.shape() {
            Ok(E::from_iter(&mut a.iter().zip(b.iter()).map(|(x, y)| x / y), a.shape.clone()))
        } else {
            Err(EngineError::ShapeMismatch(a.shape().clone(), b.shape().clone()))
        }
    }
}