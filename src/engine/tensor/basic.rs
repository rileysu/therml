use crate::engine::{Engine, basic::Basic, EngineError};

use super::{EngineTensor, factory::EngineTensorFactory};

impl<E: EngineTensorFactory<Unit = f32>> Engine for Basic<E> {
    type Unit = E::Unit;

    fn abs(a: &EngineTensor<f32>) -> Result<EngineTensor<f32>, crate::engine::EngineError> {
        Ok(E::from_iter(&mut a.iter().map(|x| x.abs()), a.shape.clone()))
    }

    fn neg(a: &EngineTensor<f32>) -> Result<EngineTensor<f32>, crate::engine::EngineError> {
        Ok(E::from_iter(&mut a.iter().map(|x| -x), a.shape.clone()))
    }

    fn add(a: &EngineTensor<f32>, b: &EngineTensor<f32>) -> Result<EngineTensor<f32>, crate::engine::EngineError> {
        if a.shape() == b.shape() {
            Ok(E::from_iter(&mut a.iter().zip(b.iter()).map(|(x, y)| x + y), a.shape.clone()))
        } else {
            Err(EngineError::ShapeMismatch(a.shape().clone(), b.shape().clone()))
        }
    }

    fn sub(a: &EngineTensor<f32>, b: &EngineTensor<f32>) -> Result<EngineTensor<f32>, crate::engine::EngineError> {
        if a.shape() == b.shape() {
            Ok(E::from_iter(&mut a.iter().zip(b.iter()).map(|(x, y)| x - y), a.shape.clone()))
        } else {
            Err(EngineError::ShapeMismatch(a.shape().clone(), b.shape().clone()))
        }
    }

    fn mul(a: &EngineTensor<f32>, b: &EngineTensor<f32>) -> Result<EngineTensor<f32>, crate::engine::EngineError> {
        if a.shape() == b.shape() {
            Ok(E::from_iter(&mut a.iter().zip(b.iter()).map(|(x, y)| x * y), a.shape.clone()))
        } else {
            Err(EngineError::ShapeMismatch(a.shape().clone(), b.shape().clone()))
        }
    }

    fn div(a: &EngineTensor<f32>, b: &EngineTensor<f32>) -> Result<EngineTensor<f32>, crate::engine::EngineError> {
        if a.shape() == b.shape() {
            Ok(E::from_iter(&mut a.iter().zip(b.iter()).map(|(x, y)| x / y), a.shape.clone()))
        } else {
            Err(EngineError::ShapeMismatch(a.shape().clone(), b.shape().clone()))
        }
    }
}