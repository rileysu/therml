use num::Signed;

use crate::engine::{Engine, basic::Basic, EngineError};

use super::{EngineTensor, factory::EngineTensorFactory, allowed_unit::AllowedUnit};

impl<T: AllowedUnit + Signed> Engine<T> for Basic {

    fn abs<E: EngineTensorFactory<T>>(a: &EngineTensor<T>) -> Result<EngineTensor<T>, crate::engine::EngineError> {
        Ok(E::from_iter(&mut a.iter().map(|x| x.abs()), a.shape.clone()))
    }

    fn neg<E: EngineTensorFactory<T>>(a: &EngineTensor<T>) -> Result<EngineTensor<T>, crate::engine::EngineError> {
        Ok(E::from_iter(&mut a.iter().map(|x| -x), a.shape.clone()))
    }

    fn add<E: EngineTensorFactory<T>>(a: &EngineTensor<T>, b: &EngineTensor<T>) -> Result<EngineTensor<T>, crate::engine::EngineError> {
        if a.shape() == b.shape() {
            Ok(E::from_iter(&mut a.iter().zip(b.iter()).map(|(x, y)| x + y), a.shape.clone()))
        } else {
            Err(EngineError::ShapeMismatch(a.shape().clone(), b.shape().clone()))
        }
    }

    fn sub<E: EngineTensorFactory<T>>(a: &EngineTensor<T>, b: &EngineTensor<T>) -> Result<EngineTensor<T>, crate::engine::EngineError> {
        if a.shape() == b.shape() {
            Ok(E::from_iter(&mut a.iter().zip(b.iter()).map(|(x, y)| x - y), a.shape.clone()))
        } else {
            Err(EngineError::ShapeMismatch(a.shape().clone(), b.shape().clone()))
        }
    }

    fn mul<E: EngineTensorFactory<T>>(a: &EngineTensor<T>, b: &EngineTensor<T>) -> Result<EngineTensor<T>, crate::engine::EngineError> {
        if a.shape() == b.shape() {
            Ok(E::from_iter(&mut a.iter().zip(b.iter()).map(|(x, y)| x * y), a.shape.clone()))
        } else {
            Err(EngineError::ShapeMismatch(a.shape().clone(), b.shape().clone()))
        }
    }

    fn div<E: EngineTensorFactory<T>>(a: &EngineTensor<T>, b: &EngineTensor<T>) -> Result<EngineTensor<T>, crate::engine::EngineError> {
        if a.shape() == b.shape() {
            Ok(E::from_iter(&mut a.iter().zip(b.iter()).map(|(x, y)| x / y), a.shape.clone()))
        } else {
            Err(EngineError::ShapeMismatch(a.shape().clone(), b.shape().clone()))
        }
    }
}