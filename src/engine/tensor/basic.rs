use crate::engine::{Engine, basic::Basic, EngineError};

use super::{EngineTensor, factory::EngineTensorFactory};

macro_rules! basic_impl {
    ($unit:ty) => {
        impl Engine<$unit> for Basic {

            //Pointwise Single
            fn abs<E: EngineTensorFactory<$unit>>(a: &EngineTensor<$unit>) -> Result<EngineTensor<$unit>, crate::engine::EngineError> {
        
                Ok(E::from_iter(a.iter().map(|x| x.abs()), a.shape().clone()))
            }
        
            fn neg<E: EngineTensorFactory<$unit>>(a: &EngineTensor<$unit>) -> Result<EngineTensor<$unit>, crate::engine::EngineError> {
                Ok(E::from_iter(a.iter().map(|x| -x), a.shape().clone()))
            }

            //Scalar
            fn add_scalar<E: EngineTensorFactory<$unit>>(s: $unit, a: &EngineTensor<$unit>) -> Result<EngineTensor<$unit>, EngineError> {
                Ok(E::from_iter(a.iter().map(|x| x + s), a.shape().clone()))
            }

            fn sub_scalar_lh<E: EngineTensorFactory<$unit>>(s: $unit, a: &EngineTensor<$unit>) -> Result<EngineTensor<$unit>, EngineError> {
                Ok(E::from_iter(a.iter().map(|x| s - x), a.shape().clone()))
            }

            fn sub_scalar_rh<E: EngineTensorFactory<$unit>>(a: &EngineTensor<$unit>, s: $unit) -> Result<EngineTensor<$unit>, EngineError> {
                Ok(E::from_iter(a.iter().map(|x| x - s), a.shape().clone()))
            }

            fn mul_scalar<E: EngineTensorFactory<$unit>>(s: $unit, a: &EngineTensor<$unit>) -> Result<EngineTensor<$unit>, EngineError> {
                Ok(E::from_iter(a.iter().map(|x| x * s), a.shape().clone()))
            }

            fn div_scalar_lh<E: EngineTensorFactory<$unit>>(s: $unit, a: &EngineTensor<$unit>) -> Result<EngineTensor<$unit>, EngineError> {
                Ok(E::from_iter(a.iter().map(|x| s / x), a.shape().clone()))
            }

            fn div_scalar_rh<E: EngineTensorFactory<$unit>>(a: &EngineTensor<$unit>, s: $unit) -> Result<EngineTensor<$unit>, EngineError> {
                Ok(E::from_iter(a.iter().map(|x| x / s), a.shape().clone()))
            }
        
            //Pointwise Double
            fn add<E: EngineTensorFactory<$unit>>(a: &EngineTensor<$unit>, b: &EngineTensor<$unit>) -> Result<EngineTensor<$unit>, crate::engine::EngineError> {
                if a.shape() == b.shape() {
                    Ok(E::from_iter(&mut a.iter().zip(b.iter()).map(|(x, y)| x + y), a.shape.clone()))
                } else {
                    Err(EngineError::ShapeMismatch(a.shape().clone(), b.shape().clone()))
                }
            }

            fn sub<E: EngineTensorFactory<$unit>>(a: &EngineTensor<$unit>, b: &EngineTensor<$unit>) -> Result<EngineTensor<$unit>, crate::engine::EngineError> {
                if a.shape() == b.shape() {
                    Ok(E::from_iter(&mut a.iter().zip(b.iter()).map(|(x, y)| x - y), a.shape.clone()))
                } else {
                    Err(EngineError::ShapeMismatch(a.shape().clone(), b.shape().clone()))
                }
            }
        
            fn mul<E: EngineTensorFactory<$unit>>(a: &EngineTensor<$unit>, b: &EngineTensor<$unit>) -> Result<EngineTensor<$unit>, crate::engine::EngineError> {
                if a.shape() == b.shape() {
                    Ok(E::from_iter(&mut a.iter().zip(b.iter()).map(|(x, y)| x * y), a.shape.clone()))
                } else {
                    Err(EngineError::ShapeMismatch(a.shape().clone(), b.shape().clone()))
                }
            }
        
            fn div<E: EngineTensorFactory<$unit>>(a: &EngineTensor<$unit>, b: &EngineTensor<$unit>) -> Result<EngineTensor<$unit>, crate::engine::EngineError> {
                if a.shape() == b.shape() {
                    Ok(E::from_iter(&mut a.iter().zip(b.iter()).map(|(x, y)| x / y), a.shape.clone()))
                } else {
                    Err(EngineError::ShapeMismatch(a.shape().clone(), b.shape().clone()))
                }
            }
        }
    };
}

macro_rules! basic_unsigned_impl {
    ($unit:ty) => {
        impl Engine<$unit> for Basic {

            //Pointwise Single
            fn abs<E: EngineTensorFactory<$unit>>(a: &EngineTensor<$unit>) -> Result<EngineTensor<$unit>, crate::engine::EngineError> {
        
                Ok(E::from_iter(&mut a.iter().map(|x| x), a.shape.clone()))
            }
        
            fn neg<E: EngineTensorFactory<$unit>>(_a: &EngineTensor<$unit>) -> Result<EngineTensor<$unit>, crate::engine::EngineError> {
                Err(crate::engine::EngineError::OperationUnsupportedForType())
            }

            //Scalar
            fn add_scalar<E: EngineTensorFactory<$unit>>(s: $unit, a: &EngineTensor<$unit>) -> Result<EngineTensor<$unit>, EngineError> {
                Ok(E::from_iter(a.iter().map(|x| x + s), a.shape().clone()))
            }

            fn sub_scalar_lh<E: EngineTensorFactory<$unit>>(s: $unit, a: &EngineTensor<$unit>) -> Result<EngineTensor<$unit>, EngineError> {
                Ok(E::from_iter(a.iter().map(|x| s - x), a.shape().clone()))
            }

            fn sub_scalar_rh<E: EngineTensorFactory<$unit>>(a: &EngineTensor<$unit>, s: $unit) -> Result<EngineTensor<$unit>, EngineError> {
                Ok(E::from_iter(a.iter().map(|x| x - s), a.shape().clone()))
            }

            fn mul_scalar<E: EngineTensorFactory<$unit>>(s: $unit, a: &EngineTensor<$unit>) -> Result<EngineTensor<$unit>, EngineError> {
                Ok(E::from_iter(a.iter().map(|x| x * s), a.shape().clone()))
            }

            fn div_scalar_lh<E: EngineTensorFactory<$unit>>(s: $unit, a: &EngineTensor<$unit>) -> Result<EngineTensor<$unit>, EngineError> {
                Ok(E::from_iter(a.iter().map(|x| s / x), a.shape().clone()))
            }

            fn div_scalar_rh<E: EngineTensorFactory<$unit>>(a: &EngineTensor<$unit>, s: $unit) -> Result<EngineTensor<$unit>, EngineError> {
                Ok(E::from_iter(a.iter().map(|x| x / s), a.shape().clone()))
            }
        
            //Pointwise Double
            fn add<E: EngineTensorFactory<$unit>>(a: &EngineTensor<$unit>, b: &EngineTensor<$unit>) -> Result<EngineTensor<$unit>, crate::engine::EngineError> {
                if a.shape() == b.shape() {
                    Ok(E::from_iter(&mut a.iter().zip(b.iter()).map(|(x, y)| x + y), a.shape.clone()))
                } else {
                    Err(EngineError::ShapeMismatch(a.shape().clone(), b.shape().clone()))
                }
            }
        
            fn sub<E: EngineTensorFactory<$unit>>(a: &EngineTensor<$unit>, b: &EngineTensor<$unit>) -> Result<EngineTensor<$unit>, crate::engine::EngineError> {
                if a.shape() == b.shape() {
                    Ok(E::from_iter(&mut a.iter().zip(b.iter()).map(|(x, y)| x - y), a.shape.clone()))
                } else {
                    Err(EngineError::ShapeMismatch(a.shape().clone(), b.shape().clone()))
                }
            }
        
            fn mul<E: EngineTensorFactory<$unit>>(a: &EngineTensor<$unit>, b: &EngineTensor<$unit>) -> Result<EngineTensor<$unit>, crate::engine::EngineError> {
                if a.shape() == b.shape() {
                    Ok(E::from_iter(&mut a.iter().zip(b.iter()).map(|(x, y)| x * y), a.shape.clone()))
                } else {
                    Err(EngineError::ShapeMismatch(a.shape().clone(), b.shape().clone()))
                }
            }
        
            fn div<E: EngineTensorFactory<$unit>>(a: &EngineTensor<$unit>, b: &EngineTensor<$unit>) -> Result<EngineTensor<$unit>, crate::engine::EngineError> {
                if a.shape() == b.shape() {
                    Ok(E::from_iter(&mut a.iter().zip(b.iter()).map(|(x, y)| x / y), a.shape.clone()))
                } else {
                    Err(EngineError::ShapeMismatch(a.shape().clone(), b.shape().clone()))
                }
            }
        }
    };
}

basic_impl!(f32);
basic_impl!(f64);
basic_impl!(i8);
basic_impl!(i16);
basic_impl!(i32);
basic_impl!(i64);
basic_unsigned_impl!(u8);
basic_unsigned_impl!(u16);
basic_unsigned_impl!(u32);
basic_unsigned_impl!(u64);