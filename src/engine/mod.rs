pub mod tensor;
pub mod basic;

mod util;

use crate::helper::Shape;
use self::tensor::{EngineTensor, allowed_unit::AllowedUnit, factory::EngineTensorFactory};
use thiserror::Error;

//Using PyTorch operations as a base
//Using a trait over an enum has little extra cost and allows for extension
//Engines provide different optimisations for Tensor operations
//Factory defines the unit as well as output tensor type
pub trait Engine<T: AllowedUnit> {
    //Pointwise Single
    fn abs<E: EngineTensorFactory<T>>(a: &EngineTensor<T>) -> Result<EngineTensor<T>, EngineError>;
    fn neg<E: EngineTensorFactory<T>>(a: &EngineTensor<T>) -> Result<EngineTensor<T>, EngineError>;
    
    //Pointwise Double
    fn add<E: EngineTensorFactory<T>>(a: &EngineTensor<T>, b: &EngineTensor<T>) -> Result<EngineTensor<T>, EngineError>;
    fn sub<E: EngineTensorFactory<T>>(a: &EngineTensor<T>, b: &EngineTensor<T>) -> Result<EngineTensor<T>, EngineError>;
    fn mul<E: EngineTensorFactory<T>>(a: &EngineTensor<T>, b: &EngineTensor<T>) -> Result<EngineTensor<T>, EngineError>;
    fn div<E: EngineTensorFactory<T>>(a: &EngineTensor<T>, b: &EngineTensor<T>) -> Result<EngineTensor<T>, EngineError>;
}

#[derive(Error, Debug)]
pub enum EngineError {
    #[error("The tensor of size {0} does not match {1}")]
    ShapeMismatch(Shape, Shape),
    #[error("The operation is not supported on this data type")]
    OperationUnsupportedForType(),
}

