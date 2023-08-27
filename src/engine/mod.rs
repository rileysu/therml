mod tensor;
mod util;

use crate::helper::Shape;
use self::tensor::EngineTensor;
use thiserror::Error;

pub struct BasicEngine;

//Using PyTorch operations as a base
//Using a trait over an enum has little extra cost and allows for extension
//Engines provide different optimisations for Tensor operations
pub trait Engine<T: Sized + Copy> {
    //Pointwise Single
    fn abs(a: &EngineTensor<T>) -> Result<EngineTensor<T>, EngineError>;
    fn neg(a: &EngineTensor<T>) -> Result<EngineTensor<T>, EngineError>;
    
    //Pointwise Double
    fn add(a: &EngineTensor<T>, b: &EngineTensor<T>) -> Result<EngineTensor<T>, EngineError>;
    fn sub(a: &EngineTensor<T>, b: &EngineTensor<T>) -> Result<EngineTensor<T>, EngineError>;
    fn mul(a: &EngineTensor<T>, b: &EngineTensor<T>) -> Result<EngineTensor<T>, EngineError>;
    fn div(a: &EngineTensor<T>, b: &EngineTensor<T>) -> Result<EngineTensor<T>, EngineError>;
}

#[derive(Error, Debug)]
pub enum EngineError {
    #[error("the tensor of size {0} does not match {1}")]
    ShapeMismatch(Shape, Shape)
}

