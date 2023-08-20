mod tensor;
mod basic;
mod util;

use crate::helper::Shape;
use self::tensor::EngineTensor;
use thiserror::Error;

//Using PyTorch operations as a base
pub trait Engine<T: Sized + Copy, E: EngineTensor<T>> {
    //Pointwise Single
    fn abs(a: &E) -> Result<E, EngineError>;
    fn neg(a: &E) -> Result<E, EngineError>;
    
    //Pointwise Double
    fn add(a: &E, b: &E) -> Result<E, EngineError>;
    fn sub(a: &E, b: &E) -> Result<E, EngineError>;
    fn mul(a: &E, b: &E) -> Result<E, EngineError>;
    fn div(a: &E, b: &E) -> Result<E, EngineError>;
}

#[derive(Error, Debug)]
pub enum EngineError {
    #[error("the tensor of size {0} does not match {1}")]
    ShapeMismatch(Shape, Shape)
}

