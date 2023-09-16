pub mod tensor;
pub mod basic;

mod util;

use crate::helper::Shape;
use self::tensor::{EngineTensor, allowed_unit::AllowedUnit};
use thiserror::Error;

//Using PyTorch operations as a base
//Using a trait over an enum has little extra cost and allows for extension
//Engines provide different optimisations for Tensor operations
//Factory defines the unit as well as output tensor type
pub trait Engine {
    type Unit: AllowedUnit;
    //Pointwise Single
    fn abs(a: &EngineTensor<Self::Unit>) -> Result<EngineTensor<Self::Unit>, EngineError>;
    fn neg(a: &EngineTensor<Self::Unit>) -> Result<EngineTensor<Self::Unit>, EngineError>;
    
    //Pointwise Double
    fn add(a: &EngineTensor<Self::Unit>, b: &EngineTensor<Self::Unit>) -> Result<EngineTensor<Self::Unit>, EngineError>;
    fn sub(a: &EngineTensor<Self::Unit>, b: &EngineTensor<Self::Unit>) -> Result<EngineTensor<Self::Unit>, EngineError>;
    fn mul(a: &EngineTensor<Self::Unit>, b: &EngineTensor<Self::Unit>) -> Result<EngineTensor<Self::Unit>, EngineError>;
    fn div(a: &EngineTensor<Self::Unit>, b: &EngineTensor<Self::Unit>) -> Result<EngineTensor<Self::Unit>, EngineError>;
}

#[derive(Error, Debug)]
pub enum EngineError {
    #[error("the tensor of size {0} does not match {1}")]
    ShapeMismatch(Shape, Shape)
}

