pub mod tensor;
pub mod unit;
pub mod basic;

mod shared;
mod util;

use crate::helper::{Shape, PositionError};
use self::{tensor::{factory::EngineTensorFactory, EngineTensor}, unit::UnitCompatible};
use thiserror::Error;

//Using PyTorch operations as a base
//Using a trait over an enum has little extra cost and allows for extension
//Engines provide different optimisations for Tensor operations
//Factory defines the unit as well as output tensor type
pub trait Engine<T: UnitCompatible> {
    //Pointwise Single
    fn abs<E: EngineTensorFactory<Unit = T>>(a: &dyn EngineTensor<Unit = T>) -> Result<Box<dyn EngineTensor<Unit = T>>, EngineError>;
    fn neg<E: EngineTensorFactory<Unit = T>>(a: &dyn EngineTensor<Unit = T>) -> Result<Box<dyn EngineTensor<Unit = T>>, EngineError>;
    
    //Pointwise Scalar
    fn add_scalar<E: EngineTensorFactory<Unit = T>>(s: T, a: &dyn EngineTensor<Unit = T>) -> Result<Box<dyn EngineTensor<Unit = T>>, EngineError>;
    fn sub_scalar_lh<E: EngineTensorFactory<Unit = T>>(s: T, a: &dyn EngineTensor<Unit = T>) -> Result<Box<dyn EngineTensor<Unit = T>>, EngineError>;
    fn sub_scalar_rh<E: EngineTensorFactory<Unit = T>>(a: &dyn EngineTensor<Unit = T>, s: T) -> Result<Box<dyn EngineTensor<Unit = T>>, EngineError>;
    fn mul_scalar<E: EngineTensorFactory<Unit = T>>(s: T, a: &dyn EngineTensor<Unit = T>) -> Result<Box<dyn EngineTensor<Unit = T>>, EngineError>;
    fn div_scalar_lh<E: EngineTensorFactory<Unit = T>>(s: T, a: &dyn EngineTensor<Unit = T>) -> Result<Box<dyn EngineTensor<Unit = T>>, EngineError>;
    fn div_scalar_rh<E: EngineTensorFactory<Unit = T>>(a: &dyn EngineTensor<Unit = T>, s: T) -> Result<Box<dyn EngineTensor<Unit = T>>, EngineError>;

    //Pointwise Double
    fn add<E: EngineTensorFactory<Unit = T>>(a: &dyn EngineTensor<Unit = T>, b: &dyn EngineTensor<Unit = T>) -> Result<Box<dyn EngineTensor<Unit = T>>, EngineError>;
    fn sub<E: EngineTensorFactory<Unit = T>>(a: &dyn EngineTensor<Unit = T>, b: &dyn EngineTensor<Unit = T>) -> Result<Box<dyn EngineTensor<Unit = T>>, EngineError>;
    fn mul<E: EngineTensorFactory<Unit = T>>(a: &dyn EngineTensor<Unit = T>, b: &dyn EngineTensor<Unit = T>) -> Result<Box<dyn EngineTensor<Unit = T>>, EngineError>;
    fn div<E: EngineTensorFactory<Unit = T>>(a: &dyn EngineTensor<Unit = T>, b: &dyn EngineTensor<Unit = T>) -> Result<Box<dyn EngineTensor<Unit = T>>, EngineError>;

    //Conv
    fn conv2d<E: EngineTensorFactory<Unit = T>>(a: &dyn EngineTensor<Unit = T>, kernel: &dyn EngineTensor<Unit = T>, padding: usize, stride: usize) -> Result<Box<dyn EngineTensor<Unit = T>>, EngineError>;
    //fn im2col_2d<E: EngineTensorFactory<Unit = T>>(a: &dyn EngineTensor<Unit = T>, kernel_shape: &Shape, padding: usize, stride: usize) -> Result<Box<dyn EngineTensor<Unit = T>>, EngineError>;
}

#[derive(Error, Debug)]
pub enum EngineError {
    #[error("The tensor of shape {0} does not match expected {1}")]
    ShapeMismatch(Shape, Shape),
    #[error("The dimension {0} does not match expected {1}")]
    DimensionMismatch(usize, usize),
    #[error("Got {0} dimensions but expected {1}")]
    DimensionsMismatch(usize, usize),
    #[error("Position operation failed: {0}")]
    Tensor(#[from] PositionError),
    #[error("The operation is not supported on this data type")]
    OperationUnsupportedForType(),
}

