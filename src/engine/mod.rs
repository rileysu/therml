pub mod tensor;
pub mod unit;

use crate::helper::{PositionError, Shape, VarArrayError};
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

    fn relu<E: EngineTensorFactory<Unit = T>>(a: &dyn EngineTensor<Unit = T>) -> Result<Box<dyn EngineTensor<Unit = T>>, EngineError>;
    fn leaky_relu<E: EngineTensorFactory<Unit = T>>(a: &dyn EngineTensor<Unit = T>, alpha: f64) -> Result<Box<dyn EngineTensor<Unit = T>>, EngineError>;
    fn sigmoid<E: EngineTensorFactory<Unit = T>>(a: &dyn EngineTensor<Unit = T>) -> Result<Box<dyn EngineTensor<Unit = T>>, EngineError>;

    //Pointwise Double
    fn add<E: EngineTensorFactory<Unit = T>>(a: &dyn EngineTensor<Unit = T>, b: &dyn EngineTensor<Unit = T>) -> Result<Box<dyn EngineTensor<Unit = T>>, EngineError>;
    fn sub<E: EngineTensorFactory<Unit = T>>(a: &dyn EngineTensor<Unit = T>, b: &dyn EngineTensor<Unit = T>) -> Result<Box<dyn EngineTensor<Unit = T>>, EngineError>;
    fn mul<E: EngineTensorFactory<Unit = T>>(a: &dyn EngineTensor<Unit = T>, b: &dyn EngineTensor<Unit = T>) -> Result<Box<dyn EngineTensor<Unit = T>>, EngineError>;
    fn div<E: EngineTensorFactory<Unit = T>>(a: &dyn EngineTensor<Unit = T>, b: &dyn EngineTensor<Unit = T>) -> Result<Box<dyn EngineTensor<Unit = T>>, EngineError>;

    fn matmul<E: EngineTensorFactory<Unit = T>>(a: &dyn EngineTensor<Unit = T>, b: &dyn EngineTensor<Unit = T>) -> Result<Box<dyn EngineTensor<Unit = T>>, EngineError>;

    //Conv
    fn conv2d<E: EngineTensorFactory<Unit = T>>(a: &dyn EngineTensor<Unit = T>, kernel: &dyn EngineTensor<Unit = T>, padding: usize, stride: usize) -> Result<Box<dyn EngineTensor<Unit = T>>, EngineError>;
    //fn im2col_2d<E: EngineTensorFactory<Unit = T>>(a: &dyn EngineTensor<Unit = T>, kernel_shape: &Shape, padding: usize, stride: usize) -> Result<Box<dyn EngineTensor<Unit = T>>, EngineError>;

    //Pool
    fn batch_norm_no_running<E: EngineTensorFactory<Unit = T>>(a: &dyn EngineTensor<Unit = T>, weight: &dyn EngineTensor<Unit = T>, bias: &dyn EngineTensor<Unit = T>, eps: f64) -> Result<Box<dyn EngineTensor<Unit = T>>, crate::engine::EngineError>;
    fn batch_norm_running<E: EngineTensorFactory<Unit = T>>(a: &dyn EngineTensor<Unit = T>, running_mean: &dyn EngineTensor<Unit = T>, running_var: &dyn EngineTensor<Unit = T>, weight: &dyn EngineTensor<Unit = T>, bias: &dyn EngineTensor<Unit = T>, momentum: f64, eps: f64) -> Result<Box<dyn EngineTensor<Unit = T>>, crate::engine::EngineError>;
}

#[derive(Error, Debug)]
pub enum EngineError {
    #[error("The tensor of shape {0} does not match expected {1}")]
    ShapeMismatch(Shape, Shape),
    #[error("The dimension {0} does not match expected {1}")]
    DimensionMismatch(usize, usize),
    #[error("The dimensions {0:?} do not match expected {1:?}")]
    DimensionsMismatch(Box<[usize]>, Box<[usize]>),
    #[error("The dimension {0} does not meet at least {1}")]
    NotEnoughDimensions(usize, usize),
    #[error("Got {0} dimensions but expected {1}")]
    NumDimensionsMismatch(usize, usize),
    #[error("Vararray operation failed: {0}")]
    VarArray(#[from] VarArrayError),
    #[error("Position operation failed: {0}")]
    Tensor(#[from] PositionError),
    #[error("The operation is not supported on this data type")]
    OperationUnsupportedForType(),
}

