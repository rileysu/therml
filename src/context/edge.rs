use crate::engine::{tensor::{allowed_unit::AllowedUnit, EngineTensor}, EngineError};

use super::comp_graph::NodeKey;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Edge<T: AllowedUnit> {
    Root,

    Abs(NodeKey, fn(&EngineTensor<T>) -> Result<EngineTensor<T>, EngineError>),
    Neg(NodeKey, fn(&EngineTensor<T>) -> Result<EngineTensor<T>, EngineError>),

    Add(NodeKey, NodeKey, fn(&EngineTensor<T>, &EngineTensor<T>) -> Result<EngineTensor<T>, EngineError>),
    Sub(NodeKey, NodeKey, fn(&EngineTensor<T>, &EngineTensor<T>) -> Result<EngineTensor<T>, EngineError>),
    Mul(NodeKey, NodeKey, fn(&EngineTensor<T>, &EngineTensor<T>) -> Result<EngineTensor<T>, EngineError>),
    Div(NodeKey, NodeKey, fn(&EngineTensor<T>, &EngineTensor<T>) -> Result<EngineTensor<T>, EngineError>),
}