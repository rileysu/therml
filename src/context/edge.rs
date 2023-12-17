use crate::engine::{tensor::{allowed_unit::AllowedUnit, EngineTensor}, EngineError};

use super::comp_graph::{NodeKey, ComputationGraphError};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Edge<T: AllowedUnit> {
    Root,

    Abs(NodeKey, fn(&dyn EngineTensor<Unit = T>) -> Result<Box<dyn EngineTensor<Unit = T>>, EngineError>),
    Neg(NodeKey, fn(&dyn EngineTensor<Unit = T>) -> Result<Box<dyn EngineTensor<Unit = T>>, EngineError>),

    AddScalar(T, NodeKey, fn(T, &dyn EngineTensor<Unit = T>) -> Result<Box<dyn EngineTensor<Unit = T>>, EngineError>),
    SubScalarLH(T, NodeKey, fn(T, &dyn EngineTensor<Unit = T>) -> Result<Box<dyn EngineTensor<Unit = T>>, EngineError>),
    SubScalarRH(NodeKey, T, fn(&dyn EngineTensor<Unit = T>, T) -> Result<Box<dyn EngineTensor<Unit = T>>, EngineError>),
    MulScalar(T, NodeKey, fn(T, &dyn EngineTensor<Unit = T>) -> Result<Box<dyn EngineTensor<Unit = T>>, EngineError>),
    DivScalarLH(T, NodeKey, fn(T, &dyn EngineTensor<Unit = T>) -> Result<Box<dyn EngineTensor<Unit = T>>, EngineError>),
    DivScalarRH(NodeKey, T, fn(&dyn EngineTensor<Unit = T>, T) -> Result<Box<dyn EngineTensor<Unit = T>>, EngineError>),

    Add(NodeKey, NodeKey, fn(&dyn EngineTensor<Unit = T>, &dyn EngineTensor<Unit = T>) -> Result<Box<dyn EngineTensor<Unit = T>>, EngineError>),
    Sub(NodeKey, NodeKey, fn(&dyn EngineTensor<Unit = T>, &dyn EngineTensor<Unit = T>) -> Result<Box<dyn EngineTensor<Unit = T>>, EngineError>),
    Mul(NodeKey, NodeKey, fn(&dyn EngineTensor<Unit = T>, &dyn EngineTensor<Unit = T>) -> Result<Box<dyn EngineTensor<Unit = T>>, EngineError>),
    Div(NodeKey, NodeKey, fn(&dyn EngineTensor<Unit = T>, &dyn EngineTensor<Unit = T>) -> Result<Box<dyn EngineTensor<Unit = T>>, EngineError>),
}

impl<T: AllowedUnit> Edge<T> {
    pub fn nodes(&self) -> EdgeNodesIterator<T> {
        EdgeNodesIterator::<T>::new(self)
    }

    pub fn is_root(&self) -> bool {
        match self {
            Edge::Root => true,
            _ => false,
        }
    }

    //Single layer computation otherwise should throw an error
    pub fn compute_tensor<'a, F: Fn(NodeKey) -> Result<&'a dyn EngineTensor<Unit = T>, ComputationGraphError>>(&'a self, resolve: F) -> Result<Box<dyn EngineTensor<Unit = T>>, ComputationGraphError> {
        match self {
            Edge::Root => Err(ComputationGraphError::RootNodeNotComputed()),
            Edge::Abs(a_key, op) |  
            Edge::Neg(a_key, op) => {
                op(resolve(*a_key)?).map_err(|e| ComputationGraphError::from(e))
            },
            Edge::AddScalar(s, a_key, op) |
            Edge::SubScalarLH(s, a_key, op) |
            Edge::MulScalar(s, a_key, op) |
            Edge::DivScalarLH(s, a_key, op) => {
                op(*s, resolve(*a_key)?).map_err(|e| ComputationGraphError::from(e))
            },
            Edge::SubScalarRH(a_key, s, op) |
            Edge::DivScalarRH(a_key, s, op) => {
                    op(resolve(*a_key)?, *s).map_err(|e| ComputationGraphError::from(e))
            }
            Edge::Add(a_key, b_key, op) |
            Edge::Sub(a_key, b_key, op) |
            Edge::Mul(a_key, b_key, op) |
            Edge::Div(a_key, b_key, op) => {

                op(resolve(*a_key)?, resolve(*b_key)?).map_err(|e| ComputationGraphError::from(e))
            },
        }
    }
}

pub struct EdgeNodesIterator<'a, T: AllowedUnit> {
    edge: &'a Edge<T>,
    pos: usize,
}

impl<'a, T: AllowedUnit> EdgeNodesIterator<'a, T> {
    pub fn new(edge: &'a Edge<T>) -> Self {
        Self {
            edge,
            pos: 0,
        }
    }
}

impl<'a, T: AllowedUnit> Iterator for EdgeNodesIterator<'a, T> {
    type Item = NodeKey;

    fn next(&mut self) -> Option<Self::Item> {
        let out = match self.edge {
            Edge::Root => None,
            Edge::Abs(a_key, _) | 
            Edge::Neg(a_key, _) |
            Edge::AddScalar(_, a_key, _) |
            Edge::SubScalarLH(_, a_key, _) |
            Edge::SubScalarRH(a_key, _, _) |
            Edge::MulScalar(_, a_key, _) |
            Edge::DivScalarLH(_, a_key, _) |
            Edge::DivScalarRH(a_key, _, _) => {
                match self.pos {
                    0 => Some(*a_key),
                    _ => None,
                }
            }
            Edge::Add(a_key, b_key, _) |
            Edge::Sub(a_key, b_key, _) |
            Edge::Mul(a_key, b_key, _) |
            Edge::Div(a_key, b_key, _) => {
                match self.pos {
                    0 => Some(*a_key),
                    1 => Some(*b_key),
                    _ => None,
                }
            }
        };

        if out.is_some() {
            self.pos += 1;
        }

        out
    }
}

