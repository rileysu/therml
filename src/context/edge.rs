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

impl<T: AllowedUnit> Edge<T> {
    fn get_children<'a>(&'a self) -> EdgeChildrenIterator<'a, T> {
        EdgeChildrenIterator::new(self)
    }
}

pub struct EdgeChildrenIterator<'a, T: AllowedUnit> {
    edge: &'a Edge<T>,
    pos: usize,
}

impl<'a, T: AllowedUnit> EdgeChildrenIterator<'a, T> {
    pub fn new(edge: &'a Edge<T>) -> Self {
        Self {
            edge,
            pos: 0,
        }
    }
}

impl<'a, T: AllowedUnit> Iterator for EdgeChildrenIterator<'a, T> {
    type Item = NodeKey;

    fn next(&mut self) -> Option<Self::Item> {
        let out = match self.edge {
            Edge::Root => None,
            Edge::Abs(key, _) | Edge::Neg(key, _) => {
                match self.pos {
                    0 => Some(*key),
                    _ => None,
                }
            }
            Edge::Add(a_key, b_key, _) | Edge::Sub(a_key, b_key, _) | Edge::Mul(a_key, b_key, _) | Edge::Div(a_key, b_key, _) => {
                match self.pos {
                    0 => Some(*a_key),
                    1 => Some(*b_key),
                    _ => None,
                }
            }
        };

        self.pos += 1;

        return out;
    }
}

