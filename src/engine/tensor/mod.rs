pub mod extension;
pub mod iter;
pub mod builder;
pub mod factory;

use crate::helper::{Interval, Position, Shape, Slice };
use self::extension::ExtensionProvider;
use self::iter::EngineTensorUnitIterator;
use std::fmt::Debug;
use super::unit::UnitCompatible;

//Unless otherwise specified every function should make as shallow of a copy as possible
pub trait EngineTensor: Debug {
    type Unit: UnitCompatible;

    fn shape(&self) -> &Shape;
    fn get(&self, pos: &Position) -> Self::Unit;
    fn iter_units(&self) -> EngineTensorUnitIterator<'_, Self::Unit>;

    fn clone(&self) -> Box<dyn EngineTensor<Unit = Self::Unit>>;
    fn mat(&self) -> Box<dyn EngineTensor<Unit = Self::Unit>>;
    fn slice(&self, intervals: &[Interval]) -> Box<dyn EngineTensor<Unit = Self::Unit>>;
    fn reshape(&self, shape: &Shape) -> Box<dyn EngineTensor<Unit = Self::Unit>>;
    fn broadcast_splice(&self, pos: usize, sub: &[usize]) -> Box<dyn EngineTensor<Unit = Self::Unit>>;

    fn extensions(&self) -> Box<dyn ExtensionProvider + '_>;
}

impl<T: UnitCompatible> PartialEq for dyn EngineTensor<Unit = T> + '_ {
    fn eq(&self, other: &Self) -> bool {
        self.shape() == other.shape() && self.iter_units().zip(other.iter_units()).all(|(a, b)| a == b)
    }
}