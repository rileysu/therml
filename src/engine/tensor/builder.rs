use crate::{engine::unit::UnitCompatible, helper::{Interval, Position, Shape}};

use super::{factory::EngineTensorFactory, EngineTensor};

pub trait EngineTensorBuilder {
    type Unit: UnitCompatible;
    type Tensor: EngineTensor<Unit = Self::Unit> + EngineTensorFactory;

    fn new(shape: Shape, init: Self::Unit) -> Self;

    fn splice_slice<I: IntoIterator<Item = Self::Unit>>(&mut self, intervals: &[Interval], replace_with: I);
    fn splice_between_positions<I: IntoIterator<Item = Self::Unit>>(&mut self, start: &Position, last: &Position, replace_with: I);

    fn construct(self) -> Self::Tensor;
}