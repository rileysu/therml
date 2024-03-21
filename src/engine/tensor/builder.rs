use std::ops::RangeBounds;

use crate::{engine::unit::UnitCompatible, helper::Shape};

use super::EngineTensor;

pub trait EngineTensorBuilder {
    type Unit: UnitCompatible;
    type Tensor: EngineTensor;

    fn new(shape: Shape) -> Self;

    fn splice<R: RangeBounds<usize>, I: IntoIterator<Item = Self::Unit>>(&mut self, range: R, replace_with: I);

    fn construct(self) -> Self::Tensor;
}