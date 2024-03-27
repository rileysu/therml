use crate::helper::Shape;
use super::{builder::EngineTensorBuilder, EngineTensor};

pub trait EngineTensorFactory: EngineTensor
where Self: Sized + 'static
{
    fn from_iter(iter: impl Iterator<Item = Self::Unit>, shape: Shape) -> Self;
    fn from_slice(data: &[Self::Unit], shape: Shape) -> Self;
    fn builder(shape: Shape, init: Self::Unit) -> impl EngineTensorBuilder<Unit = Self::Unit>;

    fn generic(self) -> Box<dyn EngineTensor<Unit = Self::Unit>> {
        Box::from(self)
    }
}