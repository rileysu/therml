use super::{allowed_unit::{AllowedArray, AllowedQuant}, Array, EngineTensor, Quant};

pub trait EngineTensorGeneric: EngineTensor
where
    Self: Sized + 'static,
{
    fn generic(self) -> Box<dyn EngineTensor<Unit = Self::Unit>> {
        Box::from(self)
    }
}

impl<T: AllowedArray> EngineTensorGeneric for Array<T> {}
impl<T: AllowedQuant> EngineTensorGeneric for Quant<T> {}
