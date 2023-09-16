use std::marker::PhantomData;
use super::tensor::factory::EngineTensorFactory;

pub struct Basic<E: EngineTensorFactory> {
    _phantom: PhantomData<E>,
}