pub mod tensor;

use crate::engine::{Engine, tensor::allowed_unit::AllowedUnit, EngineError};

struct Context{

}

struct ContextExecutor<'a, E: Engine> {
    context: &'a mut Context,
}

struct ContextTensor {

}

impl Context {
    pub fn new() -> Self {

    }

    pub fn new_executor<E: Engine>(&mut self) -> ContextExecutor<E> {
        ContextExecutor::<E>::new(self)
    }
}

impl<E: Engine> ContextExecutor<'_, E> {
    pub fn new(context: &mut Context) -> Self {
        Self {
            context,
        }
    }

    pub fn add() -> Result<, EngineError> {
        
    }
}