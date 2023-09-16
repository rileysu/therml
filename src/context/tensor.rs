use crate::engine::tensor::{allowed_unit::AllowedUnit, EngineTensor};

enum Source<U: AllowedUnit> {
    Root,

    Abs(ContextTensor<U>),
    Neg(ContextTensor<U>),

    Add(ContextTensor<U>, ContextTensor<U>),
    Sub(ContextTensor<U>, ContextTensor<U>),
    Mul(ContextTensor<U>, ContextTensor<U>),
    Div(ContextTensor<U>, ContextTensor<U>),
}

struct ContextTensor<U: AllowedUnit> {
    concrete: Option<EngineTensor<U>>,
    source: Box<Source<U>>,
}

impl<U: AllowedUnit> ContextTensor<U> {

}