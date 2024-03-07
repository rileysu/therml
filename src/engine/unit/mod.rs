use std::{
    fmt::Debug,
    iter::{Product, Sum},
    ops::{Add, Div, Mul, Rem, Sub},
};

use self::{core_value::CoreValue, signed_op::SignedOp};

pub mod core_value;
pub mod signed_op;

pub trait Base:
    Sized
    + Copy
    + Debug
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Rem<Output = Self>
    + Sum
    + Product
    + PartialEq
    + PartialOrd
    + 'static
{
}
impl<
        T: Sized
            + Copy
            + Debug
            + Add<Output = Self>
            + Sub<Output = Self>
            + Mul<Output = Self>
            + Div<Output = Self>
            + Rem<Output = Self>
            + Sum
            + Product
            + PartialEq
            + PartialOrd
            + 'static,
    > Base for T
{
}

pub trait UnitCompatible: SignedOp + CoreValue {}
impl<T: SignedOp + CoreValue> UnitCompatible for T {}
