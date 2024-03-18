use std::{
    fmt::Debug,
    iter::{Product, Sum},
    ops::{Add, Div, Mul, Rem, Sub},
};

use self::{core_value::CoreValue, signed_op::SignedOp};

pub mod core_value;
pub mod signed_op;

pub trait Base: Sized + Copy + Debug + 'static {}
impl<T: Sized + Copy + Debug + 'static> Base for T {}

pub trait UnitCompatible:
    Base
    + SignedOp
    + CoreValue
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Rem<Output = Self>
    + Sum
    + Product
    + PartialEq
    + PartialOrd
{
}
impl<
        T: Base
            + SignedOp
            + CoreValue
            + Add<Output = Self>
            + Sub<Output = Self>
            + Mul<Output = Self>
            + Div<Output = Self>
            + Rem<Output = Self>
            + Sum
            + Product
            + PartialEq
            + PartialOrd,
    > UnitCompatible for T
{
}
