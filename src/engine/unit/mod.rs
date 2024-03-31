use std::{
    fmt::Debug,
    iter::{Product, Sum},
    ops::{Add, Div, Mul, Rem, Sub},
};

use self::{core_cast::CoreCast, core_func::CoreFunc, core_value::CoreValue, exponential_op::ExponentialOp, scale::Scale, signed_op::SignedOp};

pub mod core_cast;
pub mod core_func;
pub mod core_value;
pub mod exponential_op;
pub mod scale;
pub mod signed_op;

pub trait Base: Sized + Copy + Debug + 'static {}
impl<T: Sized + Copy + Debug + 'static> Base for T {}

pub trait UnitCompatible:
    Base
    + SignedOp
    + CoreCast<usize>
    + CoreFunc
    + CoreValue
    + ExponentialOp
    + Scale
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
            + CoreCast<usize>
            + CoreFunc
            + CoreValue
            + ExponentialOp
            + Scale
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
