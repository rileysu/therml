use std::{iter::{Product, Sum}, ops::{Add, Div, Mul, Rem, Sub}};

use super::{core_value::CoreValue, exponential_op::ExponentialOp, scale::Scale, Base};

pub trait CoreFunc: CoreValue + ExponentialOp + Scale + Add<Output = Self>
+ Sub<Output = Self>
+ Mul<Output = Self>
+ Div<Output = Self>
+ Rem<Output = Self>
+ Sum
+ Product
+ PartialEq
+ PartialOrd
+ Base  {
    //Activation
    fn relu(self) -> Self {
        if self > Self::zero() { self } else { Self::zero() }
    }
    fn leaky_relu(self, alpha: f64) -> Self {
        if self > Self::zero() { self } else { self.scale_double(alpha) }
    }
    fn sigmoid(self) -> Self {
        let exp = self.exp();

        exp / (Self::one() + exp)
    }

    //Gen
    fn sqrt(self) -> Self;

    //Trig
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn tan(self) -> Self;

    fn sinh(self) -> Self;
    fn cosh(self) -> Self;
    fn tanh(self) -> Self;

    fn asin(self) -> Self;
    fn acos(self) -> Self;
    fn atan(self) -> Self;

    fn asinh(self) -> Self;
    fn acosh(self) -> Self;
    fn atanh(self) -> Self;
}

macro_rules! core_func_float {
    ($unit:ty) => {
        impl CoreFunc for $unit {
            fn sqrt(self) -> Self {
                self.sqrt()
            }

            fn sin(self) -> Self {
                self.sin()
            }
        
            fn cos(self) -> Self {
                self.cos()
            }
        
            fn tan(self) -> Self {
                self.tan()
            }
        
            fn sinh(self) -> Self {
                self.sinh()
            }
        
            fn cosh(self) -> Self {
                self.cosh()
            }
        
            fn tanh(self) -> Self {
                self.tanh()
            }
        
            fn asin(self) -> Self {
                self.asin()
            }
        
            fn acos(self) -> Self {
                self.acos()
            }
        
            fn atan(self) -> Self {
                self.atan()
            }
        
            fn asinh(self) -> Self {
                self.asinh()
            }
        
            fn acosh(self) -> Self {
                self.acosh()
            }
        
            fn atanh(self) -> Self {
                self.atanh()
            }
        }
    };
}

macro_rules! core_func_int {
    ($unit:ty) => {
        impl CoreFunc for $unit {
            fn sqrt(self) -> Self {
                (self as f64).sqrt() as $unit
            }

            fn sin(self) -> Self {
                (self as f64).sin() as $unit
            }
        
            fn cos(self) -> Self {
                (self as f64).cos() as $unit
            }
        
            fn tan(self) -> Self {
                (self as f64).tan() as $unit
            }
        
            fn sinh(self) -> Self {
                (self as f64).sinh() as $unit
            }
        
            fn cosh(self) -> Self {
                (self as f64).cosh() as $unit
            }
        
            fn tanh(self) -> Self {
                (self as f64).tanh() as $unit
            }
        
            fn asin(self) -> Self {
                (self as f64).asin() as $unit
            }
        
            fn acos(self) -> Self {
                (self as f64).acos() as $unit
            }
        
            fn atan(self) -> Self {
                (self as f64).atan() as $unit
            }
        
            fn asinh(self) -> Self {
                (self as f64).asinh() as $unit
            }
        
            fn acosh(self) -> Self {
                (self as f64).acosh() as $unit
            }
        
            fn atanh(self) -> Self {
                (self as f64).atanh() as $unit
            }
        }
    };
}

core_func_float!(f32);
core_func_float!(f64);

core_func_int!(i8);
core_func_int!(i16);
core_func_int!(i32);
core_func_int!(i64);
core_func_int!(i128);
core_func_int!(isize);

core_func_int!(u8);
core_func_int!(u16);
core_func_int!(u32);
core_func_int!(u64);
core_func_int!(u128);
core_func_int!(usize);