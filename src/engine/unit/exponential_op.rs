use super::Base;

pub trait ExponentialOp: Base {
    fn exp(self) -> Self;
}

macro_rules! exponential_op_int {
    ($unit:ty) => {
        impl ExponentialOp for $unit {
            fn exp(self) -> Self {
                (self as f64).exp() as $unit
            }
        }
    };
}

impl ExponentialOp for f32 {
    fn exp(self) -> Self {
        self.exp()
    }
}

impl ExponentialOp for f64 {
    fn exp(self) -> Self {
        self.exp()
    }
}

exponential_op_int!(i8);
exponential_op_int!(i16);
exponential_op_int!(i32);
exponential_op_int!(i64);
exponential_op_int!(i128);
exponential_op_int!(isize);

exponential_op_int!(u8);
exponential_op_int!(u16);
exponential_op_int!(u32);
exponential_op_int!(u64);
exponential_op_int!(u128);
exponential_op_int!(usize);