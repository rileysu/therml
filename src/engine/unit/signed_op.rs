use super::Base;

pub trait SignedOp: Base {
    fn abs(self) -> Self;
    fn neg(self) -> Self;
}

macro_rules! signed_op_signed {
    ($unit:ty) => {
        impl SignedOp for $unit {
            fn abs(self) -> Self {
                self.abs()
            }

            fn neg(self) -> Self {
                -self
            }
        }
    };
}

macro_rules! signed_op_unsigned {
    ($unit:ty) => {
        impl SignedOp for $unit {
            fn abs(self) -> Self {
                self
            }

            fn neg(self) -> Self {
                self
            }
        }
    };
}

signed_op_signed!{f32}
signed_op_signed!{f64}

signed_op_signed!{i8}
signed_op_signed!{i16}
signed_op_signed!{i32}
signed_op_signed!{i64}
signed_op_signed!{i128}
signed_op_signed!{isize}

signed_op_unsigned!{u8}
signed_op_unsigned!{u16}
signed_op_unsigned!{u32}
signed_op_unsigned!{u64}
signed_op_unsigned!{u128}
signed_op_unsigned!{usize}