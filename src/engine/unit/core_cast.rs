use super::Base;

pub trait CoreCast<T>: Base {
    fn from(from: T) -> Self;
    fn to(self) -> T;
}

macro_rules! core_cast {
    ($unit:ty) => {
        impl CoreCast<usize> for $unit {
            fn from(from: usize) -> Self {
                from as Self
            }

            fn to(self) -> usize {
                self as usize
            }
        }
    };
}

core_cast!(f32);
core_cast!(f64);

core_cast!(i8);
core_cast!(i16);
core_cast!(i32);
core_cast!(i64);
core_cast!(i128);
core_cast!(isize);

core_cast!(u8);
core_cast!(u16);
core_cast!(u32);
core_cast!(u64);
core_cast!(u128);
core_cast!(usize);