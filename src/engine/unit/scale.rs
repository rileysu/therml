use super::Base;

pub trait Scale: Base {
    fn scale_single(self, scale: f32) -> Self;
    fn scale_double(self, scale: f64) -> Self;
}

macro_rules! scale {
    ($unit:ty) => {
        impl Scale for $unit {
            fn scale_single(self, scale: f32) -> Self {
                (self as f32 * scale) as $unit
            }
        
            fn scale_double(self, scale: f64) -> Self {
                (self as f64 * scale) as $unit
            }
        }
    };
}

scale!(f32);
scale!(f64);

scale!(i8);
scale!(i16);
scale!(i32);
scale!(i64);
scale!(i128);
scale!(isize);

scale!(u8);
scale!(u16);
scale!(u32);
scale!(u64);
scale!(u128);
scale!(usize);