pub trait CoreValue {
    fn zero() -> Self;
    fn one() -> Self;
}

macro_rules! core_value_float {
    ($unit:ty) => {
        impl CoreValue for $unit {
            fn zero() -> Self {
                0.
            }
        
            fn one() -> Self {
                1.
            }
        }
    };
}

macro_rules! core_value_int {
    ($unit:ty) => {
        impl CoreValue for $unit {
            fn zero() -> Self {
                0
            }
        
            fn one() -> Self {
                1
            }
        }
    };
}

core_value_float!{f32}
core_value_float!{f64}

core_value_int!{i8}
core_value_int!{i16}
core_value_int!{i32}
core_value_int!{i64}
core_value_int!{i128}
core_value_int!{isize}

core_value_int!{u8}
core_value_int!{u16}
core_value_int!{u32}
core_value_int!{u64}
core_value_int!{u128}
core_value_int!{usize}