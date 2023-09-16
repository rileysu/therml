pub trait AllowedUnit: Sized + Copy {}
impl AllowedUnit for f32 {}
impl AllowedUnit for f64 {}
impl AllowedUnit for i8 {}
impl AllowedUnit for i16 {}
impl AllowedUnit for i32 {}
impl AllowedUnit for i64 {}
impl AllowedUnit for u8 {}
impl AllowedUnit for u16 {}
impl AllowedUnit for u32 {}
impl AllowedUnit for u64 {}

pub trait AllowedArray: AllowedUnit {}
impl<T: AllowedUnit> AllowedArray for T {}

pub trait AllowedQuant: AllowedUnit {}
impl AllowedQuant for f32 {}
impl AllowedQuant for f64 {}