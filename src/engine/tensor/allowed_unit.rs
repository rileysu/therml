use num::Num;

pub trait AllowedUnit: Num + Sized + Copy {}
impl<T: Num + Sized + Copy> AllowedUnit for T {}

pub trait AllowedArray: AllowedUnit {}
impl<T: AllowedUnit> AllowedArray for T {}

pub trait AllowedQuant: AllowedUnit {}
impl AllowedQuant for f32 {}
impl AllowedQuant for f64 {}