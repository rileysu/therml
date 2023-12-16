use std::fmt::Debug;

use num::Num;

pub trait AllowedUnit: Num + Sized + Copy + Debug + 'static {}
impl<T: Num + Sized + Copy + Debug + 'static> AllowedUnit for T {}

pub trait AllowedArray: AllowedUnit {}
impl<T: AllowedUnit> AllowedArray for T {}

pub trait AllowedQuant: AllowedUnit {}
impl AllowedQuant for f32 {}
impl AllowedQuant for f64 {}