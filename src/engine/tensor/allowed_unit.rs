use crate::engine::unit::UnitCompatible;

pub trait AllowedArray: UnitCompatible {}
impl<T: UnitCompatible> AllowedArray for T {}

pub trait AllowedQuant: UnitCompatible {}
impl AllowedQuant for f32 {}
impl AllowedQuant for f64 {}