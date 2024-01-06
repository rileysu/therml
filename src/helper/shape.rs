use super::{VarArray, VarArrayCompatible};

use std::fmt::Display;

use crate::helper::Position;

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Shape(VarArray);

impl Shape {
    pub fn new(data: VarArray) -> Self {
        Self(data)
    }

    pub fn elements(&self) -> usize {
        self.0.iter().product()
    }

    //first valid position
    pub fn first(&self) -> Position {
        Position::new(vec![0; self.len()].as_slice().into())
    }

    //last valid position
    pub fn last(&self) -> Position {
        Position::new(self.iter().map(|x| x - 1).collect())
    }
}

impl VarArrayCompatible for Shape {
    fn new(varr: VarArray) -> Self {
        Self(varr)
    }

    fn vararray(&self) -> &VarArray {
        &self.0
    }

    fn vararray_mut(&mut self) -> &mut VarArray {
        &mut self.0
    }
}

impl From<&[usize]> for Shape {
    fn from(value: &[usize]) -> Self {
        Self(VarArray::from(value))
    }
}

impl Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let conv_sizes: Box<[String]> = self.iter().map(|x| x.to_string()).collect();

        write!(f, "({})", conv_sizes.join(","))
    }
}

macro_rules! shape {
    () => {
        Shape::from([].as_slice())
    };
    ($($x:expr),+) => {
        Shape::from([$($x),+].as_slice())
    };
}
pub(crate) use shape;

