use std::{ops::Index, fmt::Display};

use crate::helper::Position;

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Shape(Box<[usize]>);

impl Shape {
    pub fn new(data: Box<[usize]>) -> Self {
        Self(data)
    }

    pub fn as_boxed_slice(&self) -> &Box<[usize]> {
        &self.0
    }

    pub fn as_mut_boxed_slice(&mut self) -> &mut Box<[usize]> {
        &mut self.0
    }

    pub fn len(&self) -> usize {
        self.0.iter().product()
    }

    //first valid position
    pub fn first(&self) -> Position {
        Position::new(vec![0; self.as_boxed_slice().len()].into())
    }

    //last valid position
    pub fn last(&self) -> Position {
        Position::new(self.as_boxed_slice().iter().map(|x| x - 1).collect())
    }
}

impl From<&[usize]> for Shape {
    fn from(value: &[usize]) -> Self {
        Self(Box::from(value))
    }
}

impl Index<usize> for Shape {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let conv_sizes: Box<[String]> = self.0.iter().map(|x| x.to_string()).collect();

        write!(f, "({})", conv_sizes.join(","))
    }
}