use crate::helper::{Shape, VarArrayCompatible};

use super::Position;

//Cloning is better in this case since we modify pos in place
//Until is inclusive because positions can't exceed bounds of a shape which would make iterating over an entire shape impossible otherwise
pub struct Iter<'a> {
    pos: Position,
    until: &'a Position,
    bounds: &'a Shape,
    is_done: bool,
}

impl<'a> Iter<'a> {
    pub fn new(pos: Position, until: &'a Position, bounds: &'a Shape) -> Self {
        Self {
            pos,
            until,
            bounds,
            is_done: bounds.len() == 0, //If it's an empty shape then nothing to iterate on
        }
    }
}


impl<'a> Iterator for Iter<'a> {
    type Item = Position;

    fn next(&mut self) -> Option<Self::Item> {
        if !self.is_done {
            self.pos.incdec_mut(self.bounds, 1).unwrap();

            if self.pos == *self.until {
                self.is_done = true;
            }

            Some(self.pos.clone())
        } else {
            None
        }
    }
}