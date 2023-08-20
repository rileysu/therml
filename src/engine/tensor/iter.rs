use crate::helper::Position;
use super::EngineTensorAccess;

pub struct EngineTensorIterator<'a, T: Sized + Copy> {
    tensor: &'a dyn EngineTensorAccess<T>,
    curr: Position,
    finish: Position,
    ended: bool,
}

impl<'a, T: Sized + Copy> EngineTensorIterator<'a, T> {
    pub fn new(tensor: &'a dyn EngineTensorAccess<T>, start: &Position, finish: &Position) -> Self {
        Self {
            tensor,
            curr: start.clone(),
            finish: finish.clone(),
            ended: false,
        }
    }
}

impl<T: Sized + Copy> Iterator for EngineTensorIterator<'_, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if !self.ended {
            let out = Some(self.tensor.get(&self.curr));

            if self.curr != self.finish {
                self.curr.incdec_mut(self.tensor.shape(), 1);
            } else {
                self.ended = true;
            }

            out
        } else {
            None
        }
    }
}