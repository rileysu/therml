use crate::helper::Position;
use super::{allowed_unit::AllowedUnit, EngineTensor};

//TODO basic impl that isn't optimised
//It can be enhanced by fetching chunks of contig memory if available

pub struct EngineTensorIterator<'a, T: AllowedUnit> {
    tensor: &'a EngineTensor<T>,
    curr: Position,
    finish: Position,
    ended: bool,
}

impl<'a, T: AllowedUnit> EngineTensorIterator<'a, T> {
    pub fn new(tensor: &'a EngineTensor<T>) -> Self {
        Self {
            tensor,
            curr: tensor.shape().first(),
            finish: tensor.shape().last(),
            ended: false,
        }
    }
}

impl<T: AllowedUnit> Iterator for EngineTensorIterator<'_, T> {
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