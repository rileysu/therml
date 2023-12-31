use crate::helper::Position;
use super::{allowed_unit::AllowedUnit, EngineTensor};

/*pub struct EngineTensorIterator<'a, T: AllowedUnit> {
    tensor: &'a dyn EngineTensor<Unit = T>,
    curr: usize,
    ended: bool,
}

impl<'a, T: AllowedUnit> EngineTensorIterator<'a, T> {
    pub fn new(tensor: &'a dyn EngineTensor<Unit = T>) -> Self {
        Self {
            tensor,
            curr: *tensor.shape().as_boxed_slice().first().unwrap(),
        }
    }
}

impl<'a, T: AllowedUnit> Iterator for EngineTensorIterator<'a, T> {
    type Item = &'a dyn EngineTensor<Unit = T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.curr < *self.tensor.shape().as_boxed_slice().first().unwrap() {
            let mut pos = self.tensor.shape().first();
            *pos.as_mut_boxed_slice().get_mut(0).unwrap() = self.curr;

            let out = Some(self.tensor.get(pos));

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
}*/

//TODO basic impl that isn't optimised
//It can be enhanced by fetching chunks of contig memory if available
pub struct EngineTensorUnitIterator<'a, T: AllowedUnit> {
    tensor: &'a dyn EngineTensor<Unit = T>,
    curr: Position,
    finish: Position,
    ended: bool,
}

impl<'a, T: AllowedUnit> EngineTensorUnitIterator<'a, T> {
    pub fn new(tensor: &'a dyn EngineTensor<Unit = T>) -> Self {
        Self {
            tensor,
            curr: tensor.shape().first(),
            finish: tensor.shape().last(),
            ended: false,
        }
    }
}

impl<T: AllowedUnit> Iterator for EngineTensorUnitIterator<'_, T> {
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