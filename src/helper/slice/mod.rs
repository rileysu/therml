use crate::helper::Shape;

use self::iter::Iter;

use super::{Position, VarArrayCompatible};

pub mod iter;

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Interval {
    start: Option<usize>,
    end: Option<usize>,
    step: Option<usize>,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Slice {
    intervals: Box<[Interval]>,
    shape: Shape,
}

impl Interval {
    pub fn new(start: Option<usize>, end: Option<usize>, step: Option<usize>) -> Self {
        Self {
            start,
            end,
            step,
        }
    }

    pub fn start_to(end: usize) -> Self {
        Self {
            start: None,
            end: Some(end),
            step: None,
        }
    }

    pub fn end_from(start: usize) -> Self {
        Self {
            start: Some(start),
            end: None,
            step: None,
        }
    }

    pub fn between(start: usize, end: usize) -> Self {
        Self {
            start: Some(start),
            end: Some(end),
            step: None,
        }
    }

    pub fn between_with_step(start: usize, end: usize, step: usize) -> Self {
        Self {
            start: Some(start),
            end: Some(end),
            step: Some(step),
        }
    }

    pub fn only(start: usize) -> Self {
        Self {
            start: Some(start),
            end: Some(start + 1),
            step: None,
        }
    }

    pub fn all() -> Self {
        Self {
            start: None,
            end: None,
            step: None,
        }
    }

    pub fn start_index(&self) -> usize {
        self.start.unwrap_or(0)
    }

    pub fn end_index(&self, dim: usize) -> usize {
        self.end.unwrap_or(dim)
    }

    pub fn step(&self) -> usize {
        self.step.unwrap_or(1)
    }

    pub fn len(&self, dim: usize) -> usize {
        let start_index = self.start_index();
        let finish_index = self.end_index(dim);
        let step_index = self.step();

        (finish_index - start_index) / step_index
    }


}

impl Slice {
    pub fn new(intervals: Box<[Interval]>, shape: Shape) -> Self {
        Self {
            intervals,
            shape,
        }
    }

    pub fn as_boxed_slice(&self) -> &Box<[Interval]> {
        &self.intervals
    }

    pub fn as_mut_boxed_slice(&mut self) -> &mut Box<[Interval]> {
        &mut self.intervals
    }

    //Should be possible to have a different len to self.shape
    pub fn inferred_shape(&self) -> Shape {
        Shape::new(self.as_boxed_slice().iter().zip(self.shape.iter()).map(|(interval, dim)| interval.len(dim)).collect())
    }

    pub fn len(&self) -> usize {
        self.inferred_shape().len()
    }

    pub fn elements(&self) -> usize {
        self.inferred_shape().elements()
    }

    pub fn start(&self) -> Position {
        Position::new(self.as_boxed_slice().iter().map(|interval| interval.start_index()).collect())
    }

    //This is called last to differentiate between end which wouldn't be a valid position
    pub fn last(&self) -> Position {
        Position::new(self.as_boxed_slice().iter().zip(self.shape.iter()).map(|(interval, dim)| interval.end_index(dim).saturating_sub(1)).collect())
    }

    pub fn iter(&self) -> Iter {
        Iter::new(self)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn interval_create_and_properties() {
        let dim = 512usize;

        let i0 = Interval::start_to(8);

        assert_eq!(i0.start_index(), 0);
        assert_eq!(i0.end_index(dim), 8);
        assert_eq!(i0.step(), 1);
        assert_eq!(i0.len(dim), 8);
        
        let start = 256usize;
        let i1 = Interval::end_from(start);

        assert_eq!(i1.start_index(), start);
        assert_eq!(i1.end_index(dim), dim);
        assert_eq!(i1.step(), 1);
        assert_eq!(i1.len(dim), dim - start);

        let start = 256usize - 128usize;
        let end = 256usize + 128usize;
        let i2 = Interval::between(start, end);

        assert_eq!(i2.start_index(), start);
        assert_eq!(i2.end_index(dim), end);
        assert_eq!(i2.step(), 1);
        assert_eq!(i2.len(dim), end - start);

        let start = 256usize - 128usize;
        let end = 256usize + 128usize;
        let step = 2usize;
        let i3 = Interval::between_with_step(start, end, step);

        assert_eq!(i3.start_index(), start);
        assert_eq!(i3.end_index(dim), end);
        assert_eq!(i3.step(), step);
        assert_eq!(i3.len(dim), (end - start) / step);

        let start = 256usize;
        let i4 = Interval::only(start);

        assert_eq!(i4.start_index(), start);
        assert_eq!(i4.end_index(dim), start + 1);
        assert_eq!(i4.step(), 1);
        assert_eq!(i4.len(dim), 1);

        let i5 = Interval::all();

        assert_eq!(i5.start_index(), 0);
        assert_eq!(i5.end_index(dim), dim);
        assert_eq!(i5.step(), 1);
        assert_eq!(i5.len(dim), dim);       
    }
}
