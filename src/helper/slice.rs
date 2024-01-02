use crate::helper::Shape;

use super::{Position, VarArrayCompatible};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Interval {
    start: Option<usize>,
    finish: Option<usize>,
    step: Option<usize>,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Slice(Box<[Interval]>);

impl Interval {
    pub fn new(start: Option<usize>, finish: Option<usize>, step: Option<usize>) -> Self {
        Self {
            start,
            finish,
            step,
        }
    }

    pub fn start_to(finish: usize) -> Self {
        Self {
            start: None,
            finish: Some(finish),
            step: None,
        }
    }

    pub fn finish_from(start: usize) -> Self {
        Self {
            start: Some(start),
            finish: None,
            step: None,
        }
    }

    pub fn between(start: usize, finish: usize) -> Self {
        Self {
            start: Some(start),
            finish: Some(finish),
            step: None,
        }
    }

    pub fn between_with_step(start: usize, finish: usize, step: usize) -> Self {
        Self {
            start: Some(start),
            finish: Some(finish),
            step: Some(step),
        }
    }

    pub fn all() -> Self {
        Self {
            start: None,
            finish: None,
            step: None,
        }
    }

    pub fn start_index(&self) -> usize {
        self.start.unwrap_or(0)
    }

    pub fn finish_index(&self, dim: usize) -> usize {
        self.finish.unwrap_or(dim)
    }

    pub fn step_index(&self) -> usize {
        self.step.unwrap_or(1)
    }

    pub fn len(&self, dim: usize) -> usize {
        let start_index = self.start_index();
        let finish_index = self.finish_index(dim);
        let step_index = self.step_index();

        (finish_index - start_index) / step_index
    }


}

impl Slice {
    pub fn new(data: Box<[Interval]>) -> Self {
        Self(data)
    }

    pub fn as_boxed_slice(&self) -> &Box<[Interval]> {
        &self.0
    }

    pub fn as_mut_boxed_slice(&mut self) -> &mut Box<[Interval]> {
        &mut self.0
    }

    pub fn inferred_shape(&self, shape: &Shape) -> Shape {
        Shape::new(self.as_boxed_slice().iter().zip(shape.iter()).map(|(interval, dim)| interval.len(dim)).collect())
    }

    pub fn len(&self, shape: &Shape) -> usize {
        self.inferred_shape(shape).len()
    }

    pub fn start(&self) -> Position {
        Position::new(self.as_boxed_slice().iter().map(|interval| interval.start_index()).collect())
    }

    //This is called last to differentiate between finish which wouldn't be a valid position
    pub fn last(&self, shape: &Shape) -> Position {
        Position::new(self.as_boxed_slice().iter().zip(shape.iter()).map(|(interval, dim)| interval.finish_index(dim).saturating_sub(1)).collect())
    }
}

impl From<&[Interval]> for Slice {
    fn from(value: &[Interval]) -> Self {
        Self(Box::from(value))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    //TODO
    #[test]
    fn interval_len() {
        let intervals = [
            Interval::all(),
            Interval::start_to(1),
            Interval::finish_from(1),
        ];
    }
}
