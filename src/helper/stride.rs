use crate::helper::Shape;

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Stride(Box<[usize]>);

impl Stride {
    pub fn as_boxed_slice(&self) -> &Box<[usize]> {
        &self.0
    }
}

impl From<&[usize]> for Stride {
    fn from(value: &[usize]) -> Self {
        Self(Box::from(value))
    }
}

impl From<Shape> for Stride {
    fn from(value: Shape) -> Self {
        let mut stride: Vec<usize> = Vec::with_capacity(value.as_boxed_slice().len());

        let mut next = 1usize;
        for dim in value.as_boxed_slice().iter().rev() {
            stride.push(next);
            next = dim * next;
        }

        stride.reverse();

        Stride(stride.into_boxed_slice())
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn from_shape_examples() {
        let examples = [
            (Shape::from([1].as_slice()), Stride::from([1].as_slice())),
            (Shape::from([20, 50, 4].as_slice()), Stride::from([200, 4, 1].as_slice()))
        ];

        for (shape, stride) in examples {
            assert_eq!(Stride::from(shape), stride);
        }
    }
}