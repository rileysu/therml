#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Shape(Box<[usize]>);

impl Shape {
    pub fn as_boxed_slice(&self) -> &Box<[usize]> {
        &self.0
    }

    pub fn total_elements(&self) -> usize {
        self.0.iter().product()
    }
}

impl From<&[usize]> for Shape {
    fn from(value: &[usize]) -> Self {
        Self(Box::from(value))
    }
}