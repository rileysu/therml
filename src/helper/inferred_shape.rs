use super::Shape;

//Inferred pos is the position where the inferred dimension is.
//eg: 
//  0 means it is before the first dimension
//  in a 3 len dims, inferred_pos = 3 means the last dimension is inferred
struct InferredShape {
    inferred_pos: usize,
    dims: Box<[usize]>,
}

impl InferredShape {
    pub fn new(inferred_pos: usize, dims: Box<[usize]>) -> Self {
        Self {
            inferred_pos,
            dims,
        }
    }

    pub fn infer(&self, length: usize) -> Shape {
        let inferred = length / self.dims.iter().product::<usize>();

        let mut new_dims = self.dims.to_vec();
        new_dims.insert(self.inferred_pos, inferred);

        Shape::from(new_dims.as_slice())
    }
}

impl From<(&[usize], &[usize])> for InferredShape {
    fn from(value: (&[usize], &[usize])) -> Self {
        Self {
            inferred_pos: value.0.len(),
            dims: value.0.iter().chain(value.1.iter()).copied().collect()
        }
    }
}