use crate::{engine::unit::UnitCompatible, helper::{Position, Shape, Slice, VarArray, VarArrayCompatible}};
use super::{extension::EmptyExtensionProvider, factory::EngineTensorFactory, generic::EngineTensorGeneric, iter::EngineTensorUnitIterator, Array, EngineTensor};

pub trait AllowedPadded: UnitCompatible {}
impl<T: UnitCompatible> AllowedPadded for T {}

#[derive(Debug)]
pub struct Padded<T: AllowedPadded> {
    tensor: Box<dyn EngineTensor<Unit = T>>,

    //Shape of the underlying tensor model (not sliced)
    shape: Shape,
    //Shape of the tensor that can be accessed
    slice_shape: Shape,
    //Since we don't have a stride we can use this to emulate steps (for slice_shape only)
    steps: VarArray,
    //Absolute is the starting point in the underlying model, relative is the point from the users perspective
    start_abs: Position,

    //Padding is applied using absolute positions only
    //If slicing is desired the underlying tensor should be sliced
    high_padding: VarArray,
    low_padding: VarArray,

    padding_val: T,
}

//TODO figure out how start_abs and padding interact...

impl<T: AllowedPadded> Padded<T> {
    pub fn pad_from(a: Box<dyn EngineTensor<Unit = T>>, padding: VarArray, padding_val: T) -> Self {
        if a.shape().len() == padding.len() {
            let shape = Shape::new(a.shape().iter().zip(padding.iter()).map(|(o, p)| o + 2 * p).collect());
            let slice_shape = shape.clone();
            let steps = VarArray::from_iter(std::iter::repeat(1).take(shape.len()));
            let start_abs = a.shape().first();

            let high_padding = a.shape().iter().zip(padding.iter()).map(|(o, p)| o + p).collect();
            let low_padding = padding.vararray().clone();
            
            Self {
                tensor: a,

                shape,
                slice_shape,
                steps,
                start_abs,

                high_padding,
                low_padding,

                padding_val,
            }
        } else {
            todo!()
        }
    }

    pub fn padding(&self) -> &VarArray {
        //Low padding is the same as the initial padding
        &self.low_padding
    }

    fn relative_to_absolute_pos(&self, rel_pos: &Position) -> Position {
        //TODO add errors
       self.start_abs.add(&rel_pos.mul(&self.steps).unwrap()).unwrap()
    }

    fn abs_pos_in_unpadded_bounds(&self, abs_pos: &Position) -> bool {
        abs_pos.iter().zip(self.low_padding.iter()).zip(self.high_padding.iter()).all(|((pos, low), hi)| (low..hi).contains(&pos))
    }
}

impl<T: AllowedPadded> EngineTensorGeneric for Padded<T> {}

impl<T: AllowedPadded> EngineTensor for Padded<T> {
    type Unit = T;

    fn shape(&self) -> &Shape {
        &self.slice_shape
    }

    fn get(&self, pos: &Position) -> Self::Unit {
        if pos.within_bounds(self.shape()) {
            let abs_pos = self.relative_to_absolute_pos(pos);

            let abs_pos_in_unpadded_bounds = self.abs_pos_in_unpadded_bounds(&abs_pos);

            if abs_pos_in_unpadded_bounds {
                let middle_pos = abs_pos.sub(&self.low_padding).unwrap();

                self.tensor.get(&middle_pos)
            } else {
                self.padding_val
            }
        } else {
            todo!()
        }
    }

    fn iter_units(&self) -> super::iter::EngineTensorUnitIterator<'_, Self::Unit> {
        EngineTensorUnitIterator::new(self)
    }

    fn clone(&self) -> Box<dyn EngineTensor<Unit = Self::Unit>> {
        Box::new(Self {
            tensor: self.tensor.clone(),

            shape: self.shape.clone(),
            slice_shape: self.slice_shape.clone(),
            steps: self.steps.clone(),
            start_abs: self.start_abs.clone(),

            high_padding: self.high_padding.clone(),
            low_padding: self.low_padding.clone(),

            padding_val: self.padding_val,
        })
    }

    //We can handle slices but changing anything more drastic needs a deep copy
    fn slice(&self, slice: &Slice) -> Box<dyn EngineTensor<Unit = Self::Unit>> {
        let slice_shape = slice.inferred_shape(self.shape());

        let steps = VarArray::from_iter(slice.as_boxed_slice().iter().map(|int| int.step()));

        let start_rel = slice.start();
        let start_abs = self.relative_to_absolute_pos(&start_rel);

        Box::new(Self {
            tensor: self.tensor.clone(),

            shape: self.shape.clone(),
            slice_shape,
            steps,
            start_abs,

            high_padding: self.high_padding.clone(),
            low_padding: self.low_padding.clone(),

            padding_val: self.padding_val,
        })
    }

    //Reshaping might be possible without a deep copy
    fn reshape(&self, shape: &Shape) -> Box<dyn EngineTensor<Unit = Self::Unit>> {
        Array::from_iter(self.iter_units(), shape.clone()).generic()
    }

    //This is possible with step = 0
    fn broadcast_splice(&self, pos: usize, sub: &[usize]) -> Box<dyn EngineTensor<Unit = Self::Unit>> {
        let mut slice_shape_buffer = Vec::from(self.slice_shape.as_slice());
        slice_shape_buffer.splice(pos..pos, sub.iter().copied());

        let slice_shape = Shape::from(slice_shape_buffer.as_slice());

        let mut steps_buffer = Vec::from(self.steps.as_slice());
        steps_buffer.splice(pos..pos, std::iter::repeat(0).take(sub.len()));

        let steps = VarArray::from(steps_buffer.as_slice());

        Box::new(Self {
            tensor: self.tensor.clone(),

            shape: self.shape.clone(),
            slice_shape,
            steps,
            start_abs: self.start_abs.clone(),

            high_padding: self.high_padding.clone(),
            low_padding: self.low_padding.clone(),

            padding_val: self.padding_val,
        })
    }

    fn extensions(&self)-> Box<dyn super::extension::ExtensionProvider + '_> {
        Box::new(EmptyExtensionProvider::from(self))
    }
}

#[cfg(test)]
mod test {
    use crate::helper::{shape, varr};

    use super::*;

    fn create_examples() -> Vec<Padded<f32>> {
        vec![
            //Padded::pad_from(Array::from_iter([0.0; 0].iter().copied(), shape![]), varr![], 0.0),
            Padded::pad_from(Array::from_iter((1..=9).map(|x| x as f32 / 9.0), shape![9]).generic(), varr![1], 0.0),
            Padded::pad_from(Array::from_iter((1..=9).map(|x| x as f32 / 9.0), shape![3, 3]).generic(), varr![1, 1], 0.0),
            Padded::pad_from(Array::from_iter((1..=105).map(|x| x as f32 / 105.0), shape![3, 5, 7]).generic(), varr![1, 2, 3], 0.0),
            Padded::pad_from(Array::from_iter((1..=105).map(|x| x as f32 / 105.0), shape![1, 1, 3, 5, 7]).generic(), varr![0, 1, 1, 2, 3], 0.0),
        ]
    }

    #[test]
    fn create_and_get() {
        let examples = create_examples();

        for example in examples {

            let shape = example.shape();

            println!("{:?}", Array::from_iter(example.iter_units(), shape.clone()));

            for curr_pos in shape.first().iter_positions(&shape.last(), &shape) {
                //println!("{:?}", curr_pos);

                let abs_pos = example.relative_to_absolute_pos(&curr_pos);

                if example.abs_pos_in_unpadded_bounds(&abs_pos) {

                } else {
                    assert_eq!(example.get(&curr_pos), 0.0);
                }
            }
        }
    }
}