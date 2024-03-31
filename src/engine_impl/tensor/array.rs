use std::{iter, sync::Arc};

use crate::{
    engine::{
        tensor::{
            builder::EngineTensorBuilder, extension::{EmptyExtensionProvider, ExtensionProvider}, factory::EngineTensorFactory, sub_tensor_iter::EngineTensorSubTensorIterator, unit_iter::EngineTensorUnitIterator, EngineTensor
        },
        unit::UnitCompatible,
    },
    helper::{Interval, Position, Shape, Slice, Stride, VarArrayCompatible},
};

#[derive(Debug)]
pub struct Array<T: AllowedArray> {
    stride: Stride,
    shape: Shape,
    data: Arc<[T]>,
    offset: usize,
}

pub trait AllowedArray: UnitCompatible {}
impl<T: UnitCompatible> AllowedArray for T {}

impl<T: AllowedArray> Array<T> {
    pub fn from_data(stride: Stride, shape: Shape, data: Arc<[T]>, offset: usize) -> Self {
        Self {
            stride,
            shape,
            data,
            offset,
        }
    }

    //Am I dumb?
    //This is wrong!!!
    //TODO
    fn is_contiguous(&self) -> bool {
        let mut check: Option<usize> = None;

        for curr in self.stride.iter() {
            match check {
                Some(prev) => {
                    if prev * prev == curr {
                        check = Some(curr);
                    } else {
                        return false;
                    }
                }
                None => {
                    check = Some(curr);
                }
            }
        }

        return true;
    }
}

impl<T: AllowedArray> EngineTensor for Array<T> {
    type Unit = T;

    fn shape(&self) -> &Shape {
        &self.shape
    }

    fn get(&self, pos: &Position) -> T {
        let index = pos.tensor_index(&self.stride).unwrap() + self.offset;

        *self.data.as_ref().get(index).unwrap()
    }

    fn iter_units(&self) -> EngineTensorUnitIterator<'_, T> {
        EngineTensorUnitIterator::new(self)
    }

    fn iter_sub_tensor(&self, intervals: &[Interval]) -> EngineTensorSubTensorIterator<'_, Self::Unit> {
        let slice = Slice::new(intervals.into(), self.shape().clone());

        EngineTensorSubTensorIterator::new(self, slice).unwrap()
    }

    fn clone(&self) -> Box<dyn EngineTensor<Unit = Self::Unit>> {
        Box::new(Self {
            stride: self.stride.clone(),
            shape: self.shape.clone(),
            data: self.data.clone(),
            offset: self.offset,
        })
    }

    fn mat(&self) -> Box<dyn EngineTensor<Unit = Self::Unit>> {
        Array::from_iter(self.iter_units(), self.shape().clone()).generic()
    }

    fn slice(&self, intervals: &[Interval]) -> Box<dyn EngineTensor<Unit = T>> {
        let slice = Slice::new(intervals.into(), self.shape().clone());

        let offset = slice.start().tensor_index(&self.stride).unwrap();

        Box::from(Self {
            stride: self.stride.clone(),
            shape: slice.inferred_shape(),
            data: self.data.clone(),
            offset,
        })
    }

    //Attempts to efficiently reuse memory if tensor is contiguous
    //If this is not an option it will copy from an iterator
    fn reshape(&self, shape: &Shape) -> Box<dyn EngineTensor<Unit = T>> {
        if shape.elements() == self.shape().elements() {
            if self.is_contiguous() {
                Box::new(Array::<T> {
                    stride: Stride::default_from_shape(shape),
                    shape: shape.clone(),
                    data: self.data.clone(),
                    offset: self.offset,
                })
            } else {
                Array::<T>::from_iter(self.iter_units(), shape.clone()).generic()
            }
        } else {
            todo!()
        }
    }

    fn trim(&self) -> Box<dyn EngineTensor<Unit = Self::Unit>> {
        let removing_indices = self.shape().iter().enumerate().filter_map(|(ind, dim)| if dim > 1 { Some(ind) } else { None }).collect::<Box<[usize]>>();

        let new_stride = Stride::from_iter(self.stride.iter().enumerate().filter_map(|(ind, s)| if removing_indices.contains(&ind) { None } else { Some(s) }));
        let new_shape = Shape::from_iter(self.shape().iter().enumerate().filter_map(|(ind, dim)| if removing_indices.contains(&ind) { None } else { Some(dim) }));

        Box::new(Array::<T> {
            stride: new_stride,
            shape: new_shape,
            data: self.data.clone(),
            offset: self.offset,
        })
    }

    fn broadcast_splice(
        &self,
        pos: usize,
        sub: &[usize],
    ) -> Box<dyn EngineTensor<Unit = Self::Unit>> {
        if pos <= self.shape().len() {
            let mut shape_buffer = self.shape().as_slice().to_vec();
            shape_buffer.splice(pos..pos, sub.iter().copied());

            let broadcast_shape = Shape::new(shape_buffer.as_slice().into());

            let mut stride_buffer = self.stride.as_slice().to_vec();
            stride_buffer.splice(pos..pos, std::iter::repeat(0).take(sub.len()));

            let broadcast_stride = Stride::new(stride_buffer.as_slice().into());

            Box::new(Self {
                stride: broadcast_stride,
                shape: broadcast_shape,
                data: self.data.clone(),
                offset: self.offset,
            })
        } else {
            todo!()
        }
    }

    fn extensions(&self) -> Box<dyn ExtensionProvider + '_> {
        Box::from(EmptyExtensionProvider::from(self))
    }
}

impl<T: AllowedArray> EngineTensorFactory for Array<T> {
    fn from_iter(iter: impl Iterator<Item = T>, shape: Shape) -> Self {
        Array {
            stride: Stride::default_from_shape(&shape),
            shape,
            data: iter.collect(),
            offset: 0,
        }
    }

    fn from_slice(data: &[T], shape: Shape) -> Self {
        Array {
            stride: Stride::default_from_shape(&shape),
            shape,
            data: Arc::from(data),
            offset: 0,
        }
    }
    
    fn builder(shape: Shape, init: T) -> impl EngineTensorBuilder<Unit = Self::Unit> {
        ArrayBuilder::new(shape, init)
    }
}

struct ArrayBuilder<T: AllowedArray> {
    shape: Shape,
    data: Vec<T>,
}

impl<T: AllowedArray> EngineTensorBuilder for ArrayBuilder<T> {
    type Unit = T;
    type Tensor = Array<Self::Unit>;

    fn new(shape: Shape, init: Self::Unit) -> Self {
        let elements = shape.elements();

        Self {
            shape,
            data: Vec::from_iter(iter::repeat(init).take(elements)),
        }
    }

    fn splice_slice<I: IntoIterator<Item = Self::Unit>>(&mut self, intervals: &[Interval], replace_with: I) {
        let slice = Slice::new(intervals.into(), self.shape.clone());
        let default_stride = Stride::default_from_shape(&self.shape);

        for (pos, replace) in slice.iter().zip(replace_with) {
            *self.data.get_mut(pos.tensor_index(&default_stride).unwrap()).unwrap() = replace;
        }
    }
    
    fn splice_between_positions<I: IntoIterator<Item = Self::Unit>>(&mut self, start: &Position, last: &Position, replace_with: I) {
        let default_stride = Stride::default_from_shape(&self.shape);

        self.data.splice((start.tensor_index(&default_stride).unwrap())..=(last.tensor_index(&default_stride).unwrap()), replace_with);
    }

    fn construct(self) -> Self::Tensor {
        Array::from_data(
            Stride::default_from_shape(&self.shape),
            self.shape,
            self.data.into(),
            0,
        )
    }
}
