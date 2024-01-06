use std::ops::IndexMut;

use itertools::Itertools;

use crate::{engine::{Engine, EngineError, EngineTensorFactory, EngineTensor, util::{err_if_incorrect_dimensions, err_if_dimension_mismatch}}, helper::{Position, Interval, Slice, Shape, Stride, VarArrayCompatible}};

pub struct Basic {}

macro_rules! conv_fn {
    ($unit:ty) => {
        fn conv2d<E: EngineTensorFactory<Unit = $unit>>(a: &dyn EngineTensor<Unit = $unit>, kernel: &dyn EngineTensor<Unit = $unit>, stride: usize) -> Result<Box<dyn EngineTensor<Unit = $unit>>, crate::engine::EngineError> {
            //a: (batches, in_channels, y, x)
            //kernel: (out_channels, in_channels, k_y, k_x)
        
            err_if_incorrect_dimensions(a.shape(), 4)?;
            err_if_incorrect_dimensions(kernel.shape(), 4)?;
            err_if_dimension_mismatch(a.shape().get(1).unwrap(), kernel.shape().get(1).unwrap())?;
        
            let y = a.shape().get(a.shape().len() - 2).unwrap();
            let x = a.shape().get(a.shape().len() - 1).unwrap();
            let k_y = kernel.shape().get(kernel.shape().len() - 2).unwrap() + 2 * (stride - 1);
            let k_x = kernel.shape().get(kernel.shape().len() - 1).unwrap() + 2 * (stride - 1);
        
            let batches = a.shape().get(0).unwrap();
            let out_channels = kernel.shape().get(0).unwrap();
            let in_channels = kernel.shape().get(1).unwrap();
        
            //(batches, out_channels, in_channels, y, x)
            let a_broadcast = a.broadcast_splice(1, &[out_channels]);
            //(batches, out_channels, in_channels, k_y, k_x)
            let kernel_broadcast = kernel.broadcast_splice(0, &[batches]);
            
            let half_k_y = k_y / 2;
            let half_k_x = k_x / 2;
        
            let y_out = y - half_k_y * 2;
            let x_out = x - half_k_x * 2;
            let out_shape = Shape::from([batches, out_channels, y_out, x_out].as_slice());
            let out_stride = Stride::default_from_shape(&out_shape);
        
            let mut reordered_sums: Box<[$unit]> = vec![<$unit>::default(); batches * out_channels * y_out * x_out].into_boxed_slice();
        
            for curr_y in 0..y_out {
                for curr_x in 0..x_out {
                    let a_sliced = a_broadcast.slice(&Slice::from([Interval::all(), Interval::all(), Interval::all(), Interval::between_with_step(curr_y, curr_y + k_y, stride), Interval::between_with_step(curr_x, curr_x + k_x, stride)].as_slice()));
        
                    let chunked_products = a_sliced.iter_units().zip(kernel_broadcast.iter_units()).map(|(a_elem, k_elem)| a_elem * k_elem).chunks(in_channels * k_y * k_x);
                    let curr_sums = chunked_products.into_iter().map(|i| -> $unit { i.sum() });
        
                    for (i, sum) in curr_sums.enumerate() {
                        let batch = i / out_channels;
                        let out_channel = i % out_channels;
        
                        let index = Position::from([batch, out_channel, curr_y, curr_x].as_slice()).tensor_index(&out_stride)?;
        
                        *reordered_sums.index_mut(index) = sum;
                    }
                }
            }
        
            //(batches, out_channels, y_out, x_out)
            Ok(E::from_slice(&reordered_sums, out_shape))
        }
    };
}

macro_rules! basic_impl {
    ($unit:ty) => {
        impl Engine<$unit> for Basic {

            //Pointwise Single
            fn abs<E: EngineTensorFactory<Unit = $unit>>(a: &dyn EngineTensor<Unit = $unit>) -> Result<Box<dyn EngineTensor<Unit = $unit>>, crate::engine::EngineError> {
        
                Ok(E::from_iter(a.iter_units().map(|x| x.abs()), a.shape().clone()))
            }
        
            fn neg<E: EngineTensorFactory<Unit = $unit>>(a: &dyn EngineTensor<Unit = $unit>) -> Result<Box<dyn EngineTensor<Unit = $unit>>, crate::engine::EngineError> {
                Ok(E::from_iter(a.iter_units().map(|x| -x), a.shape().clone()))
            }

            //Scalar
            fn add_scalar<E: EngineTensorFactory<Unit = $unit>>(s: $unit, a: &dyn EngineTensor<Unit = $unit>) -> Result<Box<dyn EngineTensor<Unit = $unit>>, EngineError> {
                Ok(E::from_iter(a.iter_units().map(|x| x + s), a.shape().clone()))
            }

            fn sub_scalar_lh<E: EngineTensorFactory<Unit = $unit>>(s: $unit, a: &dyn EngineTensor<Unit = $unit>) -> Result<Box<dyn EngineTensor<Unit = $unit>>, EngineError> {
                Ok(E::from_iter(a.iter_units().map(|x| s - x), a.shape().clone()))
            }

            fn sub_scalar_rh<E: EngineTensorFactory<Unit = $unit>>(a: &dyn EngineTensor<Unit = $unit>, s: $unit) -> Result<Box<dyn EngineTensor<Unit = $unit>>, EngineError> {
                Ok(E::from_iter(a.iter_units().map(|x| x - s), a.shape().clone()))
            }

            fn mul_scalar<E: EngineTensorFactory<Unit = $unit>>(s: $unit, a: &dyn EngineTensor<Unit = $unit>) -> Result<Box<dyn EngineTensor<Unit = $unit>>, EngineError> {
                Ok(E::from_iter(a.iter_units().map(|x| x * s), a.shape().clone()))
            }

            fn div_scalar_lh<E: EngineTensorFactory<Unit = $unit>>(s: $unit, a: &dyn EngineTensor<Unit = $unit>) -> Result<Box<dyn EngineTensor<Unit = $unit>>, EngineError> {
                Ok(E::from_iter(a.iter_units().map(|x| s / x), a.shape().clone()))
            }

            fn div_scalar_rh<E: EngineTensorFactory<Unit = $unit>>(a: &dyn EngineTensor<Unit = $unit>, s: $unit) -> Result<Box<dyn EngineTensor<Unit = $unit>>, EngineError> {
                Ok(E::from_iter(a.iter_units().map(|x| x / s), a.shape().clone()))
            }
        
            //Pointwise Double
            fn add<E: EngineTensorFactory<Unit = $unit>>(a: &dyn EngineTensor<Unit = $unit>, b: &dyn EngineTensor<Unit = $unit>) -> Result<Box<dyn EngineTensor<Unit = $unit>>, crate::engine::EngineError> {
                if a.shape() == b.shape() {
                    Ok(E::from_iter(&mut a.iter_units().zip(b.iter_units()).map(|(x, y)| x + y), a.shape().clone()))
                } else {
                    Err(EngineError::ShapeMismatch(a.shape().clone(), b.shape().clone()))
                }
            }

            fn sub<E: EngineTensorFactory<Unit = $unit>>(a: &dyn EngineTensor<Unit = $unit>, b: &dyn EngineTensor<Unit = $unit>) -> Result<Box<dyn EngineTensor<Unit = $unit>>, crate::engine::EngineError> {
                if a.shape() == b.shape() {
                    Ok(E::from_iter(&mut a.iter_units().zip(b.iter_units()).map(|(x, y)| x - y), a.shape().clone()))
                } else {
                    Err(EngineError::ShapeMismatch(a.shape().clone(), b.shape().clone()))
                }
            }
        
            fn mul<E: EngineTensorFactory<Unit = $unit>>(a: &dyn EngineTensor<Unit = $unit>, b: &dyn EngineTensor<Unit = $unit>) -> Result<Box<dyn EngineTensor<Unit = $unit>>, crate::engine::EngineError> {
                if a.shape() == b.shape() {
                    Ok(E::from_iter(&mut a.iter_units().zip(b.iter_units()).map(|(x, y)| x * y), a.shape().clone()))
                } else {
                    Err(EngineError::ShapeMismatch(a.shape().clone(), b.shape().clone()))
                }
            }
        
            fn div<E: EngineTensorFactory<Unit = $unit>>(a: &dyn EngineTensor<Unit = $unit>, b: &dyn EngineTensor<Unit = $unit>) -> Result<Box<dyn EngineTensor<Unit = $unit>>, crate::engine::EngineError> {
                if a.shape() == b.shape() {
                    Ok(E::from_iter(&mut a.iter_units().zip(b.iter_units()).map(|(x, y)| x / y), a.shape().clone()))
                } else {
                    Err(EngineError::ShapeMismatch(a.shape().clone(), b.shape().clone()))
                }
            }

            conv_fn!($unit);
        }
    };
}

macro_rules! basic_unsigned_impl {
    ($unit:ty) => {
        impl Engine<$unit> for Basic {

            //Pointwise Single
            fn abs<E: EngineTensorFactory<Unit = $unit>>(a: &dyn EngineTensor<Unit = $unit>) -> Result<Box<dyn EngineTensor<Unit = $unit>>, crate::engine::EngineError> {
        
                Ok(E::from_iter(&mut a.iter_units().map(|x| x), a.shape().clone()))
            }
        
            fn neg<E: EngineTensorFactory<Unit = $unit>>(_a: &dyn EngineTensor<Unit = $unit>) -> Result<Box<dyn EngineTensor<Unit = $unit>>, crate::engine::EngineError> {
                Err(crate::engine::EngineError::OperationUnsupportedForType())
            }

            //Scalar
            fn add_scalar<E: EngineTensorFactory<Unit = $unit>>(s: $unit, a: &dyn EngineTensor<Unit = $unit>) -> Result<Box<dyn EngineTensor<Unit = $unit>>, EngineError> {
                Ok(E::from_iter(a.iter_units().map(|x| x + s), a.shape().clone()))
            }

            fn sub_scalar_lh<E: EngineTensorFactory<Unit = $unit>>(s: $unit, a: &dyn EngineTensor<Unit = $unit>) -> Result<Box<dyn EngineTensor<Unit = $unit>>, EngineError> {
                Ok(E::from_iter(a.iter_units().map(|x| s - x), a.shape().clone()))
            }

            fn sub_scalar_rh<E: EngineTensorFactory<Unit = $unit>>(a: &dyn EngineTensor<Unit = $unit>, s: $unit) -> Result<Box<dyn EngineTensor<Unit = $unit>>, EngineError> {
                Ok(E::from_iter(a.iter_units().map(|x| x - s), a.shape().clone()))
            }

            fn mul_scalar<E: EngineTensorFactory<Unit = $unit>>(s: $unit, a: &dyn EngineTensor<Unit = $unit>) -> Result<Box<dyn EngineTensor<Unit = $unit>>, EngineError> {
                Ok(E::from_iter(a.iter_units().map(|x| x * s), a.shape().clone()))
            }

            fn div_scalar_lh<E: EngineTensorFactory<Unit = $unit>>(s: $unit, a: &dyn EngineTensor<Unit = $unit>) -> Result<Box<dyn EngineTensor<Unit = $unit>>, EngineError> {
                Ok(E::from_iter(a.iter_units().map(|x| s / x), a.shape().clone()))
            }

            fn div_scalar_rh<E: EngineTensorFactory<Unit = $unit>>(a: &dyn EngineTensor<Unit = $unit>, s: $unit) -> Result<Box<dyn EngineTensor<Unit = $unit>>, EngineError> {
                Ok(E::from_iter(a.iter_units().map(|x| x / s), a.shape().clone()))
            }
        
            //Pointwise Double
            fn add<E: EngineTensorFactory<Unit = $unit>>(a: &dyn EngineTensor<Unit = $unit>, b: &dyn EngineTensor<Unit = $unit>) -> Result<Box<dyn EngineTensor<Unit = $unit>>, crate::engine::EngineError> {
                if a.shape() == b.shape() {
                    Ok(E::from_iter(&mut a.iter_units().zip(b.iter_units()).map(|(x, y)| x + y), a.shape().clone()))
                } else {
                    Err(EngineError::ShapeMismatch(a.shape().clone(), b.shape().clone()))
                }
            }
        
            fn sub<E: EngineTensorFactory<Unit = $unit>>(a: &dyn EngineTensor<Unit = $unit>, b: &dyn EngineTensor<Unit = $unit>) -> Result<Box<dyn EngineTensor<Unit = $unit>>, crate::engine::EngineError> {
                if a.shape() == b.shape() {
                    Ok(E::from_iter(&mut a.iter_units().zip(b.iter_units()).map(|(x, y)| x - y), a.shape().clone()))
                } else {
                    Err(EngineError::ShapeMismatch(a.shape().clone(), b.shape().clone()))
                }
            }
        
            fn mul<E: EngineTensorFactory<Unit = $unit>>(a: &dyn EngineTensor<Unit = $unit>, b: &dyn EngineTensor<Unit = $unit>) -> Result<Box<dyn EngineTensor<Unit = $unit>>, crate::engine::EngineError> {
                if a.shape() == b.shape() {
                    Ok(E::from_iter(&mut a.iter_units().zip(b.iter_units()).map(|(x, y)| x * y), a.shape().clone()))
                } else {
                    Err(EngineError::ShapeMismatch(a.shape().clone(), b.shape().clone()))
                }
            }
        
            fn div<E: EngineTensorFactory<Unit = $unit>>(a: &dyn EngineTensor<Unit = $unit>, b: &dyn EngineTensor<Unit = $unit>) -> Result<Box<dyn EngineTensor<Unit = $unit>>, crate::engine::EngineError> {
                if a.shape() == b.shape() {
                    Ok(E::from_iter(&mut a.iter_units().zip(b.iter_units()).map(|(x, y)| x / y), a.shape().clone()))
                } else {
                    Err(EngineError::ShapeMismatch(a.shape().clone(), b.shape().clone()))
                }
            }

            conv_fn!($unit);
        }
    };
}

basic_impl!(f32);
basic_impl!(f64);
basic_impl!(i8);
basic_impl!(i16);
basic_impl!(i32);
basic_impl!(i64);
basic_unsigned_impl!(u8);
basic_unsigned_impl!(u16);
basic_unsigned_impl!(u32);
basic_unsigned_impl!(u64);

#[cfg(test)]
mod test {
    use crate::{helper::shape, engine::tensor::Array};

    use super::*;

    #[test]
    pub fn conv() {
        let a = Array::from_iter((1..=65536).map(|x| (x as f32) / 65536.0).cycle().take(1 * 3 * 256 * 256), shape![1, 3, 256, 256]);
        let kernel = Array::from_iter((1..=9).map(|x| x as f32).cycle().take(1 * 3 * 3 * 3), shape![1, 3, 3, 3]);

        let res =  Basic::conv2d::<Array<f32>>(a.as_ref(), kernel.as_ref(), 2).unwrap();

        println!("{:?}", res.shape());
        //println!("{:?}", res.iter_unit().collect::<Vec<f32>>());
    }
}