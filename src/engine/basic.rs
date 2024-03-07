use itertools::Itertools;

use crate::{engine::{Engine, EngineError, EngineTensorFactory, EngineTensor, util::{err_if_incorrect_dimensions, err_if_dimension_mismatch}}, helper::{shape, Interval, Position, Shape, Slice, Stride, VarArrayCompatible}};

use super::{shared::im2col_2d, unit::UnitCompatible};

pub struct Basic {}

impl<T: UnitCompatible> Engine<T> for Basic {
    //Pointwise Single
    fn abs<E: EngineTensorFactory<Unit = T>>(a: &dyn EngineTensor<Unit = T>) -> Result<Box<dyn EngineTensor<Unit = T>>, crate::engine::EngineError> {

        Ok(E::from_iter(a.iter_units().map(|x| x.abs()), a.shape().clone()))
    }

    fn neg<E: EngineTensorFactory<Unit = T>>(a: &dyn EngineTensor<Unit = T>) -> Result<Box<dyn EngineTensor<Unit = T>>, crate::engine::EngineError> {
        Ok(E::from_iter(a.iter_units().map(|x| x.neg()), a.shape().clone()))
    }

    //Scalar
    fn add_scalar<E: EngineTensorFactory<Unit = T>>(s: T, a: &dyn EngineTensor<Unit = T>) -> Result<Box<dyn EngineTensor<Unit = T>>, EngineError> {
        Ok(E::from_iter(a.iter_units().map(|x| x + s), a.shape().clone()))
    }

    fn sub_scalar_lh<E: EngineTensorFactory<Unit = T>>(s: T, a: &dyn EngineTensor<Unit = T>) -> Result<Box<dyn EngineTensor<Unit = T>>, EngineError> {
        Ok(E::from_iter(a.iter_units().map(|x| s - x), a.shape().clone()))
    }

    fn sub_scalar_rh<E: EngineTensorFactory<Unit = T>>(a: &dyn EngineTensor<Unit = T>, s: T) -> Result<Box<dyn EngineTensor<Unit = T>>, EngineError> {
        Ok(E::from_iter(a.iter_units().map(|x| x - s), a.shape().clone()))
    }

    fn mul_scalar<E: EngineTensorFactory<Unit = T>>(s: T, a: &dyn EngineTensor<Unit = T>) -> Result<Box<dyn EngineTensor<Unit = T>>, EngineError> {
        Ok(E::from_iter(a.iter_units().map(|x| x * s), a.shape().clone()))
    }

    fn div_scalar_lh<E: EngineTensorFactory<Unit = T>>(s: T, a: &dyn EngineTensor<Unit = T>) -> Result<Box<dyn EngineTensor<Unit = T>>, EngineError> {
        Ok(E::from_iter(a.iter_units().map(|x| s / x), a.shape().clone()))
    }

    fn div_scalar_rh<E: EngineTensorFactory<Unit = T>>(a: &dyn EngineTensor<Unit = T>, s: T) -> Result<Box<dyn EngineTensor<Unit = T>>, EngineError> {
        Ok(E::from_iter(a.iter_units().map(|x| x / s), a.shape().clone()))
    }

    //Pointwise Double
    fn add<E: EngineTensorFactory<Unit = T>>(a: &dyn EngineTensor<Unit = T>, b: &dyn EngineTensor<Unit = T>) -> Result<Box<dyn EngineTensor<Unit = T>>, crate::engine::EngineError> {
        if a.shape() == b.shape() {
            Ok(E::from_iter(&mut a.iter_units().zip(b.iter_units()).map(|(x, y)| x + y), a.shape().clone()))
        } else {
            Err(EngineError::ShapeMismatch(a.shape().clone(), b.shape().clone()))
        }
    }

    fn sub<E: EngineTensorFactory<Unit = T>>(a: &dyn EngineTensor<Unit = T>, b: &dyn EngineTensor<Unit = T>) -> Result<Box<dyn EngineTensor<Unit = T>>, crate::engine::EngineError> {
        if a.shape() == b.shape() {
            Ok(E::from_iter(&mut a.iter_units().zip(b.iter_units()).map(|(x, y)| x - y), a.shape().clone()))
        } else {
            Err(EngineError::ShapeMismatch(a.shape().clone(), b.shape().clone()))
        }
    }

    fn mul<E: EngineTensorFactory<Unit = T>>(a: &dyn EngineTensor<Unit = T>, b: &dyn EngineTensor<Unit = T>) -> Result<Box<dyn EngineTensor<Unit = T>>, crate::engine::EngineError> {
        if a.shape() == b.shape() {
            Ok(E::from_iter(&mut a.iter_units().zip(b.iter_units()).map(|(x, y)| x * y), a.shape().clone()))
        } else {
            Err(EngineError::ShapeMismatch(a.shape().clone(), b.shape().clone()))
        }
    }

    fn div<E: EngineTensorFactory<Unit = T>>(a: &dyn EngineTensor<Unit = T>, b: &dyn EngineTensor<Unit = T>) -> Result<Box<dyn EngineTensor<Unit = T>>, crate::engine::EngineError> {
        if a.shape() == b.shape() {
            Ok(E::from_iter(&mut a.iter_units().zip(b.iter_units()).map(|(x, y)| x / y), a.shape().clone()))
        } else {
            Err(EngineError::ShapeMismatch(a.shape().clone(), b.shape().clone()))
        }
    }

    //Conv
    //a: (batches, in_channels, y, x)
    //kernel: (out_channels, in_channels, k_y, k_x)
    fn conv2d<E: EngineTensorFactory<Unit = T>>(a: &dyn EngineTensor<Unit = T>, kernel: &dyn EngineTensor<Unit = T>, padding: usize, stride: usize) -> Result<Box<dyn EngineTensor<Unit = T>>, crate::engine::EngineError> {
        err_if_incorrect_dimensions(a.shape(), 4)?;
        err_if_incorrect_dimensions(kernel.shape(), 4)?;
        err_if_dimension_mismatch(a.shape().get(1).unwrap(), kernel.shape().get(1).unwrap())?;

        let batches = a.shape().get(0).unwrap();
        let in_channels = a.shape().get(1).unwrap();

        let out_channels = kernel.shape().get(0).unwrap();
        let k_y = kernel.shape().get(2).unwrap();
        let k_x = kernel.shape().get(3).unwrap();

        //(batches, out_channels, in_channels, out_y, out_x, patch_len)
        let proc = im2col_2d::<T, E>(a, kernel.shape(), padding, stride).broadcast_splice(1, &[out_channels]);

        let out_y = proc.shape().get(3).unwrap();
        let out_x = proc.shape().get(4).unwrap();
        let patch_len = proc.shape().get(5).unwrap();

        //(batches, out_channels, in_channels, out_y, out_x, patch_len)
        let kernels = kernel.reshape(&shape![out_channels, in_channels, k_y * k_x]).broadcast_splice(0, &[batches]).broadcast_splice(3, [out_y, out_x].as_slice());
        
        println!("{:?}", out_y);
        println!("{:?}", proc.shape());
        println!("{:?}", kernels.shape());

        let chunked_iter = proc.iter_units().zip(kernels.iter_units()).map(|(x, y)| x * y).chunks(patch_len);
        let out_data = chunked_iter.into_iter().map(|i| i.sum());

        //(batches, out_channels, out_y, out_x)
        Ok(E::from_iter(out_data, shape![batches, out_channels, out_y, out_x]))
    }
}

#[cfg(test)]
mod test {
    use crate::{helper::shape, engine::tensor::Array};

    use super::*;

    #[test]
    pub fn conv() {
        let a = Array::from_iter((1..=65536).map(|x| (x as f32) / 65536.0).cycle().take(4 * 3 * 256 * 256), shape![4, 3, 256, 256]);
        let kernel = Array::from_iter((1..=9).map(|x| x as f32).cycle().take(1 * 3 * 3 * 3), shape![1, 3, 3, 3]);

        let res =  Basic::conv2d::<Array<f32>>(a.as_ref(), kernel.as_ref(), 2, 1).unwrap();

        println!("{:?}", res.shape());
        //println!("{:?}", res.iter_unit().collect::<Vec<f32>>());
    }
}