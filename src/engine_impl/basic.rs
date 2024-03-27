use itertools::Itertools;

use crate::{engine::{tensor::{factory::EngineTensorFactory, EngineTensor}, unit::UnitCompatible, Engine, EngineError}, engine_impl::{shared::im2col_2d, util::{err_if_dimension_mismatch, err_if_dimensions_mistmatch, err_if_incorrect_num_dimensions, err_if_too_few_dimensions}}, helper::{shape, varr, Shape, VarArray, VarArrayCompatible}};

pub struct Basic {}

impl<T: UnitCompatible> Engine<T> for Basic {
    //Pointwise Single
    fn abs<E: EngineTensorFactory<Unit = T>>(a: &dyn EngineTensor<Unit = T>) -> Result<Box<dyn EngineTensor<Unit = T>>, crate::engine::EngineError> {

        Ok(E::from_iter(a.iter_units().map(|x| x.abs()), a.shape().clone()).generic())
    }

    fn neg<E: EngineTensorFactory<Unit = T>>(a: &dyn EngineTensor<Unit = T>) -> Result<Box<dyn EngineTensor<Unit = T>>, crate::engine::EngineError> {
        Ok(E::from_iter(a.iter_units().map(|x| x.neg()), a.shape().clone()).generic())
    }

    fn relu<E: EngineTensorFactory<Unit = T>>(a: &dyn EngineTensor<Unit = T>) -> Result<Box<dyn EngineTensor<Unit = T>>, EngineError> {
        Ok(E::from_iter(a.iter_units().map(|x: T| if x > T::zero() { x } else { T::zero() }), a.shape().clone()).generic())
    }
    
    fn leaky_relu<E: EngineTensorFactory<Unit = T>>(a: &dyn EngineTensor<Unit = T>, alpha: f32) -> Result<Box<dyn EngineTensor<Unit = T>>, EngineError> {
        Ok(E::from_iter(a.iter_units().map(|x: T| if x > T::zero() { x } else { x.scale_single(alpha) }), a.shape().clone()).generic())
    }
    
    fn sigmoid<E: EngineTensorFactory<Unit = T>>(a: &dyn EngineTensor<Unit = T>) -> Result<Box<dyn EngineTensor<Unit = T>>, EngineError> {
        Ok(E::from_iter(a.iter_units().map(|x: T| {
            let x_exp = x.exp();

            x_exp / (T::one() + x_exp)
        }), a.shape().clone()).generic())
    }

    //Pointwise Double
    fn add<E: EngineTensorFactory<Unit = T>>(a: &dyn EngineTensor<Unit = T>, b: &dyn EngineTensor<Unit = T>) -> Result<Box<dyn EngineTensor<Unit = T>>, crate::engine::EngineError> {
        if a.shape() == b.shape() {
            Ok(E::from_iter(&mut a.iter_units().zip(b.iter_units()).map(|(x, y)| x + y), a.shape().clone()).generic())
        } else {
            Err(EngineError::ShapeMismatch(a.shape().clone(), b.shape().clone()))
        }
    }

    fn sub<E: EngineTensorFactory<Unit = T>>(a: &dyn EngineTensor<Unit = T>, b: &dyn EngineTensor<Unit = T>) -> Result<Box<dyn EngineTensor<Unit = T>>, crate::engine::EngineError> {
        if a.shape() == b.shape() {
            Ok(E::from_iter(&mut a.iter_units().zip(b.iter_units()).map(|(x, y)| x - y), a.shape().clone()).generic())
        } else {
            Err(EngineError::ShapeMismatch(a.shape().clone(), b.shape().clone()))
        }
    }

    fn mul<E: EngineTensorFactory<Unit = T>>(a: &dyn EngineTensor<Unit = T>, b: &dyn EngineTensor<Unit = T>) -> Result<Box<dyn EngineTensor<Unit = T>>, crate::engine::EngineError> {
        if a.shape() == b.shape() {
            Ok(E::from_iter(&mut a.iter_units().zip(b.iter_units()).map(|(x, y)| x * y), a.shape().clone()).generic())
        } else {
            Err(EngineError::ShapeMismatch(a.shape().clone(), b.shape().clone()))
        }
    }

    fn div<E: EngineTensorFactory<Unit = T>>(a: &dyn EngineTensor<Unit = T>, b: &dyn EngineTensor<Unit = T>) -> Result<Box<dyn EngineTensor<Unit = T>>, crate::engine::EngineError> {
        if a.shape() == b.shape() {
            Ok(E::from_iter(&mut a.iter_units().zip(b.iter_units()).map(|(x, y)| x / y), a.shape().clone()).generic())
        } else {
            Err(EngineError::ShapeMismatch(a.shape().clone(), b.shape().clone()))
        }
    }

    fn matmul<E: EngineTensorFactory<Unit = T>>(a: &dyn EngineTensor<Unit = T>, b: &dyn EngineTensor<Unit = T>) -> Result<Box<dyn EngineTensor<Unit = T>>, EngineError> {
        err_if_too_few_dimensions(a.shape(), 2)?;
        err_if_too_few_dimensions(b.shape(), 2)?;

        let mut a = a.clone();
        let mut b = b.clone();

        if a.shape().len() < b.shape().len() {
            a = a.broadcast_splice(0, &b.shape().as_slice()[0..(b.shape().len() - a.shape().len())]);
        }

        if b.shape().len() < a.shape().len() {
            b = b.broadcast_splice(0, &a.shape().as_slice()[0..(a.shape().len() - b.shape().len())]);
        }

        err_if_dimensions_mistmatch(&a.shape().as_slice()[0..(a.shape().len() - 2)], &b.shape().as_slice()[0..(b.shape().len() - 2)])?;
        err_if_dimension_mismatch(a.shape().get(a.shape().len() - 1).unwrap(), b.shape().get(b.shape().len() - 2).unwrap())?;

        let out_shape = Shape::new(VarArray::concat(&VarArray::from(&a.shape().as_slice()[0..(a.shape().len() - 2)]), &varr![a.shape().get(a.shape().len() - 2).unwrap(), b.shape().get(b.shape().len() - 1).unwrap()]));

        let builder = E::builder(out_shape, T::default());

        todo!()
    }

    //Conv
    //a: (batches, in_channels, y, x)
    //kernel: (out_channels, in_channels, k_y, k_x)
    fn conv2d<E: EngineTensorFactory<Unit = T>>(a: &dyn EngineTensor<Unit = T>, kernel: &dyn EngineTensor<Unit = T>, padding: usize, stride: usize) -> Result<Box<dyn EngineTensor<Unit = T>>, crate::engine::EngineError> {
        err_if_incorrect_num_dimensions(a.shape(), 4)?;
        err_if_incorrect_num_dimensions(kernel.shape(), 4)?;
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
        Ok(E::from_iter(out_data, shape![batches, out_channels, out_y, out_x]).generic())
    }
}

#[cfg(test)]
mod test {
    use crate::{engine_impl::tensor::array::Array, helper::shape};

    use super::*;

    #[test]
    pub fn conv() {
        let a = Array::from_iter((1..=65536).map(|x| (x as f32) / 65536.0).cycle().take(4 * 3 * 256 * 256), shape![4, 3, 256, 256]);
        let kernel = Array::from_iter((1..=9).map(|x| x as f32).cycle().take(1 * 3 * 3 * 3), shape![1, 3, 3, 3]);

        let res =  Basic::conv2d::<Array<f32>>(a.generic().as_ref(), kernel.generic().as_ref(), 2, 1).unwrap();

        println!("{:?}", res.shape());
        //println!("{:?}", res.iter_unit().collect::<Vec<f32>>());
    }
}