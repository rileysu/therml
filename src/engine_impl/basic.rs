use std::iter;

use itertools::Itertools;

use crate::{engine::{tensor::{factory::EngineTensorFactory, EngineTensor}, unit::UnitCompatible, Engine, EngineError}, engine_impl::{shared::im2col_2d, util::{err_if_dimension_mismatch, err_if_dimensions_mistmatch, err_if_incorrect_num_dimensions, err_if_too_few_dimensions}}, helper::{shape, varr, Interval, Shape, VarArray, VarArrayCompatible}};
use crate::engine::tensor::builder::EngineTensorBuilder;
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
    
    fn leaky_relu<E: EngineTensorFactory<Unit = T>>(a: &dyn EngineTensor<Unit = T>, alpha: f64) -> Result<Box<dyn EngineTensor<Unit = T>>, EngineError> {
        Ok(E::from_iter(a.iter_units().map(|x: T| x.leaky_relu(alpha)), a.shape().clone()).generic())
    }
    
    fn sigmoid<E: EngineTensorFactory<Unit = T>>(a: &dyn EngineTensor<Unit = T>) -> Result<Box<dyn EngineTensor<Unit = T>>, EngineError> {
        Ok(E::from_iter(a.iter_units().map(|x: T| x.sigmoid()), a.shape().clone()).generic())
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

        let out_rows = a.shape().get(a.shape().len() - 2).unwrap();
        let out_columns = b.shape().get(b.shape().len() - 1).unwrap();

        let a_batches_shape = Shape::from(&a.shape().as_slice()[0..(a.shape().len() - 2)]);
        let b_batches_shape = Shape::from(&b.shape().as_slice()[0..(b.shape().len() - 2)]);

        let a_columns = a.shape().get(a.shape().len() - 1).unwrap();
        let b_rows = b.shape().get(b.shape().len() - 2).unwrap();

        err_if_dimensions_mistmatch(a_batches_shape.as_slice(), b_batches_shape.as_slice())?;
        err_if_dimension_mismatch(a_columns, b_rows)?;

        let out_shape = Shape::new(VarArray::concat(a_batches_shape.vararray(), &varr![out_rows, out_columns]));

        let mut builder = E::builder(out_shape, T::default());

        let mut a_intervals = [
            (0..a_batches_shape.len()).map(|_| Interval::all()).collect::<Vec<Interval>>().as_slice(), 
            [Interval::only(out_rows), Interval::all()].as_slice()
        ].concat().into_boxed_slice();
        let a_intervals_row_index = a_intervals.len() - 2;

        let mut b_intervals = [
            (0..b_batches_shape.len()).map(|_| Interval::all()).collect::<Vec<Interval>>().as_slice(), 
            [Interval::all(), Interval::only(out_columns)].as_slice()
        ].concat().into_boxed_slice();
        let b_intervals_column_index = b_intervals.len() - 1;

        let mut out_intervals: Box<[Interval]> = [
            (0..b_batches_shape.len()).map(|_| Interval::all()).collect::<Vec<Interval>>().as_slice(), 
            [Interval::only(out_rows), Interval::only(out_columns)].as_slice()
        ].concat().into_boxed_slice();
        let out_intervals_row_index = out_intervals.len() - 2;
        let out_intervals_column_index = out_intervals.len() - 1;

        for row in 0..out_rows {
            for column in 0..out_columns {
                // sum(a(row, 1..n) * b(1..n, col))

                *a_intervals.get_mut(a_intervals_row_index).unwrap() = Interval::only(row);
                *b_intervals.get_mut(b_intervals_column_index).unwrap() = Interval::only(column);

                let a_slice = a.slice(&a_intervals);
                let b_slice = b.slice(&b_intervals);

                let chunks = a_slice.iter_units().zip(b_slice.iter_units()).map(|(a_e, b_e)| a_e * b_e).chunks(a_columns);

                *out_intervals.get_mut(out_intervals_row_index).unwrap() = Interval::only(row);
                *out_intervals.get_mut(out_intervals_column_index).unwrap() = Interval::only(column);

                builder.splice_slice(&out_intervals, chunks.into_iter().map(|c| c.sum()))
            }
        }

        Ok(builder.construct().generic())
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

    #[test]
    pub fn matmul_basic() {
        let a = Array::from_slice(&[1., 2., 3., 4., 5., 6.], shape![2, 3]);
        let b = Array::from_slice(&[10., 11., 20., 21., 30., 31.], shape![3, 2]);

        let expected = Array::from_slice(&[140., 146., 320., 335.], shape![2, 2]);

        let res = Basic::matmul::<Array<f32>>(&a, &b).unwrap();

        assert!(res == expected.generic());

        let a = Array::from_slice(&[1., 2., 3., 4., 5., 6., 2., 4., 6., 8., 10., 12.], shape![2, 2, 3]);
        let b = Array::from_slice(&[10., 11., 20., 21., 30., 31.], shape![3, 2]);

        let expected = Array::from_slice(&[140., 146., 320., 335., 280., 292., 640., 670.], shape![2, 2, 2]);

        let res = Basic::matmul::<Array<f32>>(&a, &b).unwrap();

        assert!(res == expected.generic());

        let a = Array::from_slice(&[1., 2., 3., 4., 5., 6.], shape![2, 3]);
        let b = Array::from_slice(&[10., 11., 20., 21., 30., 31., 20., 22., 40., 42., 60., 62.], shape![2, 3, 2]);

        let expected = Array::from_slice(&[140., 146., 320., 335., 280., 292., 640., 670.], shape![2, 2, 2]);

        let res = Basic::matmul::<Array<f32>>(&a, &b).unwrap();

        assert!(res == expected.generic());

        let a = Array::from_slice(&[1., 2., 3., 4., 5., 6., 2., 4., 6., 8., 10., 12.], shape![2, 2, 3]);
        let b = Array::from_slice(&[10., 11., 20., 21., 30., 31., 20., 22., 40., 42., 60., 62.], shape![2, 3, 2]);

        let expected = Array::from_slice(&[140., 146., 320., 335., 560., 584., 1280., 1340.], shape![2, 2, 2]);

        let res = Basic::matmul::<Array<f32>>(&a, &b).unwrap();

        assert!(res == expected.generic());
    }
}