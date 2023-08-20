use std::ops::Neg;

use crate::engine::{
    basic::BasicEngine,
    tensor::{EngineTensorAccess, EngineTensorConstruct},
    util::return_if_matched_shape,
    Engine, EngineError,
};

use super::ArrayTensor;

impl Engine<f32, ArrayTensor<f32>> for BasicEngine {
    fn abs(a: &ArrayTensor<f32>) -> Result<ArrayTensor<f32>, EngineError> {
        Ok(ArrayTensor {
            data: a.get_data_slice().iter().map(|x| x.abs()).collect(),
            shape: a.shape().clone(),
            stride: a.stride().clone(),
            offset: 0,
        })
    }

    fn neg(a: &ArrayTensor<f32>) -> Result<ArrayTensor<f32>, EngineError> {
        Ok(ArrayTensor::<f32>::from_iter(
            &mut a.get_data_slice().iter().map(|x| x.neg()),
            a.shape().clone(),
        ))
    }

    fn add(a: &ArrayTensor<f32>, b: &ArrayTensor<f32>) -> Result<ArrayTensor<f32>, EngineError> {
        let out = ArrayTensor::<f32>::from_iter(
            &mut a
                .get_data_slice()
                .iter()
                .zip(b.get_data_slice().iter())
                .map(|(x, y)| x + y),
            a.shape().clone(),
        );

        return_if_matched_shape(a.shape(), b.shape(), out)
    }

    fn sub(a: &ArrayTensor<f32>, b: &ArrayTensor<f32>) -> Result<ArrayTensor<f32>, EngineError> {
        let out = ArrayTensor::<f32>::from_iter(
            &mut a
                .get_data_slice()
                .iter()
                .zip(b.get_data_slice().iter())
                .map(|(x, y)| x - y),
            a.shape().clone(),
        );

        return_if_matched_shape(a.shape(), b.shape(), out)
    }

    fn mul(a: &ArrayTensor<f32>, b: &ArrayTensor<f32>) -> Result<ArrayTensor<f32>, EngineError> {
        let out = ArrayTensor::<f32>::from_iter(
            &mut a
                .get_data_slice()
                .iter()
                .zip(b.get_data_slice().iter())
                .map(|(x, y)| x * y),
            a.shape().clone(),
        );

        return_if_matched_shape(a.shape(), b.shape(), out)
    }

    fn div(a: &ArrayTensor<f32>, b: &ArrayTensor<f32>) -> Result<ArrayTensor<f32>, EngineError> {
        let out = ArrayTensor::<f32>::from_iter(
            &mut a
                .get_data_slice()
                .iter()
                .zip(b.get_data_slice().iter())
                .map(|(x, y)| x / y),
            a.shape().clone(),
        );

        return_if_matched_shape(a.shape(), b.shape(), out)
    }
}
