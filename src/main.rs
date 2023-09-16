use crate::{engine::{tensor::{factory::{Array, Quant, EngineTensorFactory}, EngineTensor}, basic::Basic, Engine}, helper::Shape};

//mod context;
mod engine;
mod helper;
mod context;

fn main() {
    let a = Array::<f32>::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0], Shape::new([4, 3].into()));
    let b = Array::<f32>::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0], Shape::new([4, 3].into()));

    let sum = Basic::<Array<f32>>::add(&a, &b).unwrap();

    let sum_neg = Basic::<Array<f32>>::neg(&sum).unwrap();

    println!("{:?}", sum_neg);
}