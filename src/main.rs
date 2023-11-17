#![allow(dead_code)]

use context::Context;
use engine::{tensor::factory::Array, basic::Basic};
use helper::Shape;

mod engine;
mod helper;
mod context;

fn main() {
    let mut context = Context::<f64, Basic>::new();

    let a = context.from_slice::<Array>([1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.].as_slice(), Shape::from([4, 3].as_slice()));
    let b = context.from_slice::<Array>([2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13.].as_slice(), Shape::from([4, 3].as_slice()));

    let mut c = a;
    for _ in 0..100000 {
        c = context.div_scalar_rh::<Array>(&c, 1.00001);
    }

    println!("{:?}", context.iter(&c).collect::<Vec<f64>>());
    //println!("{:?}", context);
}