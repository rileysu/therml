#![allow(dead_code)]
#![allow(unused_macros)]

use engine::{tensor::Array, basic::Basic};
use helper::Shape;

use crate::{comp_graph::CompGraph, engine::tensor::factory::EngineTensorFactory};

mod engine;
mod helper;
mod comp_graph;

fn main() {
    let mut graph = CompGraph::<f64>::new();

    let a = graph.create_root(Box::new(Array::from_slice([1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.].as_slice(), Shape::from([4, 3].as_slice()))));
    let b = graph.create_root(Box::new(Array::from_slice([2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13.].as_slice(), Shape::from([4, 3].as_slice()))));

    let mut c = a;
    for _ in 0..100000 {
        //c = graph.div_scalar_rh::<Basic, Array<_>>(&c, 1.00001);
    }

    graph.non_populating_eval(&c).unwrap();

    println!("{:?}", graph.iter(&c).collect::<Vec<f64>>());
    //println!("{:?}", context);
}