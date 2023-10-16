#![allow(dead_code)]

use context::Context;
use engine::{tensor::factory::Array, basic::Basic};
use helper::Shape;



mod engine;
mod helper;
mod context;

fn main() {
    let mut context = Context::new();

    let a = context.from_slice::<Array<i32>>([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12].as_slice(), Shape::from([4, 3].as_slice()));
    let b = context.from_slice::<Array<i32>>([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13].as_slice(), Shape::from([4, 3].as_slice()));

    let mut c = context.add::<Basic<Array<_>>>(&a, &b);

    for _ in 0..20 {
        c = context.div::<Basic<Array<_>>>(&c, &c);
    }

    println!("{:?}", context.iter(&c).collect::<Vec<i32>>());
    //println!("{:?}", context);
}