use std::marker::PhantomData;

use crate::{engine::{tensor::{allowed_unit::AllowedUnit, factory::EngineTensorFactory}, Engine}, helper::Shape};

use self::comp_graph::{CompGraph, NodeKey};

mod comp_graph;
mod edge;

#[derive(Debug)]
pub struct Context<T: AllowedUnit, E: Engine<T>> {
    comp_graph: CompGraph<T>,
    default_engine: PhantomData<E>,
}

pub struct ContextTensor {
    node: NodeKey,
}

impl ContextTensor {
    pub fn new(node: NodeKey) -> Self {
        Self {
            node,
        }
    }

    pub fn node(&self) -> NodeKey {
        self.node
    }
}

impl<T: AllowedUnit, E: Engine<T>> Context<T, E> {
    pub fn new() -> Self {
        Self {
            comp_graph: CompGraph::new(),
            default_engine: PhantomData,
        }
    }

    pub fn eval(&mut self, tensor: &ContextTensor) {
        self.comp_graph.populating_eval(tensor.node).unwrap();
    }

    pub fn from_iter<F: EngineTensorFactory<T>>(&mut self, iter: &mut dyn Iterator<Item = T>, shape: Shape) -> ContextTensor {
        ContextTensor::new(self.comp_graph.create_root(F::from_iter(iter, shape)))
    }

    pub fn from_slice<F: EngineTensorFactory<T>>(&mut self, slice: &[T], shape: Shape) -> ContextTensor {
        ContextTensor::new(self.comp_graph.create_root(F::from_slice(slice, shape)))
    }

    pub fn iter(&mut self, tensor: &ContextTensor) -> Box<dyn Iterator<Item = T> + '_> {
        self.eval(tensor);

        Box::from(self.comp_graph.get_node(tensor.node()).unwrap().tensor().unwrap().iter())
    }

    pub fn abs<F: EngineTensorFactory<T>>(&mut self, a: &ContextTensor) -> ContextTensor {
        ContextTensor::new(self.comp_graph.abs::<E, F>(a.node()))
    }

    pub fn neg<F: EngineTensorFactory<T>>(&mut self, a: &ContextTensor) -> ContextTensor {
        ContextTensor::new(self.comp_graph.neg::<E, F>(a.node()))
    }

    pub fn add<F: EngineTensorFactory<T>>(&mut self, a: &ContextTensor, b: &ContextTensor) -> ContextTensor {
        ContextTensor::new(self.comp_graph.add::<E, F>(a.node(), b.node()))
    }

    pub fn sub<F: EngineTensorFactory<T>>(&mut self, a: &ContextTensor, b: &ContextTensor) -> ContextTensor {
        ContextTensor::new(self.comp_graph.sub::<E, F>(a.node(), b.node()))
    }

    pub fn mul<F: EngineTensorFactory<T>>(&mut self, a: &ContextTensor, b: &ContextTensor) -> ContextTensor {
        ContextTensor::new(self.comp_graph.mul::<E, F>(a.node(), b.node()))
    }

    pub fn div<F: EngineTensorFactory<T>>(&mut self, a: &ContextTensor, b: &ContextTensor) -> ContextTensor {
        ContextTensor::new(self.comp_graph.div::<E, F>(a.node(), b.node()))
    }
}