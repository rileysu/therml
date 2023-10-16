use crate::{engine::{tensor::{allowed_unit::AllowedUnit, factory::EngineTensorFactory}, Engine}, helper::{Shape, Slice}};

use self::comp_graph::{CompGraph, Node, NodeIndex};

mod comp_graph;

#[derive(Debug)]
pub struct Context<T: AllowedUnit> {
    comp_graph: CompGraph<T>,
}

pub struct ContextTensor {
    node: NodeIndex,
}

impl ContextTensor {
    pub fn new(node: NodeIndex) -> Self {
        Self {
            node,
        }
    }

    pub fn node(&self) -> NodeIndex {
        self.node
    }
}

impl<T: AllowedUnit> Context<T> {
    pub fn new() -> Self {
        Self {
            comp_graph: CompGraph::new(),
        }
    }

    pub fn eval(&mut self, tensor: &ContextTensor) {
        self.comp_graph.populating_eval(tensor.node);
    }

    pub fn from_iter<E: EngineTensorFactory<Unit = T>>(&mut self, iter: &mut dyn Iterator<Item = T>, shape: Shape) -> ContextTensor {
        ContextTensor::new(self.comp_graph.create_root(E::from_iter(iter, shape)))
    }

    pub fn from_slice<E: EngineTensorFactory<Unit = T>>(&mut self, slice: &[T], shape: Shape) -> ContextTensor {
        ContextTensor::new(self.comp_graph.create_root(E::from_slice(slice, shape)))
    }

    pub fn iter(&mut self, tensor: &ContextTensor) -> Box<dyn Iterator<Item = T> + '_> {
        self.eval(tensor);

        Box::from(self.comp_graph.get_node(tensor.node()).unwrap().tensor().unwrap().iter())
    }

    pub fn abs<E: Engine<Unit = T>>(&mut self, a: &ContextTensor) -> ContextTensor {
        ContextTensor::new(self.comp_graph.abs::<E>(a.node()))
    }

    pub fn neg<E: Engine<Unit = T>>(&mut self, a: &ContextTensor) -> ContextTensor {
        ContextTensor::new(self.comp_graph.neg::<E>(a.node()))
    }

    pub fn add<E: Engine<Unit = T>>(&mut self, a: &ContextTensor, b: &ContextTensor) -> ContextTensor {
        ContextTensor::new(self.comp_graph.add::<E>(a.node(), b.node()))
    }

    pub fn sub<E: Engine<Unit = T>>(&mut self, a: &ContextTensor, b: &ContextTensor) -> ContextTensor {
        ContextTensor::new(self.comp_graph.sub::<E>(a.node(), b.node()))
    }

    pub fn mul<E: Engine<Unit = T>>(&mut self, a: &ContextTensor, b: &ContextTensor) -> ContextTensor {
        ContextTensor::new(self.comp_graph.mul::<E>(a.node(), b.node()))
    }

    pub fn div<E: Engine<Unit = T>>(&mut self, a: &ContextTensor, b: &ContextTensor) -> ContextTensor {
        ContextTensor::new(self.comp_graph.div::<E>(a.node(), b.node()))
    }
}