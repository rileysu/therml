use crate::engine::{tensor::{EngineTensor, allowed_unit::AllowedUnit}, Engine, EngineError};

pub type NodeIndex = usize;

#[derive(Debug)]
pub struct Node<T: AllowedUnit> {
    tensor: Option<EngineTensor<T>>,
    edge: Edge<T>,
}

impl<T: AllowedUnit> Node<T> {
    pub fn create_root(tensor: EngineTensor<T>) -> Self {
        Self {
            tensor: Some(tensor),
            edge: Edge::Root,
        }
    }

    pub fn create_node(edge: Edge<T>) -> Self {
        Self {
            tensor: None,
            edge,
        }
    }

    pub fn tensor(&self) -> Option<&EngineTensor<T>> {
        self.tensor.as_ref()
    }

    pub fn set_tensor(&mut self, tensor: EngineTensor<T>) {
        self.tensor = Some(tensor)
    }

    pub fn edge(&self) -> &Edge<T> {
        &self.edge
    }
}

//Tensor might be able to be combined inside edge so root without a defined tensor isn't possible
#[derive(Clone, Copy, Debug)]
pub enum Edge<T: AllowedUnit> {
    Root,

    Abs(NodeIndex, fn(&EngineTensor<T>) -> Result<EngineTensor<T>, EngineError>),
    Neg(NodeIndex, fn(&EngineTensor<T>) -> Result<EngineTensor<T>, EngineError>),

    Add(NodeIndex, NodeIndex, fn(&EngineTensor<T>, &EngineTensor<T>) -> Result<EngineTensor<T>, EngineError>),
    Sub(NodeIndex, NodeIndex, fn(&EngineTensor<T>, &EngineTensor<T>) -> Result<EngineTensor<T>, EngineError>),
    Mul(NodeIndex, NodeIndex, fn(&EngineTensor<T>, &EngineTensor<T>) -> Result<EngineTensor<T>, EngineError>),
    Div(NodeIndex, NodeIndex, fn(&EngineTensor<T>, &EngineTensor<T>) -> Result<EngineTensor<T>, EngineError>),
}

#[derive(Debug)]
pub struct CompGraph<T: AllowedUnit> {
    nodes: Vec<Node<T>>,
}

impl<T: AllowedUnit> CompGraph<T> {
    pub fn new() -> Self {
        Self {
            nodes: vec![],
        }
    }

    pub fn get_node(&self, node_idx: NodeIndex) -> Option<&Node<T>> {
        self.nodes.get(node_idx)
    }

    pub fn get_node_mut(&mut self, node_idx: NodeIndex) -> Option<&mut Node<T>> {
        self.nodes.get_mut(node_idx)
    }

    //Root is a node that is a starting point for computation
    pub fn create_root(&mut self, tensor: EngineTensor<T>) -> NodeIndex {
        let idx = self.nodes.len();

        self.nodes.push(Node::create_root(tensor));

        idx
    }

    pub fn create_node(&mut self, edge: Edge<T>) -> NodeIndex {
        let idx = self.nodes.len();

        self.nodes.push(Node::create_node(edge));

        idx
    }

    //TODO Fix to handle errors and clean up code
    pub fn populating_eval(&mut self, target_idx: NodeIndex) {
        let mut to_eval = vec![target_idx];
        let mut discovered = vec![target_idx];

        while let Some(node_idx) = to_eval.pop() {
            let node = self.get_node(node_idx).unwrap();

            let mut push_to_eval_visited = |idxs: &[NodeIndex]| {
                for idx in idxs {
                    to_eval.push(*idx);
                    
                    visited.push(*idx);
                }
            };

            //If the tensor is not populated within the comp graph
            if node.tensor().is_none() {
                match *node.edge() {
                    Edge::Root => panic!(),
                    Edge::Abs(a_idx, _) => push_to_eval_visited(&[a_idx]),
                    Edge::Neg(a_idx, _) => push_to_eval_visited(&[a_idx]),
                    Edge::Add(a_idx, b_idx, _) => push_to_eval_visited(&[a_idx, b_idx]),
                    Edge::Sub(a_idx, b_idx, _) => push_to_eval_visited(&[a_idx, b_idx]),
                    Edge::Mul(a_idx, b_idx, _) => push_to_eval_visited(&[a_idx, b_idx]),
                    Edge::Div(a_idx, b_idx, _) => push_to_eval_visited(&[a_idx, b_idx]),
                }
            }
        }

        println!("{:?}", visited);

        //Reversing the DFS should evaluate nodes in an order that allows for computation down to the target node
        for node_idx in visited.iter().copied().rev() {
            
            let (node_no_tensor, node_edge) = {
                let node = self.get_node(node_idx).unwrap();

                (node.tensor().is_none(), *node.edge())
            };

            if node_no_tensor {
                let comp_tensor = match node_edge {
                    Edge::Root => unreachable!(),
                    Edge::Abs(a_idx, op) => op(self.get_node(a_idx).unwrap().tensor().unwrap()),
                    Edge::Neg(a_idx, op) => op(self.get_node(a_idx).unwrap().tensor().unwrap()),
                    Edge::Add(a_idx, b_idx, op) => op(self.get_node(a_idx).unwrap().tensor().unwrap(), self.get_node(b_idx).unwrap().tensor().unwrap()),
                    Edge::Sub(a_idx, b_idx, op) => op(self.get_node(a_idx).unwrap().tensor().unwrap(), self.get_node(b_idx).unwrap().tensor().unwrap()),
                    Edge::Mul(a_idx, b_idx, op) => op(self.get_node(a_idx).unwrap().tensor().unwrap(), self.get_node(b_idx).unwrap().tensor().unwrap()),
                    Edge::Div(a_idx, b_idx, op) => op(self.get_node(a_idx).unwrap().tensor().unwrap(), self.get_node(b_idx).unwrap().tensor().unwrap()),
                };

                match comp_tensor {
                    Ok(tensor) => self.get_node_mut(node_idx).unwrap().set_tensor(tensor),
                    Err(_) => panic!(),
                }
            }
        }
    }

    pub fn abs<E: Engine<Unit = T>>(&mut self, a: NodeIndex) -> NodeIndex {
        self.create_node(Edge::Abs(a, E::abs))
    }

    pub fn neg<E: Engine<Unit = T>>(&mut self, a: NodeIndex) -> NodeIndex {
        self.create_node(Edge::Neg(a, E::neg))
    }

    pub fn add<E: Engine<Unit = T>>(&mut self, a: NodeIndex, b: NodeIndex) -> NodeIndex {
        self.create_node(Edge::Add(a, b, E::add))
    }

    pub fn sub<E: Engine<Unit = T>>(&mut self, a: NodeIndex, b: NodeIndex) -> NodeIndex {
        self.create_node(Edge::Sub(a, b, E::sub))
    }

    pub fn mul<E: Engine<Unit = T>>(&mut self, a: NodeIndex, b: NodeIndex) -> NodeIndex {
        self.create_node(Edge::Mul(a, b, E::mul))
    }

    pub fn div<E: Engine<Unit = T>>(&mut self, a: NodeIndex, b: NodeIndex) -> NodeIndex {
        self.create_node(Edge::Div(a, b, E::div))
    }
}
