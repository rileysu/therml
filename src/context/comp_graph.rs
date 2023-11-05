use std::collections::{HashSet, HashMap};

use slotmap::{SlotMap, new_key_type};
use thiserror::Error;

use crate::engine::{tensor::{EngineTensor, allowed_unit::AllowedUnit, factory::EngineTensorFactory}, Engine, EngineError};

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

    pub fn is_root(&self) -> bool {
        *self.edge() == Edge::Root
    }
}

//Tensor might be able to be combined inside edge so root without a defined tensor isn't possible
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Edge<T: AllowedUnit> {
    Root,

    Abs(NodeKey, fn(&EngineTensor<T>) -> Result<EngineTensor<T>, EngineError>),
    Neg(NodeKey, fn(&EngineTensor<T>) -> Result<EngineTensor<T>, EngineError>),

    Add(NodeKey, NodeKey, fn(&EngineTensor<T>, &EngineTensor<T>) -> Result<EngineTensor<T>, EngineError>),
    Sub(NodeKey, NodeKey, fn(&EngineTensor<T>, &EngineTensor<T>) -> Result<EngineTensor<T>, EngineError>),
    Mul(NodeKey, NodeKey, fn(&EngineTensor<T>, &EngineTensor<T>) -> Result<EngineTensor<T>, EngineError>),
    Div(NodeKey, NodeKey, fn(&EngineTensor<T>, &EngineTensor<T>) -> Result<EngineTensor<T>, EngineError>),
}

new_key_type! { pub struct NodeKey; }

#[derive(Debug)]
pub struct CompGraph<T: AllowedUnit> {
    nodes: SlotMap<NodeKey, Node<T>>,
}

impl<T: AllowedUnit> CompGraph<T> {
    pub fn new() -> Self {
        Self {
            nodes: SlotMap::with_key(),
        }
    }

    pub fn get_node(&self, node_key: NodeKey) -> Option<&Node<T>> {
        self.nodes.get(node_key)
    }

    pub fn get_node_mut(&mut self, node_key: NodeKey) -> Option<&mut Node<T>> {
        self.nodes.get_mut(node_key)
    }

    //Root is a node that is a starting point for computation
    pub fn create_root(&mut self, tensor: EngineTensor<T>) -> NodeKey {
        self.nodes.insert(Node::create_root(tensor))
    }

    pub fn create_node(&mut self, edge: Edge<T>) -> NodeKey {
        self.nodes.insert(Node::create_node(edge))
    }

    //Single layer computation otherwise should throw an error
    fn compute_tensor(&self, target_key: NodeKey) -> Result<EngineTensor<T>, ComputationGraphError> {
        let target_node = self.get_node(target_key).ok_or(ComputationGraphError::NodeKeyDoesNotExist(target_key))?;

        match target_node.edge() {
            Edge::Root => Err(ComputationGraphError::RootNodeNotComputed(target_key)),
            Edge::Abs(a_key, op) |  
            Edge::Neg(a_key, op) => {
                let a_node = self.get_node(*a_key).ok_or(ComputationGraphError::ParentNodeDoesNotExist(*a_key))?;

                op(a_node.tensor().ok_or(ComputationGraphError::ParentNodeNotComputed(*a_key))?).map_err(|e| ComputationGraphError::from(e))
            },
            Edge::Add(a_key, b_key, op) |
            Edge::Sub(a_key, b_key, op) |
            Edge::Mul(a_key, b_key, op) |
            Edge::Div(a_key, b_key, op) => {
                let a_node = self.get_node(*a_key).ok_or(ComputationGraphError::ParentNodeDoesNotExist(*a_key))?;
                let b_node = self.get_node(*b_key).ok_or(ComputationGraphError::ParentNodeDoesNotExist(*b_key))?;

                op(a_node.tensor().ok_or(ComputationGraphError::ParentNodeNotComputed(*a_key))?, b_node.tensor().ok_or(ComputationGraphError::ParentNodeNotComputed(*b_key))?).map_err(|e| ComputationGraphError::from(e))
            },
        }
    }

    //TODO Fix to handle errors and clean up code
    //Kahn's Algorithm
    pub fn populating_eval(&mut self, target_key: NodeKey) {
        let mut open_nodes = Vec::<NodeKey>::new();

        let mut node_children = HashMap::<NodeKey, Vec<NodeKey>>::new();
        let mut visited_nodes: HashSet<NodeKey> = HashSet::new();
        let mut to_eval_nodes = vec![target_key];

        //Populate start nodes and node_children
        while let Some(node_key) = to_eval_nodes.pop() {
            let node = self.get_node(node_key).unwrap();

            match node.edge() {
                Edge::Root => {
                    open_nodes.push(node_key);
                },
                Edge::Abs(a_key, _) | Edge::Neg(a_key, _) => {
                    if let Some(children) = node_children.get_mut(&a_key) {
                        children.push(node_key);
                    } else {
                        node_children.insert(*a_key, vec![node_key]);
                    }

                    if !visited_nodes.contains(&a_key) {
                        to_eval_nodes.push(*a_key);
                    }
                },
                Edge::Add(a_key, b_key, _) | Edge::Sub(a_key, b_key, _) | Edge::Mul(a_key, b_key, _) | Edge::Div(a_key, b_key, _) => {
                    if let Some(children) = node_children.get_mut(&a_key) {
                        children.push(node_key);
                    } else {
                        node_children.insert(*a_key, vec![node_key]);
                    }

                    if b_key != a_key {
                        if let Some(children) = node_children.get_mut(&b_key) {
                            children.push(node_key);
                        } else {
                            node_children.insert(*b_key, vec![node_key]);
                        }
                    }

                    if !visited_nodes.contains(&a_key) {
                        to_eval_nodes.push(*a_key);
                        visited_nodes.insert(*a_key);
                    }

                    if !visited_nodes.contains(&b_key) {
                        to_eval_nodes.push(*b_key);
                        visited_nodes.insert(*a_key);
                    }
                },
            }
        }

        let mut processed_nodes = HashSet::<NodeKey>::from_iter(open_nodes.clone());
        let mut sorted_nodes = Vec::<NodeKey>::new();

        while let Some(node_key) = open_nodes.pop() {
            sorted_nodes.push(node_key);

            if !self.get_node(node_key).unwrap().is_root() {
                let comp_tensor = self.compute_tensor(node_key).unwrap();
                self.get_node_mut(node_key).unwrap().set_tensor(comp_tensor);
            }

            processed_nodes.insert(node_key);

            if let Some(children_keys) = node_children.get(&node_key) {
                for child_key in children_keys {
                    let child_node = self.get_node(*child_key).unwrap();

                    match child_node.edge() {
                        Edge::Root => {
                            //It should be impossible for a root to be a child
                            todo!()
                        },
                        Edge::Abs(_, _) | 
                        Edge::Neg(_, _) => {
                            //Since we know we just processed the parent there is no need to check
                            open_nodes.push(*child_key);
                        },
                        Edge::Add(a_key, b_key, _) | 
                        Edge::Sub(a_key, b_key, _) | 
                        Edge::Mul(a_key, b_key, _) | 
                        Edge::Div(a_key, b_key, _) => {
                            if processed_nodes.contains(a_key) && processed_nodes.contains(b_key) {
                                open_nodes.push(*child_key);
                            }
                        },
                    }
                }
            }
        }
    }

    pub fn abs<E: Engine<T>, F: EngineTensorFactory<T>>(&mut self, a: NodeKey) -> NodeKey {
        self.create_node(Edge::Abs(a, E::abs::<F>))
    }

    pub fn neg<E: Engine<T>, F: EngineTensorFactory<T>>(&mut self, a: NodeKey) -> NodeKey {
        self.create_node(Edge::Neg(a, E::neg::<F>))
    }

    pub fn add<E: Engine<T>, F: EngineTensorFactory<T>>(&mut self, a: NodeKey, b: NodeKey) -> NodeKey {
        self.create_node(Edge::Add(a, b, E::add::<F>))
    }

    pub fn sub<E: Engine<T>, F: EngineTensorFactory<T>>(&mut self, a: NodeKey, b: NodeKey) -> NodeKey {
        self.create_node(Edge::Sub(a, b, E::sub::<F>))
    }

    pub fn mul<E: Engine<T>, F: EngineTensorFactory<T>>(&mut self, a: NodeKey, b: NodeKey) -> NodeKey {
        self.create_node(Edge::Mul(a, b, E::mul::<F>))
    }

    pub fn div<E: Engine<T>, F: EngineTensorFactory<T>>(&mut self, a: NodeKey, b: NodeKey) -> NodeKey {
        self.create_node(Edge::Div(a, b, E::div::<F>))
    }
}

#[derive(Error, Debug)]
pub enum ComputationGraphError {
    #[error("Node key does not exist in this computation graph")]
    NodeKeyDoesNotExist(NodeKey),
    #[error("Root node doesn't contain computed tensor")]
    RootNodeNotComputed(NodeKey),
    #[error("Parent node does not exist in this computation graph")]
    ParentNodeDoesNotExist(NodeKey),
    #[error("Parent node not computed when expected to be")]
    ParentNodeNotComputed(NodeKey),
    #[error("Error in computation: {0}")]
    ComputationError(#[from]EngineError),
}