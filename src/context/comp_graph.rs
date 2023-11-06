use std::collections::{HashSet, HashMap};

use slotmap::{SlotMap, new_key_type};
use thiserror::Error;

use crate::engine::{tensor::{EngineTensor, allowed_unit::AllowedUnit, factory::EngineTensorFactory}, Engine, EngineError};

use super::edge::Edge;

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

    //TODO Fix error handling and clean up code
    //Kahn's Algorithm
    pub fn populating_eval(&mut self, target_key: NodeKey) -> Result<(), ComputationGraphError> {
        //Nodes that have all dependencies satisfied
        let mut open_nodes = Vec::<NodeKey>::new();

        //Cache of children from a node (since it is usually linked backwards)
        let mut node_children = HashMap::<NodeKey, Vec<NodeKey>>::new();
        //Nodes that have been visited in the initial search (as to avoid duplicates in evaluation)
        let mut visited_nodes: HashSet<NodeKey> = HashSet::new();
        //Nodes still to be searched with the initial search
        let mut to_eval_nodes = vec![target_key];

        //Populate start nodes and node_children
        while let Some(node_key) = to_eval_nodes.pop() {
            let node = self.get_node(node_key).ok_or(ComputationGraphError::NodeKeyDoesNotExist(target_key))?;

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

        //Nodes already processed (in order to find more open nodes)
        let mut processed_nodes = HashSet::<NodeKey>::from_iter(open_nodes.clone());

        while let Some(node_key) = open_nodes.pop() {
            if !self.get_node(node_key).ok_or(ComputationGraphError::NodeKeyDoesNotExist(target_key))?.is_root() {
                let comp_tensor = self.compute_tensor(node_key)?;
                self.get_node_mut(node_key).ok_or(ComputationGraphError::NodeKeyDoesNotExist(target_key))?.set_tensor(comp_tensor);
            }

            processed_nodes.insert(node_key);

            if let Some(children_keys) = node_children.get(&node_key) {
                for child_key in children_keys {
                    let child_node = self.get_node(*child_key).ok_or(ComputationGraphError::NodeKeyDoesNotExist(target_key))?;

                    match child_node.edge() {
                        Edge::Root => {
                            //It should be impossible for a root to be a child
                            return Err(ComputationGraphError::RootNodeIsChild(*child_key))
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

        Ok(())
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
    #[error("Root node was found as the child of another node")]
    RootNodeIsChild(NodeKey),
    #[error("Error in computation: {0}")]
    ComputationError(#[from]EngineError),
}

#[cfg(test)]
mod test {
    use num::traits::Pow;

    use crate::{engine::{tensor::factory::Array, basic::Basic}, helper::Shape};

    use super::*;

    pub fn init_simple_graph() -> (NodeKey, NodeKey, NodeKey, EngineTensor<f32>, CompGraph<f32>) {
        let mut graph = CompGraph::<f32>::new();

        let root1 = graph.create_root(Array::from_slice([0.0, 1.0, 2.0, 3.0].as_slice(), Shape::from([2, 2].as_slice())));
        let root2 = graph.create_root(Array::from_slice([0.0, 1.0, 2.0, 3.0].as_slice(), Shape::from([2, 2].as_slice())));

        let added = graph.add::<Basic, Array>(root1, root2);

        let expected = Array::from_slice([0.0, 2.0, 4.0, 6.0].as_slice(), Shape::from([2, 2].as_slice()));

        return (root1, root2, added, expected, graph);
    }

    pub fn init_complex_graph() -> (NodeKey, EngineTensor<f32>, EngineTensor<f32>, CompGraph<f32>) {
        let mut graph = CompGraph::<f32>::new();

        let root1 = graph.create_root(Array::from_slice([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0].as_slice(), Shape::from([3, 3].as_slice())));
        let root2 = graph.create_root(Array::from_slice([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0].as_slice(), Shape::from([3, 3].as_slice())));
        let root3 = graph.create_root(Array::from_slice([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0].as_slice(), Shape::from([3, 3].as_slice())));
        let root4 = graph.create_root(Array::from_slice([1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0].as_slice(), Shape::from([3, 3].as_slice())));

        let op1 = graph.div::<Basic, Array>(root4, root1);
        let op2 = graph.mul::<Basic, Array>(op1, root2);
        let op3 = graph.sub::<Basic, Array>(op2, root3);

        let op4 = graph.mul::<Basic, Array>(op3, op3);

        let op5 = graph.div::<Basic, Array>(op4, root1);

        return (op5, Array::from_slice([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0].as_slice(), Shape::from([3, 3].as_slice())), Array::from_slice([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0].as_slice(), Shape::from([3, 3].as_slice())), graph);
    }

    #[test]
    fn simple_no_eval() {
        let (_, _, added, _, graph) = init_simple_graph();

        assert!(graph.get_node(added).is_some());

        let node = graph.get_node(added).unwrap();

        assert!(node.tensor().is_none());
    }

    #[test]
    fn simple_eval() {
        let (_, _, added, expected, mut graph) = init_simple_graph();

        graph.populating_eval(added).unwrap();

        assert!(graph.get_node(added).is_some());

        let node = graph.get_node(added).unwrap();

        assert!(node.tensor().is_some());

        assert_eq!(*node.tensor().unwrap(), expected);
    }

    #[test]
    fn complex_eval() {
        let (node_key, expected_original, _, mut graph) = init_complex_graph();

        graph.populating_eval(node_key).unwrap();
        
        let node = graph.get_node(node_key).unwrap();

        assert_eq!(*node.tensor().unwrap(), expected_original);
    }

    #[test]
    fn large_depth_eval() {
        let (node_key, _, expected_unit,  mut graph) = init_complex_graph();

        let mut out = node_key;
        for _ in  0..20000 {
            out = graph.div::<Basic, Array>(out, out);
        }

        graph.populating_eval(out).unwrap();

        let node = graph.get_node(out).unwrap();

        assert_eq!(*node.tensor().unwrap(), expected_unit);
    }

    #[test]
    fn large_bredth_eval() {
        let (node_key, expected_original, _,  mut graph) = init_complex_graph();

        let power = 12u16;

        let mut curr_node_keys: Vec<NodeKey>;
        let mut new_node_keys = vec![node_key; 2_usize.pow(power as u32)];

        while new_node_keys.len() > 1 {
            curr_node_keys = new_node_keys;
            new_node_keys = Vec::<NodeKey>::new();

            for keys in curr_node_keys.chunks_exact(2) {
                let a_key = keys[0];
                let b_key = keys[1];

                new_node_keys.push(graph.add::<Basic, Array>(a_key, b_key));
            }
        }

        let node_key = new_node_keys.last().unwrap();

        graph.populating_eval(*node_key).unwrap();

        let node = graph.get_node(*node_key).unwrap();

        let expected = Array::from_iter( &mut expected_original.iter().map(|x| x * 2.0f32.pow(power)), expected_original.shape().clone());

        assert_eq!(*node.tensor().unwrap(), expected);
    }
}