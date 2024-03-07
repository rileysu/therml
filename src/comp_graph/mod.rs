mod edge;

use std::collections::{HashSet, HashMap};

use slotmap::{SlotMap, new_key_type};
use thiserror::Error;

use crate::engine::{tensor::{factory::EngineTensorFactory, iter::EngineTensorUnitIterator, EngineTensor}, unit::UnitCompatible, Engine, EngineError};

use self::edge::Edge;

#[derive(Debug)]
pub struct Node<T: UnitCompatible> {
    tensor: Option<Box<dyn EngineTensor<Unit = T>>>,
    edge: Edge<T>,
}

impl<T: UnitCompatible> Node<T> {
    fn create_root(tensor: Box<dyn EngineTensor<Unit = T>>) -> Self {
        Self {
            tensor: Some(tensor),
            edge: Edge::Root,
        }
    }

    fn create_node(edge: Edge<T>) -> Self {
        Self {
            tensor: None,
            edge,
        }
    }

    fn tensor(&self) -> Option<&dyn EngineTensor<Unit = T>> {
        self.tensor.as_deref()
    }

    fn set_tensor(&mut self, tensor: Box<dyn EngineTensor<Unit = T>>) {
        self.tensor = Some(tensor)
    }

    fn clear_tensor(&mut self) -> Result<(), ComputationGraphError> {
        if self.is_root() {
            return Err(ComputationGraphError::CannotClearRoot())
        }

        self.tensor = None;
        Ok(())
    }

    fn edge(&self) -> &Edge<T> {
        &self.edge
    }

    fn is_root(&self) -> bool {
        *self.edge() == Edge::Root
    }
}

//Tensor might be able to be combined inside edge so root without a defined tensor isn't possible

new_key_type! { pub struct NodeKey; }

#[derive(Debug)]
pub struct CompGraph<T: UnitCompatible> {
    nodes: SlotMap<NodeKey, Node<T>>,
}

impl<T: UnitCompatible> CompGraph<T> {
    pub fn new() -> Self {
        Self {
            nodes: SlotMap::with_key(),
        }
    }

    fn get_node(&self, node_key: &NodeKey) -> Option<&Node<T>> {
        self.nodes.get(*node_key)
    }

    fn get_node_error(&self, node_key: &NodeKey) -> Result<&Node<T>, ComputationGraphError> {
        self.nodes.get(*node_key).ok_or(ComputationGraphError::NodeDoesNotExist(*node_key))
    }

    fn get_node_mut(&mut self, node_key: &NodeKey) -> Option<&mut Node<T>> {
        self.nodes.get_mut(*node_key)
    }

    fn get_node_mut_error(&mut self, node_key: &NodeKey) -> Result<&mut Node<T>, ComputationGraphError> {
        self.nodes.get_mut(*node_key).ok_or(ComputationGraphError::NodeDoesNotExist(*node_key))
    }

    //Root is a node that is a starting point for computation
    fn create_root_node(&mut self, tensor: Box<dyn EngineTensor<Unit = T>>) -> NodeKey {
        self.nodes.insert(Node::create_root(tensor))
    }

    pub fn create_root(&mut self, tensor: Box<dyn EngineTensor<Unit = T>>) -> CompGraphTensor {
        CompGraphTensor::new(self.create_root_node(tensor))
    }

    fn create_node(&mut self, edge: Edge<T>) -> NodeKey {
        self.nodes.insert(Node::create_node(edge))
    }

    pub fn iter(&self, tensor: &CompGraphTensor) -> EngineTensorUnitIterator<T> {
        EngineTensorUnitIterator::new(self.get_node(tensor.node_key()).unwrap().tensor().unwrap())
    }

    //First return is open nodes, second is node_to_children
    //The algorithm is more efficient if done at the same time
    fn generate_node_to_children(&self, target: &NodeKey) -> Result<(Vec<NodeKey>, HashMap::<NodeKey, Vec<NodeKey>>), ComputationGraphError> {
        //Nodes still to be searched with the initial search
        let mut to_eval = vec![*target];

        //Nodes that have been visited in the initial search (as to avoid duplicates in evaluation)
        let mut visited: HashSet<NodeKey> = HashSet::new();

        //Nodes without dependencies
        let mut open = Vec::<NodeKey>::new();

        //Node to children map
        let mut node_to_children = HashMap::<NodeKey, Vec<NodeKey>>::new();

        while let Some(node_key) = to_eval.pop() {
            let node = self.get_node(&node_key).ok_or(ComputationGraphError::NodeDoesNotExist(*target))?;

            if node.edge().is_root() {
                open.push(node_key);
            } else {
                for parent_key in node.edge().nodes() {
                    if let Some(children) = node_to_children.get_mut(&parent_key) {
                        //This should be performant for small lists
                        if !children.contains(&node_key) {
                            children.push(node_key);
                        }
                    } else {
                        node_to_children.insert(parent_key, vec![node_key]);
                    }

                    if !visited.contains(&parent_key) {
                        to_eval.push(parent_key);
                        visited.insert(parent_key);
                    }
                }
            }
        }

        Ok((open, node_to_children))
    }

    //Uses Kahn's Algorithm
    fn populating_eval_node(&mut self, target: NodeKey) -> Result<(), ComputationGraphError> {
        //Nodes that have all dependencies satisfied
        let (open_roots, node_to_children) = self.generate_node_to_children(&target)?;

        //Current open set of nodes
        let mut open = open_roots.clone();

        //Nodes already processed (in order to find more open nodes)
        let mut processed_nodes = HashSet::<NodeKey>::from_iter(open.clone());

        while let Some(node_key) = open.pop() {
            let node = self.get_node(&node_key).ok_or(ComputationGraphError::NodeDoesNotExist(target))?;

            if !node.is_root() {
                let comp_tensor = node.edge().compute_tensor(
                    |k| Ok(self.get_node(&k).ok_or(ComputationGraphError::NodeDoesNotExist(k))?.tensor().ok_or(ComputationGraphError::NodeNotComputed(k))?)
                )?;
                self.get_node_mut(&node_key).ok_or(ComputationGraphError::NodeDoesNotExist(target))?.set_tensor(comp_tensor);
            }

            processed_nodes.insert(node_key);

            if let Some(children_keys) = node_to_children.get(&node_key) {
                for child_key in children_keys {
                    let child_node = self.get_node(child_key).ok_or(ComputationGraphError::NodeDoesNotExist(target))?;

                    if child_node.edge().is_root() {
                        return Err(ComputationGraphError::RootNodeIsChild(*child_key));
                    } else {
                        //If all parents of the child are processed and the child hasn't been processed before then add it to be processed
                        if child_node.edge().nodes().all(|k| processed_nodes.contains(&k)) && !processed_nodes.contains(child_key) {
                            open.push(*child_key);
                        }
                    }
                }
            }
        }

        Ok(())
    }

    pub fn populating_eval(&mut self, target: &CompGraphTensor) -> Result<(), ComputationGraphError> {
        self.populating_eval_node(*target.node_key())
    }

    fn non_populating_eval_node(&mut self, target: NodeKey) -> Result<(), ComputationGraphError> {
        //Nodes that have all dependencies satisfied
        let (open_roots, node_to_children) = self.generate_node_to_children(&target)?;

        //Current open set of nodes
        let mut open = open_roots.clone();

        //Nodes already processed (in order to find more open nodes)
        let mut processed_nodes = HashSet::<NodeKey>::from_iter(open.clone());

        //Cache for calculated nodes
        //Should be cleared once no dependencies left
        let mut comp_cache = HashMap::<NodeKey, Box<dyn EngineTensor<Unit = T>>>::new();

        while let Some(node_key) = open.pop() {
            let node = self.get_node(&node_key).ok_or(ComputationGraphError::NodeDoesNotExist(target))?;

            if !node.is_root() {
                let comp_tensor = node.edge().compute_tensor(
                    |k| {
                        match comp_cache.get(&k) {
                            Some(tensor) => Ok(tensor.as_ref()),
                            None => Ok(self.get_node(&k).ok_or(ComputationGraphError::NodeDoesNotExist(k))?.tensor().ok_or(ComputationGraphError::NodeNotComputed(k))?),
                        }
                    }
                )?;
                comp_cache.insert(node_key, comp_tensor);

                //All children are defined in the cache so the parent is no longer needed
                for parent_key in node.edge().nodes() {
                    if node_to_children.get(&parent_key).unwrap().iter().all(|k| comp_cache.contains_key(k)) {
                        comp_cache.remove(&parent_key);
                    }
                }
            }

            processed_nodes.insert(node_key);

            if let Some(children_keys) = node_to_children.get(&node_key) {
                for child_key in children_keys {
                    let child_node = self.get_node(child_key).ok_or(ComputationGraphError::NodeDoesNotExist(target))?;

                    if child_node.edge().is_root() {
                        return Err(ComputationGraphError::RootNodeIsChild(*child_key));
                    } else {
                        //If all parents of the child are processed and the child hasn't been processed before then add it to be processed
                        if child_node.edge().nodes().all(|k| processed_nodes.contains(&k)) && !processed_nodes.contains(child_key) {
                            open.push(*child_key);
                        }
                    }
                }
            }
        }

        self.get_node_mut(&target).ok_or(ComputationGraphError::NodeDoesNotExist(target))?.set_tensor(comp_cache.remove(&target).ok_or(ComputationGraphError::NodeDoesNotExist(target))?);

        Ok(())
    }

    pub fn non_populating_eval(&mut self, target: &CompGraphTensor) -> Result<(), ComputationGraphError> {
        self.non_populating_eval_node(*target.node_key())
    }

    pub fn abs<E: Engine<T>, F: EngineTensorFactory<Unit = T>>(&mut self, a: &CompGraphTensor) -> CompGraphTensor {
        CompGraphTensor::new(self.create_node(Edge::Abs(*a.node_key(), E::abs::<F>)))
    }

    pub fn neg<E: Engine<T>, F: EngineTensorFactory<Unit = T>>(&mut self, a: &CompGraphTensor) -> CompGraphTensor {
        CompGraphTensor::new(self.create_node(Edge::Neg(*a.node_key(), E::neg::<F>)))
    }

    pub fn add_scalar<E: Engine<T>, F: EngineTensorFactory<Unit = T>>(&mut self, s: T, a: &CompGraphTensor) -> CompGraphTensor {
        CompGraphTensor::new(self.create_node(Edge::AddScalar(s, *a.node_key(), E::add_scalar::<F>)))
    }

    pub fn sub_scalar_lh<E: Engine<T>, F: EngineTensorFactory<Unit = T>>(&mut self, s: T, a: &CompGraphTensor) -> CompGraphTensor {
        CompGraphTensor::new(self.create_node(Edge::SubScalarLH(s, *a.node_key(), E::sub_scalar_lh::<F>)))
    }

    pub fn sub_scalar_rh<E: Engine<T>, F: EngineTensorFactory<Unit = T>>(&mut self, a: &CompGraphTensor, s: T) -> CompGraphTensor {
        CompGraphTensor::new(self.create_node(Edge::SubScalarRH(*a.node_key(), s, E::sub_scalar_rh::<F>)))
    }

    pub fn mul_scalar<E: Engine<T>, F: EngineTensorFactory<Unit = T>>(&mut self, s: T, a: &CompGraphTensor) -> CompGraphTensor {
        CompGraphTensor::new(self.create_node(Edge::MulScalar(s, *a.node_key(), E::mul_scalar::<F>)))
    }

    pub fn div_scalar_lh<E: Engine<T>, F: EngineTensorFactory<Unit = T>>(&mut self, s: T, a: &CompGraphTensor) -> CompGraphTensor {
        CompGraphTensor::new(self.create_node(Edge::DivScalarLH(s, *a.node_key(), E::div_scalar_lh::<F>)))
    }

    pub fn div_scalar_rh<E: Engine<T>, F: EngineTensorFactory<Unit = T>>(&mut self, a: &CompGraphTensor, s: T) -> CompGraphTensor {
        CompGraphTensor::new(self.create_node(Edge::DivScalarRH(*a.node_key(), s, E::div_scalar_rh::<F>)))
    }

    pub fn add<E: Engine<T>, F: EngineTensorFactory<Unit = T>>(&mut self, a: &CompGraphTensor, b: &CompGraphTensor) -> CompGraphTensor {
        CompGraphTensor::new(self.create_node(Edge::Add(*a.node_key(), *b.node_key(), E::add::<F>)))
    }

    pub fn sub<E: Engine<T>, F: EngineTensorFactory<Unit = T>>(&mut self, a: &CompGraphTensor, b: &CompGraphTensor) -> CompGraphTensor {
        CompGraphTensor::new(self.create_node(Edge::Sub(*a.node_key(), *b.node_key(), E::sub::<F>)))
    }

    pub fn mul<E: Engine<T>, F: EngineTensorFactory<Unit = T>>(&mut self, a: &CompGraphTensor, b: &CompGraphTensor) -> CompGraphTensor {
        CompGraphTensor::new(self.create_node(Edge::Mul(*a.node_key(), *b.node_key(), E::mul::<F>)))
    }

    pub fn div<E: Engine<T>, F: EngineTensorFactory<Unit = T>>(&mut self, a: &CompGraphTensor, b: &CompGraphTensor) -> CompGraphTensor {
        CompGraphTensor::new(self.create_node(Edge::Div(*a.node_key(), *b.node_key(), E::div::<F>)))
    }
}

//External handle for nodes
//Lifetime tied to graph
#[derive(Debug, Clone)]
pub struct CompGraphTensor {
    node_key: NodeKey,
}

impl CompGraphTensor {
    fn new(node_key: NodeKey) -> Self {
        Self {
            node_key,
        }
    }

    fn node_key(&self) -> &NodeKey {
        &self.node_key
    }
}

#[derive(Error, Debug)]
pub enum ComputationGraphError {
    #[error("Node does not exist in this computation graph")]
    NodeDoesNotExist(NodeKey),
    #[error("Root node doesn't contain computed tensor")]
    RootNodeNotComputed(),
    #[error("Node not computed when expected to be")]
    NodeNotComputed(NodeKey),
    #[error("Root node was found as the child of another node")]
    RootNodeIsChild(NodeKey),
    #[error("Tried to clear root node")]
    CannotClearRoot(),
    #[error("Error in computation: {0}")]
    ComputationError(#[from]EngineError),
}

#[cfg(test)]
mod test {
    use num::traits::Pow;

    use crate::{engine::{tensor::Array, basic::Basic}, helper::Shape};

    use super::*;

    pub fn init_simple_graph() -> (CompGraphTensor, CompGraphTensor, CompGraphTensor, Box<dyn EngineTensor<Unit = f32>>, CompGraph<f32>) {
        let mut graph = CompGraph::<f32>::new();

        let root1 = graph.create_root(Array::from_slice([0.0, 1.0, 2.0, 3.0].as_slice(), Shape::from([2, 2].as_slice())));
        let root2 = graph.create_root(Array::from_slice([0.0, 1.0, 2.0, 3.0].as_slice(), Shape::from([2, 2].as_slice())));

        let added = graph.add::<Basic, Array<f32>>(&root1, &root2);

        let expected = Array::from_slice([0.0, 2.0, 4.0, 6.0].as_slice(), Shape::from([2, 2].as_slice()));

        return (root1, root2, added, expected, graph);
    }

    pub fn init_complex_graph() -> (CompGraphTensor, Box<dyn EngineTensor<Unit = f32>>, Box<dyn EngineTensor<Unit = f32>>, CompGraph<f32>) {
        let mut graph = CompGraph::<f32>::new();

        let root1 = graph.create_root(Array::from_slice([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0].as_slice(), Shape::from([3, 3].as_slice())));
        let root2 = graph.create_root(Array::from_slice([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0].as_slice(), Shape::from([3, 3].as_slice())));
        let root3 = graph.create_root(Array::from_slice([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0].as_slice(), Shape::from([3, 3].as_slice())));
        let root4 = graph.create_root(Array::from_slice([1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0].as_slice(), Shape::from([3, 3].as_slice())));

        let op1 = graph.div::<Basic, Array<f32>>(&root4, &root1);
        let op2 = graph.mul::<Basic, Array<f32>>(&op1, &root2);
        let op3 = graph.sub::<Basic, Array<f32>>(&op2, &root3);

        let op4 = graph.mul_scalar::<Basic, Array<f32>>(2.0, &op3);
        let op5 = graph.div_scalar_rh::<Basic, Array<f32>>(&op4, 2.0);

        let op6 = graph.mul::<Basic, Array<f32>>(&op5, &op5);

        let op7 = graph.div::<Basic, Array<f32>>(&op6, &root1);

        return (op7, Array::from_slice([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0].as_slice(), Shape::from([3, 3].as_slice())), Array::from_slice([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0].as_slice(), Shape::from([3, 3].as_slice())), graph);
    }

    #[test]
    fn simple_no_eval() {
        let (_, _, added, _, graph) = init_simple_graph();

        assert!(graph.get_node(added.node_key()).is_some());

        let node = graph.get_node(added.node_key()).unwrap();

        assert!(node.tensor().is_none());
    }

    #[test]
    fn simple_eval() {
        let (_, _, added, expected, mut graph) = init_simple_graph();

        graph.non_populating_eval(&added).unwrap();

        assert!(graph.get_node(added.node_key()).is_some());

        let node = graph.get_node_mut(added.node_key()).unwrap();

        assert!(node.tensor().is_some());

        assert_eq!(node.tensor().unwrap(), expected.as_ref());

        node.clear_tensor().unwrap();

        graph.populating_eval(&added).unwrap();

        assert!(graph.get_node(added.node_key()).is_some());

        let node = graph.get_node(added.node_key()).unwrap();

        assert!(node.tensor().is_some());

        assert_eq!(node.tensor().unwrap(), expected.as_ref());
    }

    #[test]
    fn large_depth_eval() {
        let (node_key, _, expected_unit,  mut graph) = init_complex_graph();

        let power = 12u16;

        let mut out = node_key;
        for _ in  0..2_usize.pow(power as u32) {
            out = graph.div::<Basic, Array<_>>(&out, &out);
        }

        graph.non_populating_eval(&out).unwrap();

        let node = graph.get_node_mut(out.node_key()).unwrap();

        assert_eq!(*node.tensor().unwrap(), *expected_unit);

        node.clear_tensor().unwrap();

        graph.populating_eval(&out).unwrap();

        let node = graph.get_node(out.node_key()).unwrap();

        assert_eq!(*node.tensor().unwrap(), *expected_unit);
    }

    #[test]
    fn large_bredth_eval() {
        let (node_key, expected_original, _,  mut graph) = init_complex_graph();

        let power = 12u16;

        let mut curr_node_keys: Vec<CompGraphTensor>;
        let mut new_node_keys = vec![node_key; 2_usize.pow(power as u32)];

        while new_node_keys.len() > 1 {
            curr_node_keys = new_node_keys;
            new_node_keys = Vec::<CompGraphTensor>::new();

            for keys in curr_node_keys.chunks_exact(2) {
                let a_key = &keys[0];
                let b_key = &keys[1];

                new_node_keys.push(graph.add::<Basic, Array<_>>(&a_key, &b_key));
            }
        }

        let tensor = new_node_keys.last().unwrap();

        let expected = Array::from_iter( &mut expected_original.iter_units().map(|x| x * 2.0f32.pow(power)), expected_original.shape().clone());

        graph.non_populating_eval(&tensor).unwrap();

        let node = graph.get_node_mut(tensor.node_key()).unwrap();

        assert_eq!(*node.tensor().unwrap(), *expected);

        node.clear_tensor().unwrap();

        graph.populating_eval(tensor).unwrap();

        let node = graph.get_node(tensor.node_key()).unwrap();

        assert_eq!(*node.tensor().unwrap(), *expected);
    }
}