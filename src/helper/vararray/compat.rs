use super::{VarArray, Unit, iter::Iter, VarArrayError};

pub trait VarArrayCompatible: Sized {
    fn new(varr: VarArray) -> Self;
    fn vararray(&self) -> &VarArray;
    fn vararray_mut(&mut self) -> &mut VarArray;

    fn len(&self) -> usize {
        self.vararray().len()
    }
    fn get(&self, index: usize) -> Result<Unit, VarArrayError> {
        self.vararray().get(index)
    }
    fn get_mut(&mut self, index: usize) -> Result<&mut Unit, VarArrayError> {
        self.vararray_mut().get_mut(index)
    }
    fn iter(&self) -> Iter {
        self.vararray().iter()
    }
    fn as_slice(&self) -> &[usize] {
        self.vararray().as_slice()
    }
    fn as_mut_slice(&mut self) -> &mut [usize] {
        self.vararray_mut().as_mut_slice()
    }
    fn concat(first: &Self, second: &Self) -> Self {
        Self::concat(first, second)
    }
    fn add(&self, rhs: &impl VarArrayCompatible) -> Result<Self, VarArrayError> {
        Ok(Self::new(self.vararray().add(rhs.vararray())?))
    }
    fn sub(&self, rhs: &impl VarArrayCompatible) -> Result<Self, VarArrayError> {
        Ok(Self::new(self.vararray().sub(rhs.vararray())?))
    }
    fn div(&self, rhs: &impl VarArrayCompatible) -> Result<Self, VarArrayError> {
        Ok(Self::new(self.vararray().div(rhs.vararray())?))
    }
    fn mul(&self, rhs: &impl VarArrayCompatible) -> Result<Self, VarArrayError> {
        Ok(Self::new(self.vararray().mul(rhs.vararray())?))
    }
    fn rem(&self, rhs: &impl VarArrayCompatible) -> Result<Self, VarArrayError> {
        Ok(Self::new(self.vararray().rem(rhs.vararray())?))
    }
}