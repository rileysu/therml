pub mod array;

use std::ops::Range;

use crate::helper::{Shape, Stride};

// No mutation should be done through this trait
// It exists purely as a way to get data for the engine
trait EngineTensor<T: Sized>
    where Self: Sized 
{
    type EngineTensorIterator: Iterator<Item = T>;

    fn shape(&self) -> Shape;
    fn stride(&self) -> Stride;

    fn is_contiguous(&self) -> bool {
        Stride::from(self.shape()) == self.stride()
    }

    // Create a view into the same memory
    fn view(&self, shape: Shape) -> Result<Self, ()>;
    
    // Iterator from element to element in order
    fn iter(&self, comp_range: &[Range<usize>]) -> Self::EngineTensorIterator;
    //fn (comp_range: &[Range<usize>]) -> Self::DataSlice;
}