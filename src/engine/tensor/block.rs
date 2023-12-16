
pub type ISum = i64;

pub trait BlockUnit {}
impl BlockUnit for u8 {}

//Low level memory view of a data section
//Used for optimising calculations between blocks
//Eg in quantised tensors blocks of a certain scale need to be multiplied
//Should be smaller of two tensors block sizes and should be the same size
pub trait EngineTensorBlock<'a, T: BlockUnit> {
    fn len() -> usize;
    fn get() -> T;
}
