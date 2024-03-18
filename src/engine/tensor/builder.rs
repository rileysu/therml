use std::ops::RangeBounds;

use crate::engine::unit::UnitCompatible;

use super::EngineTensor;

trait EngineTensorBuilder {
    type Unit: UnitCompatible;
    type Tensor: EngineTensor;

    fn splice<R: RangeBounds<usize>, I: IntoIterator<Item = Self::Unit>>(range: R, replace_with: I);

    fn construct() -> Box<dyn EngineTensor<Unit = Self::Unit>>;
}

struct ArrayBuilder;