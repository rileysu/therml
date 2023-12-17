pub mod quant;

pub use quant::QuantExtension;

use super::EngineTensor;

pub trait ExtensionProvider {
    fn quant(&self) -> Option<Box<dyn QuantExtension>> {
        None
    }
}

//This is a default provider for tensors without extensions
#[derive(Debug)]
pub struct EmptyExtensionProvider<'a, E: EngineTensor> {
    engine_tensor: &'a E,
}

impl<'a, E: EngineTensor> ExtensionProvider for EmptyExtensionProvider<'a, E> {}

impl<'a, E: EngineTensor> From<&'a E> for EmptyExtensionProvider<'a, E> {
    fn from(value: &'a E) -> Self {
        Self {
            engine_tensor: value,
        }
    }
}
