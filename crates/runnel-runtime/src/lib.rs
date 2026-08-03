//! Safe scalar reference runtime for the deterministic tiny MoE adapter.

mod error;
mod model;
mod tensor;
mod tokenizer;

pub use error::{Result, RuntimeError};
pub use model::{
    Generation, RouteDecision, SequenceState, StepOutput, TinyConfig, TinyModel, stable_top_k,
};
pub use runnel_kernels::{BackendKind, BackendRequest};
pub use tensor::{Tensor, TensorCatalog};
pub use tokenizer::{BOS_TOKEN, EOS_TOKEN, TinyTokenizer};
