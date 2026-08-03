//! Safe scalar reference runtime for the deterministic tiny MoE adapter.

mod attention;
mod error;
mod model;
mod sampling;
mod state;
mod tensor;
mod tokenizer;

pub use attention::streaming_causal_attention;
pub use error::{Result, RuntimeError};
pub use model::{Generation, RouteDecision, StepOutput, TinyConfig, TinyModel, stable_top_k};
pub use runnel_kernels::{BackendKind, BackendRequest};
pub use sampling::{
    RetainedCandidate, SampleConfig, SamplingError, SamplingPolicy, SamplingPreview,
    SamplingResult, SamplingWorkspace,
};
pub use state::{KvHistory, SequenceState, StateId, StateLayout};
pub use tensor::{Tensor, TensorCatalog};
pub use tokenizer::{BOS_TOKEN, EOS_TOKEN, TinyTokenizer};
