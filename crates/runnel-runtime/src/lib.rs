//! Safe scalar reference runtime for the deterministic tiny MoE adapter.

mod adapter;
mod attention;
mod error;
mod model;
mod sampling;
mod state;
mod tensor;
mod tokenizer;

pub use adapter::{
    AdapterExecutionLayout, AdapterTransactionId, AdapterWorkIdentity, DecoderAdapter,
    StateLayoutAccounting,
};
pub use attention::streaming_causal_attention;
pub use error::{Result, RuntimeError};
pub use model::{
    Generation, RouteDecision, StepOutput, TinyConfig, TinyExpertContribution, TinyExpertTask,
    TinyModel, TinyPendingStateCommit, TinyPreparedToken, TinyStateCommitPermit, TinyWorkspace,
    stable_top_k,
};
pub use runnel_kernels::{BackendKind, BackendRequest};
pub use sampling::{
    RetainedCandidate, SampleConfig, SamplingError, SamplingPolicy, SamplingPreview,
    SamplingResult, SamplingWorkspace,
};
pub use state::{KvHistory, SequenceState, StateId, StateLayout};
pub use tensor::{Tensor, TensorCatalog};
pub use tokenizer::{BOS_TOKEN, EOS_TOKEN, TinyTokenizer};
