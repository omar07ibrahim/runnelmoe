use thiserror::Error;

pub type Result<T> = std::result::Result<T, RuntimeError>;

#[derive(Debug, Error, PartialEq)]
pub enum RuntimeError {
    #[error("invalid tiny-model configuration: {0}")]
    InvalidConfig(String),
    #[error("artifact is incompatible with the tiny adapter: {0}")]
    InvalidArtifact(String),
    #[error("missing tensor role {0}")]
    MissingTensor(String),
    #[error("unexpected tensor role {0}")]
    UnexpectedTensor(String),
    #[error("tensor {role} is invalid: {reason}")]
    InvalidTensor { role: String, reason: String },
    #[error("token {token} is outside vocabulary size {vocab_size}")]
    InvalidToken { token: u32, vocab_size: usize },
    #[error("unsupported character at byte {byte_offset}")]
    UnsupportedCharacter { byte_offset: usize },
    #[error("token sequence is empty")]
    EmptySequence,
    #[error("context limit {limit} would be exceeded")]
    ContextLimit { limit: usize },
    #[error("sequence state belongs to a different model instance")]
    StateMismatch,
    #[error("non-finite value produced by {0}")]
    NonFinite(&'static str),
}
