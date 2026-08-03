use core::fmt;

/// A buffer named in a structural kernel error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BufferRole {
    Weights,
    Input,
    Output,
    Workspace,
}

impl fmt::Display for BufferRole {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Weights => "weights",
            Self::Input => "input",
            Self::Output => "output",
            Self::Workspace => "workspace",
        })
    }
}

/// A fixed status returned by the versioned native ABI.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(i32)]
pub enum NativeStatus {
    Success = 0,
    NullPointer = 1,
    InvalidDimension = 2,
    SizeOverflow = 3,
    ByteLengthMismatch = 4,
    NaturalAlignmentFailure = 5,
    AddressRangeOverflow = 6,
    RangeOverlap = 7,
    UnavailableIsa = 8,
}

impl NativeStatus {
    /// Decodes a status defined by ABI version 1.
    #[must_use]
    pub const fn from_code(code: i32) -> Option<Self> {
        match code {
            0 => Some(Self::Success),
            1 => Some(Self::NullPointer),
            2 => Some(Self::InvalidDimension),
            3 => Some(Self::SizeOverflow),
            4 => Some(Self::ByteLengthMismatch),
            5 => Some(Self::NaturalAlignmentFailure),
            6 => Some(Self::AddressRangeOverflow),
            7 => Some(Self::RangeOverlap),
            8 => Some(Self::UnavailableIsa),
            _ => None,
        }
    }
}

impl fmt::Display for NativeStatus {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Success => "success",
            Self::NullPointer => "null pointer",
            Self::InvalidDimension => "invalid or zero dimension",
            Self::SizeOverflow => "size overflow",
            Self::ByteLengthMismatch => "byte-length mismatch",
            Self::NaturalAlignmentFailure => "natural-alignment failure",
            Self::AddressRangeOverflow => "address-range overflow",
            Self::RangeOverlap => "buffer ranges overlap",
            Self::UnavailableIsa => "AVX2 unavailable",
        })
    }
}

/// A closed error from compact storage, dispatch, or GEMV execution.
#[derive(Clone, Debug, PartialEq)]
pub enum KernelError {
    InvalidDimensions {
        rows: usize,
        columns: usize,
    },
    SizeOverflow {
        buffer: BufferRole,
    },
    MatrixLengthMismatch {
        expected_words: usize,
        actual_words: usize,
    },
    ByteLengthMismatch {
        expected_bytes: usize,
        actual_bytes: usize,
    },
    BufferLengthMismatch {
        buffer: BufferRole,
        expected_elements: usize,
        actual_elements: usize,
    },
    NullPointer {
        buffer: BufferRole,
    },
    NaturalAlignmentFailure {
        buffer: BufferRole,
    },
    NonFiniteWeight {
        index: usize,
        bits: u16,
    },
    NonFiniteInput {
        index: usize,
        bits: u32,
    },
    NonFiniteOutput {
        index: usize,
        bits: u32,
    },
    NonFiniteConversionInput {
        bits: u32,
    },
    ConversionRoundedToInfinity {
        bits: u32,
    },
    AddressRangeOverflow {
        buffer: BufferRole,
    },
    RangeOverlap {
        first: BufferRole,
        second: BufferRole,
    },
    BackendUnavailable,
    NativeFailure(NativeStatus),
    NativeContractViolation {
        status: i32,
    },
}

impl fmt::Display for KernelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimensions { rows, columns } => {
                write!(
                    formatter,
                    "GEMV dimensions must be positive, got {rows}x{columns}"
                )
            }
            Self::SizeOverflow { buffer } => write!(formatter, "{buffer} size overflows usize"),
            Self::MatrixLengthMismatch {
                expected_words,
                actual_words,
            } => write!(
                formatter,
                "BF16 matrix needs {expected_words} words, got {actual_words}"
            ),
            Self::ByteLengthMismatch {
                expected_bytes,
                actual_bytes,
            } => write!(
                formatter,
                "BF16 matrix needs {expected_bytes} bytes, got {actual_bytes}"
            ),
            Self::BufferLengthMismatch {
                buffer,
                expected_elements,
                actual_elements,
            } => write!(
                formatter,
                "{buffer} needs {expected_elements} elements, got {actual_elements}"
            ),
            Self::NullPointer { buffer } => write!(formatter, "{buffer} pointer is null"),
            Self::NaturalAlignmentFailure { buffer } => {
                write!(formatter, "{buffer} pointer is not naturally aligned")
            }
            Self::NonFiniteWeight { index, bits } => {
                write!(formatter, "weight {index} is nonfinite BF16 0x{bits:04x}")
            }
            Self::NonFiniteInput { index, bits } => {
                write!(formatter, "input {index} is nonfinite f32 0x{bits:08x}")
            }
            Self::NonFiniteOutput { index, bits } => {
                write!(formatter, "output {index} is nonfinite f32 0x{bits:08x}")
            }
            Self::NonFiniteConversionInput { bits } => {
                write!(formatter, "cannot quantize nonfinite f32 0x{bits:08x}")
            }
            Self::ConversionRoundedToInfinity { bits } => {
                write!(formatter, "finite f32 0x{bits:08x} rounds to BF16 infinity")
            }
            Self::AddressRangeOverflow { buffer } => {
                write!(formatter, "{buffer} address range overflows usize")
            }
            Self::RangeOverlap { first, second } => {
                write!(formatter, "{first} and {second} ranges overlap")
            }
            Self::BackendUnavailable => {
                formatter.write_str("requested AVX2 backend is unavailable")
            }
            Self::NativeFailure(status) => {
                write!(formatter, "native kernel rejected call: {status}")
            }
            Self::NativeContractViolation { status } => write!(
                formatter,
                "native kernel returned unknown ABI status {status}"
            ),
        }
    }
}

impl std::error::Error for KernelError {}
