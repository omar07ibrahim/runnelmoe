use crate::{Bf16, BufferRole, KernelError};

/// A finite, dense, row-major BF16 matrix in compact host-endian storage.
#[derive(Clone, Debug, PartialEq)]
pub struct Bf16Matrix {
    rows: usize,
    columns: usize,
    words: Box<[u16]>,
}

impl Bf16Matrix {
    /// Builds compact storage from host-endian words.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid dimensions, size/length mismatch, or any
    /// infinity or NaN word.
    pub fn from_words(
        rows: usize,
        columns: usize,
        words: impl Into<Box<[u16]>>,
    ) -> Result<Self, KernelError> {
        validate_dimensions(rows, columns)?;
        let expected = checked_elements(rows, columns)?;
        let words = words.into();
        if words.len() != expected {
            return Err(KernelError::MatrixLengthMismatch {
                expected_words: expected,
                actual_words: words.len(),
            });
        }
        validate_finite_words(&words)?;
        Ok(Self {
            rows,
            columns,
            words,
        })
    }

    /// Decodes compact little-endian artifact bytes without widening them.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid dimensions, size/length mismatch, or any
    /// infinity or NaN word.
    pub fn from_le_bytes(rows: usize, columns: usize, bytes: &[u8]) -> Result<Self, KernelError> {
        validate_dimensions(rows, columns)?;
        let elements = checked_elements(rows, columns)?;
        let expected_bytes =
            elements
                .checked_mul(size_of::<u16>())
                .ok_or(KernelError::SizeOverflow {
                    buffer: BufferRole::Weights,
                })?;
        if bytes.len() != expected_bytes {
            return Err(KernelError::ByteLengthMismatch {
                expected_bytes,
                actual_bytes: bytes.len(),
            });
        }

        let words = bytes
            .chunks_exact(2)
            .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
            .collect::<Vec<_>>();
        Self::from_words(rows, columns, words)
    }

    /// Quantizes a row-major f32 matrix with ties-to-even rounding.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid dimensions or length, nonfinite input, or
    /// a finite input that rounds to BF16 infinity.
    pub fn try_from_f32(rows: usize, columns: usize, values: &[f32]) -> Result<Self, KernelError> {
        validate_dimensions(rows, columns)?;
        let expected = checked_elements(rows, columns)?;
        if values.len() != expected {
            return Err(KernelError::MatrixLengthMismatch {
                expected_words: expected,
                actual_words: values.len(),
            });
        }

        let mut words = Vec::with_capacity(expected);
        for &value in values {
            words.push(Bf16::try_from_f32_rne(value)?.to_bits());
        }
        Self::from_words(rows, columns, words)
    }

    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    #[must_use]
    pub const fn columns(&self) -> usize {
        self.columns
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.words.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.words.is_empty()
    }

    #[must_use]
    pub fn words(&self) -> &[u16] {
        &self.words
    }

    #[must_use]
    pub fn get(&self, row: usize, column: usize) -> Option<Bf16> {
        if row >= self.rows || column >= self.columns {
            return None;
        }
        Some(Bf16::from_bits(self.words[row * self.columns + column]))
    }
}

pub(crate) fn validate_dimensions(rows: usize, columns: usize) -> Result<(), KernelError> {
    if rows == 0 || columns == 0 {
        return Err(KernelError::InvalidDimensions { rows, columns });
    }
    Ok(())
}

pub(crate) fn checked_elements(rows: usize, columns: usize) -> Result<usize, KernelError> {
    rows.checked_mul(columns).ok_or(KernelError::SizeOverflow {
        buffer: BufferRole::Weights,
    })
}

fn validate_finite_words(words: &[u16]) -> Result<(), KernelError> {
    for (index, &bits) in words.iter().enumerate() {
        if !Bf16::from_bits(bits).is_finite() {
            return Err(KernelError::NonFiniteWeight { index, bits });
        }
    }
    Ok(())
}
