use crate::{Bf16, BufferRole, KernelError};

/// A finite, dense, row-major BF16 matrix in compact host-endian storage.
#[derive(Clone, Debug)]
pub struct Bf16Matrix {
    rows: usize,
    columns: usize,
    storage: Box<[u16]>,
    word_offset: usize,
}

impl PartialEq for Bf16Matrix {
    fn eq(&self, other: &Self) -> bool {
        self.rows == other.rows && self.columns == other.columns && self.words() == other.words()
    }
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
        Self::from_storage(rows, columns, words, 0)
    }

    /// Builds a compact matrix whose logical words begin at an owned-storage
    /// offset.
    ///
    /// The storage must contain exactly `word_offset` prefix words followed by
    /// the checked row-major matrix length. Prefix words are padding: they are
    /// retained for allocation alignment experiments but are never exposed,
    /// validated as weights, or passed to a kernel. Equality is defined over
    /// dimensions and logical words, not prefix contents.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid dimensions, offset/size overflow, an
    /// inexact owned-storage length, or a nonfinite logical matrix word.
    pub fn from_storage(
        rows: usize,
        columns: usize,
        storage: impl Into<Box<[u16]>>,
        word_offset: usize,
    ) -> Result<Self, KernelError> {
        validate_dimensions(rows, columns)?;
        let logical_words = checked_elements(rows, columns)?;
        let expected_storage_words =
            word_offset
                .checked_add(logical_words)
                .ok_or(KernelError::SizeOverflow {
                    buffer: BufferRole::Weights,
                })?;
        let storage = storage.into();
        if storage.len() != expected_storage_words {
            return Err(KernelError::MatrixLengthMismatch {
                expected_words: expected_storage_words,
                actual_words: storage.len(),
            });
        }
        let logical = &storage[word_offset..expected_storage_words];
        validate_finite_words(logical)?;
        Ok(Self {
            rows,
            columns,
            storage,
            word_offset,
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
        self.rows * self.columns
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[must_use]
    pub fn words(&self) -> &[u16] {
        &self.storage[self.word_offset..]
    }

    #[must_use]
    pub fn get(&self, row: usize, column: usize) -> Option<Bf16> {
        if row >= self.rows || column >= self.columns {
            return None;
        }
        Some(Bf16::from_bits(self.words()[row * self.columns + column]))
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
