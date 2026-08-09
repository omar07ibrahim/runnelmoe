//! Compact BF16 storage and one auditable expert GEMV dispatch boundary.

#![deny(unsafe_code)]

mod bf16;
mod dispatch;
mod error;
#[allow(unsafe_code)]
mod ffi;
mod matrix;

pub use bf16::{Bf16, Bf16Class};
pub use dispatch::{
    BackendKind, BackendRequest, Capabilities, native_candidate_compiled, select_backend,
};
pub use error::{BufferRole, KernelError, NativeStatus};
pub use matrix::Bf16Matrix;

/// An immutable input proven finite once before dispatch.
#[derive(Clone, Copy, Debug)]
pub struct FiniteInput<'a> {
    values: &'a [f32],
}

impl<'a> FiniteInput<'a> {
    /// Validates and borrows a finite f32 input vector.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::NonFiniteInput`] at the first infinity or NaN.
    pub fn new(values: &'a [f32]) -> Result<Self, KernelError> {
        for (index, &value) in values.iter().enumerate() {
            if !value.is_finite() {
                return Err(KernelError::NonFiniteInput {
                    index,
                    bits: value.to_bits(),
                });
            }
        }
        Ok(Self { values })
    }

    #[must_use]
    pub const fn as_slice(self) -> &'a [f32] {
        self.values
    }

    #[must_use]
    pub const fn len(self) -> usize {
        self.values.len()
    }

    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.values.is_empty()
    }
}

/// Caller-owned scratch reused by prepared GEMVs without allocation growth.
#[derive(Debug)]
pub struct GemvWorkspace {
    temporary_output: Vec<f32>,
    max_rows: usize,
}

impl GemvWorkspace {
    /// Reserves scratch for at most `max_rows` output elements.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::SizeOverflow`] when the byte count cannot be
    /// represented, or [`KernelError::AllocationFailure`] when the requested
    /// allocation cannot be reserved. No workspace becomes visible on error.
    pub fn try_new(max_rows: usize) -> Result<Self, KernelError> {
        let requested_bytes =
            max_rows
                .checked_mul(size_of::<f32>())
                .ok_or(KernelError::SizeOverflow {
                    buffer: BufferRole::Workspace,
                })?;
        let mut temporary_output = Vec::new();
        temporary_output.try_reserve_exact(max_rows).map_err(|_| {
            KernelError::AllocationFailure {
                buffer: BufferRole::Workspace,
                requested_bytes,
            }
        })?;
        temporary_output.resize(max_rows, 0.0);
        Ok(Self {
            temporary_output,
            max_rows,
        })
    }

    #[must_use]
    pub fn rows(&self) -> usize {
        self.temporary_output.len()
    }

    #[must_use]
    pub const fn max_rows(&self) -> usize {
        self.max_rows
    }

    #[must_use]
    pub fn capacity(&self) -> usize {
        self.temporary_output.capacity()
    }

    /// Changes the logical row count without moving or growing scratch.
    ///
    /// Prepared operations still require an exact logical row count. A row
    /// count above the maximum declared to [`Self::try_new`] is rejected
    /// without changing the logical row count or allocation.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::WorkspaceRowsExceedMaximum`] when `rows` is
    /// above the construction-time maximum.
    pub fn set_rows(&mut self, rows: usize) -> Result<(), KernelError> {
        if rows > self.max_rows {
            return Err(KernelError::WorkspaceRowsExceedMaximum {
                max_rows: self.max_rows,
                requested_rows: rows,
            });
        }
        self.temporary_output.resize(rows, 0.0);
        Ok(())
    }
}

/// A dimension-checked GEMV with dispatch resolved outside execution.
#[derive(Debug)]
pub struct PreparedGemv<'matrix, 'input> {
    matrix: &'matrix Bf16Matrix,
    input: FiniteInput<'input>,
    backend: BackendKind,
}

impl<'matrix, 'input> PreparedGemv<'matrix, 'input> {
    /// Checks dimensions and resolves dispatch using detected capabilities.
    ///
    /// # Errors
    ///
    /// Returns an error for an input-length mismatch or unavailable forced
    /// backend.
    pub fn new(
        matrix: &'matrix Bf16Matrix,
        input: FiniteInput<'input>,
        request: BackendRequest,
    ) -> Result<Self, KernelError> {
        Self::with_capabilities(matrix, input, request, Capabilities::detected())
    }

    /// Checks dimensions and resolves dispatch using a restrict-only token.
    ///
    /// # Errors
    ///
    /// Returns an error for an input-length mismatch or unavailable forced
    /// backend.
    pub fn with_capabilities(
        matrix: &'matrix Bf16Matrix,
        input: FiniteInput<'input>,
        request: BackendRequest,
        capabilities: Capabilities,
    ) -> Result<Self, KernelError> {
        if input.len() != matrix.columns() {
            return Err(KernelError::BufferLengthMismatch {
                buffer: BufferRole::Input,
                expected_elements: matrix.columns(),
                actual_elements: input.len(),
            });
        }
        let backend = select_backend(request, capabilities)?;
        Ok(Self {
            matrix,
            input,
            backend,
        })
    }

    #[must_use]
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    #[must_use]
    pub const fn rows(&self) -> usize {
        self.matrix.rows()
    }

    #[must_use]
    pub const fn columns(&self) -> usize {
        self.matrix.columns()
    }

    /// Computes into scratch, validates it, then publishes the complete output.
    ///
    /// # Errors
    ///
    /// Returns an error for wrong output/workspace lengths, native rejection,
    /// or a nonfinite computed value. `output` remains unchanged on every error.
    pub fn run(
        &self,
        workspace: &mut GemvWorkspace,
        output: &mut [f32],
    ) -> Result<(), KernelError> {
        require_output_lengths(self.rows(), workspace, output)?;
        let temporary = &mut workspace.temporary_output;
        match self.backend {
            BackendKind::Scalar => scalar_compute(self.matrix, self.input, temporary),
            BackendKind::Avx2 => ffi::gemv(
                self.matrix.words(),
                self.input.as_slice(),
                temporary,
                self.rows(),
                self.columns(),
            )?,
        }
        validate_finite_output(temporary)?;
        output.copy_from_slice(temporary);
        Ok(())
    }
}

/// Independently invokes the ascending-column safe Rust reference path.
///
/// # Errors
///
/// Returns an error for wrong input/output/workspace lengths or a nonfinite
/// computed value. `output` remains unchanged on every error.
pub fn scalar_gemv(
    matrix: &Bf16Matrix,
    input: FiniteInput<'_>,
    workspace: &mut GemvWorkspace,
    output: &mut [f32],
) -> Result<(), KernelError> {
    if input.len() != matrix.columns() {
        return Err(KernelError::BufferLengthMismatch {
            buffer: BufferRole::Input,
            expected_elements: matrix.columns(),
            actual_elements: input.len(),
        });
    }
    require_output_lengths(matrix.rows(), workspace, output)?;
    scalar_compute(matrix, input, &mut workspace.temporary_output);
    validate_finite_output(&workspace.temporary_output)?;
    output.copy_from_slice(&workspace.temporary_output);
    Ok(())
}

fn require_output_lengths(
    rows: usize,
    workspace: &GemvWorkspace,
    output: &[f32],
) -> Result<(), KernelError> {
    if workspace.rows() != rows {
        return Err(KernelError::BufferLengthMismatch {
            buffer: BufferRole::Workspace,
            expected_elements: rows,
            actual_elements: workspace.rows(),
        });
    }
    if output.len() != rows {
        return Err(KernelError::BufferLengthMismatch {
            buffer: BufferRole::Output,
            expected_elements: rows,
            actual_elements: output.len(),
        });
    }
    Ok(())
}

fn scalar_compute(matrix: &Bf16Matrix, input: FiniteInput<'_>, output: &mut [f32]) {
    for (row, destination) in output.iter_mut().enumerate() {
        let start = row * matrix.columns();
        let row_words = &matrix.words()[start..start + matrix.columns()];
        let mut sum = 0.0_f32;
        for (&word, &value) in row_words.iter().zip(input.as_slice()) {
            let product = Bf16::from_bits(word).to_f32() * value;
            sum += product;
        }
        *destination = sum;
    }
}

fn validate_finite_output(output: &[f32]) -> Result<(), KernelError> {
    for (index, &value) in output.iter().enumerate() {
        if !value.is_finite() {
            return Err(KernelError::NonFiniteOutput {
                index,
                bits: value.to_bits(),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod workspace_tests {
    use super::{BufferRole, GemvWorkspace, KernelError};

    fn allocation_signature(workspace: &GemvWorkspace) -> (*const f32, usize) {
        (
            workspace.temporary_output.as_ptr(),
            workspace.temporary_output.capacity(),
        )
    }

    #[test]
    fn workspace_construction_rejects_overflow_and_impossible_reservation() {
        assert!(matches!(
            GemvWorkspace::try_new(usize::MAX),
            Err(KernelError::SizeOverflow {
                buffer: BufferRole::Workspace
            })
        ));

        let impossible_rows = (isize::MAX as usize / size_of::<f32>()) + 1;
        let impossible_bytes = impossible_rows * size_of::<f32>();
        assert!(matches!(
            GemvWorkspace::try_new(impossible_rows),
            Err(KernelError::AllocationFailure {
                buffer: BufferRole::Workspace,
                requested_bytes,
            }) if requested_bytes == impossible_bytes
        ));
    }

    #[test]
    fn above_maximum_rejection_preserves_rows_and_allocation() {
        let mut workspace = GemvWorkspace::try_new(12).unwrap();
        workspace.set_rows(8).unwrap();
        let before = allocation_signature(&workspace);

        assert_eq!(
            workspace.set_rows(13),
            Err(KernelError::WorkspaceRowsExceedMaximum {
                max_rows: 12,
                requested_rows: 13,
            })
        );
        assert_eq!(workspace.rows(), 8);
        assert_eq!(allocation_signature(&workspace), before);
    }

    #[test]
    fn maximum_to_narrow_to_maximum_reuses_one_allocation() {
        let mut workspace = GemvWorkspace::try_new(12).unwrap();
        let initial = allocation_signature(&workspace);

        workspace.set_rows(8).unwrap();
        assert_eq!(workspace.rows(), 8);
        assert_eq!(allocation_signature(&workspace), initial);

        workspace.set_rows(12).unwrap();
        assert_eq!(workspace.rows(), 12);
        assert_eq!(allocation_signature(&workspace), initial);
    }
}
