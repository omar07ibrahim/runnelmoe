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

/// Caller-owned scratch reused by prepared GEMVs.
#[derive(Clone, Debug)]
pub struct GemvWorkspace {
    temporary_output: Vec<f32>,
}

impl GemvWorkspace {
    #[must_use]
    pub fn new(rows: usize) -> Self {
        Self {
            temporary_output: vec![0.0; rows],
        }
    }

    #[must_use]
    pub fn rows(&self) -> usize {
        self.temporary_output.len()
    }

    /// Changes the logical row count while retaining the allocation whenever
    /// the requested size fits its existing capacity.
    ///
    /// Prepared operations still require an exact logical row count. Callers
    /// that need to reuse one maximum-sized allocation across differently
    /// shaped GEMVs can resize it immediately before each invocation without
    /// weakening that validation contract.
    pub fn resize_rows(&mut self, rows: usize) {
        self.temporary_output.resize(rows, 0.0);
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
