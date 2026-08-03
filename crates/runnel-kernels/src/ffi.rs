use core::ffi::c_void;

#[cfg(any(runnel_native_avx2, test))]
use crate::NativeStatus;
use crate::{BufferRole, KernelError};

#[cfg(runnel_native_avx2)]
unsafe extern "C" {
    fn runnel_bf16_gemv_avx2_v1(
        weights: *const c_void,
        weight_bytes: usize,
        input: *const c_void,
        input_bytes: usize,
        output: *mut c_void,
        output_bytes: usize,
        rows: usize,
        columns: usize,
    ) -> i32;
}

pub(crate) fn gemv(
    weights: &[u16],
    input: &[f32],
    output: &mut [f32],
    rows: usize,
    columns: usize,
) -> Result<(), KernelError> {
    if rows == 0 || columns == 0 {
        return Err(KernelError::InvalidDimensions { rows, columns });
    }
    let weight_bytes = checked_bytes(
        rows.checked_mul(columns).ok_or(KernelError::SizeOverflow {
            buffer: BufferRole::Weights,
        })?,
        size_of::<u16>(),
        BufferRole::Weights,
    )?;
    let input_bytes = checked_bytes(columns, size_of::<f32>(), BufferRole::Input)?;
    let output_bytes = checked_bytes(rows, size_of::<f32>(), BufferRole::Output)?;

    require_bytes(
        weights.len(),
        size_of::<u16>(),
        weight_bytes,
        BufferRole::Weights,
    )?;

    require_pointer(
        weights.as_ptr().cast::<c_void>() as usize,
        align_of::<u16>(),
        BufferRole::Weights,
    )?;
    require_pointer(
        input.as_ptr().cast::<c_void>() as usize,
        align_of::<f32>(),
        BufferRole::Input,
    )?;
    require_pointer(
        output.as_mut_ptr().cast::<c_void>() as usize,
        align_of::<f32>(),
        BufferRole::Output,
    )?;
    require_bytes(
        input.len(),
        size_of::<f32>(),
        input_bytes,
        BufferRole::Input,
    )?;
    require_bytes(
        output.len(),
        size_of::<f32>(),
        output_bytes,
        BufferRole::Output,
    )?;

    let weight_range = checked_range(
        weights.as_ptr().cast::<c_void>() as usize,
        weight_bytes,
        BufferRole::Weights,
    )?;
    let input_range = checked_range(
        input.as_ptr().cast::<c_void>() as usize,
        input_bytes,
        BufferRole::Input,
    )?;
    let output_range = checked_range(
        output.as_mut_ptr().cast::<c_void>() as usize,
        output_bytes,
        BufferRole::Output,
    )?;
    require_disjoint(
        weight_range,
        input_range,
        BufferRole::Weights,
        BufferRole::Input,
    )?;
    require_disjoint(
        weight_range,
        output_range,
        BufferRole::Weights,
        BufferRole::Output,
    )?;
    require_disjoint(
        input_range,
        output_range,
        BufferRole::Input,
        BufferRole::Output,
    )?;

    call_native(
        weights,
        weight_bytes,
        input,
        input_bytes,
        output,
        output_bytes,
        rows,
        columns,
    )
}

fn require_pointer(
    address: usize,
    alignment: usize,
    buffer: BufferRole,
) -> Result<(), KernelError> {
    if address == 0 {
        return Err(KernelError::NullPointer { buffer });
    }
    if !address.is_multiple_of(alignment) {
        return Err(KernelError::NaturalAlignmentFailure { buffer });
    }
    Ok(())
}

fn checked_bytes(
    elements: usize,
    element_size: usize,
    buffer: BufferRole,
) -> Result<usize, KernelError> {
    elements
        .checked_mul(element_size)
        .ok_or(KernelError::SizeOverflow { buffer })
}

fn require_bytes(
    elements: usize,
    element_size: usize,
    expected: usize,
    buffer: BufferRole,
) -> Result<(), KernelError> {
    let actual = checked_bytes(elements, element_size, buffer)?;
    if actual != expected {
        return Err(KernelError::BufferLengthMismatch {
            buffer,
            expected_elements: expected / element_size,
            actual_elements: elements,
        });
    }
    Ok(())
}

fn checked_range(
    start: usize,
    bytes: usize,
    buffer: BufferRole,
) -> Result<(usize, usize), KernelError> {
    let end = start
        .checked_add(bytes)
        .ok_or(KernelError::AddressRangeOverflow { buffer })?;
    Ok((start, end))
}

fn require_disjoint(
    first: (usize, usize),
    second: (usize, usize),
    first_role: BufferRole,
    second_role: BufferRole,
) -> Result<(), KernelError> {
    if first.0 < second.1 && second.0 < first.1 {
        return Err(KernelError::RangeOverlap {
            first: first_role,
            second: second_role,
        });
    }
    Ok(())
}

#[cfg(runnel_native_avx2)]
#[allow(clippy::too_many_arguments)]
fn call_native(
    weights: &[u16],
    weight_bytes: usize,
    input: &[f32],
    input_bytes: usize,
    output: &mut [f32],
    output_bytes: usize,
    rows: usize,
    columns: usize,
) -> Result<(), KernelError> {
    // SAFETY: `gemv` has checked all three exact byte extents and address
    // arithmetic. The typed nonempty slices provide natural alignment and live
    // storage. Rust's shared/mutable borrows keep weights/input immutable and
    // output exclusively writable for the call, and pairwise range checks have
    // rejected overlap. The C symbol does not retain any pointer.
    let status = unsafe {
        runnel_bf16_gemv_avx2_v1(
            weights.as_ptr().cast::<c_void>(),
            weight_bytes,
            input.as_ptr().cast::<c_void>(),
            input_bytes,
            output.as_mut_ptr().cast::<c_void>(),
            output_bytes,
            rows,
            columns,
        )
    };
    map_status(status)
}

#[cfg(not(runnel_native_avx2))]
#[allow(clippy::too_many_arguments)]
fn call_native(
    _weights: &[u16],
    _weight_bytes: usize,
    _input: &[f32],
    _input_bytes: usize,
    _output: &mut [f32],
    _output_bytes: usize,
    _rows: usize,
    _columns: usize,
) -> Result<(), KernelError> {
    Err(KernelError::BackendUnavailable)
}

#[cfg(any(runnel_native_avx2, test))]
fn map_status(code: i32) -> Result<(), KernelError> {
    match NativeStatus::from_code(code) {
        Some(NativeStatus::Success) => Ok(()),
        Some(status) => Err(KernelError::NativeFailure(status)),
        None => Err(KernelError::NativeContractViolation { status: code }),
    }
}

#[cfg(test)]
mod tests {
    use super::map_status;
    use crate::{KernelError, NativeStatus};

    #[test]
    fn status_mapping_is_closed() {
        assert_eq!(map_status(0), Ok(()));
        for code in 1..=8 {
            assert_eq!(
                map_status(code),
                Err(KernelError::NativeFailure(
                    NativeStatus::from_code(code).expect("known status")
                ))
            );
        }
        assert_eq!(
            map_status(99),
            Err(KernelError::NativeContractViolation { status: 99 })
        );
    }
}
