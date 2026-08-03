use runnel_kernels::{
    BackendKind, BackendRequest, Bf16, Bf16Matrix, BufferRole, Capabilities, FiniteInput,
    GemvWorkspace, KernelError, PreparedGemv, native_candidate_compiled, scalar_gemv,
    select_backend,
};

#[test]
fn exact_row_major_case_prevents_transpose_errors() {
    let matrix = matrix_from_i16(3, 4, &[1, 2, 3, 4, -2, 1, 0, 3, 7, -1, 2, -4]);
    let input_values = [2.0, -1.0, 0.5, 3.0];
    let input = FiniteInput::new(&input_values).unwrap();
    let mut workspace = GemvWorkspace::try_new(3).unwrap();
    let mut output = [f32::NAN; 3];
    scalar_gemv(&matrix, input, &mut workspace, &mut output).unwrap();
    assert_eq!(
        output.map(f32::to_bits),
        [13.5_f32, 4.0, 4.0].map(f32::to_bits)
    );

    if Capabilities::detected().avx2_available() {
        let prepared = PreparedGemv::new(&matrix, input, BackendRequest::Avx2).unwrap();
        let mut native_output = [f32::NAN; 3];
        prepared
            .run(&mut workspace, &mut native_output)
            .expect("exact AVX2 case");
        assert_eq!(native_output.map(f32::to_bits), output.map(f32::to_bits));
    }
}

#[test]
fn one_word_owned_prefix_preserves_each_backend_result() {
    let logical_words = [1_i16, 2, 3, 4, -2, 1, 0, 3, 7, -1, 2, -4]
        .map(|value| Bf16::from_f32_rne(f32::from(value)).to_bits());
    let natural = Bf16Matrix::from_words(3, 4, logical_words.to_vec()).unwrap();
    let mut storage = Vec::with_capacity(logical_words.len() + 1);
    storage.push(0x7fc1);
    storage.extend_from_slice(&logical_words);
    let offset = Bf16Matrix::from_storage(3, 4, storage, 1).unwrap();
    let input_values = [2.0, -1.0, 0.5, 3.0];
    let input = FiniteInput::new(&input_values).unwrap();

    for request in [BackendRequest::Scalar, BackendRequest::Avx2] {
        if request == BackendRequest::Avx2 && !Capabilities::detected().avx2_available() {
            continue;
        }
        let mut natural_workspace = GemvWorkspace::try_new(3).unwrap();
        let mut offset_workspace = GemvWorkspace::try_new(3).unwrap();
        let mut natural_output = [f32::NAN; 3];
        let mut offset_output = [f32::NAN; 3];
        PreparedGemv::new(&natural, input, request)
            .unwrap()
            .run(&mut natural_workspace, &mut natural_output)
            .unwrap();
        PreparedGemv::new(&offset, input, request)
            .unwrap()
            .run(&mut offset_workspace, &mut offset_output)
            .unwrap();
        assert_eq!(
            offset_output.map(f32::to_bits),
            natural_output.map(f32::to_bits)
        );
    }
}

#[test]
fn every_tail_and_boundary_meets_each_backends_f64_bound() {
    let mut state = 0x6a09_e667_f3bc_c909_u64;
    let columns = (1..=33).chain([255, 256, 257, 4095, 4096, 4097]);
    for rows in [1, 2, 3, 7, 8, 17] {
        for columns in columns.clone() {
            let words = (0..rows * columns)
                .map(|_| {
                    let signed = i16::from(next_u8(&mut state)) - 128;
                    Bf16::from_f32_rne(f32::from(signed) / 64.0).to_bits()
                })
                .collect::<Vec<_>>();
            let input_values = (0..columns)
                .map(|_| {
                    let signed = i16::from(next_u8(&mut state)) - 128;
                    f32::from(signed) / 32.0
                })
                .collect::<Vec<_>>();
            check_backend(
                rows,
                columns,
                words.clone(),
                &input_values,
                BackendRequest::Scalar,
            );
            if Capabilities::detected().avx2_available() {
                check_backend(rows, columns, words, &input_values, BackendRequest::Avx2);
            }
        }
    }
}

#[test]
fn finite_subnormals_and_cancellation_are_diagnostic_cases() {
    let mut subnormal_words = vec![0_u16; 3 * 8];
    subnormal_words[0] = 0x3f80;
    subnormal_words[8 + 1] = 0x0001;
    subnormal_words[16 + 1] = 0x8001;
    let subnormal_matrix =
        Bf16Matrix::from_words(3, 8, subnormal_words).expect("finite BF16 subnormals");
    let subnormal_input_values = [f32::from_bits(1), 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let subnormal_input = FiniteInput::new(&subnormal_input_values).unwrap();
    for request in [BackendRequest::Scalar, BackendRequest::Avx2] {
        if request == BackendRequest::Avx2 && !Capabilities::detected().avx2_available() {
            continue;
        }
        let prepared = PreparedGemv::new(&subnormal_matrix, subnormal_input, request).unwrap();
        let mut workspace = GemvWorkspace::try_new(3).unwrap();
        let mut output = [f32::NAN; 3];
        prepared.run(&mut workspace, &mut output).unwrap();
        assert_eq!(
            output.map(f32::to_bits),
            [0x0000_0001, 0x0001_0000, 0x8001_0000],
            "backend={:?}",
            prepared.backend()
        );
    }

    let mut near_limit_words = vec![0_u16; 2 * 8];
    near_limit_words[0] = 0x7f7f;
    near_limit_words[8] = 0xff7f;
    let near_limit_matrix = Bf16Matrix::from_words(2, 8, near_limit_words).unwrap();
    let near_limit_values = [0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let near_limit_input = FiniteInput::new(&near_limit_values).unwrap();
    for request in [BackendRequest::Scalar, BackendRequest::Avx2] {
        if request == BackendRequest::Avx2 && !Capabilities::detected().avx2_available() {
            continue;
        }
        let prepared = PreparedGemv::new(&near_limit_matrix, near_limit_input, request).unwrap();
        let mut workspace = GemvWorkspace::try_new(2).unwrap();
        let mut output = [f32::NAN; 2];
        prepared.run(&mut workspace, &mut output).unwrap();
        assert_eq!(
            output.map(f32::to_bits),
            [0x7eff_0000, 0xfeff_0000],
            "backend={:?}",
            prepared.backend()
        );
    }

    let cancellation_words = (0..17)
        .map(|index| {
            let value = if index % 2 == 0 { 1.0 } else { -1.0 };
            Bf16::from_f32_rne(value).to_bits()
        })
        .collect::<Vec<_>>();
    let mut cancellation_input = vec![16_777_216.0; 17];
    cancellation_input[8] = 1.0;
    check_backend(
        1,
        17,
        cancellation_words.clone(),
        &cancellation_input,
        BackendRequest::Scalar,
    );
    if Capabilities::detected().avx2_available() {
        check_backend(
            1,
            17,
            cancellation_words,
            &cancellation_input,
            BackendRequest::Avx2,
        );
    }
}

#[test]
fn dispatch_is_explicit_and_capability_can_only_be_removed() {
    let detected = Capabilities::detected();
    let disabled = detected.without_avx2();
    assert!(!disabled.avx2_available());
    assert_eq!(
        select_backend(BackendRequest::Auto, disabled),
        Ok(BackendKind::Scalar)
    );
    assert_eq!(
        select_backend(BackendRequest::Scalar, detected),
        Ok(BackendKind::Scalar)
    );
    assert_eq!(
        select_backend(BackendRequest::Avx2, disabled),
        Err(KernelError::BackendUnavailable)
    );
    assert!(!native_candidate_compiled() || cfg!(all(target_arch = "x86_64", target_os = "linux")));
    if detected.avx2_available() {
        assert_eq!(
            select_backend(BackendRequest::Auto, detected),
            Ok(BackendKind::Avx2)
        );
    }
}

#[test]
fn errors_do_not_publish_partial_output() {
    let matrix = Bf16Matrix::from_words(1, 8, vec![0x7f7f; 8]).unwrap();
    let input_values = [f32::MAX; 8];
    let input = FiniteInput::new(&input_values).unwrap();
    for request in [BackendRequest::Scalar, BackendRequest::Avx2] {
        if request == BackendRequest::Avx2 && !Capabilities::detected().avx2_available() {
            continue;
        }
        let prepared = PreparedGemv::new(&matrix, input, request).unwrap();
        let mut workspace = GemvWorkspace::try_new(1).unwrap();
        let mut output = [123.0];
        assert!(matches!(
            prepared.run(&mut workspace, &mut output),
            Err(KernelError::NonFiniteOutput { index: 0, .. })
        ));
        assert_eq!(output.map(f32::to_bits), [123.0_f32.to_bits()]);
    }

    let prepared = PreparedGemv::new(&matrix, input, BackendRequest::Scalar).unwrap();
    let mut workspace = GemvWorkspace::try_new(1).unwrap();
    let mut short_output = Vec::<f32>::new();
    assert!(matches!(
        prepared.run(&mut workspace, &mut short_output),
        Err(KernelError::BufferLengthMismatch {
            buffer: BufferRole::Output,
            ..
        })
    ));

    let mut wrong_workspace = GemvWorkspace::try_new(2).unwrap();
    let mut output = [123.0];
    assert!(matches!(
        prepared.run(&mut wrong_workspace, &mut output),
        Err(KernelError::BufferLengthMismatch {
            buffer: BufferRole::Workspace,
            ..
        })
    ));
    assert_eq!(output.map(f32::to_bits), [123.0_f32.to_bits()]);
}

#[test]
fn finite_and_dimension_validation_is_closed() {
    assert!(matches!(
        FiniteInput::new(&[0.0, f32::NAN]),
        Err(KernelError::NonFiniteInput { index: 1, .. })
    ));
    assert!(matches!(
        Bf16Matrix::from_words(0, 1, Vec::<u16>::new()),
        Err(KernelError::InvalidDimensions {
            rows: 0,
            columns: 1
        })
    ));

    let matrix = matrix_from_i16(1, 2, &[1, 2]);
    let short_values = [1.0];
    let short = FiniteInput::new(&short_values).unwrap();
    assert!(matches!(
        PreparedGemv::new(&matrix, short, BackendRequest::Scalar),
        Err(KernelError::BufferLengthMismatch {
            buffer: BufferRole::Input,
            ..
        })
    ));
}

#[test]
fn one_workspace_can_be_explicitly_reused_across_shapes() {
    let wide = matrix_from_i16(12, 8, &[1; 12 * 8]);
    let narrow = matrix_from_i16(8, 12, &[1; 8 * 12]);
    let wide_input_values = [0.25_f32; 8];
    let narrow_input_values = [0.5_f32; 12];

    for request in [BackendRequest::Scalar, BackendRequest::Avx2] {
        if request == BackendRequest::Avx2 && !Capabilities::detected().avx2_available() {
            continue;
        }

        let wide_prepared = PreparedGemv::new(
            &wide,
            FiniteInput::new(&wide_input_values).unwrap(),
            request,
        )
        .unwrap();
        let narrow_prepared = PreparedGemv::new(
            &narrow,
            FiniteInput::new(&narrow_input_values).unwrap(),
            request,
        )
        .unwrap();
        let mut workspace = GemvWorkspace::try_new(12).unwrap();
        let capacity = workspace.capacity();
        assert_eq!(workspace.max_rows(), 12);

        let mut wide_output = [f32::NAN; 12];
        wide_prepared.run(&mut workspace, &mut wide_output).unwrap();
        assert_eq!(wide_output.map(f32::to_bits), [2.0_f32.to_bits(); 12]);

        workspace.set_rows(8).unwrap();
        assert_eq!(workspace.rows(), 8);
        assert_eq!(workspace.capacity(), capacity);
        let mut narrow_output = [f32::NAN; 8];
        narrow_prepared
            .run(&mut workspace, &mut narrow_output)
            .unwrap();
        assert_eq!(narrow_output.map(f32::to_bits), [6.0_f32.to_bits(); 8]);

        workspace.set_rows(12).unwrap();
        assert_eq!(workspace.rows(), 12);
        assert_eq!(workspace.capacity(), capacity);
        wide_output.fill(f32::NAN);
        wide_prepared.run(&mut workspace, &mut wide_output).unwrap();
        assert_eq!(wide_output.map(f32::to_bits), [2.0_f32.to_bits(); 12]);
    }
}

fn check_backend(
    rows: usize,
    columns: usize,
    words: Vec<u16>,
    input_values: &[f32],
    request: BackendRequest,
) {
    let matrix = Bf16Matrix::from_words(rows, columns, words).unwrap();
    let input = FiniteInput::new(input_values).unwrap();
    let prepared = PreparedGemv::new(&matrix, input, request).unwrap();
    let mut workspace = GemvWorkspace::try_new(rows).unwrap();
    let mut output = vec![0.0; rows];
    prepared.run(&mut workspace, &mut output).unwrap();

    let unit_roundoff = 2.0_f64.powi(-24);
    let operation_count = f64::from(u32::try_from(2 * columns).expect("test shape fits u32"));
    let gamma = operation_count * unit_roundoff / (1.0 - operation_count * unit_roundoff);
    for (row, &actual) in output.iter().enumerate() {
        let mut reference = 0.0_f64;
        let mut sum_abs = 0.0_f64;
        for (column, &input_value) in input_values.iter().enumerate() {
            let weight = f64::from(matrix.get(row, column).unwrap().to_f32());
            let value = f64::from(input_value);
            reference += weight * value;
            sum_abs += (weight * value).abs();
        }
        let error = (f64::from(actual) - reference).abs();
        let bound = gamma * sum_abs + 1.0e-7;
        assert!(
            error <= bound,
            "backend={:?} shape={rows}x{columns} row={row} error={error:e} bound={bound:e}",
            prepared.backend()
        );
    }
}

fn matrix_from_i16(rows: usize, columns: usize, values: &[i16]) -> Bf16Matrix {
    let words = values
        .iter()
        .map(|&value| Bf16::from_f32_rne(f32::from(value)).to_bits())
        .collect::<Vec<_>>();
    Bf16Matrix::from_words(rows, columns, words).unwrap()
}

fn next_u8(state: &mut u64) -> u8 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    state.to_le_bytes()[3]
}
