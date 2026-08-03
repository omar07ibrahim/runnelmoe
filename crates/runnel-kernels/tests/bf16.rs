use std::mem::size_of;

use runnel_kernels::{Bf16, Bf16Class, Bf16Matrix, KernelError};

#[test]
fn exhaustive_decode_is_exact_and_classified() {
    for raw in u16::MIN..=u16::MAX {
        let value = Bf16::from_bits(raw);
        assert_eq!(value.to_f32().to_bits(), u32::from(raw) << 16);
        let exponent = raw & 0x7f80;
        let fraction = raw & 0x007f;
        let expected = match (exponent, fraction) {
            (0, 0) => Bf16Class::Zero,
            (0, _) => Bf16Class::Subnormal,
            (0x7f80, 0) => Bf16Class::Infinity,
            (0x7f80, _) => Bf16Class::Nan,
            _ => Bf16Class::Normal,
        };
        assert_eq!(value.classify(), expected, "raw=0x{raw:04x}");
        assert_eq!(
            value.is_finite(),
            !matches!(expected, Bf16Class::Infinity | Bf16Class::Nan)
        );
        if value.is_finite() {
            assert_eq!(
                Bf16::from_f32_rne(value.to_f32()).to_bits(),
                raw,
                "finite round trip raw=0x{raw:04x}"
            );
        }
    }
}

#[test]
fn conversion_is_ties_to_even_across_signs_and_exponent_carry() {
    let cases = [
        (0x3f80_0000, 0x3f80),
        (0xbf80_0000, 0xbf80),
        (0x0000_0000, 0x0000),
        (0x8000_0000, 0x8000),
        (0x3f80_8000, 0x3f80),
        (0xbf80_8000, 0xbf80),
        (0x3f81_8000, 0x3f82),
        (0xbf81_8000, 0xbf82),
        (0x3fff_8000, 0x4000),
        (0xbfff_8000, 0xc000),
        (0x0000_0001, 0x0000),
        (0x0000_8000, 0x0000),
        (0x0001_8000, 0x0002),
        (0x0001_0000, 0x0001),
        (0x7f7f_0000, 0x7f7f),
    ];
    for (f32_bits, expected) in cases {
        assert_eq!(
            Bf16::from_f32_rne(f32::from_bits(f32_bits)).to_bits(),
            expected,
            "f32=0x{f32_bits:08x}"
        );
    }
}

#[test]
fn validated_conversion_rejects_nonfinite_and_rounding_overflow() {
    assert!(matches!(
        Bf16::try_from_f32_rne(f32::INFINITY),
        Err(KernelError::NonFiniteConversionInput { .. })
    ));
    assert_eq!(Bf16::from_f32_rne(f32::MAX).classify(), Bf16Class::Infinity);
    assert!(matches!(
        Bf16::try_from_f32_rne(f32::MAX),
        Err(KernelError::ConversionRoundedToInfinity { .. })
    ));
}

#[test]
fn matrix_loads_little_endian_and_rejects_nonfinite_words() {
    let matrix = Bf16Matrix::from_le_bytes(1, 2, &[0x80, 0x3f, 0x00, 0x80]).unwrap();
    assert_eq!(matrix.words(), &[0x3f80, 0x8000]);
    assert_eq!(
        matrix.get(0, 0).unwrap().to_f32().to_bits(),
        1.0_f32.to_bits()
    );
    assert_eq!(matrix.get(0, 1).unwrap().to_f32().to_bits(), 0x8000_0000);

    assert!(matches!(
        Bf16Matrix::from_words(1, 1, vec![0x7f80]),
        Err(KernelError::NonFiniteWeight {
            index: 0,
            bits: 0x7f80
        })
    ));
    assert!(matches!(
        Bf16Matrix::from_words(1, 1, vec![0x7fc1]),
        Err(KernelError::NonFiniteWeight { .. })
    ));
}

#[test]
fn owned_prefix_exposes_only_the_exact_logical_matrix() {
    let storage = vec![0x7f80, 0x3f80, 0xc000].into_boxed_slice();
    let allocation_address = storage.as_ptr() as usize;
    let matrix = Bf16Matrix::from_storage(1, 2, storage, 1).unwrap();
    let logical_address = matrix.words().as_ptr() as usize;

    assert_eq!(logical_address - allocation_address, size_of::<u16>());
    assert_eq!(matrix.words(), &[0x3f80, 0xc000]);
    assert_eq!(matrix.len(), 2);
    assert_eq!(
        matrix,
        Bf16Matrix::from_words(1, 2, vec![0x3f80, 0xc000]).unwrap()
    );
}

#[test]
fn owned_prefix_rejects_bad_offsets_lengths_and_nonfinite_logical_words() {
    assert!(matches!(
        Bf16Matrix::from_storage(1, 2, vec![0_u16; 2], 1),
        Err(KernelError::MatrixLengthMismatch {
            expected_words: 3,
            actual_words: 2,
        })
    ));
    assert!(matches!(
        Bf16Matrix::from_storage(1, 2, vec![0_u16; 4], 1),
        Err(KernelError::MatrixLengthMismatch {
            expected_words: 3,
            actual_words: 4,
        })
    ));
    assert!(matches!(
        Bf16Matrix::from_storage(1, 1, vec![0_u16; 1], usize::MAX),
        Err(KernelError::SizeOverflow { .. })
    ));
    assert!(matches!(
        Bf16Matrix::from_storage(1, 1, vec![0_u16, 0x7fc1], 1),
        Err(KernelError::NonFiniteWeight {
            index: 0,
            bits: 0x7fc1,
        })
    ));
}
