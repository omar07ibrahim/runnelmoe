use crate::KernelError;

/// One raw IEEE-style BF16 storage word.
///
/// This type may represent finite and nonfinite encodings. Matrices accept only
/// finite words, keeping exhaustive decoding separate from artifact validation.
#[derive(Clone, Copy, Default, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct Bf16(u16);

/// The encoding class of a BF16 word.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Bf16Class {
    Zero,
    Subnormal,
    Normal,
    Infinity,
    Nan,
}

impl Bf16 {
    #[must_use]
    pub const fn from_bits(bits: u16) -> Self {
        Self(bits)
    }

    #[must_use]
    pub const fn to_bits(self) -> u16 {
        self.0
    }

    /// Widens a BF16 word exactly by placing its bits in the high half of f32.
    #[must_use]
    pub const fn to_f32(self) -> f32 {
        f32::from_bits((self.0 as u32) << 16)
    }

    #[must_use]
    pub const fn classify(self) -> Bf16Class {
        let exponent = self.0 & 0x7f80;
        let fraction = self.0 & 0x007f;
        match (exponent, fraction) {
            (0, 0) => Bf16Class::Zero,
            (0, _) => Bf16Class::Subnormal,
            (0x7f80, 0) => Bf16Class::Infinity,
            (0x7f80, _) => Bf16Class::Nan,
            _ => Bf16Class::Normal,
        }
    }

    #[must_use]
    pub const fn is_finite(self) -> bool {
        !matches!(self.classify(), Bf16Class::Infinity | Bf16Class::Nan)
    }

    /// Converts f32 to BF16 with round-to-nearest, ties-to-even.
    ///
    /// This raw conversion retains a nonfinite class. Use [`Self::try_from_f32_rne`]
    /// when constructing validated storage.
    #[must_use]
    pub const fn from_f32_rne(value: f32) -> Self {
        let bits = value.to_bits();
        let exponent = bits & 0x7f80_0000;
        let fraction = bits & 0x007f_ffff;

        if exponent == 0x7f80_0000 && fraction != 0 {
            let mut word = (bits >> 16) as u16;
            if word.trailing_zeros() >= 7 {
                word |= 1;
            }
            return Self(word);
        }

        let retained_lsb = (bits >> 16) & 1;
        let rounded = bits.wrapping_add(0x7fff + retained_lsb);
        Self((rounded >> 16) as u16)
    }

    /// Converts a finite f32 to a finite BF16 word, rejecting range overflow.
    ///
    /// # Errors
    ///
    /// Returns an error if `value` is nonfinite or rounds to BF16 infinity.
    pub fn try_from_f32_rne(value: f32) -> Result<Self, KernelError> {
        if !value.is_finite() {
            return Err(KernelError::NonFiniteConversionInput {
                bits: value.to_bits(),
            });
        }
        let converted = Self::from_f32_rne(value);
        if !converted.is_finite() {
            return Err(KernelError::ConversionRoundedToInfinity {
                bits: value.to_bits(),
            });
        }
        Ok(converted)
    }
}

impl core::fmt::Debug for Bf16 {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(formatter, "Bf16(0x{:04x})", self.0)
    }
}

impl From<Bf16> for f32 {
    fn from(value: Bf16) -> Self {
        value.to_f32()
    }
}
