use std::{collections::BTreeMap, fmt};

use crate::{Result, RuntimeError};

#[derive(PartialEq)]
pub struct Tensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}

impl fmt::Debug for Tensor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Tensor")
            .field("shape", &self.shape)
            .field("data", &"<redacted>")
            .finish()
    }
}

impl Tensor {
    pub fn new(role: &str, shape: Vec<usize>, data: Vec<f32>) -> Result<Self> {
        if shape.is_empty() {
            return Err(invalid(role, "rank must be positive"));
        }
        let elements = shape.iter().try_fold(1_usize, |total, dimension| {
            if *dimension == 0 {
                None
            } else {
                total.checked_mul(*dimension)
            }
        });
        if elements != Some(data.len()) {
            return Err(invalid(
                role,
                &format!(
                    "shape {:?} has {:?} elements but data has {}",
                    shape,
                    elements,
                    data.len()
                ),
            ));
        }
        if data.iter().any(|value| !value.is_finite()) {
            return Err(invalid(role, "values must be finite"));
        }
        Ok(Self { shape, data })
    }

    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    #[must_use]
    pub fn data(&self) -> &[f32] {
        &self.data
    }
}

pub type TensorCatalog = BTreeMap<String, Tensor>;

fn invalid(role: &str, reason: &str) -> RuntimeError {
    RuntimeError::InvalidTensor {
        role: role.into(),
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_reports_geometry_without_tensor_payload() {
        let tensor = Tensor::new("secret", vec![2], vec![1_234.5, -6_789.0]).unwrap();
        let debug = format!("{tensor:?}");
        assert!(debug.contains("shape: [2]"));
        assert!(debug.contains("<redacted>"));
        assert!(!debug.contains("1234.5"));
        assert!(!debug.contains("6789"));
    }
}
