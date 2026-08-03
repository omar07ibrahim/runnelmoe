use std::collections::BTreeMap;

use crate::{Result, RuntimeError};

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    shape: Box<[usize]>,
    data: Box<[f32]>,
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
        Ok(Self {
            shape: shape.into_boxed_slice(),
            data: data.into_boxed_slice(),
        })
    }

    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    #[must_use]
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub(crate) fn row(&self, row: usize) -> &[f32] {
        let width = self.shape[1];
        &self.data[row * width..(row + 1) * width]
    }
}

pub type TensorCatalog = BTreeMap<String, Tensor>;

fn invalid(role: &str, reason: &str) -> RuntimeError {
    RuntimeError::InvalidTensor {
        role: role.into(),
        reason: reason.into(),
    }
}
