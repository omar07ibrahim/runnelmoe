use crate::StoreError;

const DEFAULT_CAS_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const DEFAULT_FILESYSTEM_RESERVE: u64 = 2 * 1024 * 1024 * 1024;

/// Persistent-byte ceiling and free-filesystem reserve for one CAS.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DiskBudget {
    pub max_bytes: u64,
    pub reserve_bytes: u64,
}

impl DiskBudget {
    #[must_use]
    pub const fn new(max_bytes: u64, reserve_bytes: u64) -> Self {
        Self {
            max_bytes,
            reserve_bytes,
        }
    }

    pub(crate) fn validate(self) -> Result<Self, StoreError> {
        if self.max_bytes == 0 {
            return Err(StoreError::invalid_config(
                "disk_budget.max_bytes",
                "must be nonzero",
            ));
        }
        Ok(self)
    }

    pub(crate) fn preflight(
        self,
        usage: DiskUsage,
        additional_bytes: u64,
    ) -> Result<(), StoreError> {
        let required = usage
            .checked_total_bytes()?
            .checked_add(additional_bytes)
            .ok_or(StoreError::ArithmeticOverflow {
                context: "adding CAS disk usage",
            })?;
        if required > self.max_bytes {
            return Err(StoreError::BudgetExceeded {
                required,
                limit: self.max_bytes,
            });
        }

        let remaining = usage.available_bytes.saturating_sub(self.reserve_bytes);
        if additional_bytes > remaining {
            return Err(StoreError::FilesystemReserve {
                required: additional_bytes,
                available: usage.available_bytes,
                reserve: self.reserve_bytes,
            });
        }
        Ok(())
    }
}

impl Default for DiskBudget {
    fn default() -> Self {
        Self::new(DEFAULT_CAS_BYTES, DEFAULT_FILESYSTEM_RESERVE)
    }
}

/// Descriptor-derived charged bytes at one transaction boundary.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DiskUsage {
    pub final_bytes: u64,
    pub staging_bytes: u64,
    pub available_bytes: u64,
    pub allocation_unit: u64,
}

impl DiskUsage {
    #[must_use]
    pub const fn total_bytes(self) -> u64 {
        self.final_bytes.saturating_add(self.staging_bytes)
    }

    pub(crate) fn checked_total_bytes(self) -> Result<u64, StoreError> {
        self.final_bytes
            .checked_add(self.staging_bytes)
            .ok_or(StoreError::ArithmeticOverflow {
                context: "summing CAS disk usage",
            })
    }

    pub(crate) fn round_up(self, bytes: u64) -> Result<u64, StoreError> {
        let unit = self.allocation_unit.max(1);
        let remainder = bytes % unit;
        if remainder == 0 {
            return Ok(bytes);
        }
        bytes
            .checked_add(unit - remainder)
            .ok_or(StoreError::ArithmeticOverflow {
                context: "rounding planned disk allocation",
            })
    }
}

#[cfg(test)]
mod tests {
    use super::{DiskBudget, DiskUsage};
    use crate::StoreError;

    #[test]
    fn budget_boundary_is_exact() {
        let budget = DiskBudget::new(10_000, 2_000);
        let usage = DiskUsage {
            final_bytes: 4_000,
            staging_bytes: 1_000,
            available_bytes: 7_000,
            allocation_unit: 1,
        };
        assert!(budget.preflight(usage, 5_000).is_ok());
        assert_eq!(
            budget.preflight(usage, 5_001).unwrap_err(),
            StoreError::BudgetExceeded {
                required: 10_001,
                limit: 10_000,
            }
        );
    }

    #[test]
    fn filesystem_reserve_boundary_is_exact() {
        let budget = DiskBudget::new(100_000, 2_000);
        let usage = DiskUsage {
            available_bytes: 7_000,
            ..DiskUsage::default()
        };
        assert!(budget.preflight(usage, 5_000).is_ok());
        assert_eq!(
            budget.preflight(usage, 5_001).unwrap_err(),
            StoreError::FilesystemReserve {
                required: 5_001,
                available: 7_000,
                reserve: 2_000,
            }
        );
    }

    #[test]
    fn planned_allocation_rounds_up() {
        let usage = DiskUsage {
            allocation_unit: 4_096,
            ..DiskUsage::default()
        };
        assert_eq!(usage.round_up(1).unwrap(), 4_096);
        assert_eq!(usage.round_up(4_096).unwrap(), 4_096);
        assert_eq!(usage.round_up(4_097).unwrap(), 8_192);
    }
}
