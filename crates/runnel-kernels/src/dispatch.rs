use crate::KernelError;

/// A caller's dispatch policy.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum BackendRequest {
    #[default]
    Auto,
    Scalar,
    Avx2,
}

/// The backend selected for one prepared operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackendKind {
    Scalar,
    Avx2,
}

/// Process capabilities, detected once and only restrictable by callers.
///
/// The fields and constructor are private: tests and applications can remove
/// AVX2, but cannot manufacture support absent from this binary and CPU.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Capabilities {
    avx2: bool,
}

impl Capabilities {
    #[must_use]
    pub fn detected() -> Self {
        Self {
            avx2: native_candidate_compiled() && cpu_has_avx2(),
        }
    }

    #[must_use]
    pub const fn without_avx2(self) -> Self {
        Self { avx2: false }
    }

    #[must_use]
    pub const fn avx2_available(self) -> bool {
        self.avx2
    }
}

/// Purely maps a request and already-bounded capability token to a backend.
///
/// # Errors
///
/// Returns [`KernelError::BackendUnavailable`] when AVX2 is forced but absent.
pub const fn select_backend(
    request: BackendRequest,
    capabilities: Capabilities,
) -> Result<BackendKind, KernelError> {
    match (request, capabilities.avx2) {
        (BackendRequest::Scalar, _) | (BackendRequest::Auto, false) => Ok(BackendKind::Scalar),
        (BackendRequest::Auto | BackendRequest::Avx2, true) => Ok(BackendKind::Avx2),
        (BackendRequest::Avx2, false) => Err(KernelError::BackendUnavailable),
    }
}

#[must_use]
pub const fn native_candidate_compiled() -> bool {
    cfg!(runnel_native_avx2)
}

#[cfg(target_arch = "x86_64")]
fn cpu_has_avx2() -> bool {
    std::arch::is_x86_feature_detected!("avx2")
}

#[cfg(not(target_arch = "x86_64"))]
const fn cpu_has_avx2() -> bool {
    false
}
