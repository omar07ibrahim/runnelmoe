use std::fmt;
use std::mem::size_of;
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::{Result, RuntimeError};

const LOGICAL_ALIGNMENT: usize = 64;
const PAGE_METADATA_BYTES: usize = 64;
const STATE_PAGE_TABLE_RESOURCE: &str = "state page table";
const STATE_KEY_PAGE_RESOURCE: &str = "state key page";
const STATE_VALUE_PAGE_RESOURCE: &str = "state value page";
const STATE_PAYLOAD_RESOURCE: &str = "state payload";
const STATE_CHARGE_RESOURCE: &str = "state logical charge";

static NEXT_STATE_ID: CheckedIdCounter = CheckedIdCounter::new(1);

/// Fully checked geometry and logical accounting for a paged K/V state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StateLayout {
    max_tokens: usize,
    page_tokens: usize,
    hidden_size: usize,
    page_count: usize,
    page_payload_bytes: usize,
    page_stride_bytes: usize,
    payload_bytes: usize,
    metadata_bytes: usize,
    charge_bytes: usize,
}

impl StateLayout {
    /// Computes page geometry without allocating or copying request data.
    pub fn new(max_tokens: usize, page_tokens: usize, hidden_size: usize) -> Result<Self> {
        if max_tokens == 0 {
            return Err(invalid_layout("max_tokens must be nonzero"));
        }
        if page_tokens == 0 {
            return Err(invalid_layout("page_tokens must be nonzero"));
        }
        if hidden_size == 0 {
            return Err(invalid_layout("hidden_size must be nonzero"));
        }
        if size_of::<StatePage>() > PAGE_METADATA_BYTES {
            return Err(invalid_layout(
                "state-page metadata exceeds its logical charge",
            ));
        }

        let page_count = max_tokens
            .checked_div(page_tokens)
            .and_then(|whole| {
                whole.checked_add(usize::from(!max_tokens.is_multiple_of(page_tokens)))
            })
            .ok_or_else(|| invalid_layout("page count overflows"))?;
        let page_elements = page_tokens
            .checked_mul(hidden_size)
            .ok_or_else(|| invalid_layout("page element count overflows"))?;
        let one_kv_bytes = page_elements
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| invalid_layout("one K/V page byte count overflows"))?;
        ensure_host_allocation(one_kv_bytes, STATE_KEY_PAGE_RESOURCE)?;
        let page_payload_bytes = one_kv_bytes
            .checked_mul(2)
            .ok_or_else(|| invalid_layout("combined K/V page byte count overflows"))?;
        let page_stride_bytes = round_up(page_payload_bytes, LOGICAL_ALIGNMENT)
            .ok_or_else(|| invalid_layout("rounded page payload overflows"))?;
        let payload_bytes = page_count
            .checked_mul(page_stride_bytes)
            .ok_or_else(|| invalid_layout("state payload byte count overflows"))?;
        let metadata_bytes = page_count
            .checked_mul(PAGE_METADATA_BYTES)
            .ok_or_else(|| invalid_layout("state metadata byte count overflows"))?;
        let charge_bytes = payload_bytes
            .checked_add(metadata_bytes)
            .ok_or_else(|| invalid_layout("state charge overflows"))?;
        ensure_host_allocation(payload_bytes, STATE_PAYLOAD_RESOURCE)?;
        ensure_host_allocation(charge_bytes, STATE_CHARGE_RESOURCE)?;
        let page_table_bytes = page_count
            .checked_mul(size_of::<StatePage>())
            .ok_or_else(|| invalid_layout("page-table allocation overflows"))?;
        ensure_host_allocation(page_table_bytes, STATE_PAGE_TABLE_RESOURCE)?;

        Ok(Self {
            max_tokens,
            page_tokens,
            hidden_size,
            page_count,
            page_payload_bytes,
            page_stride_bytes,
            payload_bytes,
            metadata_bytes,
            charge_bytes,
        })
    }

    #[must_use]
    pub fn max_tokens(self) -> usize {
        self.max_tokens
    }

    #[must_use]
    pub fn page_tokens(self) -> usize {
        self.page_tokens
    }

    #[must_use]
    pub fn hidden_size(self) -> usize {
        self.hidden_size
    }

    #[must_use]
    pub fn page_count(self) -> usize {
        self.page_count
    }

    #[must_use]
    pub fn page_payload_bytes(self) -> usize {
        self.page_payload_bytes
    }

    #[must_use]
    pub fn page_stride_bytes(self) -> usize {
        self.page_stride_bytes
    }

    #[must_use]
    pub fn payload_bytes(self) -> usize {
        self.payload_bytes
    }

    #[must_use]
    pub fn metadata_bytes(self) -> usize {
        self.metadata_bytes
    }

    #[must_use]
    pub fn charge_bytes(self) -> usize {
        self.charge_bytes
    }

    fn page_elements(self) -> usize {
        // Construction proved this product fits.
        self.page_tokens * self.hidden_size
    }

    fn one_kv_page_bytes(self) -> usize {
        // Construction proved both products fit.
        self.page_elements() * size_of::<f32>()
    }
}

/// A nonzero identity for adapter state.
///
/// Runtime-allocated `SequenceState` values are process-unique. External
/// decoder adapters that construct work identities are responsible for the
/// same no-reuse property within each model instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StateId(NonZeroU64);

impl StateId {
    pub(crate) const fn from_nonzero(value: NonZeroU64) -> Self {
        Self(value)
    }

    #[must_use]
    pub fn get(self) -> u64 {
        self.0.get()
    }
}

/// Token-major, eagerly allocated K/V state.
///
/// `new` preserves the compatibility constructor as an unbound, allocation-free
/// value. A model-owned constructor or the first model operation binds it to
/// checked geometry. The type intentionally does not implement `Clone`.
pub struct SequenceState {
    model_instance_id: Option<u64>,
    state_id: Option<StateId>,
    revision: u64,
    layout: Option<StateLayout>,
    pages: Vec<StatePage>,
    len: usize,
}

impl SequenceState {
    #[must_use]
    pub fn new() -> Self {
        Self {
            model_instance_id: None,
            state_id: None,
            revision: 0,
            layout: None,
            pages: Vec::new(),
            len: 0,
        }
    }

    /// Eagerly allocates every page before publishing a bound state identity.
    pub(crate) fn try_new(layout: StateLayout, model_instance_id: u64) -> Result<Self> {
        let mut probe = NoopAllocationProbe;
        Self::try_new_with_probe(layout, model_instance_id, &mut probe)
    }

    pub(crate) fn try_new_with_probe<P: AllocationProbe>(
        layout: StateLayout,
        model_instance_id: u64,
        probe: &mut P,
    ) -> Result<Self> {
        Self::try_new_with_counter(layout, model_instance_id, probe, &NEXT_STATE_ID)
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[must_use]
    pub fn state_id(&self) -> Option<StateId> {
        self.state_id
    }

    #[must_use]
    pub fn revision(&self) -> u64 {
        self.revision
    }

    #[must_use]
    pub fn model_instance_id(&self) -> Option<u64> {
        self.model_instance_id
    }

    #[must_use]
    pub fn layout(&self) -> Option<StateLayout> {
        self.layout
    }

    #[must_use]
    pub fn context_limit(&self) -> Option<usize> {
        self.layout.map(StateLayout::max_tokens)
    }

    #[must_use]
    pub fn page_tokens(&self) -> Option<usize> {
        self.layout.map(StateLayout::page_tokens)
    }

    #[must_use]
    pub fn accounted_payload_bytes(&self) -> usize {
        self.layout.map_or(0, StateLayout::payload_bytes)
    }

    #[must_use]
    pub fn accounted_charge_bytes(&self) -> usize {
        self.layout.map_or(0, StateLayout::charge_bytes)
    }

    #[must_use]
    pub fn history(&self) -> KvHistory<'_> {
        KvHistory { state: self }
    }

    /// Captures logical and private preallocated state for rollback tests.
    #[cfg(test)]
    pub(crate) fn test_fingerprint(&self) -> TestStateFingerprint {
        let mut page_bits = Vec::new();
        let total_elements = self
            .pages
            .iter()
            .map(|page| page.keys.len() + page.values.len())
            .sum();
        page_bits.reserve_exact(total_elements);
        let allocations = self
            .pages
            .iter()
            .map(|page| TestPageAllocation {
                key_pointer: page.keys.as_ptr() as usize,
                key_len: page.keys.len(),
                key_capacity: page.keys.capacity(),
                value_pointer: page.values.as_ptr() as usize,
                value_len: page.values.len(),
                value_capacity: page.values.capacity(),
            })
            .collect();
        for page in &self.pages {
            page_bits.extend(page.keys.iter().map(|value| value.to_bits()));
            page_bits.extend(page.values.iter().map(|value| value.to_bits()));
        }
        TestStateFingerprint {
            model_instance_id: self.model_instance_id,
            state_id: self.state_id,
            revision: self.revision,
            layout: self.layout,
            len: self.len,
            page_bits,
            allocations,
        }
    }

    pub(crate) fn validate_append<'state, 'pending>(
        &'state mut self,
        expected_state_id: StateId,
        expected_revision: u64,
        position: usize,
        key: &'pending [f32],
        value: &'pending [f32],
    ) -> Result<StateAppendPermit<'state, 'pending>> {
        let layout = self.layout.ok_or(RuntimeError::InvalidState(
            "state must be bound before append",
        ))?;
        if self.state_id != Some(expected_state_id) {
            return Err(RuntimeError::StateMismatch);
        }
        if self.revision != expected_revision {
            return Err(RuntimeError::StateRevisionMismatch {
                expected: expected_revision,
                actual: self.revision,
            });
        }
        if position != self.len {
            return Err(RuntimeError::InvalidState(
                "position must equal the logical state length",
            ));
        }
        if position >= layout.max_tokens {
            return Err(RuntimeError::ContextLimit {
                limit: layout.max_tokens,
            });
        }
        if key.len() != layout.hidden_size || value.len() != layout.hidden_size {
            return Err(RuntimeError::InvalidState(
                "key and value widths must match hidden_size",
            ));
        }
        if key.iter().chain(value).any(|item| !item.is_finite()) {
            return Err(RuntimeError::NonFinite("state append"));
        }
        let next_revision = self
            .revision
            .checked_add(1)
            .ok_or(RuntimeError::StateRevisionExhausted)?;
        let page_index = position / layout.page_tokens;
        let token_in_page = position % layout.page_tokens;
        let start = token_in_page
            .checked_mul(layout.hidden_size)
            .ok_or(RuntimeError::InvalidState("state offset overflows"))?;
        let end = start
            .checked_add(layout.hidden_size)
            .ok_or(RuntimeError::InvalidState("state range overflows"))?;
        let page = self
            .pages
            .get(page_index)
            .ok_or(RuntimeError::InvalidState("state page is missing"))?;
        if end > page.keys.len() || end > page.values.len() {
            return Err(RuntimeError::InvalidState(
                "state page geometry is inconsistent",
            ));
        }

        Ok(StateAppendPermit {
            state: self,
            key,
            value,
            page_index,
            start,
            end,
            position,
            next_revision,
        })
    }

    fn try_new_with_counter<P: AllocationProbe>(
        layout: StateLayout,
        model_instance_id: u64,
        probe: &mut P,
        ids: &CheckedIdCounter,
    ) -> Result<Self> {
        if model_instance_id == 0 {
            return Err(RuntimeError::InvalidState(
                "model instance identity must be nonzero",
            ));
        }

        let page_table_bytes = layout
            .page_count
            .checked_mul(size_of::<StatePage>())
            .ok_or_else(|| invalid_layout("page-table allocation overflows"))?;
        probe.before_reserve(AllocationSite::PageTable, 0, page_table_bytes)?;
        let mut pages = Vec::new();
        pages
            .try_reserve_exact(layout.page_count)
            .map_err(|error| {
                map_reserve_error(error, STATE_PAGE_TABLE_RESOURCE, page_table_bytes)
            })?;

        for page_index in 0..layout.page_count {
            let keys = allocate_page_buffer(
                layout,
                probe,
                AllocationSite::KeyPage,
                page_index,
                STATE_KEY_PAGE_RESOURCE,
            )?;
            let values = allocate_page_buffer(
                layout,
                probe,
                AllocationSite::ValuePage,
                page_index,
                STATE_VALUE_PAGE_RESOURCE,
            )?;
            pages.push(StatePage { keys, values });
        }

        // Identity assignment is deliberately last: every fallible allocation
        // has completed, so a failed construction cannot consume or publish it.
        let state_id = ids.next()?;
        Ok(Self {
            model_instance_id: Some(model_instance_id),
            state_id: Some(state_id),
            revision: 0,
            layout: Some(layout),
            pages,
            len: 0,
        })
    }
}

#[cfg(test)]
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct TestStateFingerprint {
    model_instance_id: Option<u64>,
    state_id: Option<StateId>,
    revision: u64,
    layout: Option<StateLayout>,
    len: usize,
    page_bits: Vec<u32>,
    allocations: Vec<TestPageAllocation>,
}

#[cfg(test)]
#[derive(Debug, PartialEq, Eq)]
struct TestPageAllocation {
    key_pointer: usize,
    key_len: usize,
    key_capacity: usize,
    value_pointer: usize,
    value_len: usize,
    value_capacity: usize,
}

impl Default for SequenceState {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for SequenceState {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SequenceState")
            .field("model_instance_id", &self.model_instance_id)
            .field("state_id", &self.state_id)
            .field("revision", &self.revision)
            .field("layout", &self.layout)
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}

/// Read-only logical K/V history. Preallocated tail capacity is never exposed.
#[derive(Clone, Copy)]
pub struct KvHistory<'a> {
    state: &'a SequenceState,
}

impl KvHistory<'_> {
    #[must_use]
    pub fn len(self) -> usize {
        self.state.len
    }

    #[must_use]
    pub fn is_empty(self) -> bool {
        self.state.len == 0
    }

    #[must_use]
    pub fn hidden_size(self) -> Option<usize> {
        self.state.layout.map(StateLayout::hidden_size)
    }
}

impl<'a> KvHistory<'a> {
    #[must_use]
    pub fn key_at(self, position: usize) -> Option<&'a [f32]> {
        self.logical_slice(position, PagePart::Key)
    }

    #[must_use]
    pub fn value_at(self, position: usize) -> Option<&'a [f32]> {
        self.logical_slice(position, PagePart::Value)
    }

    fn logical_slice(self, position: usize, part: PagePart) -> Option<&'a [f32]> {
        if position >= self.state.len {
            return None;
        }
        let layout = self.state.layout?;
        let page_index = position / layout.page_tokens;
        let token_in_page = position % layout.page_tokens;
        let start = token_in_page.checked_mul(layout.hidden_size)?;
        let end = start.checked_add(layout.hidden_size)?;
        let page = self.state.pages.get(page_index)?;
        match part {
            PagePart::Key => page.keys.get(start..end),
            PagePart::Value => page.values.get(start..end),
        }
    }
}

#[derive(Debug)]
pub(crate) struct StateAppendPermit<'state, 'pending> {
    state: &'state mut SequenceState,
    key: &'pending [f32],
    value: &'pending [f32],
    page_index: usize,
    start: usize,
    end: usize,
    position: usize,
    next_revision: u64,
}

impl StateAppendPermit<'_, '_> {
    /// Applies the exact K/V slices checked when this permit was created.
    ///
    /// The permit borrows the exact immutable pending slices checked by
    /// `validate_append`, so apply cannot substitute different bytes or fail a
    /// length check. A later outer transaction permit must therefore borrow a
    /// separately stored pending commit rather than form a self-reference.
    pub(crate) fn apply(self) {
        let page = &mut self.state.pages[self.page_index];
        page.keys[self.start..self.end].copy_from_slice(self.key);
        page.values[self.start..self.end].copy_from_slice(self.value);
        self.state.len = self.position + 1;
        self.state.revision = self.next_revision;
    }
}

struct StatePage {
    keys: Vec<f32>,
    values: Vec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PagePart {
    Key,
    Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AllocationSite {
    PageTable,
    KeyPage,
    ValuePage,
}

pub(crate) trait AllocationProbe {
    fn before_reserve(
        &mut self,
        site: AllocationSite,
        page_index: usize,
        bytes: usize,
    ) -> Result<()>;
}

struct NoopAllocationProbe;

impl AllocationProbe for NoopAllocationProbe {
    fn before_reserve(
        &mut self,
        _site: AllocationSite,
        _page_index: usize,
        _bytes: usize,
    ) -> Result<()> {
        Ok(())
    }
}

struct CheckedIdCounter {
    next: AtomicU64,
}

impl CheckedIdCounter {
    const fn new(first: u64) -> Self {
        Self {
            next: AtomicU64::new(first),
        }
    }

    fn next(&self) -> Result<StateId> {
        let mut candidate = self.next.load(Ordering::Relaxed);
        loop {
            let identity = NonZeroU64::new(candidate)
                .map(StateId)
                .ok_or(RuntimeError::StateIdentityExhausted)?;
            let successor = candidate.checked_add(1).unwrap_or(0);
            match self.next.compare_exchange_weak(
                candidate,
                successor,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Ok(identity),
                Err(observed) => candidate = observed,
            }
        }
    }
}

fn allocate_page_buffer<P: AllocationProbe>(
    layout: StateLayout,
    probe: &mut P,
    site: AllocationSite,
    page_index: usize,
    resource: &'static str,
) -> Result<Vec<f32>> {
    let bytes = layout.one_kv_page_bytes();
    probe.before_reserve(site, page_index, bytes)?;
    let mut buffer = Vec::new();
    buffer
        .try_reserve_exact(layout.page_elements())
        .map_err(|error| map_reserve_error(error, resource, bytes))?;
    buffer.resize(layout.page_elements(), 0.0);
    Ok(buffer)
}

fn ensure_host_allocation(bytes: usize, resource: &'static str) -> Result<()> {
    if bytes > isize::MAX as usize {
        return Err(RuntimeError::ResourceExhausted { resource, bytes });
    }
    Ok(())
}

fn round_up(value: usize, alignment: usize) -> Option<usize> {
    let remainder = value % alignment;
    if remainder == 0 {
        Some(value)
    } else {
        value.checked_add(alignment - remainder)
    }
}

fn invalid_layout(reason: &'static str) -> RuntimeError {
    RuntimeError::InvalidStateLayout(reason)
}

fn map_reserve_error(
    _error: std::collections::TryReserveError,
    resource: &'static str,
    bytes: usize,
) -> RuntimeError {
    RuntimeError::ResourceExhausted { resource, bytes }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FailNth {
        fail_at: usize,
        calls: usize,
        sites: Vec<(AllocationSite, usize)>,
    }

    impl FailNth {
        fn new(fail_at: usize) -> Self {
            Self {
                fail_at,
                calls: 0,
                sites: Vec::new(),
            }
        }
    }

    impl AllocationProbe for FailNth {
        fn before_reserve(
            &mut self,
            site: AllocationSite,
            page_index: usize,
            bytes: usize,
        ) -> Result<()> {
            self.sites.push((site, page_index));
            let call = self.calls;
            self.calls += 1;
            if call == self.fail_at {
                let resource = match site {
                    AllocationSite::PageTable => STATE_PAGE_TABLE_RESOURCE,
                    AllocationSite::KeyPage => STATE_KEY_PAGE_RESOURCE,
                    AllocationSite::ValuePage => STATE_VALUE_PAGE_RESOURCE,
                };
                return Err(RuntimeError::ResourceExhausted { resource, bytes });
            }
            Ok(())
        }
    }

    fn v3_layout() -> StateLayout {
        StateLayout::new(1_024, 16, 8).unwrap()
    }

    fn append_pattern(state: &mut SequenceState, position: usize) {
        let key = std::array::from_fn::<_, 8, _>(|index| {
            (position as f32 + index as f32 + 1.0) / 2_048.0
        });
        let value =
            std::array::from_fn::<_, 8, _>(|index| (position as f32 - index as f32) / 1_024.0);
        let state_id = state.state_id().unwrap();
        let revision = state.revision();
        state
            .validate_append(state_id, revision, position, &key, &value)
            .unwrap()
            .apply();
    }

    fn allocation_signature(state: &SequenceState) -> Vec<(usize, usize, usize, usize)> {
        let mut signature = Vec::with_capacity(state.pages.len() + 1);
        signature.push((state.pages.as_ptr() as usize, state.pages.capacity(), 0, 0));
        signature.extend(state.pages.iter().map(|page| {
            (
                page.keys.as_ptr() as usize,
                page.keys.capacity(),
                page.values.as_ptr() as usize,
                page.values.capacity(),
            )
        }));
        signature
    }

    fn private_digest(state: &SequenceState) -> u64 {
        let mut digest = 0xcbf2_9ce4_8422_2325_u64;
        for byte in state
            .pages
            .iter()
            .flat_map(|page| page.keys.iter().chain(&page.values))
            .flat_map(|value| value.to_bits().to_le_bytes())
        {
            digest ^= u64::from(byte);
            digest = digest.wrapping_mul(0x0000_0100_0000_01b3);
        }
        digest
    }

    #[test]
    fn ps01_v3_geometry_and_accounting_are_exact() {
        let layout = v3_layout();
        assert_eq!(layout.page_count(), 64);
        assert_eq!(layout.page_payload_bytes(), 1_024);
        assert_eq!(layout.page_stride_bytes(), 1_024);
        assert_eq!(layout.payload_bytes(), 65_536);
        assert_eq!(layout.metadata_bytes(), 4_096);
        assert_eq!(layout.charge_bytes(), 69_632);

        let rounded = StateLayout::new(1, 1, 1).unwrap();
        assert_eq!(rounded.page_payload_bytes(), 8);
        assert_eq!(rounded.page_stride_bytes(), 64);
        assert_eq!(rounded.charge_bytes(), 128);
    }

    #[test]
    fn ps02_all_pages_are_eager_and_allocation_shape_never_changes() {
        let layout = v3_layout();
        let mut state = SequenceState::try_new(layout, 7).unwrap();
        assert_eq!(state.pages.len(), 64);
        assert!(state.pages.iter().all(|page| {
            page.keys.len() == 128
                && page.values.len() == 128
                && page.keys.capacity() >= 128
                && page.values.capacity() >= 128
        }));
        let before = allocation_signature(&state);
        let payload = state.accounted_payload_bytes();
        let charge = state.accounted_charge_bytes();
        for position in 0..1_024 {
            append_pattern(&mut state, position);
        }
        assert_eq!(allocation_signature(&state), before);
        assert_eq!(state.accounted_payload_bytes(), payload);
        assert_eq!(state.accounted_charge_bytes(), charge);
    }

    #[test]
    fn ps03_boundaries_map_to_expected_pages_and_hide_partial_tails() {
        let mut state = SequenceState::try_new(v3_layout(), 8).unwrap();
        for position in 0..1_024 {
            append_pattern(&mut state, position);
        }
        let history = state.history();
        for one_based in [15, 16, 17, 255, 256, 257, 1_023, 1_024] {
            let position = one_based - 1;
            let key = history.key_at(position).unwrap();
            let expected = (position as f32 + 1.0) / 2_048.0;
            assert_eq!(key[0], expected, "position {one_based}");
            let page_index = position / 16;
            let scalar_offset = (position % 16) * 8;
            assert_eq!(
                state.pages[page_index].keys[scalar_offset], expected,
                "physical position {one_based}"
            );
        }
        assert!(history.key_at(1_024).is_none());
        assert!(history.value_at(1_024).is_none());

        let mut tail = SequenceState::try_new(StateLayout::new(17, 16, 8).unwrap(), 9).unwrap();
        for position in 0..17 {
            append_pattern(&mut tail, position);
        }
        assert_eq!(tail.history().len(), 17);
        assert!(tail.history().key_at(17).is_none());
        assert!(tail.history().value_at(31).is_none());
    }

    #[test]
    fn ps04_context_failure_preserves_every_private_state_field() {
        let mut state = SequenceState::try_new(v3_layout(), 10).unwrap();
        for position in 0..1_024 {
            append_pattern(&mut state, position);
        }
        let identity = state.state_id();
        let revision = state.revision();
        let len = state.len();
        let digest = private_digest(&state);
        let key = [0.0; 8];
        let value = [0.0; 8];
        let error = state
            .validate_append(identity.unwrap(), revision, 1_024, &key, &value)
            .unwrap_err();
        assert_eq!(error, RuntimeError::ContextLimit { limit: 1_024 });
        assert_eq!(state.state_id(), identity);
        assert_eq!(state.revision(), revision);
        assert_eq!(state.len(), len);
        assert_eq!(private_digest(&state), digest);
    }

    #[test]
    fn append_permit_commits_only_the_slices_it_validated() {
        let mut state = SequenceState::try_new(StateLayout::new(1, 1, 2).unwrap(), 10).unwrap();
        let key = [1.0, 2.0];
        let value = [3.0, 4.0];
        let unrelated_key = [9.0, 9.0];
        let unrelated_value = [8.0, 8.0];
        let state_id = state.state_id().unwrap();
        let permit = state.validate_append(state_id, 0, 0, &key, &value).unwrap();
        assert_eq!(unrelated_key, [9.0, 9.0]);
        assert_eq!(unrelated_value, [8.0, 8.0]);
        permit.apply();
        assert_eq!(state.history().key_at(0).unwrap(), key);
        assert_eq!(state.history().value_at(0).unwrap(), value);
    }

    #[test]
    fn ps05_identities_are_nonzero_unique_foreign_safe_and_never_wrap() {
        let first = SequenceState::try_new(StateLayout::new(1, 1, 1).unwrap(), 11).unwrap();
        let mut second = SequenceState::try_new(StateLayout::new(1, 1, 1).unwrap(), 11).unwrap();
        assert_ne!(first.state_id(), second.state_id());
        assert_ne!(first.state_id().unwrap().get(), 0);

        let key = [1.0];
        let value = [2.0];
        let error = second
            .validate_append(first.state_id().unwrap(), 0, 0, &key, &value)
            .unwrap_err();
        assert_eq!(error, RuntimeError::StateMismatch);
        assert_eq!(second.len(), 0);
        assert_eq!(second.revision(), 0);

        let second_id = second.state_id().unwrap();
        second.revision = u64::MAX;
        let error = second
            .validate_append(second_id, u64::MAX, 0, &key, &value)
            .unwrap_err();
        assert_eq!(error, RuntimeError::StateRevisionExhausted);
        assert_eq!(second.len(), 0);
        assert_eq!(second.revision(), u64::MAX);

        let counter = CheckedIdCounter::new(u64::MAX);
        assert_eq!(counter.next().unwrap().get(), u64::MAX);
        assert_eq!(
            counter.next().unwrap_err(),
            RuntimeError::StateIdentityExhausted
        );
        assert_eq!(
            counter.next().unwrap_err(),
            RuntimeError::StateIdentityExhausted
        );
    }

    #[test]
    fn al01_rejects_zero_overflow_and_unrepresentable_host_allocations() {
        assert!(StateLayout::new(0, 16, 8).is_err());
        assert!(StateLayout::new(1, 0, 8).is_err());
        assert!(StateLayout::new(1, 16, 0).is_err());
        assert!(StateLayout::new(usize::MAX, usize::MAX, usize::MAX).is_err());
        let aggregate_too_large = (isize::MAX as usize / 16) + 1;
        assert!(matches!(
            StateLayout::new(2, 1, aggregate_too_large),
            Err(RuntimeError::ResourceExhausted {
                resource: STATE_PAYLOAD_RESOURCE,
                ..
            })
        ));
        assert_eq!(
            ensure_host_allocation(usize::MAX, "test").unwrap_err(),
            RuntimeError::ResourceExhausted {
                resource: "test",
                bytes: usize::MAX,
            }
        );
    }

    #[test]
    fn al02_every_reservation_site_rolls_back_before_identity_publication() {
        let layout = StateLayout::new(17, 16, 8).unwrap();
        let reservation_count = 1 + 2 * layout.page_count();
        for fail_at in 0..reservation_count {
            let ids = CheckedIdCounter::new(41);
            let mut probe = FailNth::new(fail_at);
            let error =
                SequenceState::try_new_with_counter(layout, 12, &mut probe, &ids).unwrap_err();
            assert!(matches!(error, RuntimeError::ResourceExhausted { .. }));
            assert_eq!(ids.next().unwrap().get(), 41, "failure site {fail_at}");
        }

        let ids = CheckedIdCounter::new(51);
        let mut probe = FailNth::new(usize::MAX);
        let state = SequenceState::try_new_with_counter(layout, 13, &mut probe, &ids).unwrap();
        assert_eq!(state.state_id().unwrap().get(), 51);
        assert_eq!(
            probe.sites,
            [
                (AllocationSite::PageTable, 0),
                (AllocationSite::KeyPage, 0),
                (AllocationSite::ValuePage, 0),
                (AllocationSite::KeyPage, 1),
                (AllocationSite::ValuePage, 1),
            ]
        );
    }

    #[test]
    fn al03_try_reserve_errors_map_to_resource_exhausted() {
        let error = Vec::<u8>::new().try_reserve_exact(usize::MAX).unwrap_err();
        assert_eq!(
            map_reserve_error(error, "test reserve", 123),
            RuntimeError::ResourceExhausted {
                resource: "test reserve",
                bytes: 123,
            }
        );
    }
}
