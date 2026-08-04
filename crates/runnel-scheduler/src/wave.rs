use std::{fmt, mem::size_of};

use runnel_runtime::AdapterWorkIdentity;

use crate::id::{EngineTransactionId, RequestId, SlotKey};
use crate::ring::ServiceReservation;

const MAX_BATCH_WIDTH: usize = 4_096;
const MAX_TASKS_PER_WAVE: usize = 262_144;
const MAX_TASKS_PER_TOKEN: usize = u16::MAX as usize + 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WaveScratchError {
    SizeOverflow {
        resource: &'static str,
    },
    AllocationFailure {
        resource: &'static str,
        bytes: usize,
    },
    CapacityInvariant(&'static str),
}

pub(crate) struct WaveSelection<P> {
    pub(crate) slot: SlotKey,
    pub(crate) request_id: RequestId,
    pub(crate) transaction_id: EngineTransactionId,
    pub(crate) adapter_identity: AdapterWorkIdentity,
    pub(crate) reservation: Option<ServiceReservation>,
    pub(crate) prepared: Option<P>,
    pub(crate) task_count: usize,
}

pub(crate) struct TaskEnvelope<T> {
    pub(crate) selection_index: usize,
    pub(crate) slot: SlotKey,
    pub(crate) request_id: RequestId,
    pub(crate) transaction_id: EngineTransactionId,
    pub(crate) adapter_identity: AdapterWorkIdentity,
    pub(crate) router_rank: u16,
    pub(crate) expert_id: u16,
    pub(crate) scatter_index: usize,
    pub(crate) task: T,
}

impl<T> TaskEnvelope<T> {
    /// Separates the adapter task from its immutable scheduler completion key.
    pub(crate) fn into_parts(self) -> (TaskCompletionKey, T) {
        (
            TaskCompletionKey {
                selection_index: self.selection_index,
                slot: self.slot,
                request_id: self.request_id,
                transaction_id: self.transaction_id,
                adapter_identity: self.adapter_identity,
                router_rank: self.router_rank,
                expert_id: self.expert_id,
                scatter_index: self.scatter_index,
            },
            self.task,
        )
    }
}

/// Scheduler identity retained while an adapter task is executed by value.
///
/// The key is intentionally not `Clone` or `Copy`: one task execution can
/// publish at most one completion through the safe wave API.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct TaskCompletionKey {
    selection_index: usize,
    slot: SlotKey,
    request_id: RequestId,
    transaction_id: EngineTransactionId,
    adapter_identity: AdapterWorkIdentity,
    router_rank: u16,
    expert_id: u16,
    scatter_index: usize,
}

impl TaskCompletionKey {
    pub(crate) const fn selection_index(&self) -> usize {
        self.selection_index
    }

    pub(crate) const fn slot(&self) -> SlotKey {
        self.slot
    }

    pub(crate) const fn request_id(&self) -> RequestId {
        self.request_id
    }

    pub(crate) const fn adapter_identity(&self) -> AdapterWorkIdentity {
        self.adapter_identity
    }

    pub(crate) const fn router_rank(&self) -> u16 {
        self.router_rank
    }

    pub(crate) const fn expert_id(&self) -> u16 {
        self.expert_id
    }
}

/// Constructor-preallocated storage reused by every synchronous token wave.
pub(crate) struct WaveScratch<P, T, C> {
    batch_width: usize,
    max_tasks_per_token: usize,
    task_capacity: usize,
    tasks_sorted: bool,
    selections: Vec<WaveSelection<P>>,
    tasks: Vec<TaskEnvelope<T>>,
    scatter: Vec<Option<C>>,
    ordered_contributions: Vec<C>,
}

type DrainedTasks<'a, T, C> = (std::vec::Drain<'a, TaskEnvelope<T>>, &'a mut [Option<C>]);

impl<P, T, C> fmt::Debug for WaveScratch<P, T, C> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("WaveScratch")
            .field("batch_width", &self.batch_width)
            .field("max_tasks_per_token", &self.max_tasks_per_token)
            .field("task_capacity", &self.task_capacity)
            .field("selection_count", &self.selections.len())
            .field("task_count", &self.tasks.len())
            .field("tasks_sorted", &self.tasks_sorted)
            .field("contribution_payload", &"<redacted>")
            .finish()
    }
}

impl<P, T, C> WaveScratch<P, T, C> {
    pub(crate) fn try_new(
        batch_width: usize,
        max_tasks_per_token: usize,
    ) -> Result<Self, WaveScratchError> {
        if batch_width == 0 || max_tasks_per_token == 0 {
            return Err(WaveScratchError::CapacityInvariant(
                "wave dimensions must be nonzero",
            ));
        }
        let task_capacity =
            batch_width
                .checked_mul(max_tasks_per_token)
                .ok_or(WaveScratchError::SizeOverflow {
                    resource: "wave task count",
                })?;
        if batch_width > MAX_BATCH_WIDTH {
            return Err(WaveScratchError::CapacityInvariant(
                "wave batch width exceeds implementation ceiling",
            ));
        }
        if max_tasks_per_token > MAX_TASKS_PER_TOKEN {
            return Err(WaveScratchError::CapacityInvariant(
                "per-token task count exceeds router-rank identity",
            ));
        }
        if task_capacity > MAX_TASKS_PER_WAVE {
            return Err(WaveScratchError::CapacityInvariant(
                "wave task count exceeds implementation ceiling",
            ));
        }

        let mut selections = Vec::new();
        reserve_exact(&mut selections, batch_width, "wave selections")?;
        let mut tasks = Vec::new();
        reserve_exact(&mut tasks, task_capacity, "wave tasks")?;
        let mut scatter = Vec::new();
        reserve_exact(&mut scatter, task_capacity, "wave contribution scatter")?;
        scatter.resize_with(task_capacity, || None);
        let mut ordered_contributions = Vec::new();
        reserve_exact(
            &mut ordered_contributions,
            max_tasks_per_token,
            "rank-order contributions",
        )?;

        Ok(Self {
            batch_width,
            max_tasks_per_token,
            task_capacity,
            tasks_sorted: false,
            selections,
            tasks,
            scatter,
            ordered_contributions,
        })
    }

    pub(crate) fn reset(&mut self) -> Result<(), WaveScratchError> {
        if self
            .selections
            .iter()
            .any(|selection| selection.reservation.is_some())
        {
            return Err(WaveScratchError::CapacityInvariant(
                "wave reset would discard a service reservation",
            ));
        }
        self.selections.clear();
        self.tasks.clear();
        self.tasks_sorted = false;
        for contribution in &mut self.scatter {
            let _ = contribution.take();
        }
        self.ordered_contributions.clear();
        Ok(())
    }

    pub(crate) fn push_selection(
        &mut self,
        pending: &mut Option<WaveSelection<P>>,
    ) -> Result<usize, WaveScratchError> {
        let selection = pending.as_ref().ok_or(WaveScratchError::CapacityInvariant(
            "caller selection slot is empty",
        ))?;
        if self.selections.len() >= self.batch_width {
            return Err(WaveScratchError::CapacityInvariant(
                "wave selection capacity exceeded",
            ));
        }
        if selection.task_count == 0 || selection.task_count > self.max_tasks_per_token {
            return Err(WaveScratchError::CapacityInvariant(
                "selection task count is out of range",
            ));
        }
        if selection.prepared.is_none() {
            return Err(WaveScratchError::CapacityInvariant(
                "selection is missing its prepared token",
            ));
        }
        let Some(reservation) = selection.reservation.as_ref() else {
            return Err(WaveScratchError::CapacityInvariant(
                "selection is missing its service reservation",
            ));
        };
        if reservation.slot_key() != selection.slot
            || reservation.request_id() != selection.request_id
        {
            return Err(WaveScratchError::CapacityInvariant(
                "selection reservation identity does not match",
            ));
        }
        if self.selections.iter().any(|existing| {
            existing.slot == selection.slot
                || existing.request_id == selection.request_id
                || existing.transaction_id == selection.transaction_id
                || existing.adapter_identity == selection.adapter_identity
        }) {
            return Err(WaveScratchError::CapacityInvariant(
                "wave selection identity is not unique",
            ));
        }
        let index = self.selections.len();
        let selection = pending.take().ok_or(WaveScratchError::CapacityInvariant(
            "caller selection slot changed during validation",
        ))?;
        self.selections.push(selection);
        self.tasks_sorted = false;
        Ok(index)
    }

    pub(crate) fn push_task(
        &mut self,
        mut envelope: TaskEnvelope<T>,
    ) -> Result<(), WaveScratchError> {
        let selection = self.selections.get(envelope.selection_index).ok_or(
            WaveScratchError::CapacityInvariant("task selection index is out of range"),
        )?;
        if envelope.slot != selection.slot
            || envelope.request_id != selection.request_id
            || envelope.transaction_id != selection.transaction_id
            || envelope.adapter_identity != selection.adapter_identity
        {
            return Err(WaveScratchError::CapacityInvariant(
                "task envelope identity does not match its selection",
            ));
        }
        let rank = usize::from(envelope.router_rank);
        if rank >= selection.task_count {
            return Err(WaveScratchError::CapacityInvariant(
                "task router rank exceeds selection task count",
            ));
        }
        let tasks_for_selection = self
            .tasks
            .iter()
            .filter(|task| task.selection_index == envelope.selection_index)
            .count();
        if rank != tasks_for_selection {
            return Err(WaveScratchError::CapacityInvariant(
                "task router ranks must be contiguous and unique",
            ));
        }
        envelope.scatter_index = envelope
            .selection_index
            .checked_mul(self.max_tasks_per_token)
            .and_then(|base| base.checked_add(rank))
            .ok_or(WaveScratchError::SizeOverflow {
                resource: "task scatter index",
            })?;
        if self.tasks.len() >= self.task_capacity {
            return Err(WaveScratchError::CapacityInvariant(
                "wave task capacity exceeded",
            ));
        }
        self.tasks.push(envelope);
        self.tasks_sorted = false;
        Ok(())
    }

    /// Removes the newest, not-yet-sorted selection after adapter-envelope
    /// validation fails, returning its prepared payload and service credit to
    /// the scheduler. Tasks for that selection must still be the task tail.
    pub(crate) fn rollback_last_selection(
        &mut self,
        selection_index: usize,
    ) -> Result<WaveSelection<P>, WaveScratchError> {
        if self.tasks_sorted || selection_index.checked_add(1) != Some(self.selections.len()) {
            return Err(WaveScratchError::CapacityInvariant(
                "only the newest unsorted selection can be rolled back",
            ));
        }
        let first_task = self
            .tasks
            .iter()
            .position(|task| task.selection_index == selection_index)
            .unwrap_or(self.tasks.len());
        if self.tasks[first_task..]
            .iter()
            .any(|task| task.selection_index != selection_index)
        {
            return Err(WaveScratchError::CapacityInvariant(
                "rolled-back selection tasks are not a contiguous tail",
            ));
        }
        self.tasks.truncate(first_task);
        self.selections
            .pop()
            .ok_or(WaveScratchError::CapacityInvariant(
                "rolled-back selection is missing",
            ))
    }

    pub(crate) fn sort_tasks(&mut self) -> Result<(), WaveScratchError> {
        for (selection_index, selection) in self.selections.iter().enumerate() {
            let observed = self
                .tasks
                .iter()
                .filter(|task| task.selection_index == selection_index)
                .count();
            if observed != selection.task_count {
                return Err(WaveScratchError::CapacityInvariant(
                    "selection task envelope count is incomplete",
                ));
            }
        }
        self.tasks.sort_unstable_by_key(task_sort_key);
        self.tasks_sorted = true;
        Ok(())
    }

    /// Drains tasks in frozen sorted order while retaining their allocation.
    ///
    /// Returning `Drain` rather than `&mut Vec` prevents execution code from
    /// growing or reserving the constructor-bounded task storage.
    pub(crate) fn drain_tasks_and_scatter(
        &mut self,
    ) -> Result<DrainedTasks<'_, T, C>, WaveScratchError> {
        if !self.tasks_sorted {
            return Err(WaveScratchError::CapacityInvariant(
                "wave tasks must be validated and sorted before execution",
            ));
        }
        self.tasks_sorted = false;
        Ok((self.tasks.drain(..), &mut self.scatter))
    }

    pub(crate) fn put_contribution(
        scatter: &mut [Option<C>],
        completion: TaskCompletionKey,
        contribution: C,
    ) -> Result<(), WaveScratchError> {
        let slot = scatter.get_mut(completion.scatter_index).ok_or(
            WaveScratchError::CapacityInvariant("contribution scatter index is out of range"),
        )?;
        if slot.is_some() {
            return Err(WaveScratchError::CapacityInvariant(
                "contribution scatter slot is already occupied",
            ));
        }
        *slot = Some(contribution);
        Ok(())
    }

    pub(crate) fn gather_contributions(
        &mut self,
        selection_index: usize,
    ) -> Result<&[C], WaveScratchError> {
        let task_count = self
            .selections
            .get(selection_index)
            .ok_or(WaveScratchError::CapacityInvariant(
                "contribution selection index is out of range",
            ))?
            .task_count;
        let base = selection_index
            .checked_mul(self.max_tasks_per_token)
            .ok_or(WaveScratchError::SizeOverflow {
                resource: "contribution gather index",
            })?;
        let required_end = base
            .checked_add(task_count)
            .ok_or(WaveScratchError::SizeOverflow {
                resource: "contribution gather index",
            })?;
        let lane_end =
            base.checked_add(self.max_tasks_per_token)
                .ok_or(WaveScratchError::SizeOverflow {
                    resource: "contribution gather index",
                })?;
        let lane = self
            .scatter
            .get(base..lane_end)
            .ok_or(WaveScratchError::CapacityInvariant(
                "contribution gather lane is out of range",
            ))?;
        if lane[..task_count].iter().any(Option::is_none) {
            return Err(WaveScratchError::CapacityInvariant(
                "required contribution is missing",
            ));
        }
        if lane[task_count..].iter().any(Option::is_some) {
            return Err(WaveScratchError::CapacityInvariant(
                "unexpected contribution lies outside declared ranks",
            ));
        }

        // No payload is moved until the complete lane has been validated.
        self.ordered_contributions.clear();
        for scatter_index in base..required_end {
            let contribution =
                self.scatter[scatter_index]
                    .take()
                    .ok_or(WaveScratchError::CapacityInvariant(
                        "prevalidated contribution disappeared",
                    ))?;
            self.ordered_contributions.push(contribution);
        }
        Ok(&self.ordered_contributions)
    }

    pub(crate) fn selections(&self) -> &[WaveSelection<P>] {
        &self.selections
    }

    pub(crate) fn selections_mut(&mut self) -> &mut [WaveSelection<P>] {
        &mut self.selections
    }

    #[cfg(test)]
    fn allocation_fingerprint(&self) -> AllocationFingerprint {
        AllocationFingerprint {
            selections_ptr: self.selections.as_ptr() as usize,
            selections_capacity: self.selections.capacity(),
            tasks_ptr: self.tasks.as_ptr() as usize,
            tasks_capacity: self.tasks.capacity(),
            scatter_ptr: self.scatter.as_ptr() as usize,
            scatter_capacity: self.scatter.capacity(),
            ordered_ptr: self.ordered_contributions.as_ptr() as usize,
            ordered_capacity: self.ordered_contributions.capacity(),
        }
    }
}

fn task_sort_key<T>(envelope: &TaskEnvelope<T>) -> (u16, u64, usize, u16, u64) {
    (
        envelope.expert_id,
        envelope.request_id.get(),
        envelope.adapter_identity.position(),
        envelope.router_rank,
        envelope.transaction_id.get(),
    )
}

fn reserve_exact<T>(
    values: &mut Vec<T>,
    count: usize,
    resource: &'static str,
) -> Result<(), WaveScratchError> {
    let bytes = count
        .checked_mul(size_of::<T>())
        .ok_or(WaveScratchError::SizeOverflow { resource })?;
    if bytes > isize::MAX as usize {
        return Err(WaveScratchError::SizeOverflow { resource });
    }
    values
        .try_reserve_exact(count)
        .map_err(|_| WaveScratchError::AllocationFailure { resource, bytes })
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AllocationFingerprint {
    selections_ptr: usize,
    selections_capacity: usize,
    tasks_ptr: usize,
    tasks_capacity: usize,
    scatter_ptr: usize,
    scatter_capacity: usize,
    ordered_ptr: usize,
    ordered_capacity: usize,
}

#[cfg(test)]
mod tests {
    use crate::id::{EngineTransactionIdIssuer, RequestIdIssuer, SlotGenerationIssuer, SlotKey};
    use crate::ring::DrrRing;

    use super::*;

    fn reserved_selections(task_counts: &[usize]) -> Vec<WaveSelection<u8>> {
        let mut requests = RequestIdIssuer::new();
        let mut transactions = EngineTransactionIdIssuer::new();
        let mut generations = SlotGenerationIssuer::new();
        let mut ring = DrrRing::try_with_capacity(task_counts.len()).unwrap();
        let mut identities = Vec::new();
        for (index, &task_count) in task_counts.iter().enumerate() {
            let request_id = requests.issue().unwrap();
            let transaction_id = transactions.issue().unwrap();
            let slot = SlotKey::new(index, generations.issue().unwrap());
            let raw = u64::try_from(index).unwrap() + 1;
            let adapter_identity =
                AdapterWorkIdentity::try_new(raw, 10, raw + 20, 0, index + 5).unwrap();
            ring.insert(slot, request_id).unwrap();
            identities.push((
                slot,
                request_id,
                transaction_id,
                adapter_identity,
                task_count,
            ));
        }
        ring.open_round().unwrap().expect("nonempty test round");
        identities
            .into_iter()
            .enumerate()
            .map(
                |(prepared, (slot, request_id, transaction_id, adapter_identity, task_count))| {
                    let visit = ring.next_visit().unwrap();
                    assert_eq!(visit.slot_key(), slot);
                    assert_eq!(visit.request_id(), request_id);
                    let (reservation, _) = ring.mark_selected(visit).unwrap();
                    WaveSelection {
                        slot,
                        request_id,
                        transaction_id,
                        adapter_identity,
                        reservation: Some(reservation),
                        prepared: Some(u8::try_from(prepared).unwrap()),
                        task_count,
                    }
                },
            )
            .collect()
    }

    fn envelope(
        selection_index: usize,
        selection: &WaveSelection<u8>,
        router_rank: u16,
        expert_id: u16,
        task: u8,
    ) -> TaskEnvelope<u8> {
        TaskEnvelope {
            selection_index,
            slot: selection.slot,
            request_id: selection.request_id,
            transaction_id: selection.transaction_id,
            adapter_identity: selection.adapter_identity,
            router_rank,
            expert_id,
            scatter_index: usize::MAX,
            task,
        }
    }

    fn push_valid(scratch: &mut WaveScratch<u8, u8, u8>, selection: WaveSelection<u8>) -> usize {
        let mut pending = Some(selection);
        let index = scratch.push_selection(&mut pending).unwrap();
        assert!(pending.is_none());
        index
    }

    fn duplicate_completion(key: &TaskCompletionKey) -> TaskCompletionKey {
        TaskCompletionKey {
            selection_index: key.selection_index,
            slot: key.slot,
            request_id: key.request_id,
            transaction_id: key.transaction_id,
            adapter_identity: key.adapter_identity,
            router_rank: key.router_rank,
            expert_id: key.expert_id,
            scatter_index: key.scatter_index,
        }
    }

    #[test]
    fn sort_key_and_rank_scatter_are_exact() {
        let mut scratch = WaveScratch::<u8, u8, u8>::try_new(2, 2).unwrap();
        let mut selections = reserved_selections(&[2]);
        let selection_value = selections.pop().unwrap();
        let tasks = [
            envelope(0, &selection_value, 0, 3, 30),
            envelope(0, &selection_value, 1, 1, 10),
        ];
        let selection = push_valid(&mut scratch, selection_value);
        for task in tasks {
            scratch.push_task(task).unwrap();
        }
        assert!(matches!(
            scratch.drain_tasks_and_scatter(),
            Err(WaveScratchError::CapacityInvariant(
                "wave tasks must be validated and sorted before execution"
            ))
        ));
        scratch.sort_tasks().unwrap();
        let (tasks, scatter) = scratch.drain_tasks_and_scatter().unwrap();
        let mut experts = [0_u16; 2];
        for (index, task) in tasks.enumerate() {
            experts[index] = task.expert_id;
            let (completion, payload) = task.into_parts();
            WaveScratch::<u8, u8, u8>::put_contribution(scatter, completion, payload + 1).unwrap();
        }
        assert_eq!(experts, [1, 3]);
        assert_eq!(scratch.gather_contributions(selection).unwrap(), [31, 11]);
    }

    #[test]
    fn complete_cycles_never_move_or_grow_storage() {
        let mut scratch = WaveScratch::<u8, u8, u8>::try_new(2, 2).unwrap();
        let original = scratch.allocation_fingerprint();
        for _ in 0..100 {
            let selections = reserved_selections(&[2, 2]);
            for selection in selections {
                let tasks = [
                    envelope(scratch.selections.len(), &selection, 0, 1, 2),
                    envelope(scratch.selections.len(), &selection, 1, 0, 3),
                ];
                push_valid(&mut scratch, selection);
                for task in tasks {
                    scratch.push_task(task).unwrap();
                }
            }
            scratch.sort_tasks().unwrap();
            let (tasks, scatter) = scratch.drain_tasks_and_scatter().unwrap();
            for task in tasks {
                let (completion, payload) = task.into_parts();
                WaveScratch::<u8, u8, u8>::put_contribution(scatter, completion, payload).unwrap();
            }
            for selection_index in 0..2 {
                assert_eq!(
                    scratch.gather_contributions(selection_index).unwrap().len(),
                    2
                );
            }
            for selection in scratch.selections_mut() {
                let _ = selection.reservation.take();
            }
            scratch.reset().unwrap();
            assert_eq!(scratch.allocation_fingerprint(), original);
        }
    }

    #[test]
    fn duplicate_rank_and_incomplete_envelopes_fail_before_sort() {
        let mut scratch = WaveScratch::<u8, u8, u8>::try_new(1, 2).unwrap();
        let mut selections = reserved_selections(&[2]);
        let selection = selections.pop().unwrap();
        let first = envelope(0, &selection, 0, 1, 10);
        let duplicate = envelope(0, &selection, 0, 2, 11);
        push_valid(&mut scratch, selection);
        scratch.push_task(first).unwrap();
        assert_eq!(
            scratch.push_task(duplicate).unwrap_err(),
            WaveScratchError::CapacityInvariant("task router ranks must be contiguous and unique")
        );
        assert_eq!(
            scratch.sort_tasks().unwrap_err(),
            WaveScratchError::CapacityInvariant("selection task envelope count is incomplete")
        );
    }

    #[test]
    fn missing_scatter_is_nondestructive_and_duplicate_scatter_is_rejected() {
        let mut scratch = WaveScratch::<u8, u8, u8>::try_new(1, 2).unwrap();
        let mut selections = reserved_selections(&[2]);
        let selection = selections.pop().unwrap();
        let tasks = [
            envelope(0, &selection, 0, 1, 10),
            envelope(0, &selection, 1, 2, 20),
        ];
        let selection_index = push_valid(&mut scratch, selection);
        for task in tasks {
            scratch.push_task(task).unwrap();
        }
        scratch.sort_tasks().unwrap();
        let completions = {
            let (tasks, _) = scratch.drain_tasks_and_scatter().unwrap();
            tasks.map(TaskEnvelope::into_parts).collect::<Vec<_>>()
        };
        let mut completions = completions.into_iter();
        let (first, first_payload) = completions.next().unwrap();
        let duplicate = duplicate_completion(&first);
        WaveScratch::<u8, u8, u8>::put_contribution(&mut scratch.scatter, duplicate, first_payload)
            .unwrap();
        assert_eq!(
            WaveScratch::<u8, u8, u8>::put_contribution(
                &mut scratch.scatter,
                first,
                first_payload,
            )
            .unwrap_err(),
            WaveScratchError::CapacityInvariant(
                "contribution scatter slot is already occupied"
            )
        );
        assert_eq!(
            scratch.gather_contributions(selection_index).unwrap_err(),
            WaveScratchError::CapacityInvariant("required contribution is missing")
        );
        assert_eq!(scratch.scatter[0], Some(first_payload));

        let (second, second_payload) = completions.next().unwrap();
        WaveScratch::<u8, u8, u8>::put_contribution(&mut scratch.scatter, second, second_payload)
            .unwrap();
        assert_eq!(
            scratch.gather_contributions(selection_index).unwrap(),
            [first_payload, second_payload]
        );
    }

    #[test]
    fn dimensions_and_every_logical_capacity_fail_closed() {
        assert_eq!(
            WaveScratch::<u8, u8, u8>::try_new(0, 1).unwrap_err(),
            WaveScratchError::CapacityInvariant("wave dimensions must be nonzero")
        );
        assert_eq!(
            WaveScratch::<u8, u8, u8>::try_new(usize::MAX, 2).unwrap_err(),
            WaveScratchError::SizeOverflow {
                resource: "wave task count"
            }
        );
        assert_eq!(
            WaveScratch::<u8, u8, u8>::try_new(MAX_BATCH_WIDTH + 1, 1).unwrap_err(),
            WaveScratchError::CapacityInvariant("wave batch width exceeds implementation ceiling")
        );
        assert_eq!(
            WaveScratch::<u8, u8, u8>::try_new(1, MAX_TASKS_PER_TOKEN + 1).unwrap_err(),
            WaveScratchError::CapacityInvariant(
                "per-token task count exceeds router-rank identity"
            )
        );

        let mut scratch = WaveScratch::<u8, u8, u8>::try_new(1, 1).unwrap();
        let mut selections = reserved_selections(&[1, 1]);
        let selection = selections.remove(0);
        push_valid(&mut scratch, selection);
        let mut pending = Some(selections.remove(0));
        assert_eq!(
            scratch.push_selection(&mut pending).unwrap_err(),
            WaveScratchError::CapacityInvariant("wave selection capacity exceeded")
        );
        assert!(pending.is_some());
    }

    #[test]
    fn selection_requires_matching_reservation_and_prepared_payload() {
        let mut scratch = WaveScratch::<u8, u8, u8>::try_new(2, 1).unwrap();
        let mut selections = reserved_selections(&[1]);
        let mut missing = selections.pop().unwrap();
        missing.reservation = None;
        let mut pending = Some(missing);
        assert_eq!(
            scratch.push_selection(&mut pending).unwrap_err(),
            WaveScratchError::CapacityInvariant("selection is missing its service reservation")
        );
        assert!(pending.is_some());

        let mut selections = reserved_selections(&[1]);
        let mut missing = selections.pop().unwrap();
        missing.prepared = None;
        let mut pending = Some(missing);
        assert_eq!(
            scratch.push_selection(&mut pending).unwrap_err(),
            WaveScratchError::CapacityInvariant("selection is missing its prepared token")
        );
        assert!(pending.is_some());

        let mut selections = reserved_selections(&[1, 1]);
        let mut first = selections.remove(0);
        let mut second = selections.remove(0);
        first.reservation = second.reservation.take();
        let mut pending = Some(first);
        assert_eq!(
            scratch.push_selection(&mut pending).unwrap_err(),
            WaveScratchError::CapacityInvariant("selection reservation identity does not match")
        );
        assert!(pending.unwrap().reservation.is_some());
    }

    #[test]
    fn rejection_returns_ownership_and_reset_refuses_to_drop_reservation() {
        let mut scratch = WaveScratch::<u8, u8, u8>::try_new(1, 1).unwrap();
        let mut first = reserved_selections(&[1]);
        push_valid(&mut scratch, first.pop().unwrap());

        let mut overflow = reserved_selections(&[1]);
        let mut pending = Some(overflow.pop().unwrap());
        assert_eq!(
            scratch.push_selection(&mut pending).unwrap_err(),
            WaveScratchError::CapacityInvariant("wave selection capacity exceeded")
        );
        let returned = pending.unwrap();
        assert!(returned.reservation.is_some());
        assert!(returned.prepared.is_some());
        assert_eq!(
            scratch.reset().unwrap_err(),
            WaveScratchError::CapacityInvariant("wave reset would discard a service reservation")
        );
        assert_eq!(scratch.selections.len(), 1);

        let _ = scratch.selections_mut()[0].reservation.take();
        scratch.reset().unwrap();
        assert!(scratch.selections.is_empty());
    }

    #[test]
    fn frozen_sort_key_contains_all_five_fields_in_order() {
        let request = RequestId::try_from_raw_for_test(7).unwrap();
        let transaction = EngineTransactionIdIssuer::from_next(11).issue().unwrap();
        let generation = SlotGenerationIssuer::new().issue().unwrap();
        let envelope = TaskEnvelope {
            selection_index: 0,
            slot: SlotKey::new(3, generation),
            request_id: request,
            transaction_id: transaction,
            adapter_identity: AdapterWorkIdentity::try_new(1, 2, 3, 4, 9).unwrap(),
            router_rank: 5,
            expert_id: 4,
            scatter_index: 0,
            task: (),
        };
        assert_eq!(task_sort_key(&envelope), (4, 7, 9, 5, 11));
    }
}
