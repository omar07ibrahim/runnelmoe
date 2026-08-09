use crate::state::KvHistory;
use crate::{Result, RuntimeError};

/// Computes stable, three-pass causal attention over committed pages plus one
/// pending K/V pair. The caller owns `output`; no context-sized scratch is used.
pub fn streaming_causal_attention(
    query: &[f32],
    committed: KvHistory<'_>,
    pending_key: &[f32],
    pending_value: &[f32],
    num_heads: usize,
    output: &mut [f32],
) -> Result<()> {
    let hidden_size = committed.hidden_size().ok_or(RuntimeError::InvalidState(
        "attention history must be bound",
    ))?;
    if pending_key.len() != hidden_size || pending_value.len() != hidden_size {
        return Err(RuntimeError::InvalidState(
            "pending attention K/V widths must match hidden size",
        ));
    }
    if pending_key
        .iter()
        .chain(pending_value)
        .any(|value| !value.is_finite())
    {
        return Err(RuntimeError::NonFinite("pending attention K/V"));
    }
    let length = committed
        .len()
        .checked_add(1)
        .ok_or(RuntimeError::InvalidState(
            "attention history length overflows",
        ))?;
    let history = PendingHistory {
        committed,
        pending_key,
        pending_value,
        hidden_size,
        length,
    };
    streaming_attention_impl(query, &history, num_heads, output)
}

trait AttentionHistory {
    fn len(&self) -> usize;
    fn hidden_size(&self) -> usize;
    fn key_at(&self, position: usize) -> Option<&[f32]>;
    fn value_at(&self, position: usize) -> Option<&[f32]>;
}

struct PendingHistory<'state, 'pending> {
    committed: KvHistory<'state>,
    pending_key: &'pending [f32],
    pending_value: &'pending [f32],
    hidden_size: usize,
    length: usize,
}

impl AttentionHistory for PendingHistory<'_, '_> {
    fn len(&self) -> usize {
        self.length
    }

    fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    fn key_at(&self, position: usize) -> Option<&[f32]> {
        if position == self.committed.len() {
            Some(self.pending_key)
        } else {
            self.committed.key_at(position)
        }
    }

    fn value_at(&self, position: usize) -> Option<&[f32]> {
        if position == self.committed.len() {
            Some(self.pending_value)
        } else {
            self.committed.value_at(position)
        }
    }
}

fn streaming_attention_impl<H: AttentionHistory>(
    query: &[f32],
    history: &H,
    num_heads: usize,
    output: &mut [f32],
) -> Result<()> {
    let hidden_size = history.hidden_size();
    if hidden_size == 0 {
        return Err(RuntimeError::InvalidState(
            "attention hidden size must be nonzero",
        ));
    }
    if num_heads == 0 || !hidden_size.is_multiple_of(num_heads) {
        return Err(RuntimeError::InvalidState(
            "attention heads must divide hidden size",
        ));
    }
    if query.len() != hidden_size || output.len() != hidden_size {
        return Err(RuntimeError::InvalidState(
            "attention query and output widths must match hidden size",
        ));
    }
    if history.len() == 0 {
        return Err(RuntimeError::InvalidState(
            "attention history must contain the pending position",
        ));
    }
    if query.iter().any(|value| !value.is_finite()) {
        return Err(RuntimeError::NonFinite("attention query"));
    }

    output.fill(0.0);
    let head_size = hidden_size / num_heads;
    let scale = (head_size as f32).sqrt().recip();
    if !scale.is_finite() {
        return Err(RuntimeError::NonFinite("attention scale"));
    }

    for head in 0..num_heads {
        let start = head
            .checked_mul(head_size)
            .ok_or(RuntimeError::InvalidState(
                "attention head offset overflows",
            ))?;
        let end = start
            .checked_add(head_size)
            .ok_or(RuntimeError::InvalidState("attention head range overflows"))?;

        // Pass one: finite maximum, in ascending logical-position order.
        let mut maximum = f32::NEG_INFINITY;
        for position in 0..history.len() {
            let score = attention_score(query, history, position, start, end, scale)?;
            maximum = maximum.max(score);
        }
        if !maximum.is_finite() {
            return Err(RuntimeError::NonFinite("attention maximum"));
        }

        // Pass two: stable denominator, with the identical visit order.
        let mut denominator = 0.0_f32;
        for position in 0..history.len() {
            let score = attention_score(query, history, position, start, end, scale)?;
            let weight = (score - maximum).exp();
            if !weight.is_finite() {
                return Err(RuntimeError::NonFinite("attention exponential"));
            }
            denominator += weight;
            if !denominator.is_finite() {
                return Err(RuntimeError::NonFinite("attention denominator"));
            }
        }
        if denominator <= 0.0 {
            return Err(RuntimeError::NonFinite("attention denominator"));
        }

        // Pass three: recompute each score, then reduce values in ascending order.
        for position in 0..history.len() {
            let score = attention_score(query, history, position, start, end, scale)?;
            let probability = (score - maximum).exp() / denominator;
            if !probability.is_finite() {
                return Err(RuntimeError::NonFinite("attention probability"));
            }
            let value = history
                .value_at(position)
                .ok_or(RuntimeError::InvalidState(
                    "attention value position is missing",
                ))?;
            if value.len() != hidden_size {
                return Err(RuntimeError::InvalidState(
                    "attention value width does not match hidden size",
                ));
            }
            for index in start..end {
                let next = output[index] + probability * value[index];
                if !next.is_finite() {
                    return Err(RuntimeError::NonFinite("attention output"));
                }
                output[index] = next;
            }
        }
    }

    Ok(())
}

fn attention_score<H: AttentionHistory>(
    query: &[f32],
    history: &H,
    position: usize,
    start: usize,
    end: usize,
    scale: f32,
) -> Result<f32> {
    let key = history.key_at(position).ok_or(RuntimeError::InvalidState(
        "attention key position is missing",
    ))?;
    if key.len() != history.hidden_size() {
        return Err(RuntimeError::InvalidState(
            "attention key width does not match hidden size",
        ));
    }
    let mut dot = 0.0_f32;
    for index in start..end {
        let product = query[index] * key[index];
        if !product.is_finite() {
            return Err(RuntimeError::NonFinite("attention dot product"));
        }
        dot += product;
        if !dot.is_finite() {
            return Err(RuntimeError::NonFinite("attention dot product"));
        }
    }
    let score = dot * scale;
    if !score.is_finite() {
        return Err(RuntimeError::NonFinite("attention score"));
    }
    Ok(score)
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use super::*;
    use crate::state::{SequenceState, StateLayout};

    fn append(state: &mut SequenceState, key: &[f32], value: &[f32]) {
        let id = state.state_id().unwrap();
        let revision = state.revision();
        let position = state.len();
        state
            .validate_append(id, revision, position, key, value)
            .unwrap()
            .apply();
    }

    fn vector(position: usize, salt: usize) -> [f32; 8] {
        std::array::from_fn(|index| {
            let raw = ((position + 1) * (index + salt + 1)) % 29;
            (raw as f32 - 14.0) / 32.0
        })
    }

    fn materialized_f64(
        query: &[f32],
        state: &SequenceState,
        pending_key: &[f32],
        pending_value: &[f32],
        num_heads: usize,
    ) -> Vec<f64> {
        let history = state.history();
        let hidden = query.len();
        let head_size = hidden / num_heads;
        let scale = 1.0_f64 / (head_size as f64).sqrt();
        let mut output = vec![0.0_f64; hidden];
        for head in 0..num_heads {
            let start = head * head_size;
            let end = start + head_size;
            let mut scores = Vec::with_capacity(history.len() + 1);
            for position in 0..=history.len() {
                let key = if position == history.len() {
                    pending_key
                } else {
                    history.key_at(position).unwrap()
                };
                let score = (start..end)
                    .map(|index| f64::from(query[index]) * f64::from(key[index]))
                    .sum::<f64>()
                    * scale;
                scores.push(score);
            }
            let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let denominator = scores
                .iter()
                .map(|score| (score - maximum).exp())
                .sum::<f64>();
            for (position, score) in scores.into_iter().enumerate() {
                let value = if position == history.len() {
                    pending_value
                } else {
                    history.value_at(position).unwrap()
                };
                let probability = (score - maximum).exp() / denominator;
                for index in start..end {
                    output[index] += probability * f64::from(value[index]);
                }
            }
        }
        output
    }

    #[test]
    fn at01_streaming_matches_materialized_f64_at_frozen_boundaries() {
        let mut state =
            SequenceState::try_new(StateLayout::new(1_024, 16, 8).unwrap(), 20).unwrap();
        for one_based in 1..=1_024 {
            let position = one_based - 1;
            let query = vector(position, 2);
            let key = vector(position, 5);
            let value = vector(position, 9);
            if [1, 15, 16, 17, 255, 256, 257, 1_023, 1_024].contains(&one_based) {
                let expected = materialized_f64(&query, &state, &key, &value, 2);
                let mut actual = [f32::NAN; 8];
                streaming_causal_attention(&query, state.history(), &key, &value, 2, &mut actual)
                    .unwrap();
                for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                    let tolerance = 1.0e-5_f64 + 1.0e-4_f64 * expected.abs();
                    assert!(
                        (f64::from(*actual) - expected).abs() <= tolerance,
                        "position {one_based}, component {index}: {actual} vs {expected}"
                    );
                }
            }
            append(&mut state, &key, &value);
        }
    }

    struct RecordingHistory {
        keys: Vec<Vec<f32>>,
        values: Vec<Vec<f32>>,
        visits: RefCell<Vec<(char, usize)>>,
    }

    impl AttentionHistory for RecordingHistory {
        fn len(&self) -> usize {
            self.keys.len()
        }

        fn hidden_size(&self) -> usize {
            self.keys[0].len()
        }

        fn key_at(&self, position: usize) -> Option<&[f32]> {
            self.visits.borrow_mut().push(('k', position));
            self.keys.get(position).map(Vec::as_slice)
        }

        fn value_at(&self, position: usize) -> Option<&[f32]> {
            self.visits.borrow_mut().push(('v', position));
            self.values.get(position).map(Vec::as_slice)
        }
    }

    #[test]
    fn at02_visits_ascending_stably_and_rejects_nonfinite_intermediates() {
        let history = RecordingHistory {
            keys: vec![vec![1_000.0], vec![1_001.0], vec![999.0]],
            values: vec![vec![1.0], vec![3.0], vec![7.0]],
            visits: RefCell::new(Vec::new()),
        };
        let mut output = [0.0];
        streaming_attention_impl(&[1.0], &history, 1, &mut output).unwrap();
        let expected = ((-1.0_f64).exp() + 3.0 + 7.0 * (-2.0_f64).exp())
            / ((-1.0_f64).exp() + 1.0 + (-2.0_f64).exp());
        assert!((f64::from(output[0]) - expected).abs() <= 1.0e-5);
        assert_eq!(
            history.visits.into_inner(),
            [
                ('k', 0),
                ('k', 1),
                ('k', 2),
                ('k', 0),
                ('k', 1),
                ('k', 2),
                ('k', 0),
                ('v', 0),
                ('k', 1),
                ('v', 1),
                ('k', 2),
                ('v', 2),
            ]
        );

        let nonfinite = RecordingHistory {
            keys: vec![vec![f32::INFINITY]],
            values: vec![vec![1.0]],
            visits: RefCell::new(Vec::new()),
        };
        assert_eq!(
            streaming_attention_impl(&[1.0], &nonfinite, 1, &mut output).unwrap_err(),
            RuntimeError::NonFinite("attention dot product")
        );
        let overflow = RecordingHistory {
            keys: vec![vec![f32::MAX]],
            values: vec![vec![1.0]],
            visits: RefCell::new(Vec::new()),
        };
        assert_eq!(
            streaming_attention_impl(&[2.0], &overflow, 1, &mut output).unwrap_err(),
            RuntimeError::NonFinite("attention dot product")
        );
        let invalid_value = RecordingHistory {
            keys: vec![vec![1.0]],
            values: vec![vec![f32::INFINITY]],
            visits: RefCell::new(Vec::new()),
        };
        assert_eq!(
            streaming_attention_impl(&[1.0], &invalid_value, 1, &mut output).unwrap_err(),
            RuntimeError::NonFinite("attention output")
        );
    }

    #[test]
    fn at03_caller_output_and_state_capacities_stay_fixed_through_context() {
        let mut state =
            SequenceState::try_new(StateLayout::new(1_024, 16, 8).unwrap(), 21).unwrap();
        let payload_bytes = state.accounted_payload_bytes();
        let charge_bytes = state.accounted_charge_bytes();
        let mut output = Vec::with_capacity(8);
        output.resize(8, 0.0);
        let output_pointer = output.as_ptr();
        let output_capacity = output.capacity();
        for position in 0..1_024 {
            let query = vector(position, 2);
            let key = vector(position, 5);
            let value = vector(position, 9);
            streaming_causal_attention(&query, state.history(), &key, &value, 2, &mut output)
                .unwrap();
            append(&mut state, &key, &value);
        }
        assert_eq!(output.as_ptr(), output_pointer);
        assert_eq!(output.capacity(), output_capacity);
        assert_eq!(state.accounted_payload_bytes(), payload_bytes);
        assert_eq!(state.accounted_charge_bytes(), charge_bytes);
    }
}
