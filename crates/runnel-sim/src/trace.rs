//! Canonical trace parser and serializer.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Serialize, de::DeserializeOwned};
use sha2::{Digest as _, Sha256};

use crate::model::{PREFETCH_MODEL, TRACE_SCHEMA};
use crate::{
    ExpertPrediction, PageClass, PageDescriptor, PageId, SimError, SimLimits, TraceEvent,
    TraceHeader, ValidatedTrace,
};

const HEADER_KIND: &str = "header";
const MAX_TRACE_ID_BYTES: usize = 64;
const SCORE_SCALE_PPM: u64 = 1_000_000;

type PageIndex = BTreeMap<PageId, usize>;
type ExpertPageIndex = BTreeMap<(u32, u32), Vec<PageId>>;

/// Parse and validate one bounded canonical cache trace.
///
/// Canonical JSONL here is deliberately narrower than arbitrary JSON: every
/// record is the compact `serde` representation of its closed Rust schema,
/// the complete file is ASCII, and every record (including the last) is
/// terminated by exactly one LF. This gives the simulator one stable byte
/// representation to hash and rejects duplicate/unknown fields through the
/// typed deserializers.
pub fn parse_trace(bytes: &[u8], limits: SimLimits) -> Result<ValidatedTrace, SimError> {
    validate_framing(bytes, limits)?;

    let content = &bytes[..bytes.len() - 1];
    let mut lines = content.split(|byte| *byte == b'\n');
    let mut line_number = 1_usize;

    let header_line = lines
        .next()
        .ok_or_else(|| SimError::invalid_trace("trace is missing its header record"))?;
    let header: TraceHeader = parse_canonical_record(header_line, line_number, limits)?;
    validate_header(&header, limits)?;

    let mut pages = Vec::with_capacity(header.page_count);
    for page_index in 0..header.page_count {
        line_number = line_number
            .checked_add(1)
            .ok_or_else(|| SimError::invalid_trace("trace line number overflow"))?;
        let line = lines.next().ok_or_else(|| {
            SimError::invalid_trace(format!("catalog ended before declared page {page_index}"))
        })?;
        pages.push(parse_canonical_record(line, line_number, limits)?);
    }

    let mut events = Vec::with_capacity(header.event_count);
    for event_index in 0..header.event_count {
        line_number = line_number
            .checked_add(1)
            .ok_or_else(|| SimError::invalid_trace("trace line number overflow"))?;
        let line = lines.next().ok_or_else(|| {
            SimError::invalid_trace(format!(
                "event stream ended before declared event {event_index}"
            ))
        })?;
        events.push(parse_canonical_record(line, line_number, limits)?);
    }

    if lines.next().is_some() {
        return Err(SimError::invalid_trace(
            "trace contains records beyond its declared catalog and event counts",
        ));
    }

    let (page_index, expert_pages) = validate_pages(&header, &pages)?;
    validate_events(&events, &page_index, &expert_pages, limits)?;

    Ok(ValidatedTrace {
        header,
        pages,
        page_index,
        expert_pages,
        events,
        sha256: hex::encode(Sha256::digest(bytes)),
    })
}

/// Serialize a validated trace into its unique canonical JSONL representation.
///
/// The serializer does not silently reorder semantic input. Catalog and event
/// order are part of the format contract, so invalid order is rejected by a
/// validation round trip.
pub fn serialize_trace(
    header: &TraceHeader,
    pages: &[PageDescriptor],
    events: &[TraceEvent],
    limits: SimLimits,
) -> Result<Vec<u8>, SimError> {
    validate_header(header, limits)?;
    if header.page_count != pages.len() {
        return Err(SimError::invalid_trace(format!(
            "header declares {} pages but serializer received {}",
            header.page_count,
            pages.len()
        )));
    }
    if header.event_count != events.len() {
        return Err(SimError::invalid_trace(format!(
            "header declares {} events but serializer received {}",
            header.event_count,
            events.len()
        )));
    }
    for event in events {
        if let TraceEvent::RouterSignal { predictions, .. } = event
            && predictions.len() > limits.max_predictions_per_signal
        {
            return Err(SimError::invalid_trace(format!(
                "router signal contains {} predictions, limit is {}",
                predictions.len(),
                limits.max_predictions_per_signal
            )));
        }
    }

    let mut output = Vec::new();
    append_record(&mut output, header, limits)?;
    for page in pages {
        append_record(&mut output, page, limits)?;
    }
    for event in events {
        append_record(&mut output, event, limits)?;
    }

    let parsed = parse_trace(&output, limits)?;
    if parsed.header != *header || parsed.pages != pages || parsed.events != events {
        return Err(SimError::invalid_trace(
            "canonical serialization failed its typed round trip",
        ));
    }
    Ok(output)
}

fn validate_framing(bytes: &[u8], limits: SimLimits) -> Result<(), SimError> {
    if bytes.len() > limits.max_trace_bytes {
        return Err(SimError::invalid_trace(format!(
            "trace is {} bytes, limit is {}",
            bytes.len(),
            limits.max_trace_bytes
        )));
    }
    if bytes.is_empty() || !bytes.ends_with(b"\n") {
        return Err(SimError::invalid_trace("canonical trace must end with LF"));
    }
    if bytes.contains(&b'\r') {
        return Err(SimError::invalid_trace(
            "canonical trace must not contain CR bytes",
        ));
    }
    if !bytes.is_ascii() {
        return Err(SimError::invalid_trace(
            "canonical trace must contain ASCII bytes only",
        ));
    }
    Ok(())
}

fn parse_canonical_record<T>(
    line: &[u8],
    line_number: usize,
    limits: SimLimits,
) -> Result<T, SimError>
where
    T: DeserializeOwned + Serialize,
{
    if line.is_empty() {
        return Err(SimError::invalid_trace(format!(
            "line {line_number} is empty"
        )));
    }
    if line.len() > limits.max_line_bytes {
        return Err(SimError::invalid_trace(format!(
            "line {line_number} is {} bytes, limit is {}",
            line.len(),
            limits.max_line_bytes
        )));
    }

    let value: T = serde_json::from_slice(line)?;
    if serde_json::to_vec(&value)? != line {
        return Err(SimError::invalid_trace(format!(
            "line {line_number} is not canonical compact JSON"
        )));
    }
    Ok(value)
}

fn validate_header(header: &TraceHeader, limits: SimLimits) -> Result<(), SimError> {
    if header.kind != HEADER_KIND {
        return Err(SimError::invalid_trace(format!(
            "header kind must be {HEADER_KIND:?}"
        )));
    }
    if header.schema != TRACE_SCHEMA {
        return Err(SimError::invalid_trace(format!(
            "unsupported trace schema {:?}",
            header.schema
        )));
    }
    if header.prefetch_model != PREFETCH_MODEL {
        return Err(SimError::invalid_trace(format!(
            "unsupported prefetch model {:?}",
            header.prefetch_model
        )));
    }
    validate_trace_id(&header.trace_id)?;
    if header.page_count > limits.max_pages {
        return Err(SimError::invalid_trace(format!(
            "header declares {} pages, limit is {}",
            header.page_count, limits.max_pages
        )));
    }
    if header.event_count > limits.max_events {
        return Err(SimError::invalid_trace(format!(
            "header declares {} events, limit is {}",
            header.event_count, limits.max_events
        )));
    }
    if header.charge_quantum == 0 {
        return Err(SimError::invalid_trace(
            "charge_quantum must be greater than zero",
        ));
    }
    Ok(())
}

fn validate_trace_id(trace_id: &str) -> Result<(), SimError> {
    let bytes = trace_id.as_bytes();
    let valid_first = bytes
        .first()
        .is_some_and(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit());
    let valid_rest = bytes.iter().skip(1).all(|byte| {
        byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(*byte, b'.' | b'_' | b'-')
    });
    if bytes.len() > MAX_TRACE_ID_BYTES || !valid_first || !valid_rest {
        return Err(SimError::invalid_trace(format!(
            "trace_id must be 1-{MAX_TRACE_ID_BYTES} bytes and match [a-z0-9][a-z0-9._-]*"
        )));
    }
    Ok(())
}

fn validate_pages(
    header: &TraceHeader,
    pages: &[PageDescriptor],
) -> Result<(PageIndex, ExpertPageIndex), SimError> {
    let mut page_index = BTreeMap::new();
    let mut expert_tuples = BTreeSet::new();
    let mut expert_ordinals: BTreeMap<(u32, u32), Vec<(u32, PageId)>> = BTreeMap::new();

    for (index, page) in pages.iter().enumerate() {
        let expected_id = u32::try_from(index)
            .map(PageId)
            .map_err(|_| SimError::invalid_trace("page index does not fit in u32"))?;
        if page.id != expected_id {
            return Err(SimError::invalid_trace(format!(
                "catalog page {index} must have id {}, found {}",
                expected_id.0, page.id.0
            )));
        }
        if page.logical_bytes == 0 || page.charge_bytes == 0 {
            return Err(SimError::invalid_trace(format!(
                "page {} byte sizes must be greater than zero",
                page.id.0
            )));
        }
        if page.logical_bytes > page.charge_bytes {
            return Err(SimError::invalid_trace(format!(
                "page {} logical_bytes exceeds charge_bytes",
                page.id.0
            )));
        }
        if page.charge_bytes % header.charge_quantum != 0 {
            return Err(SimError::invalid_trace(format!(
                "page {} charge_bytes is not divisible by charge_quantum",
                page.id.0
            )));
        }

        page_index.insert(page.id, index);
        if let PageClass::Expert {
            layer,
            expert,
            ordinal,
        } = &page.class
        {
            if !expert_tuples.insert((*layer, *expert, *ordinal)) {
                return Err(SimError::invalid_trace(format!(
                    "duplicate expert page tuple ({layer}, {expert}, {ordinal})"
                )));
            }
            expert_ordinals
                .entry((*layer, *expert))
                .or_default()
                .push((*ordinal, page.id));
        }
    }

    let expert_pages = expert_ordinals
        .into_iter()
        .map(|(key, mut entries)| {
            entries.sort_unstable_by_key(|(ordinal, page_id)| (*ordinal, *page_id));
            (
                key,
                entries.into_iter().map(|(_, page_id)| page_id).collect(),
            )
        })
        .collect();
    Ok((page_index, expert_pages))
}

fn validate_events(
    events: &[TraceEvent],
    page_index: &PageIndex,
    expert_pages: &ExpertPageIndex,
    limits: SimLimits,
) -> Result<(), SimError> {
    let mut latest_realized_step = BTreeMap::<u64, u64>::new();
    let mut realized_steps = BTreeSet::<(u64, u64)>::new();
    let mut signal_targets = BTreeSet::<(u64, u64, u32)>::new();

    for (index, event) in events.iter().enumerate() {
        let expected_sequence = u64::try_from(index)
            .map_err(|_| SimError::invalid_trace("event index does not fit in u64"))?;
        if event.sequence() != expected_sequence {
            return Err(SimError::invalid_trace(format!(
                "event {index} must have sequence {expected_sequence}, found {}",
                event.sequence()
            )));
        }

        match event {
            TraceEvent::Demand {
                request,
                step,
                page,
                ..
            } => {
                if !page_index.contains_key(page) {
                    return Err(SimError::invalid_trace(format!(
                        "demand event {index} references unknown page {}",
                        page.0
                    )));
                }
                if latest_realized_step
                    .get(request)
                    .is_some_and(|latest| step < latest)
                {
                    return Err(SimError::invalid_trace(format!(
                        "request {request} demand step {step} regresses from its latest realized step"
                    )));
                }
                latest_realized_step.insert(*request, *step);
                realized_steps.insert((*request, *step));
            }
            TraceEvent::RouterSignal {
                request,
                target_step,
                layer,
                predictions,
                ..
            } => {
                if latest_realized_step
                    .get(request)
                    .is_some_and(|latest| target_step <= latest)
                {
                    return Err(SimError::invalid_trace(format!(
                        "request {request} router target step {target_step} is not later than its latest realized step"
                    )));
                }
                if !signal_targets.insert((*request, *target_step, *layer)) {
                    return Err(SimError::invalid_trace(format!(
                        "duplicate router signal target ({request}, {target_step}, {layer})"
                    )));
                }
                validate_predictions(index, *layer, predictions, expert_pages, limits)?;
            }
        }
    }
    for (request, target_step, layer) in signal_targets {
        if !realized_steps.contains(&(request, target_step)) {
            return Err(SimError::invalid_trace(format!(
                "router signal target ({request}, {target_step}, {layer}) has no later realized demand"
            )));
        }
    }
    Ok(())
}

fn validate_predictions(
    event_index: usize,
    layer: u32,
    predictions: &[ExpertPrediction],
    expert_pages: &ExpertPageIndex,
    limits: SimLimits,
) -> Result<(), SimError> {
    if predictions.len() > limits.max_predictions_per_signal {
        return Err(SimError::invalid_trace(format!(
            "router event {event_index} contains {} predictions, limit is {}",
            predictions.len(),
            limits.max_predictions_per_signal
        )));
    }

    let mut experts = BTreeSet::new();
    let mut score_sum = 0_u64;
    for prediction in predictions {
        if u64::from(prediction.score_ppm) > SCORE_SCALE_PPM {
            return Err(SimError::invalid_trace(format!(
                "router event {event_index} score exceeds 1,000,000 ppm"
            )));
        }
        score_sum = score_sum
            .checked_add(u64::from(prediction.score_ppm))
            .ok_or_else(|| SimError::invalid_trace("router score sum overflow"))?;
        if score_sum > SCORE_SCALE_PPM {
            return Err(SimError::invalid_trace(format!(
                "router event {event_index} scores sum above 1,000,000 ppm"
            )));
        }
        if !experts.insert(prediction.expert) {
            return Err(SimError::invalid_trace(format!(
                "router event {event_index} repeats expert {}",
                prediction.expert
            )));
        }
        if !expert_pages.contains_key(&(layer, prediction.expert)) {
            return Err(SimError::invalid_trace(format!(
                "router event {event_index} references expert {} absent from layer {layer}",
                prediction.expert
            )));
        }
    }
    Ok(())
}

fn append_record<T: Serialize>(
    output: &mut Vec<u8>,
    value: &T,
    limits: SimLimits,
) -> Result<(), SimError> {
    let line = serde_json::to_vec(value)?;
    if line.len() > limits.max_line_bytes {
        return Err(SimError::invalid_trace(format!(
            "serialized record is {} bytes, line limit is {}",
            line.len(),
            limits.max_line_bytes
        )));
    }
    let new_len = output
        .len()
        .checked_add(line.len())
        .and_then(|length| length.checked_add(1))
        .ok_or_else(|| SimError::invalid_trace("serialized trace length overflow"))?;
    if new_len > limits.max_trace_bytes {
        return Err(SimError::invalid_trace(format!(
            "serialized trace exceeds {} byte limit",
            limits.max_trace_bytes
        )));
    }
    output.extend_from_slice(&line);
    output.push(b'\n');
    Ok(())
}
