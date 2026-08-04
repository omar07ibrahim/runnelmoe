use std::collections::BTreeMap;
use std::fmt::Write as _;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;
use sha2::{Digest, Sha256};

use runnel_fixture::FixtureArtifact;

const ACTION_DOMAIN: &[u8] = b"runnel-m5-actor-stress-v1\0";
const DESCRIPTOR_DOMAIN: &[u8] = b"runnel-m5-actor-request-v1\0";
const FIXTURE_BYTES: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../fixtures/scheduler/actor-stress-v1.json"
));
const TINY_V3_SPEC_BYTES: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../fixtures/tiny-v3/spec.json"
));
const FIXTURE_FILE_DIGEST: &str =
    "sha256:eca1faeee91a41d19d98be7ffdad6fc5cebb9027f3e7a634c01ea1cc394fb574";
const FIXTURE_ID: &str = "sha256:5010492fb74eda207511b26811992ed4779814185b9f184663b37a37747bd051";

#[derive(Clone, Debug, Eq, PartialEq)]
struct NullableU32(Option<u32>);

impl NullableU32 {
    const fn some(value: u32) -> Self {
        Self(Some(value))
    }

    const fn null() -> Self {
        Self(None)
    }
}

impl Serialize for NullableU32 {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.0.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for NullableU32 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Option::<u32>::deserialize(deserializer).map(Self)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
enum Sampling {
    Greedy,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct Descriptor {
    deadline_ns: NullableU32,
    index: u32,
    max_new_tokens: u32,
    prompt: Vec<u32>,
    sampling: Sampling,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "lowercase", deny_unknown_fields)]
enum Action {
    Submit {
        exhausted: bool,
        ordinal: u32,
        producer: u32,
        request_index: NullableU32,
        submit_attempt: u32,
    },
    Cancel {
        ordinal: u32,
        producer: u32,
        request_index: u32,
    },
    Drop {
        ordinal: u32,
        producer: u32,
        request_index: u32,
    },
    Drain {
        ordinal: u32,
        producer: u32,
        request_index: u32,
    },
    Wake {
        ordinal: u32,
        producer: u32,
        selector_index: u32,
    },
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct ByKind {
    cancel: usize,
    drain: usize,
    drop: usize,
    submit: usize,
    wake: usize,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct ActionCounts {
    by_kind: ByKind,
    exhausted_submits: usize,
    total: usize,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct ActionVectors {
    count: usize,
    digest: String,
    first: Vec<Action>,
    last: Vec<Action>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct ActorConfig {
    adapter: String,
    admission_reserve_bytes: u64,
    backend: String,
    batch_width: usize,
    command_capacity: usize,
    logical_memory_limit_bytes: u64,
    max_active_requests: usize,
    max_context_tokens: usize,
    max_new_tokens: usize,
    max_outstanding_requests: usize,
    max_prompt_tokens: usize,
    max_queued_requests: usize,
    max_retained_terminal_results: usize,
    model_identity: ModelIdentity,
    output_capacity_per_request: usize,
    page_pool_partition_bytes: u64,
    state_page_tokens: usize,
    trace_capacity: usize,
    waves_per_step: usize,
    worker_count: usize,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct ModelIdentity {
    artifact_id: String,
    object_digest: String,
    object_length: u64,
    page_table_digest: String,
    page_table_length: u64,
    spec_file_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct Algorithms {
    action_generation: String,
    descriptor_generation: String,
    producer_assignment: String,
    sequence_digest: String,
    unbiased_mapping: String,
    word_stream: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct DescriptorVectors {
    count: usize,
    digest: String,
    first: Vec<Descriptor>,
    last: Vec<Descriptor>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct Domains {
    action_words: String,
    request_descriptors: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct KindCodes {
    #[serde(rename = "0")]
    zero: String,
    #[serde(rename = "1")]
    one: String,
    #[serde(rename = "2")]
    two: String,
    #[serde(rename = "3")]
    three: String,
    #[serde(rename = "4")]
    four: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct Parameters {
    action_count: usize,
    kind_bound: u64,
    pin_count: usize,
    request_count: usize,
    selector_bound: u64,
    words_per_digest: usize,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct ProducerCounts {
    producer_0: usize,
    producer_1: usize,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct Fixture {
    action_counts: ActionCounts,
    action_vectors: ActionVectors,
    actor_config: ActorConfig,
    algorithms: Algorithms,
    descriptor_vectors: DescriptorVectors,
    domains: Domains,
    fixture_id: String,
    kind_codes: KindCodes,
    parameters: Parameters,
    producer_counts: ProducerCounts,
    schema: String,
    specification: String,
}

struct WordStream {
    counter: u64,
    next_word: usize,
    words: [u64; 4],
}

impl WordStream {
    const fn new() -> Self {
        Self {
            counter: 0,
            next_word: 4,
            words: [0; 4],
        }
    }

    fn next(&mut self) -> u64 {
        if self.next_word == self.words.len() {
            let mut hasher = Sha256::new();
            hasher.update(ACTION_DOMAIN);
            hasher.update(self.counter.to_le_bytes());
            let digest = hasher.finalize();
            for (index, word) in self.words.iter_mut().enumerate() {
                let offset = index * size_of::<u64>();
                *word = u64::from_le_bytes(
                    digest[offset..offset + size_of::<u64>()]
                        .try_into()
                        .expect("SHA-256 word has eight bytes"),
                );
            }
            self.counter = self
                .counter
                .checked_add(1)
                .expect("the finite corpus cannot exhaust its word counter");
            self.next_word = 0;
        }

        let word = self.words[self.next_word];
        self.next_word += 1;
        word
    }

    fn sample(&mut self, bound: u64) -> u64 {
        assert_ne!(bound, 0, "sampling bound must be nonzero");
        let range = 1_u128 << 64;
        let cutoff = range / u128::from(bound) * u128::from(bound);
        loop {
            let word = self.next();
            if u128::from(word) < cutoff {
                return word % bound;
            }
        }
    }
}

fn descriptors(count: usize) -> Vec<Descriptor> {
    (0..count)
        .map(|index| {
            let index = u32::try_from(index).expect("fixture request index fits u32");
            let mut hasher = Sha256::new();
            hasher.update(DESCRIPTOR_DOMAIN);
            hasher.update(index.to_le_bytes());
            let digest = hasher.finalize();
            let prompt_len = usize::from(1 + digest[0] % 4);
            let prompt = digest[1..=prompt_len]
                .iter()
                .map(|byte| 1 + u32::from(*byte) % 31)
                .collect();
            let remaining_context =
                u8::try_from(17 - prompt_len).expect("descriptor context bound fits u8");
            let max_new_tokens = 1 + u32::from(digest[5] % remaining_context);
            Descriptor {
                deadline_ns: NullableU32::null(),
                index,
                max_new_tokens,
                prompt,
                sampling: Sampling::Greedy,
            }
        })
        .collect()
}

fn actions(count: usize) -> Vec<Action> {
    let mut stream = WordStream::new();
    let mut submit_attempt = 0_u32;
    (0..count)
        .map(|ordinal| {
            let ordinal = u32::try_from(ordinal).expect("fixture action ordinal fits u32");
            match stream.sample(5) {
                0 => {
                    let request_index = if submit_attempt < 64 {
                        NullableU32::some(submit_attempt)
                    } else {
                        NullableU32::null()
                    };
                    let action = Action::Submit {
                        exhausted: request_index.0.is_none(),
                        ordinal,
                        producer: submit_attempt % 2,
                        request_index,
                        submit_attempt,
                    };
                    submit_attempt = submit_attempt
                        .checked_add(1)
                        .expect("finite corpus cannot exhaust submit attempts");
                    action
                }
                kind => {
                    let request_index = u32::try_from(stream.sample(64))
                        .expect("selector bound guarantees a u32 value");
                    match kind {
                        1 => Action::Cancel {
                            ordinal,
                            producer: 1 - request_index % 2,
                            request_index,
                        },
                        2 => Action::Drop {
                            ordinal,
                            producer: request_index % 2,
                            request_index,
                        },
                        3 => Action::Drain {
                            ordinal,
                            producer: request_index % 2,
                            request_index,
                        },
                        4 => Action::Wake {
                            ordinal,
                            producer: ordinal % 2,
                            selector_index: request_index,
                        },
                        _ => unreachable!("sampled kind is below five"),
                    }
                }
            }
        })
        .collect()
}

fn sort_json(value: &mut Value) {
    match value {
        Value::Array(elements) => {
            for element in elements {
                sort_json(element);
            }
        }
        Value::Object(object) => {
            let mut entries: Vec<_> = std::mem::take(object).into_iter().collect();
            entries.sort_unstable_by(|left, right| left.0.cmp(&right.0));
            for (key, mut child) in entries {
                sort_json(&mut child);
                object.insert(key, child);
            }
        }
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {}
    }
}

fn canonical_json_ascii<T: Serialize>(value: &T) -> Vec<u8> {
    let mut value = serde_json::to_value(value).expect("test corpus must serialize");
    sort_json(&mut value);
    let mut bytes = serde_json::to_vec_pretty(&value).expect("test corpus must encode as JSON");
    assert!(bytes.is_ascii(), "canonical corpus JSON must be ASCII");
    bytes.push(b'\n');
    bytes
}

fn sha256_label(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut label = String::with_capacity("sha256:".len() + digest.len() * 2);
    label.push_str("sha256:");
    for byte in digest {
        write!(&mut label, "{byte:02x}").expect("writing to a String cannot fail");
    }
    label
}

fn expected_actor_config() -> ActorConfig {
    let fixture_identity = FixtureArtifact::build_v3().identity();
    ActorConfig {
        adapter: "tiny-v3".into(),
        admission_reserve_bytes: 1_048_576,
        backend: "scalar".into(),
        batch_width: 8,
        command_capacity: 8,
        logical_memory_limit_bytes: 8_388_608,
        max_active_requests: 8,
        max_context_tokens: 19,
        max_new_tokens: 16,
        max_outstanding_requests: 16,
        max_prompt_tokens: 4,
        max_queued_requests: 16,
        max_retained_terminal_results: 16,
        model_identity: ModelIdentity {
            artifact_id: fixture_identity.artifact_id.to_string(),
            object_digest: fixture_identity.object_digest.to_string(),
            object_length: fixture_identity.object_length,
            page_table_digest: fixture_identity.page_table_digest.to_string(),
            page_table_length: fixture_identity.page_table_length,
            spec_file_sha256: sha256_label(TINY_V3_SPEC_BYTES),
        },
        output_capacity_per_request: 2,
        page_pool_partition_bytes: 0,
        state_page_tokens: 4,
        trace_capacity: 1_024,
        waves_per_step: 4,
        worker_count: 1,
    }
}

fn action_statistics(actions: &[Action]) -> (ActionCounts, ProducerCounts) {
    let mut by_kind = ByKind {
        cancel: 0,
        drain: 0,
        drop: 0,
        submit: 0,
        wake: 0,
    };
    let mut exhausted_submits = 0;
    let mut producers = [0_usize; 2];

    for action in actions {
        let (producer, exhausted) = match action {
            Action::Submit {
                exhausted,
                producer,
                ..
            } => {
                by_kind.submit += 1;
                (*producer, *exhausted)
            }
            Action::Cancel { producer, .. } => {
                by_kind.cancel += 1;
                (*producer, false)
            }
            Action::Drop { producer, .. } => {
                by_kind.drop += 1;
                (*producer, false)
            }
            Action::Drain { producer, .. } => {
                by_kind.drain += 1;
                (*producer, false)
            }
            Action::Wake { producer, .. } => {
                by_kind.wake += 1;
                (*producer, false)
            }
        };
        let producer = usize::try_from(producer).expect("producer index fits usize");
        producers[producer] += 1;
        exhausted_submits += usize::from(exhausted);
    }

    (
        ActionCounts {
            by_kind,
            exhausted_submits,
            total: actions.len(),
        },
        ProducerCounts {
            producer_0: producers[0],
            producer_1: producers[1],
        },
    )
}

#[test]
fn committed_actor_stress_fixture_matches_the_independent_contract() {
    let fixture: Fixture =
        serde_json::from_slice(FIXTURE_BYTES).expect("fixture must match the closed Rust schema");

    assert_eq!(sha256_label(FIXTURE_BYTES), FIXTURE_FILE_DIGEST);
    assert_eq!(canonical_json_ascii(&fixture), FIXTURE_BYTES);
    assert_eq!(fixture.fixture_id, FIXTURE_ID);

    let mut identity_value = serde_json::to_value(&fixture).expect("fixture must serialize");
    identity_value
        .as_object_mut()
        .expect("fixture is a JSON object")
        .remove("fixture_id")
        .expect("fixture identity field is present");
    assert_eq!(
        sha256_label(&canonical_json_ascii(&identity_value)),
        FIXTURE_ID
    );

    assert_eq!(fixture.schema, "runnel.actor-stress-vectors/2");
    assert_eq!(fixture.specification, "runnel-m5-actor-stress-v1");
    assert_eq!(fixture.domains.action_words.as_bytes(), ACTION_DOMAIN);
    assert_eq!(
        fixture.domains.request_descriptors.as_bytes(),
        DESCRIPTOR_DOMAIN
    );
    assert_eq!(
        fixture.kind_codes,
        KindCodes {
            zero: "submit".into(),
            one: "cancel".into(),
            two: "drop".into(),
            three: "drain".into(),
            four: "wake".into(),
        }
    );
    assert_eq!(
        fixture.parameters,
        Parameters {
            action_count: 1_024,
            kind_bound: 5,
            pin_count: 4,
            request_count: 64,
            selector_bound: 64,
            words_per_digest: 4,
        }
    );
    assert_eq!(fixture.actor_config, expected_actor_config());
    assert_eq!(
        fixture.algorithms,
        Algorithms {
            action_generation: "unbiased-kind-then-selector-submit-cursor-v1".into(),
            descriptor_generation: "sha256-domain-u32le-index-byte-formulas-v1".into(),
            producer_assignment:
                "submit-attempt-home-drop-drain-home-cancel-opposite-wake-ordinal-v1".into(),
            sequence_digest: "sha256-canonical-json-ascii-v1".into(),
            unbiased_mapping: "reject-ge-floor-2^64-over-bound-times-bound-v1".into(),
            word_stream: "sha256-domain-u64le-counter-four-u64le-byte-order-v1".into(),
        }
    );

    let descriptors = descriptors(fixture.parameters.request_count);
    assert_eq!(fixture.descriptor_vectors.count, descriptors.len());
    let maximum_output_publications = descriptors
        .iter()
        .map(|descriptor| descriptor.max_new_tokens)
        .sum::<u32>();
    assert_eq!(maximum_output_publications, 547);
    assert_eq!(
        maximum_output_publications
            + 2 * u32::try_from(descriptors.len()).expect("descriptor count fits u32"),
        675,
        "bounded observer lifetime covers output, terminal, and first-EOF records"
    );
    assert_eq!(
        fixture.descriptor_vectors.digest,
        sha256_label(&canonical_json_ascii(&descriptors))
    );
    assert_eq!(fixture.descriptor_vectors.first, descriptors[..4]);
    assert_eq!(fixture.descriptor_vectors.last, descriptors[60..]);

    let actions = actions(fixture.parameters.action_count);
    assert_eq!(fixture.action_vectors.count, actions.len());
    assert_eq!(
        fixture.action_vectors.digest,
        sha256_label(&canonical_json_ascii(&actions))
    );
    assert_eq!(fixture.action_vectors.first, actions[..4]);
    assert_eq!(fixture.action_vectors.last, actions[1_020..]);

    let (action_counts, producer_counts) = action_statistics(&actions);
    assert_eq!(fixture.action_counts, action_counts);
    assert_eq!(fixture.producer_counts, producer_counts);
}

#[test]
fn closed_schema_rejects_unknown_field_mutations() {
    let fixture: Value = serde_json::from_slice(FIXTURE_BYTES).expect("fixture JSON must parse");

    let mut root_mutation = fixture.clone();
    root_mutation
        .as_object_mut()
        .expect("fixture is an object")
        .insert("unknown_root".into(), Value::Bool(true));
    assert!(serde_json::from_value::<Fixture>(root_mutation).is_err());

    let mut config_mutation = fixture.clone();
    config_mutation
        .get_mut("actor_config")
        .and_then(Value::as_object_mut)
        .expect("actor config is an object")
        .insert("unknown_limit".into(), Value::from(1));
    assert!(serde_json::from_value::<Fixture>(config_mutation).is_err());

    let mut identity_mutation = fixture.clone();
    identity_mutation
        .pointer_mut("/actor_config/model_identity")
        .and_then(Value::as_object_mut)
        .expect("model identity is an object")
        .insert("unknown_identity".into(), Value::Null);
    assert!(serde_json::from_value::<Fixture>(identity_mutation).is_err());

    for pointer in [
        "/actor_config/model_identity/object_length",
        "/actor_config/model_identity/page_table_length",
    ] {
        let mut boolean_length = fixture.clone();
        *boolean_length
            .pointer_mut(pointer)
            .expect("model length is present") = Value::Bool(true);
        assert!(serde_json::from_value::<Fixture>(boolean_length).is_err());
    }

    let mut vector_mutation = fixture;
    vector_mutation
        .pointer_mut("/action_vectors/first/0")
        .and_then(Value::as_object_mut)
        .expect("first action is an object")
        .insert("unknown_action_field".into(), Value::Null);
    assert!(serde_json::from_value::<Fixture>(vector_mutation).is_err());
}

#[test]
fn rejection_sampler_discards_values_at_or_above_the_exact_cutoff() {
    fn accepts(word: u64, bound: u64) -> bool {
        let range = 1_u128 << 64;
        let cutoff = range / u128::from(bound) * u128::from(bound);
        u128::from(word) < cutoff
    }

    assert!(accepts(u64::MAX - 1, 5));
    assert!(!accepts(u64::MAX, 5));
    assert!(accepts(u64::MAX, 64));
}

#[test]
fn canonical_object_writer_sorts_keys_recursively() {
    let value = BTreeMap::from([
        ("z", Value::from(1)),
        ("a", serde_json::json!({ "z": 2, "a": 3 })),
    ]);
    assert_eq!(
        canonical_json_ascii(&value),
        b"{\n  \"a\": {\n    \"a\": 3,\n    \"z\": 2\n  },\n  \"z\": 1\n}\n"
    );
}
