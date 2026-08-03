use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use serde::de::{self, DeserializeSeed, MapAccess, SeqAccess, Visitor};

use crate::error::FormatError;

pub(crate) const MAX_SAFE_INTEGER: u64 = 9_007_199_254_740_991;
pub(crate) const MAX_NESTING_DEPTH: usize = 12;
pub(crate) const MAX_SCHEMA_STRING_BYTES: usize = 256;
const MAX_JSON_COLLECTION_ENTRIES: usize = 65_536;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum JsonValue {
    Bool(bool),
    Integer(u64),
    String(String),
    Array(Vec<Self>),
    Object(BTreeMap<String, Self>),
}

#[derive(Clone, Debug)]
enum JsonIssue {
    DuplicateKey(String),
    NestingTooDeep,
    Null,
    FloatingPoint,
    IntegerOutOfRange,
    StringTooLong(usize),
    NonAsciiString,
    CollectionTooLarge,
}

#[derive(Default)]
struct ParseState {
    issue: std::cell::RefCell<Option<JsonIssue>>,
}

impl ParseState {
    fn fail<E: de::Error>(&self, issue: JsonIssue) -> E {
        let mut stored = self.issue.borrow_mut();
        if stored.is_none() {
            *stored = Some(issue);
        }
        E::custom("RMOA JSON policy violation")
    }

    fn validate_string<E: de::Error>(&self, value: &str) -> Result<(), E> {
        if value.len() > MAX_SCHEMA_STRING_BYTES {
            return Err(self.fail(JsonIssue::StringTooLong(value.len())));
        }
        if !value.is_ascii() {
            return Err(self.fail(JsonIssue::NonAsciiString));
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct ValueSeed<'a> {
    state: &'a ParseState,
    depth: usize,
}

impl<'de> DeserializeSeed<'de> for ValueSeed<'_> {
    type Value = JsonValue;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_any(ValueVisitor {
            state: self.state,
            depth: self.depth,
        })
    }
}

struct ValueVisitor<'a> {
    state: &'a ParseState,
    depth: usize,
}

impl<'de> Visitor<'de> for ValueVisitor<'_> {
    type Value = JsonValue;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("an RMOA JSON value")
    }

    fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
        Ok(JsonValue::Bool(value))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        if value > MAX_SAFE_INTEGER {
            return Err(self.state.fail(JsonIssue::IntegerOutOfRange));
        }
        Ok(JsonValue::Integer(value))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        let value =
            u64::try_from(value).map_err(|_| self.state.fail::<E>(JsonIssue::IntegerOutOfRange))?;
        self.visit_u64(value)
    }

    fn visit_u128<E>(self, value: u128) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        let value =
            u64::try_from(value).map_err(|_| self.state.fail::<E>(JsonIssue::IntegerOutOfRange))?;
        self.visit_u64(value)
    }

    fn visit_i128<E>(self, value: i128) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        let value =
            u64::try_from(value).map_err(|_| self.state.fail::<E>(JsonIssue::IntegerOutOfRange))?;
        self.visit_u64(value)
    }

    fn visit_f64<E>(self, _value: f64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Err(self.state.fail(JsonIssue::FloatingPoint))
    }

    fn visit_none<E>(self) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Err(self.state.fail(JsonIssue::Null))
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Err(self.state.fail(JsonIssue::Null))
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        self.state.validate_string(value)?;
        Ok(JsonValue::String(value.to_owned()))
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        self.state.validate_string(&value)?;
        Ok(JsonValue::String(value))
    }

    fn visit_seq<A>(self, mut values: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        if self.depth > MAX_NESTING_DEPTH {
            return Err(self.state.fail(JsonIssue::NestingTooDeep));
        }
        let mut result = Vec::new();
        while let Some(value) = values.next_element_seed(ValueSeed {
            state: self.state,
            depth: self.depth + 1,
        })? {
            if result.len() == MAX_JSON_COLLECTION_ENTRIES {
                return Err(self.state.fail(JsonIssue::CollectionTooLarge));
            }
            result.push(value);
        }
        Ok(JsonValue::Array(result))
    }

    fn visit_map<A>(self, mut values: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        if self.depth > MAX_NESTING_DEPTH {
            return Err(self.state.fail(JsonIssue::NestingTooDeep));
        }
        let mut result = BTreeMap::new();
        let mut keys = BTreeSet::new();
        while let Some(key) = values.next_key::<String>()? {
            self.state.validate_string(&key)?;
            if !keys.insert(key.clone()) {
                return Err(self.state.fail(JsonIssue::DuplicateKey(key)));
            }
            if result.len() == MAX_JSON_COLLECTION_ENTRIES {
                return Err(self.state.fail(JsonIssue::CollectionTooLarge));
            }
            let value = values.next_value_seed(ValueSeed {
                state: self.state,
                depth: self.depth + 1,
            })?;
            result.insert(key, value);
        }
        Ok(JsonValue::Object(result))
    }
}

pub(crate) fn parse(bytes: &[u8]) -> Result<JsonValue, FormatError> {
    let state = ParseState::default();
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    let parsed = ValueSeed {
        state: &state,
        depth: 1,
    }
    .deserialize(&mut deserializer);

    match parsed.and_then(|value| deserializer.end().map(|()| value)) {
        Ok(value) => Ok(value),
        Err(error) => {
            if let Some(issue) = state.issue.into_inner() {
                return Err(match issue {
                    JsonIssue::DuplicateKey(key) => FormatError::DuplicateKey { key },
                    JsonIssue::NestingTooDeep => FormatError::NestingTooDeep {
                        limit: MAX_NESTING_DEPTH,
                    },
                    JsonIssue::Null => FormatError::NullNotAllowed,
                    JsonIssue::FloatingPoint => FormatError::FloatingPointNotAllowed,
                    JsonIssue::IntegerOutOfRange => FormatError::IntegerOutOfRange {
                        max: MAX_SAFE_INTEGER,
                    },
                    JsonIssue::StringTooLong(actual) => FormatError::StringTooLong {
                        actual,
                        limit: MAX_SCHEMA_STRING_BYTES,
                    },
                    JsonIssue::NonAsciiString => FormatError::NonAsciiString,
                    JsonIssue::CollectionTooLarge => FormatError::CountLimit {
                        field: "JSON collection",
                        actual: MAX_JSON_COLLECTION_ENTRIES + 1,
                        limit: MAX_JSON_COLLECTION_ENTRIES,
                    },
                });
            }
            Err(FormatError::JsonSyntax {
                line: error.line(),
                column: error.column(),
                message: error.to_string(),
            })
        }
    }
}

/// Serializes the ASCII-only subset of RFC 8785 used by the M1 schema.
pub(crate) fn canonical_bytes(value: &JsonValue) -> Vec<u8> {
    let mut output = Vec::new();
    write_value(value, &mut output);
    output.push(b'\n');
    output
}

fn write_value(value: &JsonValue, output: &mut Vec<u8>) {
    match value {
        JsonValue::Bool(value) => output.extend_from_slice(if *value { b"true" } else { b"false" }),
        JsonValue::Integer(value) => output.extend_from_slice(value.to_string().as_bytes()),
        JsonValue::String(value) => write_string(value, output),
        JsonValue::Array(values) => {
            output.push(b'[');
            for (index, value) in values.iter().enumerate() {
                if index != 0 {
                    output.push(b',');
                }
                write_value(value, output);
            }
            output.push(b']');
        }
        JsonValue::Object(values) => {
            output.push(b'{');
            for (index, (key, value)) in values.iter().enumerate() {
                if index != 0 {
                    output.push(b',');
                }
                write_string(key, output);
                output.push(b':');
                write_value(value, output);
            }
            output.push(b'}');
        }
    }
}

fn write_string(value: &str, output: &mut Vec<u8>) {
    const HEX: &[u8; 16] = b"0123456789abcdef";

    output.push(b'"');
    for byte in value.bytes() {
        match byte {
            b'"' => output.extend_from_slice(br#"\""#),
            b'\\' => output.extend_from_slice(br#"\\"#),
            0x08 => output.extend_from_slice(br"\b"),
            0x09 => output.extend_from_slice(br"\t"),
            0x0a => output.extend_from_slice(br"\n"),
            0x0c => output.extend_from_slice(br"\f"),
            0x0d => output.extend_from_slice(br"\r"),
            0x00..=0x1f => {
                output.extend_from_slice(br"\u00");
                output.push(HEX[usize::from(byte >> 4)]);
                output.push(HEX[usize::from(byte & 0x0f)]);
            }
            _ => output.push(byte),
        }
    }
    output.push(b'"');
}
