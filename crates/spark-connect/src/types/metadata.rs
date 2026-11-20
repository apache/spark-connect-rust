// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! The metadata of a struct field, and the JSON of Python's `json` module, of which PySpark writes
//! and reads it.

use std::collections::{BTreeMap, HashMap};
use std::fmt;

use spark_connect_core::error::{Result, SparkError};

/// The deepest nesting of arrays and objects that [`Json::parse`] reads: that of the default
/// recursion limit of Python, which its `json.dumps` cannot write deeper, and of the JSON reader of
/// the server.
const MAX_DEPTH: usize = 1000;

/// A JSON value as Python's `json` module reads and writes it, of the values that PySpark's
/// metadata holds: an int of any size, a float of its value (also `NaN`, `Infinity` and
/// `-Infinity`, which Python's JSON writes and reads), and objects of their keys in their order,
/// as a dict keeps them, a key more than once too, as `json.dumps` writes `{1: "a", "1": "b"}`.
#[derive(Debug, Clone)]
pub(crate) enum Json {
    Null,
    Bool(bool),
    /// An int, as its decimal digits, without leading zeros or the sign of a zero.
    Int(String),
    Float(f64),
    Str(String),
    Array(Vec<Json>),
    Object(Vec<(String, Json)>),
    /// The JSON text of a value, which both writers write as it is: the `json()` of a type that
    /// the Python client lowered by its JSON (see [`DataType::json_as_pyspark`](super::DataType)).
    Raw(String),
}

impl Json {
    /// The JSON value of the text `text`, as Python's `json.loads` reads it: standard JSON and the
    /// `NaN`, `Infinity` and `-Infinity` of Python's floats. The error is a message of the place.
    pub(crate) fn parse(text: &str) -> std::result::Result<Json, String> {
        let mut parser = Parser {
            text: text.as_bytes(),
            at: 0,
        };
        parser.skip_whitespace();
        let value = parser.value(0)?;
        parser.skip_whitespace();
        if parser.at != parser.text.len() {
            return Err(parser.error("extra data"));
        }
        Ok(value)
    }

    /// The JSON text of the value as Python's `json.dumps(value)` writes it: of the separators
    /// `", "` and `": "`, of the keys in their order, and of the escapes of its `ensure_ascii`.
    pub(crate) fn write_python(&self, out: &mut String) {
        self.write(out, false);
    }

    /// The JSON text of the value as PySpark's `json()` writes it, of
    /// `json.dumps(value, separators=(",", ":"), sort_keys=True)`: the keys of each object
    /// sorted, a key of several values in the order of its values.
    pub(crate) fn write_sorted(&self, out: &mut String) {
        self.write(out, true);
    }

    fn write(&self, out: &mut String, sorted: bool) {
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(true) => out.push_str("true"),
            Json::Bool(false) => out.push_str("false"),
            Json::Int(digits) => out.push_str(digits),
            Json::Float(value) => out.push_str(&python_float_json(*value)),
            Json::Str(text) => write_python_str(out, text),
            Json::Array(items) => {
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push_str(if sorted { "," } else { ", " });
                    }
                    item.write(out, sorted);
                }
                out.push(']');
            }
            Json::Object(entries) => Json::write_object(out, entries, sorted),
            Json::Raw(text) => out.push_str(text),
        }
    }

    /// Write the object of the members `entries`, as [`Json::write`] writes one.
    fn write_object(out: &mut String, entries: &[(String, Json)], sorted: bool) {
        let mut entries: Vec<&(String, Json)> = entries.iter().collect();
        if sorted {
            // A stable sort of the keys, as `sort_keys` sorts the str keys of a dict: UTF-8
            // orders the code points as Python's comparison of strs does. Python sorts a key that
            // is not a str, such as an int, as it is (`9` before `10`) before `json.dumps` writes
            // it as a str, which the JSON does not keep, so the Python client sends the text of
            // its own `json()` of a type of such keys (`Unparsed`, written as it is).
            entries.sort_by(|a, b| a.0.cmp(&b.0));
        }
        out.push('{');
        for (i, (key, value)) in entries.into_iter().enumerate() {
            if i > 0 {
                out.push_str(if sorted { "," } else { ", " });
            }
            write_python_str(out, key);
            out.push_str(if sorted { ":" } else { ": " });
            value.write(out, sorted);
        }
        out.push('}');
    }

    /// The value as a `serde_json` value, which holds an int of 64 bits and a finite float: a
    /// larger int is its nearest float, as `serde_json` reads its text, and a float that is not
    /// finite is null, as `serde_json` makes a value of it. A key of several values has the last.
    pub(crate) fn to_serde(&self) -> serde_json::Value {
        match self {
            Json::Null => serde_json::Value::Null,
            Json::Bool(flag) => serde_json::Value::Bool(*flag),
            Json::Int(digits) => {
                if let Ok(int) = digits.parse::<i64>() {
                    int.into()
                } else if let Ok(int) = digits.parse::<u64>() {
                    int.into()
                } else {
                    digits
                        .parse::<f64>()
                        .map_or(serde_json::Value::Null, |f| f.into())
                }
            }
            Json::Float(value) => (*value).into(),
            Json::Str(text) => serde_json::Value::String(text.clone()),
            Json::Array(items) => {
                serde_json::Value::Array(items.iter().map(Json::to_serde).collect())
            }
            Json::Object(entries) => serde_json::Value::Object(
                entries
                    .iter()
                    .map(|(key, value)| (key.clone(), value.to_serde()))
                    .collect(),
            ),
            Json::Raw(text) => Json::parse(text).map_or(serde_json::Value::Null, |v| v.to_serde()),
        }
    }

    /// The value of the `serde_json` value `value`, which it holds exactly.
    pub(crate) fn from_serde(value: &serde_json::Value) -> Json {
        match value {
            serde_json::Value::Null => Json::Null,
            serde_json::Value::Bool(flag) => Json::Bool(*flag),
            serde_json::Value::Number(number) => {
                if let Some(int) = number.as_i64() {
                    Json::Int(int.to_string())
                } else if let Some(int) = number.as_u64() {
                    Json::Int(int.to_string())
                } else {
                    Json::Float(number.as_f64().unwrap_or(f64::NAN))
                }
            }
            serde_json::Value::String(text) => Json::Str(text.clone()),
            serde_json::Value::Array(items) => {
                Json::Array(items.iter().map(Json::from_serde).collect())
            }
            serde_json::Value::Object(map) => Json::Object(
                map.iter()
                    .map(|(key, value)| (key.clone(), Json::from_serde(value)))
                    .collect(),
            ),
        }
    }

    /// The value of the object member `key`, its last one, as `json.loads` keeps it.
    pub(crate) fn member(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Object(entries) => entries.iter().rev().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// The value of the object member `key`, its last one, to change.
    pub(crate) fn member_mut(&mut self, key: &str) -> Option<&mut Json> {
        match self {
            Json::Object(entries) => entries
                .iter_mut()
                .rev()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v),
            _ => None,
        }
    }

    /// The entries of an object as the dict of `json.loads` holds them: each key once, at its
    /// first place, of its last value.
    fn dict_entries(entries: &[(String, Json)]) -> Vec<(&str, &Json)> {
        let mut out: Vec<(&str, &Json)> = Vec::with_capacity(entries.len());
        let mut places: HashMap<&str, usize> = HashMap::with_capacity(entries.len());
        for (key, value) in entries {
            match places.get(key.as_str()) {
                Some(&place) => out[place].1 = value,
                None => {
                    places.insert(key, out.len());
                    out.push((key, value));
                }
            }
        }
        out
    }

    /// Whether `self` and `other` are equal as the values of the dicts of `json.loads`: an int
    /// and a float are not, as `serde_json`'s numbers of their types are not, and two `NaN`s are.
    fn same(&self, other: &Json) -> bool {
        match (self, other) {
            (Json::Null, Json::Null) => true,
            (Json::Bool(a), Json::Bool(b)) => a == b,
            (Json::Int(a), Json::Int(b)) => a == b,
            (Json::Float(a), Json::Float(b)) => a == b || (a.is_nan() && b.is_nan()),
            (Json::Str(a), Json::Str(b)) => a == b,
            (Json::Array(a), Json::Array(b)) => {
                a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.same(b))
            }
            (Json::Object(a), Json::Object(b)) => entries_match(a, b, Json::same),
            (Json::Raw(a), Json::Raw(b)) => a == b,
            _ => false,
        }
    }

    /// Whether `self` and `other` are equal as Python's `==` compares the values of `json.loads`
    /// of them: numbers by their values, whatever their types, a bool as the int of its value, and
    /// arrays and objects of equal items. Two `NaN`s are equal, which only the values of one
    /// object are in Python.
    fn eq_as_python(&self, other: &Json) -> bool {
        match (self, other) {
            (Json::Null, Json::Null) => true,
            (Json::Str(a), Json::Str(b)) => a == b,
            (Json::Array(a), Json::Array(b)) => {
                a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.eq_as_python(b))
            }
            (Json::Object(a), Json::Object(b)) => entries_match(a, b, Json::eq_as_python),
            (Json::Raw(a), Json::Raw(b)) => a == b,
            _ => match (self.python_number(), other.python_number()) {
                (Some(a), Some(b)) => a.eq_as_python(&b),
                _ => false,
            },
        }
    }

    /// The number of a number or a bool, as Python compares it.
    fn python_number(&self) -> Option<PythonNumber<'_>> {
        match self {
            Json::Bool(false) => Some(PythonNumber::Int("0")),
            Json::Bool(true) => Some(PythonNumber::Int("1")),
            Json::Int(digits) => Some(PythonNumber::Int(digits)),
            Json::Float(value) => Some(PythonNumber::Float(*value)),
            _ => None,
        }
    }
}

/// Whether the entries `a` and `b` of two objects hold the same keys, of values that `eq` finds
/// equal, as the dicts of `json.loads` of them.
fn entries_match(a: &[(String, Json)], b: &[(String, Json)], eq: fn(&Json, &Json) -> bool) -> bool {
    let a = Json::dict_entries(a);
    // Of the last value of each key, as the dict of `json.loads` holds it.
    let b: HashMap<&str, &Json> = b.iter().map(|(key, value)| (key.as_str(), value)).collect();
    a.len() == b.len()
        && a.iter()
            .all(|(key, value)| b.get(key).is_some_and(|other| eq(value, other)))
}

/// A number as Python compares it: an int, of its digits, or a float.
enum PythonNumber<'a> {
    Int(&'a str),
    Float(f64),
}

impl PythonNumber<'_> {
    /// As Python compares an int and a float, exactly: `2**53 + 1` does not equal
    /// `float(2**53 + 1)`, which is `2**53`, nor `10**30` the float `1e30`.
    fn eq_as_python(&self, other: &PythonNumber<'_>) -> bool {
        match (self, other) {
            (PythonNumber::Int(a), PythonNumber::Int(b)) => a == b,
            (PythonNumber::Float(a), PythonNumber::Float(b)) => {
                a == b || (a.is_nan() && b.is_nan())
            }
            (PythonNumber::Int(int), PythonNumber::Float(float))
            | (PythonNumber::Float(float), PythonNumber::Int(int)) => {
                // The digits of an integral float are exact at a precision of 0, of which only a
                // zero has a sign that an int does not.
                float.is_finite() && float.fract() == 0.0 && {
                    let digits = format!("{float:.0}");
                    digits
                        .strip_prefix("-")
                        .filter(|d| *d == "0")
                        .unwrap_or(&digits)
                        == *int
                }
            }
        }
    }
}

/// The text of the float `value` as Python's `json.dumps` writes it: its `repr`, and `NaN`,
/// `Infinity` or `-Infinity` of a float that is not finite.
fn python_float_json(value: f64) -> String {
    if value.is_nan() {
        "NaN".to_string()
    } else if value.is_infinite() {
        if value > 0.0 { "Infinity" } else { "-Infinity" }.to_string()
    } else {
        crate::expression::python_float_repr(value)
    }
}

/// Write the JSON string of `text` as Python's `json.dumps` writes a str, of its default
/// `ensure_ascii`: the escapes `\"`, `\\`, `\n`, `\r`, `\t`, `\b` and `\f`, and `\uXXXX` in
/// lowercase hex for each other character that is not printable ASCII, `\x7f` too (a surrogate
/// pair above U+FFFF).
fn write_python_str(out: &mut String, text: &str) {
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            ' '..='~' => out.push(c),
            _ => {
                let mut units = [0u16; 2];
                for unit in c.encode_utf16(&mut units) {
                    out.push_str(&format!("\\u{unit:04x}"));
                }
            }
        }
    }
    out.push('"');
}

/// A reader of the JSON of Python's `json.loads`.
struct Parser<'a> {
    text: &'a [u8],
    at: usize,
}

/// The result of a step of [`Parser`], of the message of its error.
type Parsed<T> = std::result::Result<T, String>;

impl Parser<'_> {
    fn error(&self, what: &str) -> String {
        format!("{what}: char {}", self.at)
    }

    fn skip_whitespace(&mut self) {
        while let Some(b' ' | b'\t' | b'\n' | b'\r') = self.text.get(self.at) {
            self.at += 1;
        }
    }

    fn eat(&mut self, literal: &str) -> bool {
        if self.text[self.at..].starts_with(literal.as_bytes()) {
            self.at += literal.len();
            true
        } else {
            false
        }
    }

    fn value(&mut self, depth: usize) -> Parsed<Json> {
        match self.text.get(self.at) {
            Some(b'{') | Some(b'[') if depth >= MAX_DEPTH => Err(self.error("too deep a nesting")),
            Some(b'{') => {
                self.at += 1;
                let mut entries = Vec::new();
                self.skip_whitespace();
                if self.eat("}") {
                    return Ok(Json::Object(entries));
                }
                loop {
                    self.skip_whitespace();
                    if self.text.get(self.at) != Some(&b'"') {
                        return Err(self.error("expecting a property name in double quotes"));
                    }
                    let key = self.string()?;
                    self.skip_whitespace();
                    if !self.eat(":") {
                        return Err(self.error("expecting ':' delimiter"));
                    }
                    self.skip_whitespace();
                    let value = self.value(depth + 1)?;
                    entries.push((key, value));
                    self.skip_whitespace();
                    if self.eat("}") {
                        return Ok(Json::Object(entries));
                    }
                    if !self.eat(",") {
                        return Err(self.error("expecting ',' delimiter"));
                    }
                }
            }
            Some(b'[') => {
                self.at += 1;
                let mut items = Vec::new();
                self.skip_whitespace();
                if self.eat("]") {
                    return Ok(Json::Array(items));
                }
                loop {
                    self.skip_whitespace();
                    items.push(self.value(depth + 1)?);
                    self.skip_whitespace();
                    if self.eat("]") {
                        return Ok(Json::Array(items));
                    }
                    if !self.eat(",") {
                        return Err(self.error("expecting ',' delimiter"));
                    }
                }
            }
            Some(b'"') => Ok(Json::Str(self.string()?)),
            _ if self.eat("null") => Ok(Json::Null),
            _ if self.eat("true") => Ok(Json::Bool(true)),
            _ if self.eat("false") => Ok(Json::Bool(false)),
            _ if self.eat("NaN") => Ok(Json::Float(f64::NAN)),
            _ if self.eat("Infinity") => Ok(Json::Float(f64::INFINITY)),
            _ if self.eat("-Infinity") => Ok(Json::Float(f64::NEG_INFINITY)),
            Some(b'-' | b'0'..=b'9') => self.number(),
            _ => Err(self.error("expecting value")),
        }
    }

    /// A number of the grammar of JSON, `-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][-+]?[0-9]+)?`: an int
    /// of no fraction nor exponent, as Python's `int()` reads it, else a float of `float()`.
    fn number(&mut self) -> Parsed<Json> {
        let start = self.at;
        let digits = |parser: &mut Self| {
            let from = parser.at;
            while let Some(b'0'..=b'9') = parser.text.get(parser.at) {
                parser.at += 1;
            }
            parser.at - from
        };
        self.eat("-");
        match self.text.get(self.at) {
            Some(b'0') => self.at += 1,
            Some(b'1'..=b'9') => {
                digits(self);
            }
            _ => return Err(self.error("expecting value")),
        }
        let mut float = false;
        if self.text.get(self.at) == Some(&b'.')
            && matches!(self.text.get(self.at + 1), Some(b'0'..=b'9'))
        {
            self.at += 1;
            digits(self);
            float = true;
        }
        if let Some(b'e' | b'E') = self.text.get(self.at) {
            let mark = self.at;
            self.at += 1;
            if let Some(b'+' | b'-') = self.text.get(self.at) {
                self.at += 1;
            }
            if digits(self) == 0 {
                // As Python's scanner, whose number ends before an exponent of no digits.
                self.at = mark;
            } else {
                float = true;
            }
        }
        let token = std::str::from_utf8(&self.text[start..self.at])
            .map_err(|_| self.error("invalid number"))?;
        if float {
            let value = token
                .parse::<f64>()
                .map_err(|_| self.error("invalid number"))?;
            return Ok(Json::Float(value));
        }
        // `int("-0")` is 0, which Python writes without a sign.
        Ok(Json::Int(if token == "-0" {
            "0".to_string()
        } else {
            token.to_string()
        }))
    }

    /// A string, of the escapes of JSON, as Python's strict `json.loads` reads it: a control
    /// character must be escaped, and a surrogate pair is the character of the pair. A lone
    /// surrogate, which Python's str holds and UTF-8 does not, is an error.
    fn string(&mut self) -> Parsed<String> {
        self.at += 1;
        let mut out = String::new();
        loop {
            let start = self.at;
            while let Some(&byte) = self.text.get(self.at) {
                if byte == b'"' || byte == b'\\' || byte < 0x20 {
                    break;
                }
                self.at += 1;
            }
            out.push_str(
                std::str::from_utf8(&self.text[start..self.at])
                    .map_err(|_| self.error("invalid UTF-8"))?,
            );
            match self.text.get(self.at) {
                Some(b'"') => {
                    self.at += 1;
                    return Ok(out);
                }
                Some(b'\\') => {
                    self.at += 1;
                    let escape = self.text.get(self.at).copied();
                    self.at += 1;
                    match escape {
                        Some(b'"') => out.push('"'),
                        Some(b'\\') => out.push('\\'),
                        Some(b'/') => out.push('/'),
                        Some(b'b') => out.push('\u{8}'),
                        Some(b'f') => out.push('\u{c}'),
                        Some(b'n') => out.push('\n'),
                        Some(b'r') => out.push('\r'),
                        Some(b't') => out.push('\t'),
                        Some(b'u') => {
                            let unit = self.hex_unit()?;
                            let c = if (0xD800..0xDC00).contains(&unit) && self.eat("\\u") {
                                let low = self.hex_unit()?;
                                if !(0xDC00..0xE000).contains(&low) {
                                    return Err(self.error("a lone surrogate"));
                                }
                                char::from_u32(0x10000 + ((unit - 0xD800) << 10) + (low - 0xDC00))
                            } else {
                                char::from_u32(unit)
                            };
                            out.push(c.ok_or_else(|| self.error("a lone surrogate"))?);
                        }
                        _ => return Err(self.error("invalid \\escape")),
                    }
                }
                Some(_) => return Err(self.error("invalid control character")),
                None => return Err(self.error("unterminated string")),
            }
        }
    }

    /// The four hex digits of a `\u` escape.
    fn hex_unit(&mut self) -> Parsed<u32> {
        let digits = self
            .text
            .get(self.at..self.at + 4)
            .and_then(|digits| std::str::from_utf8(digits).ok())
            .filter(|digits| digits.bytes().all(|b| b.is_ascii_hexdigit()))
            .ok_or_else(|| self.error("invalid \\uXXXX escape"))?;
        self.at += 4;
        u32::from_str_radix(digits, 16).map_err(|_| self.error("invalid \\uXXXX escape"))
    }
}

/// The metadata of a [`StructField`](super::StructField), mirroring the `metadata` dict of PySpark's
/// `StructField`: a JSON object, which keeps its values as Python's `json` module writes and reads
/// them, so that each keeps its value: an int of any size, and a float of its value, also the
/// `NaN`, `Infinity` and `-Infinity` of Python's JSON. Its keys stay in their order, as the dict
/// keeps them.
///
/// A plan sends it as PySpark sends the dict, as the text of `json.dumps(metadata)`
/// ([`Metadata::json`]), and the JSON of a type writes it as PySpark's `json()` does, of its keys
/// sorted. So metadata of the Python client is sent as PySpark sends it, where the server keeps the
/// low 64 bits of an int beyond a long, and refuses a float that is not finite.
///
/// It is made of `serde_json` values, which it holds exactly (`Metadata::from` a map), or of the
/// text of a JSON object ([`Metadata::from_json`]), and [`Metadata::get`] gives a value as a
/// `serde_json` value, which holds an int of 64 bits and a finite float only.
///
/// ```
/// use spark_connect::types::Metadata;
///
/// let metadata = Metadata::from_json(r#"{"comment": "id", "max": 100000000000000000000}"#)?;
/// assert_eq!(metadata.json(), r#"{"comment": "id", "max": 100000000000000000000}"#);
/// assert_eq!(metadata.get("comment"), Some(serde_json::json!("id")));
/// let mut metadata = Metadata::new();
/// metadata.insert("comment", serde_json::json!("id"));
/// assert_eq!(metadata.json(), r#"{"comment": "id"}"#);
/// # Ok::<(), spark_connect::SparkError>(())
/// ```
#[derive(Clone, Default)]
pub struct Metadata {
    /// The members of the object, in their order.
    entries: Vec<(String, Json)>,
}

impl Metadata {
    /// No metadata, as PySpark's `{}`.
    pub fn new() -> Metadata {
        Metadata::default()
    }

    /// The metadata of the JSON text of an object, of standard JSON and of the `NaN`, `Infinity`
    /// and `-Infinity` of Python's JSON, as PySpark's `json.loads` of the metadata of a field
    /// reads it. Its values are kept as they are written, and its keys in their order.
    pub fn from_json(json: &str) -> Result<Metadata> {
        match Json::parse(json) {
            Ok(Json::Object(entries)) => Ok(Metadata { entries }),
            Ok(_) => Err(SparkError::value_msg(format!(
                "the metadata of a field is not a JSON object: {json}"
            ))),
            Err(error) => Err(SparkError::value_msg(format!(
                "the metadata of a field is not JSON: {error}"
            ))),
        }
    }

    /// The JSON text of the metadata as PySpark's `json.dumps(metadata)` writes it, which it sends
    /// in a plan: of the separators `", "` and `": "`, of the keys in their order, of the escapes
    /// of its `ensure_ascii`, of each int as it is and of each float as Python's `repr` writes it,
    /// or as `NaN`, `Infinity` and `-Infinity`.
    pub fn json(&self) -> String {
        let mut out = String::new();
        Json::write_object(&mut out, &self.entries, false);
        out
    }

    /// Whether the metadata has no keys.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The number of the keys of the metadata.
    pub fn len(&self) -> usize {
        Json::dict_entries(&self.entries).len()
    }

    /// The keys of the metadata, in their order, each once.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        Json::dict_entries(&self.entries)
            .into_iter()
            .map(|(key, _)| key)
    }

    /// Whether the metadata has the key `key`.
    pub fn contains_key(&self, key: &str) -> bool {
        self.entries.iter().any(|(k, _)| k == key)
    }

    /// The value of the key `key`, as a `serde_json` value: an int beyond 64 bits is its nearest
    /// float, and a float that is not finite is null (see [`Metadata::get_json`]).
    pub fn get(&self, key: &str) -> Option<serde_json::Value> {
        self.value(key).map(Json::to_serde)
    }

    /// The JSON text of the value of the key `key`, as `json.dumps` writes it, which keeps it
    /// exactly.
    pub fn get_json(&self, key: &str) -> Option<String> {
        self.value(key).map(|value| {
            let mut out = String::new();
            value.write_python(&mut out);
            out
        })
    }

    /// Set the value of the key `key`, at its place if the metadata has it, else after the others,
    /// as a dict sets it, and return the value it had.
    pub fn insert(
        &mut self,
        key: impl Into<String>,
        value: serde_json::Value,
    ) -> Option<serde_json::Value> {
        self.insert_json(key.into(), Json::from_serde(&value))
            .map(|old| old.to_serde())
    }

    /// Remove the key `key`, and return its value.
    pub fn remove(&mut self, key: &str) -> Option<serde_json::Value> {
        self.remove_json(key).map(|old| old.to_serde())
    }

    /// The metadata as a map of `serde_json` values (see [`Metadata::get`]).
    pub fn to_map(&self) -> BTreeMap<String, serde_json::Value> {
        Json::dict_entries(&self.entries)
            .into_iter()
            .map(|(key, value)| (key.to_string(), value.to_serde()))
            .collect()
    }

    /// The value of the key `key`, of its last member, as `json.loads` keeps it.
    fn value(&self, key: &str) -> Option<&Json> {
        self.entries
            .iter()
            .rev()
            .find(|(k, _)| k == key)
            .map(|(_, value)| value)
    }

    /// [`Metadata::insert`] of a JSON value: the key stays at its first place, as the dict of
    /// `json.loads` holds it.
    pub(crate) fn insert_json(&mut self, key: String, value: Json) -> Option<Json> {
        let later = self.remove_later(&key);
        match self.entries.iter_mut().find(|(k, _)| *k == key) {
            Some(entry) => {
                let first = std::mem::replace(&mut entry.1, value);
                Some(later.unwrap_or(first))
            }
            None => {
                self.entries.push((key, value));
                None
            }
        }
    }

    /// [`Metadata::remove`] of a JSON value.
    pub(crate) fn remove_json(&mut self, key: &str) -> Option<Json> {
        let mut old = None;
        self.entries.retain_mut(|(k, value)| {
            if k == key {
                old = Some(std::mem::replace(value, Json::Null));
                false
            } else {
                true
            }
        });
        old
    }

    /// Remove the members of `key` after its first, and return the value of its last.
    fn remove_later(&mut self, key: &str) -> Option<Json> {
        let first = self.entries.iter().position(|(k, _)| k == key)?;
        let mut old = None;
        let mut index = 0;
        self.entries.retain_mut(|(k, value)| {
            let keep = index <= first || k != key;
            if !keep {
                old = Some(std::mem::replace(value, Json::Null));
            }
            index += 1;
            keep
        });
        old
    }

    /// The metadata as the JSON object of its members.
    pub(crate) fn as_object(&self) -> Json {
        Json::Object(self.entries.clone())
    }

    /// The metadata of the members of a JSON object, or none of another value.
    pub(crate) fn of_json(value: Json) -> Option<Metadata> {
        match value {
            Json::Object(entries) => Some(Metadata { entries }),
            _ => None,
        }
    }

    /// Whether the metadata equal `other` as Python's `==` compares the dicts of `json.loads` of
    /// them (see [`DataType::eq_as_python`](super::DataType::eq_as_python)). Not a stable API.
    pub(crate) fn eq_as_python(&self, other: &Metadata) -> bool {
        entries_match(&self.entries, &other.entries, Json::eq_as_python)
    }
}

/// As the dicts of `json.loads` of them compare: of the same keys, in any order, of the same
/// values, of which an int and a float are not, as `serde_json`'s numbers of their types are not.
impl PartialEq for Metadata {
    fn eq(&self, other: &Metadata) -> bool {
        entries_match(&self.entries, &other.entries, Json::same)
    }
}

impl fmt::Debug for Metadata {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Metadata({})", self.json())
    }
}

/// Of the keys of the map, each once, in its order.
impl From<BTreeMap<String, serde_json::Value>> for Metadata {
    fn from(map: BTreeMap<String, serde_json::Value>) -> Metadata {
        let entries = map
            .into_iter()
            .map(|(key, value)| (key, Json::from_serde(&value)))
            .collect();
        Metadata { entries }
    }
}

/// Of the keys of the map, each once, in its order.
impl From<serde_json::Map<String, serde_json::Value>> for Metadata {
    fn from(map: serde_json::Map<String, serde_json::Value>) -> Metadata {
        let entries = map
            .into_iter()
            .map(|(key, value)| (key, Json::from_serde(&value)))
            .collect();
        Metadata { entries }
    }
}

/// As [`Metadata::insert`] of each pair in turn: a key of several pairs is at its first place, of
/// its last value.
impl<K: Into<String>> FromIterator<(K, serde_json::Value)> for Metadata {
    fn from_iter<I: IntoIterator<Item = (K, serde_json::Value)>>(iter: I) -> Metadata {
        let mut entries: Vec<(String, Json)> = Vec::new();
        let mut places: HashMap<String, usize> = HashMap::new();
        for (key, value) in iter {
            let (key, value) = (key.into(), Json::from_serde(&value));
            match places.get(&key) {
                Some(&place) => entries[place].1 = value,
                None => {
                    places.insert(key.clone(), entries.len());
                    entries.push((key, value));
                }
            }
        }
        Metadata { entries }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parsed(text: &str) -> Metadata {
        Metadata::from_json(text).unwrap()
    }

    #[test]
    fn json_is_pythons_json_dumps_of_the_dict() {
        // As `json.dumps` of the dicts that `json.loads` makes of these, of PySpark's plans.
        for (text, python) in [
            ("{}", "{}"),
            (r#"{"x":1}"#, r#"{"x": 1}"#),
            (
                r#"{"x": 1000000000000000000000000000000, "f": 1e-07, "b": [1, 1180591620717411303424]}"#,
                r#"{"x": 1000000000000000000000000000000, "f": 1e-07, "b": [1, 1180591620717411303424]}"#,
            ),
            (
                r#"{"b":1,"a":{"d":2,"c":[]}}"#,
                r#"{"b": 1, "a": {"d": 2, "c": []}}"#,
            ),
            (
                r#"{"x": NaN, "y": [Infinity, -Infinity]}"#,
                r#"{"x": NaN, "y": [Infinity, -Infinity]}"#,
            ),
            (
                r#"{"x":1.0E-7,"y":1E16,"z":-0.0,"w":-0,"v":0.1}"#,
                r#"{"x": 1e-07, "y": 1e+16, "z": -0.0, "w": 0, "v": 0.1}"#,
            ),
            (
                r#"{"k":"é中😀\u007f\n\"\\/"}"#,
                r#"{"k": "\u00e9\u4e2d\ud83d\ude00\u007f\n\"\\/"}"#,
            ),
            (r#"{"1": "a", "1": "b"}"#, r#"{"1": "a", "1": "b"}"#),
            (
                r#" { "a" : [ true , false , null ] } "#,
                r#"{"a": [true, false, null]}"#,
            ),
            (
                r#"{"big": 1e400, "e": 1e5}"#,
                r#"{"big": Infinity, "e": 100000.0}"#,
            ),
        ] {
            assert_eq!(parsed(text).json(), python, "{text}");
        }
    }

    #[test]
    fn write_sorted_is_pysparks_json_of_the_dict() {
        // As `json.dumps(value, separators=(",", ":"), sort_keys=True)`.
        let mut out = String::new();
        parsed(r#"{"b": 1, "a": {"d": 2, "c": [1.5, 100000000000000000000]}, "é": "é"}"#)
            .as_object()
            .write_sorted(&mut out);
        assert_eq!(
            out,
            r#"{"a":{"c":[1.5,100000000000000000000],"d":2},"b":1,"\u00e9":"\u00e9"}"#
        );
    }

    #[test]
    fn from_json_refuses_what_python_does_not_read() {
        for text in [
            "[1]",
            "1",
            "{",
            r#"{"a" 1}"#,
            r#"{"a": 1,}"#,
            r#"{"a": 01}"#,
            r#"{"a": .5}"#,
            r#"{"a": +1}"#,
            r#"{"a": 1.}"#,
            r#"{"a": 1e}"#,
            r#"{"a": nan}"#,
            r#"{"a": "\ud800"}"#,
            r#"{"a": "\ud800\u0041"}"#,
            "{\"a\": \"\u{1}\"}",
            r#"{"a": 1} x"#,
            r#"{"a": "\x"}"#,
        ] {
            assert!(Metadata::from_json(text).is_err(), "{text}");
        }
        // Of a depth that Python writes, and no deeper than the server reads.
        let deep = format!("{{\"a\": {}1{}}}", "[".repeat(998), "]".repeat(998));
        assert!(Metadata::from_json(&deep).is_ok());
        let deeper = format!("{{\"a\": {}1{}}}", "[".repeat(1000), "]".repeat(1000));
        assert!(Metadata::from_json(&deeper).is_err());
    }

    #[test]
    fn serde_values_are_held_exactly_and_given_back_as_serde_holds_them() {
        let metadata: Metadata = [
            ("i", serde_json::json!(-5)),
            ("u", serde_json::json!(u64::MAX)),
            ("f", serde_json::json!(1e-7)),
            ("s", serde_json::json!({"k": [true, null]})),
        ]
        .into_iter()
        .collect();
        assert_eq!(
            metadata.json(),
            r#"{"i": -5, "u": 18446744073709551615, "f": 1e-07, "s": {"k": [true, null]}}"#
        );
        assert_eq!(metadata.get("u"), Some(serde_json::json!(u64::MAX)));
        let big = parsed(r#"{"x": 1000000000000000000000000000000, "n": NaN}"#);
        assert_eq!(big.get("x"), Some(serde_json::json!(1e30)));
        assert_eq!(big.get("n"), Some(serde_json::Value::Null));
        assert_eq!(
            big.get_json("x").as_deref(),
            Some("1000000000000000000000000000000")
        );
        assert_eq!(big.get_json("n").as_deref(), Some("NaN"));
        assert_eq!(big.get("missing"), None);
        let map = BTreeMap::from([("b".to_string(), serde_json::json!(2))]);
        assert_eq!(Metadata::from(map.clone()).to_map(), map);
    }

    #[test]
    fn keys_are_those_of_the_dict_of_json_loads() {
        let mut metadata = parsed(r#"{"a": 1, "b": 2, "a": 3}"#);
        assert_eq!(metadata.len(), 2);
        assert_eq!(metadata.keys().collect::<Vec<_>>(), ["a", "b"]);
        assert_eq!(metadata.get("a"), Some(serde_json::json!(3)));
        assert!(metadata.contains_key("b"));
        assert_eq!(
            metadata.insert("a", serde_json::json!(4)),
            Some(serde_json::json!(3))
        );
        assert_eq!(metadata.json(), r#"{"a": 4, "b": 2}"#);
        assert_eq!(metadata.insert("c", serde_json::json!(5)), None);
        assert_eq!(metadata.json(), r#"{"a": 4, "b": 2, "c": 5}"#);
        assert_eq!(metadata.remove("b"), Some(serde_json::json!(2)));
        assert_eq!(metadata.remove("b"), None);
        assert_eq!(metadata.json(), r#"{"a": 4, "c": 5}"#);
        assert!(!metadata.is_empty());
        assert!(Metadata::new().is_empty());
        assert_eq!(format!("{:?}", metadata), r#"Metadata({"a": 4, "c": 5})"#);
    }

    #[test]
    fn metadata_of_many_keys_compare_and_count_in_linear_time() {
        // Each key once, and by a hash: a comparison of the keys of one with each of the other
        // took seconds for 20,000 keys, and would take minutes for these.
        let text = |reversed: bool| {
            let mut keys: Vec<usize> = (0..100_000).collect();
            if reversed {
                keys.reverse();
            }
            let entries: Vec<String> = keys.iter().map(|i| format!("\"k{i}\": {i}")).collect();
            format!("{{{}}}", entries.join(", "))
        };
        let (a, b) = (parsed(&text(false)), parsed(&text(true)));
        let started = std::time::Instant::now();
        assert!(a == b && a.eq_as_python(&b));
        assert_eq!((a.len(), a.keys().count()), (100_000, 100_000));
        let collected: Metadata = (0..100_000)
            .map(|i| (format!("k{}", i % 50_000), serde_json::json!(i)))
            .collect();
        assert_eq!(collected.len(), 50_000);
        assert_eq!(collected.get("k0"), Some(serde_json::json!(50_000)));
        assert!(started.elapsed() < std::time::Duration::from_secs(20));
    }

    #[test]
    fn eq_compares_the_dicts_of_json_loads() {
        assert_eq!(
            parsed(r#"{"a": 1, "b": [2]}"#),
            parsed(r#"{"b":[2],"a":1}"#)
        );
        assert_eq!(parsed(r#"{"a": 1, "a": 2}"#), parsed(r#"{"a": 2}"#));
        assert_eq!(parsed(r#"{"a": NaN}"#), parsed(r#"{"a": NaN}"#));
        assert_eq!(parsed(r#"{"a": 1e-07}"#), parsed(r#"{"a": 1.0E-7}"#));
        assert_ne!(parsed(r#"{"a": 1}"#), parsed(r#"{"a": 1.0}"#));
        assert_ne!(parsed(r#"{"a": 1}"#), parsed(r#"{"a": 1, "b": 1}"#));
        assert_ne!(parsed(r#"{"a": [1]}"#), parsed(r#"{"a": [1, 2]}"#));
    }

    #[test]
    fn eq_as_python_compares_numbers_by_their_values() {
        let eq = |a: &str, b: &str| parsed(a).eq_as_python(&parsed(b));
        assert!(eq(r#"{"a": 1}"#, r#"{"a": 1.0}"#));
        assert!(eq(r#"{"a": true}"#, r#"{"a": 1}"#));
        assert!(eq(r#"{"a": false}"#, r#"{"a": -0.0}"#));
        assert!(eq(r#"{"a": 0}"#, r#"{"a": -0.0}"#));
        assert!(eq(r#"{"a": -3}"#, r#"{"a": -3.0}"#));
        assert!(eq(r#"{"a": [1, {"b": 2}]}"#, r#"{"a": [1.0, {"b": 2.0}]}"#));
        assert!(eq(
            r#"{"a": 1000000000000000019884624838656}"#,
            r#"{"a": 1e30}"#
        ));
        assert!(!eq(
            r#"{"a": 1000000000000000000000000000000}"#,
            r#"{"a": 1e30}"#
        ));
        assert!(!eq(
            r#"{"a": 9007199254740993}"#,
            r#"{"a": 9007199254740992.0}"#
        ));
        assert!(!eq(r#"{"a": 3}"#, r#"{"a": -3.0}"#));
        assert!(!eq(r#"{"a": 1}"#, r#"{"a": "1"}"#));
        assert!(!eq(r#"{"a": 1}"#, r#"{"a": Infinity}"#));
        assert!(!eq(r#"{"a": null}"#, r#"{"a": 0}"#));
    }
}
