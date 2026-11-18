//! The text format of a message, in either of protobuf's two writers of it ([`Style`]): the set
//! fields in the order of their numbers, a message as `name { ... }`, a repeated field as one line
//! per value, a map as one entry message per key, a string or bytes value C-escaped, an enum value
//! by its name. On one line (`as_one_line`), the fields are separated by spaces.

use pyo3::prelude::*;
use pyo3::types::PyFloat;

use super::containers::{MapContainer, MapValue};
use super::message::{new_instance, PyMessage, Slot};
use super::pool::{pool, Kind};
use super::scalar::{key_order, Scalar};
use super::{codec, grow};

/// Which of protobuf's writers of the text format to follow.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Style {
    /// upb's (`upb_TextEncode`), which `str()` and `repr()` of a message give: doubles as C's
    /// `%.15g` (or `%.17g`) and floats as `%.6g` (or `%.9g`), every map entry of its key and its
    /// value, in upb's order of the keys, and a `google.protobuf.Any` as its `type_url` and
    /// `value`.
    Upb,
    /// Python's `text_format.MessageToString`: doubles as `repr`, floats of the fewest digits
    /// that read back as the float, the map entries in the order of their keys without a key or
    /// value of the default, and an `Any` of a known type expanded to its message.
    TextFormat,
}

pub(crate) fn to_text(
    py: Python<'_>,
    message: &Bound<'_, PyMessage>,
    style: Style,
    one_line: bool,
) -> PyResult<String> {
    let mut out = String::new();
    let w = Writer {
        py,
        style,
        one_line,
    };
    w.message(&message.borrow(), 0, &mut out)?;
    if one_line {
        while out.ends_with(' ') {
            out.pop();
        }
    }
    Ok(out)
}

fn start_line(out: &mut String, one_line: bool, indent: usize) {
    if !one_line {
        out.extend(std::iter::repeat_n("  ", indent));
    }
}

fn end_line(out: &mut String, one_line: bool) {
    out.push(if one_line { ' ' } else { '\n' });
}

struct Writer<'py> {
    py: Python<'py>,
    style: Style,
    one_line: bool,
}

impl Writer<'_> {
    /// Write a `google.protobuf.Any` that holds a message of a known type as `text_format` does,
    /// as `[type_url] { ... }` of the message, and return whether it was one.
    fn any(&self, message: &PyMessage, indent: usize, out: &mut String) -> PyResult<bool> {
        if self.style != Style::TextFormat || message.desc.full_name != "google.protobuf.Any" {
            return Ok(false);
        }
        let value_of = |name: &str| {
            message
                .desc
                .by_name
                .get(name)
                .and_then(|&i| match message.slots.get(i) {
                    Some(Slot::Scalar(v)) => Some(v.clone()),
                    _ => None,
                })
        };
        let (Some(Scalar::Str(url)), value) = (value_of("type_url"), value_of("value")) else {
            return Ok(false);
        };
        let Some((_, type_name)) = url.rsplit_once('/') else {
            return Ok(false);
        };
        if pool().message(type_name).is_none() {
            return Ok(false);
        }
        let bytes = match value {
            Some(Scalar::Bytes(bytes)) => bytes,
            _ => Vec::new(),
        };
        let packed = new_instance(self.py, type_name)?;
        if codec::merge_unchecked(self.py, &packed, &bytes).is_err() {
            return Ok(false);
        }
        start_line(out, self.one_line, indent);
        out.push('[');
        out.push_str(&url);
        out.push_str("] {");
        end_line(out, self.one_line);
        self.message(&packed.borrow(), indent + 1, out)?;
        start_line(out, self.one_line, indent);
        out.push('}');
        end_line(out, self.one_line);
        Ok(true)
    }

    fn message(&self, message: &PyMessage, indent: usize, out: &mut String) -> PyResult<()> {
        let py = self.py;
        grow(|| {
            if self.any(message, indent, out)? {
                return Ok(());
            }
            for (index, slot) in message.slots.iter() {
                let field = &message.desc.fields[index];
                match slot {
                    Slot::Scalar(value) => {
                        if field.has_presence() || !value.is_default() {
                            self.scalar(
                                &field.name,
                                field.kind,
                                &field.type_name,
                                value,
                                indent,
                                out,
                            )?;
                        }
                    }
                    Slot::Message(child) => {
                        self.message_field(&field.name, &child.bind(py).borrow(), indent, out)?
                    }
                    Slot::Repeated(container) => {
                        for value in &container.bind(py).borrow().items {
                            self.scalar(
                                &field.name,
                                field.kind,
                                &field.type_name,
                                value,
                                indent,
                                out,
                            )?;
                        }
                    }
                    Slot::Composite(container) => {
                        for child in &container.bind(py).borrow().items {
                            self.message_field(&field.name, &child.bind(py).borrow(), indent, out)?;
                        }
                    }
                    Slot::Map(map) => self.map(&field.name, &map.bind(py).borrow(), indent, out)?,
                }
            }
            Ok(())
        })
    }

    /// The entries of a map, as entry messages: upb's writes the key and the value of each, in
    /// upb's order of the keys; `text_format` skips a key or a value of the default, in Python's
    /// order of the keys.
    fn map(&self, name: &str, map: &MapContainer, indent: usize, out: &mut String) -> PyResult<()> {
        let mut entries: Vec<&(Scalar, MapValue)> = map.entries.iter().collect();
        match self.style {
            Style::Upb => entries.sort_by(|a, b| upb_text_key_order(&a.0, &b.0)),
            Style::TextFormat => entries.sort_by(|a, b| key_order(&a.0, &b.0)),
        }
        let all = self.style == Style::Upb;
        for (key, value) in entries {
            start_line(out, self.one_line, indent);
            out.push_str(name);
            out.push_str(" {");
            end_line(out, self.one_line);
            if all || !key.is_default() {
                self.scalar("key", map.map.key, "", key, indent + 1, out)?;
            }
            match value {
                MapValue::Scalar(value) => {
                    if all || !value.is_default() {
                        self.scalar(
                            "value",
                            map.map.value,
                            &map.map.value_type,
                            value,
                            indent + 1,
                            out,
                        )?;
                    }
                }
                MapValue::Message(child) => {
                    self.message_field("value", &child.bind(self.py).borrow(), indent + 1, out)?;
                }
            }
            start_line(out, self.one_line, indent);
            out.push('}');
            end_line(out, self.one_line);
        }
        Ok(())
    }

    fn message_field(
        &self,
        name: &str,
        message: &PyMessage,
        indent: usize,
        out: &mut String,
    ) -> PyResult<()> {
        start_line(out, self.one_line, indent);
        out.push_str(name);
        out.push_str(" {");
        end_line(out, self.one_line);
        self.message(message, indent + 1, out)?;
        start_line(out, self.one_line, indent);
        out.push('}');
        end_line(out, self.one_line);
        Ok(())
    }

    fn scalar(
        &self,
        name: &str,
        kind: Kind,
        enum_type: &str,
        value: &Scalar,
        indent: usize,
        out: &mut String,
    ) -> PyResult<()> {
        start_line(out, self.one_line, indent);
        out.push_str(name);
        out.push_str(": ");
        out.push_str(&format_scalar(self.py, self.style, kind, enum_type, value)?);
        end_line(out, self.one_line);
        Ok(())
    }
}

/// upb's order of the keys of a map in its text format (`_upb_mapsorter_cmpstr`): of two strings
/// that differ in a byte of their common length, the one of the greater byte first, else the
/// shorter first; of numbers and bools, the lesser first, where upb compares the bits of signed
/// integers as unsigned ones, so that a negative key comes after the others (`0, 1, 100, -5,
/// -1`).
fn upb_text_key_order(a: &Scalar, b: &Scalar) -> std::cmp::Ordering {
    match (a, b) {
        (Scalar::Str(x), Scalar::Str(y)) => upb_bytes_order(x.as_bytes(), y.as_bytes()),
        (Scalar::Bytes(x), Scalar::Bytes(y)) => upb_bytes_order(x, y),
        (Scalar::I32(x), Scalar::I32(y)) => (*x as u32).cmp(&(*y as u32)),
        (Scalar::I64(x), Scalar::I64(y)) => (*x as u64).cmp(&(*y as u64)),
        _ => key_order(a, b),
    }
}

fn upb_bytes_order(x: &[u8], y: &[u8]) -> std::cmp::Ordering {
    let common = x.len().min(y.len());
    match x[..common].cmp(&y[..common]) {
        std::cmp::Ordering::Equal => x.len().cmp(&y.len()),
        unequal => unequal.reverse(),
    }
}

/// upb's order of the map entries of a deterministic serialization, the reverse of that of its
/// text format: upb sorts the entries as for the text, and writes a message from its end.
pub(crate) fn upb_wire_key_order(a: &Scalar, b: &Scalar) -> std::cmp::Ordering {
    upb_text_key_order(a, b).reverse()
}

fn format_scalar(
    py: Python<'_>,
    style: Style,
    kind: Kind,
    enum_type: &str,
    value: &Scalar,
) -> PyResult<String> {
    Ok(match value {
        Scalar::F64(v) => match style {
            Style::Upb => {
                upb_round_trip(py, *v, 15, 17, |s| s.parse::<f64>().is_ok_and(|r| r == *v))?
            }
            Style::TextFormat => PyFloat::new(py, *v).repr()?.to_string(),
        },
        Scalar::F32(v) => match style {
            Style::Upb => upb_round_trip(py, *v as f64, 6, 9, |s| {
                s.parse::<f32>().is_ok_and(|r| r == *v)
            })?,
            Style::TextFormat => format_float(py, *v)?,
        },
        Scalar::I64(v) => v.to_string(),
        Scalar::U64(v) => v.to_string(),
        Scalar::I32(v) => v.to_string(),
        Scalar::U32(v) => v.to_string(),
        Scalar::Bool(v) => if *v { "true" } else { "false" }.to_string(),
        Scalar::Str(v) => format!("\"{}\"", c_escape_str(v)),
        Scalar::Bytes(v) => format!("\"{}\"", c_escape(v)),
        Scalar::Enum(v) => {
            let name = (kind == Kind::Enum)
                .then(|| pool().enumeration(enum_type).and_then(|e| e.name_of(*v)))
                .flatten();
            name.map_or_else(|| v.to_string(), str::to_string)
        }
    })
}

/// A floating point value as upb's text format writes it (`_upb_EncodeRoundTripDouble` and
/// `_upb_EncodeRoundTripFloat`): C's `%.{short}g`, or `%.{long}g` when that does not read back as
/// the value, and `nan` of a NaN.
fn upb_round_trip(
    py: Python<'_>,
    value: f64,
    short: usize,
    long: usize,
    reads_back: impl Fn(&str) -> bool,
) -> PyResult<String> {
    if value.is_nan() {
        return Ok("nan".to_string());
    }
    let format = py.import("builtins")?.getattr("format")?;
    let text: String = format.call1((value, format!(".{short}g")))?.extract()?;
    if reads_back(&text) {
        return Ok(text);
    }
    format.call1((value, format!(".{long}g")))?.extract()
}

/// A float field's value as protobuf's text format writes it, of `type_checkers.ToShortestFloat`:
/// the value with 6 significant digits, or as many more as reading it back as a float takes.
fn format_float(py: Python<'_>, value: f32) -> PyResult<String> {
    if value.is_nan() {
        return Ok("nan".to_string());
    }
    let format = py.import("builtins")?.getattr("format")?;
    let original = value as f64;
    let mut rounded = original;
    for precision in 6..=17 {
        let text: String = format
            .call1((original, format!(".{precision}g")))?
            .extract()?;
        rounded = text.parse::<f64>().unwrap_or(original);
        if rounded as f32 == value {
            break;
        }
    }
    Ok(PyFloat::new(py, rounded).repr()?.to_string())
}

/// The string `value` escaped as protobuf's text format writes a string field: the C escapes of
/// the newline, carriage return, tab, quotes and backslash, an octal escape of each other control
/// character and of DEL, and every other character, non-ASCII ones included, as it is.
fn c_escape_str(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for c in value.chars() {
        match c {
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            '"' => escaped.push_str("\\\""),
            '\'' => escaped.push_str("\\'"),
            '\\' => escaped.push_str("\\\\"),
            '\u{0}'..='\u{1f}' | '\u{7f}' => escaped.push_str(&format!("\\{:03o}", c as u32)),
            _ => escaped.push(c),
        }
    }
    escaped
}

/// `bytes` escaped as protobuf's text format writes a bytes field: the C escapes of the newline,
/// carriage return, tab, quotes and backslash, and an octal escape of every other byte outside
/// the printable ASCII range.
fn c_escape(bytes: &[u8]) -> String {
    let mut escaped = String::with_capacity(bytes.len());
    for &byte in bytes {
        match byte {
            b'\n' => escaped.push_str("\\n"),
            b'\r' => escaped.push_str("\\r"),
            b'\t' => escaped.push_str("\\t"),
            b'"' => escaped.push_str("\\\""),
            b'\'' => escaped.push_str("\\'"),
            b'\\' => escaped.push_str("\\\\"),
            32..=126 => escaped.push(byte as char),
            _ => escaped.push_str(&format!("\\{byte:03o}")),
        }
    }
    escaped
}
