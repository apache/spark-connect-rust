//! Row type mirroring `pyspark.sql.Row`.
//!
//! A Row represents a single record: an ordered collection of (field_name, value) pairs.
//! Values are accessed by index or by field name. Supports all Spark SQL value types.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

/// A Value in a Row, supporting all Spark SQL types.
#[derive(Clone, PartialEq)]
pub enum Value {
    /// NULL value
    Null,
    /// Boolean
    Bool(bool),
    /// Byte (i8)
    Byte(i8),
    /// Short (i16)
    Short(i16),
    /// Integer (i32)
    Integer(i32),
    /// Long (i64)
    Long(i64),
    /// Float (f32)
    Float(f32),
    /// Double (f64)
    Double(f64),
    /// String
    String(String),
    /// Binary (bytes)
    Binary(Vec<u8>),
    /// Date (days since epoch)
    Date(i32),
    /// Timestamp (microseconds since epoch)
    Timestamp(i64),
    /// Decimal (string value with optional precision and scale)
    Decimal {
        value: String,
        precision: Option<i32>,
        scale: Option<i32>,
    },
    /// Array of values
    List(Vec<Value>),
    /// Map of key-value pairs
    Map(BTreeMap<String, Value>),
    /// Struct (nested Row), with ordered names and values.
    /// Construct with `Value::Struct(Row::new(fields, values).into())`.
    Struct(StructValue),
    /// A VARIANT value carried as its raw (value, metadata) binary components, matching
    /// `pyspark.sql.types.VariantVal(value, metadata)`. Decoding to JSON/Python is done
    /// lazily on the Python side (VariantVal.toJson/toPython via variant_utils).
    Variant { value: Vec<u8>, metadata: Vec<u8> },
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Null => write!(f, "Null"),
            Self::Bool(v) => f.debug_tuple("Bool").field(v).finish(),
            Self::Byte(v) => f.debug_tuple("Byte").field(v).finish(),
            Self::Short(v) => f.debug_tuple("Short").field(v).finish(),
            Self::Integer(v) => f.debug_tuple("Integer").field(v).finish(),
            Self::Long(v) => f.debug_tuple("Long").field(v).finish(),
            Self::Float(v) => f.debug_tuple("Float").field(v).finish(),
            Self::Double(v) => f.debug_tuple("Double").field(v).finish(),
            Self::String(v) => f.debug_tuple("String").field(v).finish(),
            Self::Binary(v) => f.debug_tuple("Binary").field(v).finish(),
            Self::Date(v) => f.debug_tuple("Date").field(v).finish(),
            Self::Timestamp(v) => f.debug_tuple("Timestamp").field(v).finish(),
            Self::Decimal {
                value,
                precision,
                scale,
            } => f
                .debug_struct("Decimal")
                .field("value", value)
                .field("precision", precision)
                .field("scale", scale)
                .finish(),
            Self::List(v) => f.debug_tuple("List").field(v).finish(),
            Self::Map(v) => f.debug_tuple("Map").field(v).finish(),
            Self::Struct(row) => {
                // Keep the previous ordered-pair representation while names are shared.
                struct Fields<'a>(&'a StructValue);

                impl fmt::Debug for Fields<'_> {
                    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                        f.debug_list()
                            .entries(self.0.fields().iter().zip(self.0.values()))
                            .finish()
                    }
                }

                f.debug_tuple("Struct").field(&Fields(row)).finish()
            }
            Self::Variant { value, metadata } => f
                .debug_struct("Variant")
                .field("value", value)
                .field("metadata", metadata)
                .finish(),
        }
    }
}

impl Value {
    /// Get this value as a bool, or None if it's not a bool.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Get this value as i64, or None if it's not an integer type.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::Byte(b) => Some(*b as i64),
            Value::Short(s) => Some(*s as i64),
            Value::Integer(i) => Some(*i as i64),
            Value::Long(l) => Some(*l),
            _ => None,
        }
    }

    /// Get this value as f64, or None if it's not a float type.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Float(f) => Some(*f as f64),
            Value::Double(d) => Some(*d),
            _ => None,
        }
    }

    /// Get this value as a string reference, or None if it's not a string.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    /// Get this value as a bytes reference, or None if it's not binary.
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            Value::Binary(b) => Some(b),
            _ => None,
        }
    }

    /// Check if this value is null.
    pub fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Null => write!(f, "NULL"),
            Value::Bool(b) => write!(f, "{}", b),
            Value::Byte(b) => write!(f, "{}", b),
            Value::Short(s) => write!(f, "{}", s),
            Value::Integer(i) => write!(f, "{}", i),
            Value::Long(l) => write!(f, "{}", l),
            Value::Float(fl) => write!(f, "{}", fl),
            Value::Double(d) => write!(f, "{}", d),
            Value::String(s) => write!(f, "{}", s),
            Value::Binary(b) => write!(f, "{:?}", b),
            Value::Date(d) => write!(f, "{}", d),
            Value::Timestamp(t) => write!(f, "{}", t),
            Value::Decimal {
                value,
                precision,
                scale,
            } => {
                write!(f, "Decimal({}", value)?;
                if let Some(p) = precision {
                    write!(f, ",{})", p)?;
                    if let Some(s) = scale {
                        write!(f, "s={}", s)?;
                    }
                } else {
                    write!(f, ")")?;
                }
                Ok(())
            }
            Value::List(l) => {
                write!(f, "[")?;
                for (i, v) in l.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", v)?;
                }
                write!(f, "]")
            }
            Value::Map(m) => {
                write!(f, "{{")?;
                for (i, (k, v)) in m.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}: {}", k, v)?;
                }
                write!(f, "}}")
            }
            Value::Struct(s) => {
                write!(f, "(")?;
                for (i, (k, v)) in s.fields().iter().zip(s.values()).enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}={}", k, v)?;
                }
                write!(f, ")")
            }
            Value::Variant { value, metadata } => {
                write!(
                    f,
                    "Variant(value={} bytes, metadata={} bytes)",
                    value.len(),
                    metadata.len()
                )
            }
        }
    }
}

/// An ordered struct value with shared immutable field names and owned values.
/// Cloning shares the names while cloning each value independently.
#[derive(Debug, Clone, PartialEq)]
pub struct StructValue {
    fields: Arc<[String]>,
    values: Vec<Value>,
}

impl StructValue {
    /// Create a struct with immutable names that can be shared with other structs.
    /// Panics if the numbers of names and values differ.
    pub fn new(fields: Arc<[String]>, values: Vec<Value>) -> Self {
        assert_eq!(
            fields.len(),
            values.len(),
            "field names and values must have the same length"
        );
        Self { fields, values }
    }

    /// Get the number of fields.
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    /// Check whether this struct has no fields.
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    /// Get the ordered field names.
    pub fn fields(&self) -> &[String] {
        &self.fields
    }

    /// Get the ordered values.
    pub fn values(&self) -> &[Value] {
        &self.values
    }

    /// Get a field value by index.
    pub fn get(&self, index: usize) -> Option<&Value> {
        self.values.get(index)
    }

    /// Get the first field value with the given name.
    pub fn get_by_name(&self, name: &str) -> Option<&Value> {
        self.fields
            .iter()
            .position(|field| field == name)
            .and_then(|index| self.values.get(index))
    }
}

// Keep local constructors free of Arc conversion allocations while sharing decoded batch names.
#[derive(Clone)]
enum RowFields {
    Owned(Vec<String>),
    Shared(Arc<[String]>),
}

impl RowFields {
    fn as_slice(&self) -> &[String] {
        match self {
            Self::Owned(fields) => fields,
            Self::Shared(fields) => fields,
        }
    }
}

impl fmt::Debug for RowFields {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.as_slice(), f)
    }
}

impl PartialEq for RowFields {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

/// A Row is an ordered collection of (field_name, value) pairs.
/// Supports access by index or by field name.
#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    /// Field names (in order)
    fields: RowFields,
    /// Values corresponding to fields (in order)
    values: Vec<Value>,
}

impl From<Row> for StructValue {
    fn from(row: Row) -> Self {
        let fields = match row.fields {
            RowFields::Owned(fields) => fields.into(),
            RowFields::Shared(fields) => fields,
        };
        Self {
            fields,
            values: row.values,
        }
    }
}

impl From<StructValue> for Row {
    fn from(value: StructValue) -> Self {
        Self {
            fields: RowFields::Shared(value.fields),
            values: value.values,
        }
    }
}

impl Row {
    /// Create a new Row from field names and values.
    /// Panics if lengths don't match.
    pub fn new(fields: Vec<String>, values: Vec<Value>) -> Self {
        assert_eq!(
            fields.len(),
            values.len(),
            "field names and values must have the same length"
        );
        Row {
            fields: RowFields::Owned(fields),
            values,
        }
    }

    /// Create a Row with immutable field names shared by rows from the same batch.
    pub(crate) fn with_shared_fields(fields: Arc<[String]>, values: Vec<Value>) -> Self {
        assert_eq!(
            fields.len(),
            values.len(),
            "field names and values must have the same length"
        );
        Row {
            fields: RowFields::Shared(fields),
            values,
        }
    }

    /// Create an empty Row.
    pub fn empty() -> Self {
        Row {
            fields: RowFields::Owned(vec![]),
            values: vec![],
        }
    }

    /// Get the number of fields in this Row.
    pub fn len(&self) -> usize {
        self.fields().len()
    }

    /// Check if this Row is empty.
    pub fn is_empty(&self) -> bool {
        self.fields().is_empty()
    }

    /// Get a field value by index.
    pub fn get(&self, index: usize) -> Option<&Value> {
        self.values.get(index)
    }

    /// Get a field value by index, or panic if out of bounds.
    pub fn get_unchecked(&self, index: usize) -> &Value {
        &self.values[index]
    }

    /// Get a field value by name.
    pub fn get_by_name(&self, name: &str) -> Option<&Value> {
        self.fields()
            .iter()
            .position(|f| f == name)
            .and_then(|i| self.values.get(i))
    }

    /// Get field names.
    pub fn fields(&self) -> &[String] {
        self.fields.as_slice()
    }

    /// Get values.
    pub fn values(&self) -> &[Value] {
        &self.values
    }

    /// Convert into values.
    pub fn into_values(self) -> Vec<Value> {
        self.values
    }
}

impl fmt::Display for Row {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, v) in self.values.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", v)?;
        }
        write!(f, "]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_row_creation() {
        let row = Row::new(
            vec!["id".to_string(), "name".to_string()],
            vec![Value::Long(1), Value::String("Alice".to_string())],
        );

        assert_eq!(row.len(), 2);
        assert_eq!(row.get(0), Some(&Value::Long(1)));
        assert_eq!(row.get(1), Some(&Value::String("Alice".to_string())));
    }

    #[test]
    fn test_row_access_by_name() {
        let row = Row::new(
            vec!["id".to_string(), "name".to_string()],
            vec![Value::Long(1), Value::String("Alice".to_string())],
        );

        assert_eq!(row.get_by_name("id"), Some(&Value::Long(1)));
        assert_eq!(
            row.get_by_name("name"),
            Some(&Value::String("Alice".to_string()))
        );
        assert_eq!(row.get_by_name("nonexistent"), None);
    }

    #[test]
    fn cloned_rows_share_field_names() {
        let fields: Arc<[String]> = vec!["id".to_string()].into();
        let row = Row::with_shared_fields(Arc::clone(&fields), vec![Value::Long(1)]);
        let cloned = row.clone();

        assert!(std::ptr::eq(fields.as_ref(), row.fields()));
        assert!(std::ptr::eq(row.fields(), cloned.fields()));
    }

    #[test]
    fn shared_field_names_preserve_order_duplicates_equality_and_debug() {
        let names = vec![
            "second".to_string(),
            "first".to_string(),
            "second".to_string(),
        ];
        let values = vec![Value::Long(2), Value::Long(1), Value::Long(3)];
        let shared = Row::with_shared_fields(names.clone().into(), values.clone());
        let owned = Row::new(names.clone(), values);

        assert_eq!(shared.fields(), names.as_slice());
        assert_eq!(shared.get_by_name("second"), Some(&Value::Long(2)));
        assert_eq!(shared.get(2), Some(&Value::Long(3)));
        assert_eq!(shared, owned);
        assert_eq!(owned, shared);
        assert_eq!(format!("{shared:?}"), format!("{owned:?}"));
        assert_eq!(
            format!("{shared:?}"),
            "Row { fields: [\"second\", \"first\", \"second\"], values: [Long(2), Long(1), Long(3)] }"
        );
        let reordered = Row::new(
            vec![
                "first".to_string(),
                "second".to_string(),
                "second".to_string(),
            ],
            owned.values().to_vec(),
        );
        assert_ne!(shared, reordered);
    }

    #[test]
    fn nested_struct_names_preserve_pair_order_equality_and_formatting() {
        let names = vec![
            "".to_string(),
            "a".to_string(),
            "a".to_string(),
            "é".to_string(),
        ];
        let values = vec![
            Value::Null,
            Value::Long(1),
            Value::Long(2),
            Value::List(vec![Value::String("original".to_string())]),
        ];
        let shared =
            Value::Struct(Row::with_shared_fields(names.clone().into(), values.clone()).into());
        let owned = Value::Struct(Row::new(names, values).into());
        assert_eq!(shared, owned);
        assert_eq!(owned, shared);
        let Value::Struct(row) = &shared else {
            panic!("expected a struct value");
        };
        assert_eq!(row.get_by_name("a"), Some(&Value::Long(1)));
        assert_eq!(row.get(2), Some(&Value::Long(2)));
        assert_eq!(shared.to_string(), "(=NULL, a=1, a=2, é=[original])");
        assert_eq!(
            format!("{shared:?}"),
            "Struct([(\"\", Null), (\"a\", Long(1)), (\"a\", Long(2)), (\"é\", List([String(\"original\")]))])"
        );
        assert_eq!(format!("{shared:#?}"), format!("{owned:#?}"));
        let empty = Value::Struct(Row::empty().into());
        assert_eq!(format!("{empty:?}"), "Struct([])");
        assert_eq!(empty.to_string(), "()");
        let one = Value::Struct(Row::new(vec!["a".to_string()], vec![Value::Integer(1)]).into());
        assert_eq!(
            format!("{one:#?}"),
            concat!(
                "Struct(\n",
                "    [\n",
                "        (\n",
                "            \"a\",\n",
                "            Integer(\n",
                "                1,\n",
                "            ),\n",
                "        ),\n",
                "    ],\n",
                ")"
            )
        );
    }

    #[test]
    fn nested_struct_clones_share_names_and_keep_child_values_independent() {
        let fields: Arc<[String]> = vec!["items".to_string()].into();
        let value = Value::Struct(
            Row::with_shared_fields(
                Arc::clone(&fields),
                vec![Value::List(vec![Value::String("original".to_string())])],
            )
            .into(),
        );
        let mut cloned = value.clone();
        let Value::Struct(original_row) = &value else {
            panic!("expected a struct value");
        };
        let Value::Struct(cloned_row) = &mut cloned else {
            panic!("expected a cloned struct value");
        };
        assert!(std::ptr::eq(fields.as_ref(), cloned_row.fields()));
        assert!(std::ptr::eq(original_row.fields(), cloned_row.fields()));
        let Value::List(items) = &mut cloned_row.values[0] else {
            panic!("expected a list value");
        };
        items[0] = Value::String("changed".to_string());
        assert_eq!(
            original_row.get(0),
            Some(&Value::List(vec![Value::String("original".to_string())]))
        );
        assert_ne!(value, cloned);
    }

    #[test]
    fn nested_struct_equality_does_not_shortcut_nan() {
        let value = Value::Struct(
            Row::with_shared_fields(
                vec!["nan".to_string()].into(),
                vec![Value::Double(f64::NAN)],
            )
            .into(),
        );
        assert_ne!(value, value.clone());
    }

    #[test]
    fn struct_row_conversions_move_values_and_preserve_name_storage() {
        let row = Row::new(
            vec!["field".to_string()],
            vec![Value::String("value".to_string())],
        );
        let names_text = row.fields()[0].as_ptr();
        let values = row.values().as_ptr();
        let value = StructValue::from(row);
        assert_eq!(value.fields()[0].as_ptr(), names_text);
        assert_eq!(value.values().as_ptr(), values);
        assert_eq!(value.len(), 1);
        assert!(!value.is_empty());
        assert_eq!(value.get(0), Some(&Value::String("value".to_string())));
        assert_eq!(value.get(1), None);
        assert_eq!(value.get_by_name("missing"), None);

        let names = value.fields().as_ptr();
        let row = Row::from(value);
        assert_eq!(row.fields().as_ptr(), names);
        assert_eq!(row.values().as_ptr(), values);
        let value = StructValue::from(row);
        assert_eq!(value.fields().as_ptr(), names);
        assert_eq!(value.values().as_ptr(), values);
    }

    #[test]
    fn struct_value_constructor_shares_names_and_preserves_empty_values() {
        let names: Arc<[String]> = vec!["name".to_string()].into();
        let value = StructValue::new(Arc::clone(&names), vec![Value::Null]);
        assert!(std::ptr::eq(names.as_ref(), value.fields()));
        assert_eq!(value.get_by_name("name"), Some(&Value::Null));
        let empty = StructValue::new(Vec::new().into(), vec![]);
        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());
        assert!(empty.fields().is_empty());
        assert!(empty.values().is_empty());
        assert_eq!(Row::from(empty), Row::empty());
    }

    #[test]
    #[should_panic(expected = "field names and values must have the same length")]
    fn struct_value_constructor_rejects_mismatched_lengths() {
        let _ = StructValue::new(vec!["name".to_string()].into(), vec![]);
    }

    #[test]
    fn shared_empty_field_names_preserve_empty_row_behavior() {
        let fields: Arc<[String]> = Vec::new().into();
        let row = Row::with_shared_fields(fields, vec![]);
        assert_eq!(row, Row::empty());
        assert!(row.fields().is_empty());
        assert!(row.into_values().is_empty());
    }

    #[test]
    #[should_panic(expected = "field names and values must have the same length")]
    fn shared_field_names_reject_mismatched_lengths() {
        let _ = Row::with_shared_fields(vec!["field".to_string()].into(), vec![]);
    }

    #[test]
    fn test_value_conversions() {
        let b = Value::Bool(true);
        assert_eq!(b.as_bool(), Some(true));

        let i = Value::Integer(42);
        assert_eq!(i.as_i64(), Some(42));

        #[allow(clippy::approx_constant)]
        let d = Value::Double(3.14);

        assert_eq!(d.as_f64(), Some(3.14));

        let s = Value::String("test".to_string());
        assert_eq!(s.as_str(), Some("test"));
    }

    #[test]
    fn test_value_date_timestamp_decimal() {
        // Test Date
        let date_val = Value::Date(18993); // days since epoch
        assert_eq!(date_val, Value::Date(18993));

        // Test Timestamp
        let ts_val = Value::Timestamp(1693526400000000); // micros since epoch
        assert_eq!(ts_val, Value::Timestamp(1693526400000000));

        // Test Decimal
        let dec_val = Value::Decimal {
            value: "123.45".to_string(),
            precision: Some(5),
            scale: Some(2),
        };
        match dec_val {
            Value::Decimal {
                value,
                precision,
                scale,
            } => {
                assert_eq!(value, "123.45");
                assert_eq!(precision, Some(5));
                assert_eq!(scale, Some(2));
            }
            _ => panic!("Expected Decimal variant"),
        }
    }

    #[test]
    fn as_i64_covers_all_integer_widths_and_rejects_others() {
        assert_eq!(Value::Byte(7).as_i64(), Some(7));
        assert_eq!(Value::Short(300).as_i64(), Some(300));
        assert_eq!(Value::Integer(70_000).as_i64(), Some(70_000));
        assert_eq!(Value::Long(5_000_000_000).as_i64(), Some(5_000_000_000));
        assert_eq!(Value::Double(1.0).as_i64(), None);
        assert_eq!(Value::Null.as_i64(), None);
    }

    #[test]
    fn as_f64_covers_float_and_double_and_rejects_others() {
        assert_eq!(Value::Float(1.5).as_f64(), Some(1.5));
        assert_eq!(Value::Double(2.5).as_f64(), Some(2.5));
        assert_eq!(Value::Long(3).as_f64(), None);
    }

    #[test]
    fn scalar_accessors_reject_wrong_types() {
        assert_eq!(Value::Integer(1).as_bool(), None);
        assert_eq!(Value::Bool(true).as_str(), None);
        assert_eq!(Value::String("x".into()).as_bytes(), None);
        assert_eq!(Value::Binary(vec![1, 2]).as_bytes(), Some(&[1u8, 2][..]));
    }

    #[test]
    fn is_null_reflects_variant() {
        assert!(Value::Null.is_null());
        assert!(!Value::Integer(0).is_null());
    }

    #[test]
    fn display_covers_every_value_variant() {
        assert_eq!(Value::Null.to_string(), "NULL");
        assert_eq!(Value::Bool(true).to_string(), "true");
        assert_eq!(Value::Byte(1).to_string(), "1");
        assert_eq!(Value::Short(2).to_string(), "2");
        assert_eq!(Value::Integer(3).to_string(), "3");
        assert_eq!(Value::Long(4).to_string(), "4");
        assert_eq!(Value::Float(1.5).to_string(), "1.5");
        assert_eq!(Value::Double(2.5).to_string(), "2.5");
        assert_eq!(Value::String("hi".into()).to_string(), "hi");
        assert_eq!(Value::Binary(vec![1, 2]).to_string(), "[1, 2]");
        assert_eq!(Value::Date(19_000).to_string(), "19000");
        assert_eq!(Value::Timestamp(123).to_string(), "123");
        assert_eq!(
            Value::List(vec![Value::Integer(1), Value::Integer(2)]).to_string(),
            "[1, 2]"
        );
        let mut m = std::collections::BTreeMap::new();
        m.insert("k".to_string(), Value::Integer(9));
        assert_eq!(Value::Map(m).to_string(), "{k: 9}");
        assert_eq!(
            Value::Struct(Row::new(vec!["a".to_string()], vec![Value::Integer(1)]).into())
                .to_string(),
            "(a=1)"
        );
    }

    #[test]
    fn display_decimal_with_and_without_precision() {
        assert_eq!(
            Value::Decimal {
                value: "123.45".into(),
                precision: Some(5),
                scale: Some(2),
            }
            .to_string(),
            "Decimal(123.45,5)s=2"
        );
        assert_eq!(
            Value::Decimal {
                value: "7".into(),
                precision: None,
                scale: None,
            }
            .to_string(),
            "Decimal(7)"
        );
    }

    #[test]
    fn row_helpers_and_display() {
        let empty = Row::empty();
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);

        let row = Row::new(
            vec!["a".to_string(), "b".to_string()],
            vec![Value::Integer(1), Value::String("x".to_string())],
        );
        assert!(!row.is_empty());
        assert_eq!(row.get(5), None);
        assert_eq!(row.get_unchecked(0), &Value::Integer(1));
        assert_eq!(row.fields(), &["a".to_string(), "b".to_string()]);
        assert_eq!(row.values().len(), 2);
        assert_eq!(row.to_string(), "[1, x]");
        assert_eq!(
            row.into_values(),
            vec![Value::Integer(1), Value::String("x".to_string())]
        );
    }

    #[test]
    #[should_panic(expected = "same length")]
    fn row_new_rejects_mismatched_lengths() {
        let _ = Row::new(vec!["a".to_string()], vec![]);
    }
}
