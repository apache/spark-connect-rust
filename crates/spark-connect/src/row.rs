//! Row type mirroring `pyspark.sql.Row`.
//!
//! A Row represents a single record: an ordered collection of (field_name, value) pairs.
//! Values are accessed by index or by field name. Supports all Spark SQL value types.

use std::collections::BTreeMap;
use std::fmt;

/// A Value in a Row, supporting all Spark SQL types.
#[derive(Debug, Clone, PartialEq)]
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
    /// Struct (nested Row)
    Struct(Vec<(String, Value)>),
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
                for (i, (k, v)) in s.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}={}", k, v)?;
                }
                write!(f, ")")
            }
        }
    }
}

/// A Row is an ordered collection of (field_name, value) pairs.
/// Supports access by index or by field name.
#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    /// Field names (in order)
    fields: Vec<String>,
    /// Values corresponding to fields (in order)
    values: Vec<Value>,
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
        Row { fields, values }
    }

    /// Create an empty Row.
    pub fn empty() -> Self {
        Row {
            fields: vec![],
            values: vec![],
        }
    }

    /// Get the number of fields in this Row.
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    /// Check if this Row is empty.
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
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
        self.fields
            .iter()
            .position(|f| f == name)
            .and_then(|i| self.values.get(i))
    }

    /// Get field names.
    pub fn fields(&self) -> &[String] {
        &self.fields
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
            Value::Struct(vec![("a".to_string(), Value::Integer(1))]).to_string(),
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
