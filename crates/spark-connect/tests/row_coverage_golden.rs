//! Coverage for the server-free Row/Value accessors (no live server needed).

use spark_connect::row::{Row, Value};

#[test]
fn row_accessors() {
    let r = Row::new(
        vec!["b".into(), "i".into(), "f".into(), "s".into(), "n".into()],
        vec![
            Value::Bool(true),
            Value::Long(7),
            Value::Double(1.5),
            Value::String("x".into()),
            Value::Null,
        ],
    );
    assert_eq!(r.len(), 5);
    assert!(!r.is_empty());
    assert_eq!(r.fields().len(), 5);
    assert_eq!(r.values().len(), 5);
    // positional + named access
    assert_eq!(r.get(0).unwrap().as_bool(), Some(true));
    assert_eq!(r.get_unchecked(1).as_i64(), Some(7));
    assert_eq!(r.get_by_name("f").unwrap().as_f64(), Some(1.5));
    assert_eq!(r.get_by_name("s").unwrap().as_str(), Some("x"));
    assert!(r.get_by_name("n").unwrap().is_null());
    assert!(r.get(99).is_none());
    assert!(r.get_by_name("missing").is_none());
    // wrong-type accessors return None
    assert_eq!(r.get(0).unwrap().as_i64(), None);
    assert_eq!(r.get(1).unwrap().as_str(), None);
    // into_values consumes
    assert_eq!(r.into_values().len(), 5);

    assert!(Row::empty().is_empty());
    assert_eq!(Row::empty().len(), 0);
}

#[test]
fn value_accessors_and_bytes() {
    assert_eq!(
        Value::Binary(vec![1, 2, 3]).as_bytes(),
        Some(&[1u8, 2, 3][..])
    );
    assert_eq!(Value::Integer(3).as_i64(), Some(3));
    assert_eq!(Value::Float(2.0).as_f64(), Some(2.0));
    assert!(Value::Null.is_null());
    assert!(!Value::Bool(false).is_null());
}
