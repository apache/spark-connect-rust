//! Golden parity test for protobuf/avro functions.
//!
//! These functions are in separate Python modules (pyspark.sql.connect.protobuf
//! and pyspark.sql.connect.avro) and are tested via unit tests asserting the
//! function name and argument structure, since capturing requires live Spark
//! Connect server and descriptor files.

use spark_connect::column::Column;
use spark_connect::expression::{ColumnReference, Expression};
use spark_connect::functions::*;

#[test]
fn test_protobuf_from_protobuf_basic() {
    let data = Column::new(Expression::ColumnReference(ColumnReference::new("data")));
    let result = from_protobuf(data, "MyMessage");

    // Verify the expression is an UnresolvedFunction with correct name
    let expr = result.expression();
    if let Expression::UnresolvedFunction(func) = expr {
        assert_eq!(func.name, "from_protobuf");
        assert_eq!(func.args.len(), 2);
    } else {
        panic!("Expected UnresolvedFunction");
    }
}

#[test]
fn test_protobuf_from_protobuf_with_descriptor() {
    let data = Column::new(Expression::ColumnReference(ColumnReference::new("data")));
    let descriptor = vec![1, 2, 3, 4, 5];
    let result = from_protobuf_with_descriptor(data, "MyMessage", descriptor);

    let expr = result.expression();
    if let Expression::UnresolvedFunction(func) = expr {
        assert_eq!(func.name, "from_protobuf");
        assert_eq!(func.args.len(), 3);
    } else {
        panic!("Expected UnresolvedFunction");
    }
}

#[test]
fn test_protobuf_from_protobuf_with_descriptor_and_options() {
    let data = Column::new(Expression::ColumnReference(ColumnReference::new("data")));
    let descriptor = vec![1, 2, 3, 4, 5];
    let options = Column::new(Expression::ColumnReference(ColumnReference::new("opts")));
    let result = from_protobuf_with_descriptor_and_options(data, "MyMessage", descriptor, options);

    let expr = result.expression();
    if let Expression::UnresolvedFunction(func) = expr {
        assert_eq!(func.name, "from_protobuf");
        assert_eq!(func.args.len(), 4);
    } else {
        panic!("Expected UnresolvedFunction");
    }
}

#[test]
fn test_protobuf_from_protobuf_with_options() {
    let data = Column::new(Expression::ColumnReference(ColumnReference::new("data")));
    let options = Column::new(Expression::ColumnReference(ColumnReference::new("opts")));
    let result = from_protobuf_with_options(data, "MyMessage", options);

    let expr = result.expression();
    if let Expression::UnresolvedFunction(func) = expr {
        assert_eq!(func.name, "from_protobuf");
        assert_eq!(func.args.len(), 3);
    } else {
        panic!("Expected UnresolvedFunction");
    }
}

#[test]
fn test_protobuf_to_protobuf_basic() {
    let data = Column::new(Expression::ColumnReference(ColumnReference::new("data")));
    let result = to_protobuf(data, "MyMessage");

    let expr = result.expression();
    if let Expression::UnresolvedFunction(func) = expr {
        assert_eq!(func.name, "to_protobuf");
        assert_eq!(func.args.len(), 2);
    } else {
        panic!("Expected UnresolvedFunction");
    }
}

#[test]
fn test_protobuf_to_protobuf_with_descriptor() {
    let data = Column::new(Expression::ColumnReference(ColumnReference::new("data")));
    let descriptor = vec![1, 2, 3, 4, 5];
    let result = to_protobuf_with_descriptor(data, "MyMessage", descriptor);

    let expr = result.expression();
    if let Expression::UnresolvedFunction(func) = expr {
        assert_eq!(func.name, "to_protobuf");
        assert_eq!(func.args.len(), 3);
    } else {
        panic!("Expected UnresolvedFunction");
    }
}

#[test]
fn test_protobuf_to_protobuf_with_descriptor_and_options() {
    let data = Column::new(Expression::ColumnReference(ColumnReference::new("data")));
    let descriptor = vec![1, 2, 3, 4, 5];
    let options = Column::new(Expression::ColumnReference(ColumnReference::new("opts")));
    let result = to_protobuf_with_descriptor_and_options(data, "MyMessage", descriptor, options);

    let expr = result.expression();
    if let Expression::UnresolvedFunction(func) = expr {
        assert_eq!(func.name, "to_protobuf");
        assert_eq!(func.args.len(), 4);
    } else {
        panic!("Expected UnresolvedFunction");
    }
}

#[test]
fn test_protobuf_to_protobuf_with_options() {
    let data = Column::new(Expression::ColumnReference(ColumnReference::new("data")));
    let options = Column::new(Expression::ColumnReference(ColumnReference::new("opts")));
    let result = to_protobuf_with_options(data, "MyMessage", options);

    let expr = result.expression();
    if let Expression::UnresolvedFunction(func) = expr {
        assert_eq!(func.name, "to_protobuf");
        assert_eq!(func.args.len(), 3);
    } else {
        panic!("Expected UnresolvedFunction");
    }
}

#[test]
fn test_avro_from_avro_basic() {
    let data = Column::new(Expression::ColumnReference(ColumnReference::new("data")));
    let result = from_avro(
        data,
        r#"{"type":"record","name":"Test","fields":[{"name":"f","type":"string"}]}"#,
    );

    let expr = result.expression();
    if let Expression::UnresolvedFunction(func) = expr {
        assert_eq!(func.name, "from_avro");
        assert_eq!(func.args.len(), 2);
    } else {
        panic!("Expected UnresolvedFunction");
    }
}

#[test]
fn test_avro_from_avro_with_options() {
    let data = Column::new(Expression::ColumnReference(ColumnReference::new("data")));
    let options = Column::new(Expression::ColumnReference(ColumnReference::new("opts")));
    let result = from_avro_with_options(data, r#"{"type":"record"}"#, options);

    let expr = result.expression();
    if let Expression::UnresolvedFunction(func) = expr {
        assert_eq!(func.name, "from_avro");
        assert_eq!(func.args.len(), 3);
    } else {
        panic!("Expected UnresolvedFunction");
    }
}

#[test]
fn test_avro_to_avro_basic() {
    let data = Column::new(Expression::ColumnReference(ColumnReference::new("data")));
    let result = to_avro(data);

    let expr = result.expression();
    if let Expression::UnresolvedFunction(func) = expr {
        assert_eq!(func.name, "to_avro");
        assert_eq!(func.args.len(), 1);
    } else {
        panic!("Expected UnresolvedFunction");
    }
}

#[test]
fn test_avro_to_avro_with_schema() {
    let data = Column::new(Expression::ColumnReference(ColumnReference::new("data")));
    let result = to_avro_with_schema(data, r#"{"type":"record"}"#);

    let expr = result.expression();
    if let Expression::UnresolvedFunction(func) = expr {
        assert_eq!(func.name, "to_avro");
        assert_eq!(func.args.len(), 2);
    } else {
        panic!("Expected UnresolvedFunction");
    }
}
