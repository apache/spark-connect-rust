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

//! The array literals of the core, as PySpark's `LiteralExpression.to_plan` writes them: of the
//! type of their elements, which the server requires. The tests of a server run when
//! SPARK_REMOTE is set.

use spark_connect::expression::{Expression, LiteralExpression};
use spark_connect::types::DataType;
use spark_connect::{Column, SparkSession};
use spark_connect_proto as proto;

fn string() -> DataType {
    DataType::String {
        collation: String::new(),
    }
}

fn array(element_type: DataType, elements: Vec<LiteralExpression>) -> LiteralExpression {
    LiteralExpression::Array {
        element_type: Box::new(element_type),
        elements,
    }
}

#[allow(deprecated)]
fn element_type(literal: &proto::expression::Literal) -> Option<proto::DataType> {
    match &literal.literal_type {
        Some(proto::expression::literal::LiteralType::Array(array)) => array.element_type.clone(),
        _ => panic!("expected an array literal, got {literal:?}"),
    }
}

#[test]
fn array_literals_are_of_the_type_of_their_elements() {
    let empty = array(string(), vec![]).to_literal_proto();
    assert_eq!(element_type(&empty), Some(string().to_proto()));

    let inner = DataType::Array {
        element_type: Box::new(DataType::Long),
        contains_null: true,
    };
    let nested = array(
        inner.clone(),
        vec![array(DataType::Long, vec![LiteralExpression::long(1)])],
    )
    .to_literal_proto();
    assert_eq!(element_type(&nested), Some(inner.to_proto()));
    let Some(proto::expression::literal::LiteralType::Array(outer)) = &nested.literal_type else {
        unreachable!()
    };
    assert_eq!(
        element_type(&outer.elements[0]),
        Some(DataType::Long.to_proto())
    );
}

#[test]
fn array_literals_run_on_the_server() {
    let Ok(remote) = std::env::var("SPARK_REMOTE") else {
        return;
    };
    let session = SparkSession::builder()
        .remote(&remote)
        .get_or_create()
        .unwrap();
    let literal = |value: LiteralExpression| Column::new(Expression::literal(value));
    let df = session.range(1).unwrap().select([
        literal(array(
            DataType::Integer,
            vec![LiteralExpression::int(1), LiteralExpression::int(2)],
        ))
        .alias("a"),
        literal(array(string(), vec![])).alias("e"),
    ]);
    assert_eq!(
        df.schema().unwrap().simple_string(),
        "struct<a:array<int>,e:array<string>>"
    );
    let rows = df.collect().unwrap();
    assert_eq!(rows.len(), 1);
}
