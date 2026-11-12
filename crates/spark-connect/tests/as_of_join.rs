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

//! The join type and the direction of an as-of join: the values that the client knows, and the
//! strings that it sends as they are, as PySpark's `_joinAsOf` sends `how` and `direction`
//! without checking them. No server is needed: a session connects when it first sends.

mod common;

use spark_connect::column::col;
use spark_connect::dataframe::AsOfJoinOptions;
use spark_connect::plan::{AsOfJoinDirection, AsOfJoinType};
use spark_connect_proto as proto;

/// The `join_type` and the `direction` that an as-of join of `join_type` and `direction` sends.
fn sent(join_type: AsOfJoinType, direction: AsOfJoinDirection) -> (String, String) {
    let spark = common::session();
    let (left, right) = (spark.range(3).unwrap(), spark.range(5).unwrap());
    let options = AsOfJoinOptions {
        join_type,
        direction,
        ..AsOfJoinOptions::new(col("id"), col("id"))
    };
    let joined = left.as_of_join(&right, options).unwrap();
    let relation = joined.to_relation().unwrap();
    let Some(proto::relation::RelType::AsOfJoin(as_of)) = relation.rel_type else {
        panic!("expected AsOfJoin: {relation:?}");
    };
    (as_of.join_type, as_of.direction)
}

#[test]
fn the_join_types_and_directions_of_an_as_of_join_are_sent_by_their_names() {
    assert_eq!(AsOfJoinType::default(), AsOfJoinType::Inner);
    assert_eq!(AsOfJoinDirection::default(), AsOfJoinDirection::Backward);
    for (join_type, name) in [
        (AsOfJoinType::Inner, "inner"),
        (AsOfJoinType::LeftOuter, "left"),
    ] {
        assert_eq!(AsOfJoinType::parse(name), join_type);
        assert_eq!(AsOfJoinType::from(name), join_type);
        assert_eq!(join_type.as_str(), name);
        assert_eq!(sent(join_type, AsOfJoinDirection::default()).0, name);
    }
    for (direction, name) in [
        (AsOfJoinDirection::Backward, "backward"),
        (AsOfJoinDirection::Forward, "forward"),
        (AsOfJoinDirection::Nearest, "nearest"),
    ] {
        assert_eq!(AsOfJoinDirection::parse(name), direction);
        assert_eq!(AsOfJoinDirection::from(name), direction);
        assert_eq!(direction.as_str(), name);
        assert_eq!(sent(AsOfJoinType::default(), direction).1, name);
    }
}

/// PySpark checks neither `how` nor `direction`, and the server reads more spellings than the
/// names above (`"LEFT"`, `"left_outer"`, `"Forward"`) and refuses the rest in the analysis
/// (`UNSUPPORTED_JOIN_TYPE`, `AS_OF_JOIN.UNSUPPORTED_DIRECTION`). So the client fails for no
/// string, and sends each byte for byte.
#[test]
fn any_other_join_type_or_direction_is_sent_as_it_is_for_the_server_to_check() {
    for how in [
        "LEFT",
        "left_outer",
        "leftouter",
        "Inner",
        "right",
        "sideways",
        "",
    ] {
        let join_type = AsOfJoinType::parse(how);
        assert_eq!(join_type, AsOfJoinType::Other(how.to_string()));
        assert_eq!(join_type.as_str(), how);
        assert_eq!(sent(join_type, AsOfJoinDirection::default()).0, how);
    }
    for name in ["Forward", "BACKWARD", "sideways", ""] {
        let direction = AsOfJoinDirection::parse(name);
        assert_eq!(direction, AsOfJoinDirection::Other(name.to_string()));
        assert_eq!(direction.as_str(), name);
        assert_eq!(sent(AsOfJoinType::default(), direction).1, name);
    }
}
