//! The origins of expressions in user code, as PySpark's `pyspark.errors.utils.current_origin`:
//! the API function that builds an expression, and its call site, which the server reports in
//! the `DataFrameQueryContext` of an error in the expression.

use std::cell::RefCell;
use std::sync::Arc;

/// The origin of an expression: the API function that built it, and its call site in user
/// code. Mirrors `pyspark.sql.connect.proto.PythonOrigin`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Origin {
    /// The name of the function, such as `col` or `__add__`.
    pub fragment: String,
    /// Where the user code called it, such as `script.py:12`.
    pub call_site: String,
}

thread_local! {
    /// The origin of the expressions that this thread builds now, as PySpark's thread-local
    /// `_current_origin`.
    static CURRENT: RefCell<Option<Arc<Origin>>> = const { RefCell::new(None) };
}

/// The origin of the expressions that this thread builds now, if any.
pub(crate) fn current() -> Option<Arc<Origin>> {
    CURRENT.with(|current| current.borrow().clone())
}

/// Sets the origin of the expressions that this thread builds while the returned guard lives,
/// as PySpark's `_with_origin` does for a call of an API function. As PySpark's, the guard then
/// unsets the origin.
pub fn set_current(origin: Origin) -> OriginGuard {
    CURRENT.with(|current| *current.borrow_mut() = Some(Arc::new(origin)));
    OriginGuard { _private: () }
}

/// The origin of the expression of `encoded`, the encoding of a `spark.connect.Expression`, which
/// its common fields carry: that which a pickled Column keeps of the origin of its expression, as
/// PySpark's pickled expression keeps its `origin`, so that a Column that its unpickle makes again
/// of its parts, as a sort order or a CASE WHEN, keeps it rather than take the origin of the call
/// that unpickles it (see [`crate::expression::Expression::with_origin`]). Only the common fields
/// are decoded, of a time of the length of the encoding. None of no origin, or of an encoding
/// that is not an expression. Not a stable API.
#[doc(hidden)]
pub fn origin_of_encoded(encoded: &[u8]) -> Option<Origin> {
    use prost::Message;
    // The common fields of an expression alone, of the number of the field of its encoding.
    #[derive(Clone, PartialEq, prost::Message)]
    struct CommonOf {
        #[prost(message, optional, tag = "18")]
        common: Option<spark_connect_proto::ExpressionCommon>,
    }
    let common = CommonOf::decode(encoded).ok()?.common?;
    match common.origin?.function? {
        spark_connect_proto::origin::Function::PythonOrigin(origin) => Some(Origin {
            fragment: origin.fragment,
            call_site: origin.call_site,
        }),
        _ => None,
    }
}

/// Unsets the current origin when dropped. See [`set_current`].
#[must_use]
pub struct OriginGuard {
    _private: (),
}

impl Drop for OriginGuard {
    fn drop(&mut self) {
        CURRENT.with(|current| *current.borrow_mut() = None);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expression::Expression;
    use spark_connect_proto as proto;

    #[test]
    fn the_origin_of_an_encoded_expression_is_that_of_its_common_fields() {
        // As PySpark's pickled expression, which keeps its origin: the origin of the encoding of
        // an expression, which a pickled Column carries, is given back of its common fields
        // alone, of a deep expression too, and an expression made again of it takes it.
        use prost::Message;
        let origin = Origin {
            fragment: "desc".to_string(),
            call_site: "script.py:7".to_string(),
        };
        let made = {
            let _origin = set_current(origin.clone());
            Expression::column("a")
        };
        let mut deep = Expression::column("b");
        for _ in 0..200 {
            deep = Expression::function("+", vec![deep, Expression::column("b")]);
        }
        let deep = deep.with_origin(Some(origin.clone()));
        let encoded = made.to_proto().encode_to_vec();
        assert_eq!(origin_of_encoded(&encoded), Some(origin.clone()));
        let deep_encoding =
            stacker::maybe_grow(1 << 20, 16 << 20, || deep.to_proto().encode_to_vec());
        assert_eq!(origin_of_encoded(&deep_encoding), Some(origin.clone()));
        let plain = Expression::column("a");
        assert_eq!(origin_of_encoded(&plain.to_proto().encode_to_vec()), None);
        assert_eq!(origin_of_encoded(b"\xff\xff"), None);
        let kept = plain.with_origin(origin_of_encoded(&encoded));
        assert_eq!(kept.origin(), Some(&origin));
        assert_eq!(kept.to_proto(), made.to_proto());
        assert!(kept.with_origin(None).origin().is_none());
    }

    #[test]
    fn expressions_carry_the_origin_of_their_construction() {
        // As PySpark's Expression, which takes the current origin when it is made, and sends
        // it as the python_origin of its common fields.
        assert!(Expression::column("before").to_proto().common.is_none());
        let built = {
            let _origin = set_current(Origin {
                fragment: "col".to_string(),
                call_site: "script.py:3".to_string(),
            });
            Expression::column("a")
        };
        let after = Expression::column("after");
        assert!(after.to_proto().common.is_none());
        assert_eq!(built, Expression::column("a"));
        let common = built.to_proto().common.expect("the origin");
        let Some(proto::origin::Function::PythonOrigin(origin)) =
            common.origin.and_then(|origin| origin.function)
        else {
            panic!("expected a PythonOrigin");
        };
        assert_eq!(origin.fragment, "col");
        assert_eq!(origin.call_site, "script.py:3");
    }
}
