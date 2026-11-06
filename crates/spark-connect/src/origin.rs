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
