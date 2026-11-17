//! What PySpark's client does in the user's process as it builds the plan of a request, besides
//! the plan: it warns of a window expression of no partitions, and it makes the conjunction of
//! the conditions of a join of a list of Columns, whose `and`s have the origin of `__and__` at
//! that time. The core builds the plans, and tells them to the [`PlanBuildListener`] that a
//! client of it sets, so that the Python binding does what PySpark's client does there. Without
//! a listener, the plans are built without them. Not a stable API.

use std::sync::{Arc, RwLock};

use spark_connect_core::error::Result;

use crate::origin::Origin;

/// What a client of the core does as a request builds a plan, where PySpark's client runs code
/// of its own. The core calls it on the thread that builds the plan, while it builds it, in the
/// order in which PySpark's client builds the plan; an error fails the request, as the error of
/// an expression that PySpark fails to build does. Not a stable API.
pub trait PlanBuildListener: Send + Sync {
    /// The origin of an expression that PySpark's client makes as it builds the plan, by its API
    /// function `fragment` (`__and__`), as its `_with_origin` gives the origin then: None for
    /// none.
    fn origin(&self, fragment: &str) -> Result<Option<Origin>>;

    /// A window expression of no partitions is built, of which PySpark's
    /// `WindowExpression.to_plan` warns once it has built its window function.
    fn window_without_partitions(&self) -> Result<()>;
}

/// The listener of the process, which every request tells.
static LISTENER: RwLock<Option<Arc<dyn PlanBuildListener>>> = RwLock::new(None);

/// Set the listener of the plans that the requests of the process build, or none.
pub fn set_listener(listener: Option<Arc<dyn PlanBuildListener>>) {
    *LISTENER.write().unwrap_or_else(|e| e.into_inner()) = listener;
}

/// The listener of the process, if one is set.
pub(crate) fn listener() -> Option<Arc<dyn PlanBuildListener>> {
    LISTENER.read().unwrap_or_else(|e| e.into_inner()).clone()
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;
    use std::thread::ThreadId;

    use spark_connect_core::error::SparkError;
    use spark_connect_proto as proto;

    use super::*;
    use crate::column::Column;
    use crate::expression::Expression;
    use crate::functions::{col, row_number};
    use crate::window::Window;

    /// A listener that records what the plan builds of one thread tell it, as other tests build
    /// plans at the same time, and that fails a window of no partitions once `fail_windows` is
    /// set.
    struct Recording {
        thread: ThreadId,
        calls: Mutex<Vec<String>>,
        fail_windows: std::sync::atomic::AtomicBool,
    }

    impl Recording {
        fn of_this_thread(&self) -> bool {
            std::thread::current().id() == self.thread
        }

        fn take(&self) -> Vec<String> {
            std::mem::take(&mut self.calls.lock().unwrap())
        }
    }

    impl PlanBuildListener for Recording {
        fn origin(&self, fragment: &str) -> Result<Option<Origin>> {
            if !self.of_this_thread() {
                return Ok(None);
            }
            let mut calls = self.calls.lock().unwrap();
            calls.push(format!("origin {fragment}"));
            Ok(Some(Origin {
                fragment: fragment.to_string(),
                call_site: format!("build.py:{}", calls.len()),
            }))
        }

        fn window_without_partitions(&self) -> Result<()> {
            if !self.of_this_thread() {
                return Ok(());
            }
            self.calls.lock().unwrap().push("window".to_string());
            if self.fail_windows.load(std::sync::atomic::Ordering::SeqCst) {
                return Err(SparkError::value_msg("no partitions"));
            }
            Ok(())
        }
    }

    fn python_origin(expression: &proto::Expression) -> Option<(String, String)> {
        let origin = expression.common.as_ref()?.origin.as_ref()?;
        let Some(proto::origin::Function::PythonOrigin(origin)) = &origin.function else {
            return None;
        };
        Some((origin.fragment.clone(), origin.call_site.clone()))
    }

    #[test]
    fn a_plan_build_tells_the_listener_as_pysparks_client_builds_the_plan() {
        let unordered = || row_number().over(Window::order_by(["id"]));
        let conjunction = Column::new(Expression::with_plan_build_origin(
            col("a").and(col("b")).expression().clone(),
            "__and__",
        ));
        // Without a listener, the window is built alone, and the conjunction is of no origin.
        assert!(python_origin(&conjunction.try_to_proto().unwrap()).is_none());

        let listener = Arc::new(Recording {
            thread: std::thread::current().id(),
            calls: Mutex::default(),
            fail_windows: Default::default(),
        });
        set_listener(Some(listener.clone()));

        // As PySpark's `WindowExpression.to_plan`, which warns of a window of no partitions at
        // each build of it, and of none of a window of partitions.
        unordered().try_to_proto().unwrap();
        unordered().try_to_proto().unwrap();
        assert_eq!(listener.take(), ["window", "window"]);
        row_number()
            .over(Window::partition_by(["id"]).order_by(["id"]))
            .try_to_proto()
            .unwrap();
        assert!(listener.take().is_empty());

        // As PySpark's `reduce` of the `&` of the conditions of a join, at each build: the `and`
        // is of the origin that the listener gives then, and its conditions keep their own.
        let built = conjunction.try_to_proto().unwrap();
        assert_eq!(
            python_origin(&built),
            Some(("__and__".to_string(), "build.py:1".to_string()))
        );
        assert_eq!(
            python_origin(&conjunction.try_to_proto().unwrap()),
            Some(("__and__".to_string(), "build.py:2".to_string()))
        );
        assert_eq!(listener.take(), ["origin __and__", "origin __and__"]);
        assert_eq!(
            conjunction.expression().render(),
            col("a").and(col("b")).expression().render()
        );
        let Some(proto::expression::ExprType::UnresolvedFunction(and)) = built.expr_type else {
            panic!("expected the function of the conjunction");
        };
        assert_eq!(and.function_name, "and");
        assert!(and
            .arguments
            .iter()
            .all(|argument| python_origin(argument).is_none()));

        // An error of the listener fails the build, as PySpark raises the error of its plan
        // build there, and the build tells the listener nothing after it.
        listener
            .fail_windows
            .store(true, std::sync::atomic::Ordering::SeqCst);
        let three = crate::functions::r#struct([unordered(), conjunction.clone(), unordered()]);
        let error = three.try_to_proto().unwrap_err();
        assert_eq!(error.message(), "no partitions");
        assert_eq!(listener.take(), ["window"]);
        set_listener(None);
    }
}
