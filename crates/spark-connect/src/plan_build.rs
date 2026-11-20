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
    use crate::expression::{Expression, ExpressionNode, MAX_EXPRESSION_DEPTH};
    use crate::functions::{col, row_number};
    use crate::window::Window;

    /// Taken by the tests that set the listener of the process, which run one at a time.
    static LISTENER_TESTS: Mutex<()> = Mutex::new(());

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
        let _serial = LISTENER_TESTS.lock().unwrap_or_else(|e| e.into_inner());
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

    /// A Python UDF of `arguments`, whose pickler records its pickling in the calls of
    /// `listener`, and fails with the error of `tag` when it `fails`.
    fn recording_udf(
        listener: &Arc<Recording>,
        tag: &'static str,
        fails: bool,
        arguments: Vec<Column>,
    ) -> Column {
        let pickler = crate::udf::CommandPickler::new(Arc::clone(listener), move |listener| {
            listener.calls.lock().unwrap().push(format!("pickle {tag}"));
            if fails {
                return Err(SparkError::value_msg(tag));
            }
            Ok(tag.as_bytes().to_vec())
        });
        let payload = crate::udf::PythonUDFPayload::pickled_at_build(
            crate::types::DataType::Long,
            crate::udf::eval_type::SQL_BATCHED_UDF,
            pickler,
            "3.11".to_string(),
        );
        Column::new(Expression::new(
            ExpressionNode::CommonInlineUserDefinedFunction(Box::new(
                crate::udf::CommonInlineUserDefinedFunctionExpression::new(
                    "f".to_string(),
                    true,
                    arguments.iter().map(|c| c.expression().clone()).collect(),
                    payload,
                ),
            )),
        ))
    }

    #[test]
    fn a_window_after_a_udf_is_told_in_its_turn_among_the_picklings_as_pysparks_build() {
        let _serial = LISTENER_TESTS.lock().unwrap_or_else(|e| e.into_inner());
        let listener = Arc::new(Recording {
            thread: std::thread::current().id(),
            calls: Mutex::default(),
            fail_windows: Default::default(),
        });
        set_listener(Some(listener.clone()));
        let unordered = || row_number().over(Window::order_by(["id"]));
        let udf = |tag, fails, arguments| recording_udf(&listener, tag, fails, arguments);
        let build =
            |columns: Vec<Column>| crate::functions::r#struct(columns).try_to_proto().map(drop);

        // As PySpark's build of a plan, which pickles each UDF and warns of each window of no
        // partitions as it meets them, once, though this one builds a plan of UDFs twice.
        build(vec![
            udf("a", false, vec![]),
            unordered(),
            udf("b", false, vec![]),
            unordered(),
        ])
        .unwrap();
        assert_eq!(
            listener.take(),
            ["pickle a", "window", "pickle b", "window"]
        );
        // The first pickling that fails ends the build: a window after it is not told.
        let error = build(vec![
            udf("a", false, vec![]),
            unordered(),
            udf("bad", true, vec![]),
            unordered(),
        ])
        .unwrap_err();
        assert_eq!(error.message(), "bad");
        assert_eq!(listener.take(), ["pickle a", "window", "pickle bad"]);
        build(vec![udf("bad", true, vec![]), unordered()]).unwrap_err();
        assert_eq!(listener.take(), ["pickle bad"]);
        // The arguments of a UDF are built before its function is pickled.
        build(vec![udf("bad", true, vec![unordered()])]).unwrap_err();
        assert_eq!(listener.take(), ["window", "pickle bad"]);

        // An error of a window after a UDF fails the build before the UDFs after it are pickled.
        listener
            .fail_windows
            .store(true, std::sync::atomic::Ordering::SeqCst);
        let error = build(vec![
            udf("a", false, vec![]),
            unordered(),
            udf("b", false, vec![]),
        ])
        .unwrap_err();
        assert_eq!(error.message(), "no partitions");
        assert_eq!(listener.take(), ["pickle a", "window"]);
        listener
            .fail_windows
            .store(false, std::sync::atomic::Ordering::SeqCst);

        // An expression too deep to be sent is the `RecursionError` of the request, and no
        // function is pickled; its windows are told, as in a build of no UDFs.
        let mut deep = col("x");
        for _ in 0..MAX_EXPRESSION_DEPTH {
            deep = deep + col("y");
        }
        let error = build(vec![udf("a", false, vec![]), unordered(), deep]).unwrap_err();
        assert_eq!(
            error.kind,
            spark_connect_core::error::SparkErrorKind::RecursionError
        );
        assert_eq!(listener.take(), ["window"]);
        set_listener(None);
    }
}
