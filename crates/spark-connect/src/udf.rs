//! User-defined function (UDF) support.
//!
//! Mirrors `pyspark.sql.connect.udf` and `pyspark.util.PythonEvalType`.
//! UDFs are cloudpickled on the Python client and wrapped into
//! `CommonInlineUserDefinedFunction` expressions for transmission to the server.

use std::sync::Arc;

use crate::expression::Expression;
use crate::types::DataType;
use spark_connect_core::error::Result;
use spark_connect_proto as proto;

/// Python evaluation type constants, matching `pyspark.util.PythonEvalType`.
/// These distinguish between different UDF execution modes (batched, pandas, arrow, etc.).
pub mod eval_type {
    /// Regular Python UDF, row-by-row (column as list).
    pub const SQL_BATCHED_UDF: i32 = 100;
    /// Arrow-optimized Python UDF (column as PyArrow table).
    pub const SQL_ARROW_BATCHED_UDF: i32 = 101;

    /// Pandas scalar UDF (Series -> Series).
    pub const SQL_SCALAR_PANDAS_UDF: i32 = 200;
    /// Pandas grouped map UDF (grouped DataFrame -> DataFrame).
    pub const SQL_GROUPED_MAP_PANDAS_UDF: i32 = 201;
    /// Pandas grouped aggregate UDF.
    pub const SQL_GROUPED_AGG_PANDAS_UDF: i32 = 202;
    /// Pandas window aggregate UDF.
    pub const SQL_WINDOW_AGG_PANDAS_UDF: i32 = 203;
    /// Pandas scalar iterator UDF.
    pub const SQL_SCALAR_PANDAS_ITER_UDF: i32 = 204;
    /// Pandas map iterator UDF.
    pub const SQL_MAP_PANDAS_ITER_UDF: i32 = 205;
    /// Pandas cogrouped map UDF.
    pub const SQL_COGROUPED_MAP_PANDAS_UDF: i32 = 206;
    /// Arrow map iterator UDF.
    pub const SQL_MAP_ARROW_ITER_UDF: i32 = 207;
    /// Pandas grouped map with state UDF.
    pub const SQL_GROUPED_MAP_PANDAS_UDF_WITH_STATE: i32 = 208;
    /// Arrow grouped map UDF.
    pub const SQL_GROUPED_MAP_ARROW_UDF: i32 = 209;
    /// Arrow cogrouped map UDF.
    pub const SQL_COGROUPED_MAP_ARROW_UDF: i32 = 210;
    /// Pandas transform with state UDF.
    pub const SQL_TRANSFORM_WITH_STATE_PANDAS_UDF: i32 = 211;
    /// Pandas transform with state init state UDF.
    pub const SQL_TRANSFORM_WITH_STATE_PANDAS_INIT_STATE_UDF: i32 = 212;
    /// Python row transform with state UDF.
    pub const SQL_TRANSFORM_WITH_STATE_PYTHON_ROW_UDF: i32 = 213;
    /// Python row transform with state init state UDF.
    pub const SQL_TRANSFORM_WITH_STATE_PYTHON_ROW_INIT_STATE_UDF: i32 = 214;
    /// Arrow grouped map iterator UDF.
    pub const SQL_GROUPED_MAP_ARROW_ITER_UDF: i32 = 215;
    /// Pandas grouped map iterator UDF.
    pub const SQL_GROUPED_MAP_PANDAS_ITER_UDF: i32 = 216;
    /// Pandas grouped aggregate iterator UDF.
    pub const SQL_GROUPED_AGG_PANDAS_ITER_UDF: i32 = 217;

    /// Arrow scalar UDF.
    pub const SQL_SCALAR_ARROW_UDF: i32 = 250;
    /// Arrow scalar iterator UDF.
    pub const SQL_SCALAR_ARROW_ITER_UDF: i32 = 251;
    /// Arrow grouped aggregate UDF.
    pub const SQL_GROUPED_AGG_ARROW_UDF: i32 = 252;
    /// Arrow window aggregate UDF.
    pub const SQL_WINDOW_AGG_ARROW_UDF: i32 = 253;
    /// Arrow grouped aggregate iterator UDF.
    pub const SQL_GROUPED_AGG_ARROW_ITER_UDF: i32 = 254;

    /// SQL table UDF (UDTF).
    pub const SQL_TABLE_UDF: i32 = 300;
    /// Arrow SQL table UDF (UDTF).
    pub const SQL_ARROW_TABLE_UDF: i32 = 301;
    /// Arrow UDTF.
    pub const SQL_ARROW_UDTF: i32 = 302;
}

/// The pickling of the command of a Python UDF or UDTF, which a request calls each time it builds
/// the plan that holds the UDF, as PySpark's `PythonUDF.to_plan` and `PythonUDTF.to_plan` pickle
/// the function for each plan that they build: the command is that of the function as it is when
/// the request is built, so a global that the function reads is the value it has then, and a
/// function that does not pickle fails the request with the error of its pickling, before
/// anything is sent (see [`crate::expression`] for the order of the picklings and of the other
/// errors of a plan). The Python client gives it, of the object that it pickles; a UDF of the
/// Rust API carries the bytes of its command, which are sent as they are. Copies share it. Not a
/// stable API.
#[doc(hidden)]
#[derive(Clone)]
pub struct CommandPickler(Arc<dyn Pickling>);

/// The object of a [`CommandPickler`] and its pickling.
trait Pickling: Send + Sync {
    fn pickle(&self) -> Result<Vec<u8>>;
    fn object(&self) -> &(dyn std::any::Any + Send + Sync);
}

struct PicklingOf<O, F> {
    object: O,
    pickle: F,
}

impl<O, F> Pickling for PicklingOf<O, F>
where
    O: std::any::Any + Send + Sync,
    F: Fn(&O) -> Result<Vec<u8>> + Send + Sync,
{
    fn pickle(&self) -> Result<Vec<u8>> {
        (self.pickle)(&self.object)
    }

    fn object(&self) -> &(dyn std::any::Any + Send + Sync) {
        &self.object
    }
}

impl CommandPickler {
    /// The pickler of `object`, which `pickle` pickles: it gives the bytes of the command, or the
    /// error that the request raises.
    pub fn new<O: std::any::Any + Send + Sync>(
        object: O,
        pickle: impl Fn(&O) -> Result<Vec<u8>> + Send + Sync + 'static,
    ) -> Self {
        CommandPickler(Arc::new(PicklingOf { object, pickle }))
    }

    /// The command as it pickles now.
    pub(crate) fn pickle(&self) -> Result<Vec<u8>> {
        self.0.pickle()
    }

    /// The object of this pickler, which a deep copy of the UDF copies, as PySpark's deep copy of
    /// a UDF copies its function (see [`PicklerCopies`]). Not a stable API.
    #[doc(hidden)]
    pub fn object(&self) -> &(dyn std::any::Any + Send + Sync) {
        self.0.object()
    }

    /// The address of the pickling of this pickler, which its copies share.
    fn address(&self) -> usize {
        Arc::as_ptr(&self.0) as *const () as usize
    }

    /// The object of this pickler, if no copy of it shares it: so the plan or the expression that
    /// holds this pickler alone holds the object, which the garbage collector of a Python client
    /// is told of (see [`exclusive_pickler_objects`]).
    fn exclusive_object(&self) -> Option<&(dyn std::any::Any + Send + Sync)> {
        (Arc::strong_count(&self.0) == 1).then(|| self.0.object())
    }
}

impl std::fmt::Debug for CommandPickler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("CommandPickler")
    }
}

impl PartialEq for CommandPickler {
    /// The same pickling: a copy of the UDF, which pickles the same function.
    fn eq(&self, other: &Self) -> bool {
        std::ptr::addr_eq(Arc::as_ptr(&self.0), Arc::as_ptr(&other.0))
    }
}

/// A plan, an expression, a subquery, a UDF or the pickler of a UDTF, of which
/// [`exclusive_pickler_objects`] and [`all_picklers`] visit the picklers.
pub(crate) enum PicklerHolder<'a> {
    Plan(&'a crate::plan::LogicalPlan),
    Expression(&'a Expression),
    Subquery(&'a crate::expression::SubqueryExpression),
    Udf(&'a CommonInlineUserDefinedFunctionExpression),
    Pickler(&'a CommandPickler),
}

/// The objects of the picklers of `holders`, of their UDFs and UDTFs, that they alone hold: those
/// below the plans and the expressions that no other plan, expression or DataFrame shares, and
/// of which no copy of the pickler shares the object. A Python client tells its garbage collector
/// of these, so that a reference cycle through the function of a UDF and the DataFrame or the
/// Column that holds it is freed, as PySpark's plan, of Python objects, is: an object reachable
/// from two of them, or from elsewhere too, is not told of, which keeps it alive, rather than
/// counting one reference twice. Computed without recursing.
pub(crate) fn exclusive_pickler_objects<'a>(
    holders: Vec<PicklerHolder<'a>>,
) -> Vec<&'a (dyn std::any::Any + Send + Sync)> {
    let mut objects = Vec::new();
    let mut pending = holders;
    while let Some(holder) = pending.pop() {
        match holder {
            PicklerHolder::Plan(plan) => {
                plan.push_own_pickler_holders(&mut pending);
                pending.extend(
                    plan.inputs()
                        .into_iter()
                        .filter(|input| Arc::strong_count(input) == 1)
                        .map(|input| PicklerHolder::Plan(input)),
                );
            }
            PicklerHolder::Expression(expression) => {
                if expression.is_exclusive() {
                    expression.node().push_pickler_holders(&mut pending);
                }
            }
            PicklerHolder::Subquery(subquery) => subquery.push_pickler_holders(&mut pending, true),
            PicklerHolder::Udf(udf) => {
                pending.extend(udf.arguments.iter().map(PicklerHolder::Expression));
                pending.extend(udf.python_udf.pickler.as_ref().map(PicklerHolder::Pickler));
            }
            PicklerHolder::Pickler(pickler) => objects.extend(pickler.exclusive_object()),
        }
    }
    objects
}

/// The picklers of `holders`, of their UDFs and UDTFs, below them and in the plans of their
/// subqueries, each once, in the order of the build of a request: a plan that several plans share,
/// and an expression that several expressions share, are visited once. Computed without
/// recursing.
pub(crate) fn all_picklers(holders: Vec<PicklerHolder<'_>>) -> Vec<CommandPickler> {
    let mut picklers = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut pending: Vec<PicklerHolder<'_>> = holders.into_iter().rev().collect();
    while let Some(holder) = pending.pop() {
        let mut below = Vec::new();
        match holder {
            PicklerHolder::Plan(plan) => {
                if !seen.insert(plan as *const crate::plan::LogicalPlan as usize) {
                    continue;
                }
                below.extend(
                    plan.inputs()
                        .into_iter()
                        .map(|input| PicklerHolder::Plan(input)),
                );
                plan.push_own_pickler_holders(&mut below);
            }
            PicklerHolder::Expression(expression) => {
                if !seen.insert(expression.node_address()) {
                    continue;
                }
                expression.node().push_pickler_holders(&mut below);
            }
            PicklerHolder::Subquery(subquery) => subquery.push_pickler_holders(&mut below, false),
            PicklerHolder::Udf(udf) => {
                below.extend(udf.arguments.iter().map(PicklerHolder::Expression));
                below.extend(udf.python_udf.pickler.as_ref().map(PicklerHolder::Pickler));
            }
            PicklerHolder::Pickler(pickler) => {
                if seen.insert(pickler.address()) {
                    picklers.push(pickler.clone());
                }
            }
        }
        // The first below is visited first.
        pending.extend(below.into_iter().rev());
    }
    picklers
}

/// The copies of the picklers of a deep copy, by the pickler that each is a copy of: of the copy
/// of the object of the pickler, as PySpark's deep copy of a plan or an expression copies the
/// function of each UDF of it with the memo of the call, so a function, of which a deep copy is
/// the function itself, gives a pickler of the same function, and a callable object a pickler of
/// its copy. Not a stable API.
#[doc(hidden)]
#[derive(Default)]
pub struct PicklerCopies {
    copies: std::collections::HashMap<usize, CommandPickler>,
}

impl PicklerCopies {
    pub fn new() -> Self {
        PicklerCopies::default()
    }

    /// Make `copy` the copy of `original`.
    pub fn insert(&mut self, original: &CommandPickler, copy: CommandPickler) {
        self.copies.insert(original.address(), copy);
    }

    /// Whether no pickler has a copy.
    pub fn is_empty(&self) -> bool {
        self.copies.is_empty()
    }

    /// The copy of `pickler`, if it has one.
    fn copy_of(&self, pickler: &CommandPickler) -> Option<&CommandPickler> {
        self.copies.get(&pickler.address())
    }
}

/// The rebuild of a plan, an expression or a subquery with the copies of its picklers, and, of a
/// deep copy of a DataFrame, with the copies of its Observations: a node above a pickler or an
/// Observation that has a copy is rebuilt, of the copies below it, and the others are shared. A
/// node that several nodes share is rebuilt once, so the copy shares its copy as they shared it,
/// as the memo of PySpark's deep copy shares the copy of an object. A rebuilt plan has the plan ID
/// of the plan that it is a copy of, as PySpark's copy of a plan keeps its plan ID.
pub(crate) struct PicklerRebuild<'c> {
    copies: &'c PicklerCopies,
    observations: &'c [(String, crate::observation::Observation)],
    plans: std::collections::HashMap<usize, Option<Arc<crate::plan::LogicalPlan>>>,
    expressions: std::collections::HashMap<usize, Option<Expression>>,
}

impl<'c> PicklerRebuild<'c> {
    /// The rebuild of the copies `copies` of picklers and `observations` of Observations, by
    /// their names.
    pub(crate) fn new(
        copies: &'c PicklerCopies,
        observations: &'c [(String, crate::observation::Observation)],
    ) -> Self {
        PicklerRebuild {
            copies,
            observations,
            plans: Default::default(),
            expressions: Default::default(),
        }
    }

    /// The rebuilt `plan`, or None if nothing below it has a copy. A rebuilt node has the plan ID
    /// in `ids` of the node that it is a copy of.
    pub(crate) fn plan(
        &mut self,
        plan: &Arc<crate::plan::LogicalPlan>,
        ids: Option<&crate::plan::PlanIds>,
    ) -> Option<Arc<crate::plan::LogicalPlan>> {
        use crate::plan::PlanPart;
        let key = Arc::as_ptr(plan) as usize;
        if let Some(rebuilt) = self.plans.get(&key) {
            return rebuilt.clone();
        }
        let rebuilt = crate::expression::with_stack_for_a_level(|| {
            let mut node = plan.as_ref().clone();
            let mut rebuilt = false;
            for part in node.parts_mut() {
                rebuilt |= match part {
                    PlanPart::Input(input) => self.replace_plan(input, ids),
                    PlanPart::Expression(expression) => self.replace_expression(expression),
                    PlanPart::Udf(udf) => self.udf(udf),
                    PlanPart::Pickler(pickler) => self.pickler(pickler),
                    PlanPart::Observation(name, observation) => self.observation(name, observation),
                    PlanPart::Observations(observations) => {
                        let mut copied = false;
                        for (name, observation) in observations.iter_mut() {
                            if let Some(copy) = self.copy_of_observation(name) {
                                *observation = copy;
                                copied = true;
                            }
                        }
                        copied
                    }
                };
            }
            rebuilt.then(|| Arc::new(node))
        });
        if let (Some(rebuilt), Some(ids)) = (&rebuilt, ids) {
            ids.assign(rebuilt, ids.id_of(plan));
        }
        self.plans.insert(key, rebuilt.clone());
        rebuilt
    }

    /// The rebuilt `expression`, or None if nothing below it has a copy.
    pub(crate) fn expression(&mut self, expression: &Expression) -> Option<Expression> {
        use crate::expression::ExpressionPart;
        let key = expression.node_address();
        if let Some(rebuilt) = self.expressions.get(&key) {
            return rebuilt.clone();
        }
        let rebuilt = crate::expression::with_stack_for_a_level(|| {
            let mut node = expression.node().clone();
            let mut rebuilt = false;
            for part in node.parts_mut() {
                rebuilt |= match part {
                    ExpressionPart::Expression(child) => self.replace_expression(child),
                    ExpressionPart::Udf(udf) => self.udf(udf),
                    ExpressionPart::Subquery(subquery) => self.subquery(subquery),
                };
            }
            rebuilt.then(|| expression.of_node(node))
        });
        self.expressions.insert(key, rebuilt.clone());
        rebuilt
    }

    /// Rebuild `subquery` in place: its plan, of the plan IDs of its session, and its
    /// expressions. Whether anything below it has a copy.
    pub(crate) fn subquery(
        &mut self,
        subquery: &mut crate::expression::SubqueryExpression,
    ) -> bool {
        let ids = subquery.plan_ids.clone();
        let mut rebuilt = self.replace_plan(&mut subquery.plan, ids.as_deref());
        for expression in subquery
            .partition_spec
            .iter_mut()
            .chain(subquery.order_spec.iter_mut())
            .chain(subquery.in_subquery_values.iter_mut())
        {
            rebuilt |= self.replace_expression(expression);
        }
        rebuilt
    }

    fn replace_plan(
        &mut self,
        plan: &mut Arc<crate::plan::LogicalPlan>,
        ids: Option<&crate::plan::PlanIds>,
    ) -> bool {
        match self.plan(plan, ids) {
            Some(rebuilt) => {
                *plan = rebuilt;
                true
            }
            None => false,
        }
    }

    fn replace_expression(&mut self, expression: &mut Expression) -> bool {
        match self.expression(expression) {
            Some(rebuilt) => {
                *expression = rebuilt;
                true
            }
            None => false,
        }
    }

    fn udf(&mut self, udf: &mut CommonInlineUserDefinedFunctionExpression) -> bool {
        let mut rebuilt = false;
        for argument in udf.arguments.iter_mut() {
            rebuilt |= self.replace_expression(argument);
        }
        rebuilt | self.pickler(&mut udf.python_udf.pickler)
    }

    fn pickler(&self, pickler: &mut Option<CommandPickler>) -> bool {
        match pickler
            .as_ref()
            .and_then(|pickler| self.copies.copy_of(pickler))
        {
            Some(copy) => {
                *pickler = Some(copy.clone());
                true
            }
            None => false,
        }
    }

    fn observation(
        &self,
        name: &str,
        observation: &mut Option<crate::observation::Observation>,
    ) -> bool {
        match (observation.is_some(), self.copy_of_observation(name)) {
            (true, Some(copy)) => {
                *observation = Some(copy);
                true
            }
            _ => false,
        }
    }

    fn copy_of_observation(&self, name: &str) -> Option<crate::observation::Observation> {
        self.observations
            .iter()
            .find(|(copied, _)| copied == name)
            .map(|(_, copy)| copy.clone())
    }
}

/// The command of a UDF or UDTF as a request builds it: that of `pickler`, which pickles it
/// then, or the bytes of `command`.
pub(crate) fn command_bytes(command: &[u8], pickler: Option<&CommandPickler>) -> bytes::Bytes {
    match pickler {
        Some(pickler) => bytes::Bytes::from(crate::expression::pickled_command(pickler)),
        None => bytes::Bytes::copy_from_slice(command),
    }
}

/// Represents a Python UDF with its serialized command and metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct PythonUDFPayload {
    /// The output data type of the UDF.
    pub output_type: DataType,
    /// The evaluation type (e.g., SQL_BATCHED_UDF, SQL_SCALAR_PANDAS_UDF).
    pub eval_type: i32,
    /// The cloudpickled command bytes: typically cloudpickle.dumps((func, output_type)). Empty
    /// for a payload of a `pickler`.
    pub command: Vec<u8>,
    /// Python version used for pickling (e.g., "3.9", "3.11").
    pub python_ver: String,
    /// The pickling of the command, which a request calls when it builds the plan, in place of
    /// `command`, as PySpark pickles the function of a UDF for each plan that it builds. Not a
    /// stable API.
    #[doc(hidden)]
    pub pickler: Option<CommandPickler>,
}

impl PythonUDFPayload {
    /// Create a new Python UDF payload.
    pub fn new(
        output_type: DataType,
        eval_type: i32,
        command: Vec<u8>,
        python_ver: String,
    ) -> Self {
        PythonUDFPayload {
            output_type,
            eval_type,
            command,
            python_ver,
            pickler: None,
        }
    }

    /// A payload whose command `pickler` pickles each time a request builds the plan of the UDF,
    /// as PySpark's `PythonUDF.to_plan` does. Not a stable API.
    #[doc(hidden)]
    pub fn pickled_at_build(
        output_type: DataType,
        eval_type: i32,
        pickler: CommandPickler,
        python_ver: String,
    ) -> Self {
        PythonUDFPayload {
            pickler: Some(pickler),
            ..PythonUDFPayload::new(output_type, eval_type, Vec::new(), python_ver)
        }
    }

    /// Convert to a proto PythonUDF message.
    pub fn to_proto(&self) -> proto::PythonUdf {
        let mut proto = proto::PythonUdf::default();
        proto.output_type = Some(self.output_type.to_proto());
        proto.eval_type = self.eval_type;
        proto.command = command_bytes(&self.command, self.pickler.as_ref());
        proto.python_ver = self.python_ver.clone();
        proto
    }
}

/// Represents a CommonInlineUserDefinedFunction expression.
/// Wraps a Python UDF with its name, determinism flag, and arguments.
#[derive(Debug, Clone, PartialEq)]
pub struct CommonInlineUserDefinedFunctionExpression {
    /// Name of the UDF (e.g., "my_func").
    pub function_name: String,
    /// Whether the UDF is deterministic.
    pub deterministic: bool,
    /// Argument expressions passed to the UDF.
    pub arguments: Vec<Expression>,
    /// The Python UDF payload (command, output type, eval type, python version).
    pub python_udf: PythonUDFPayload,
}

impl CommonInlineUserDefinedFunctionExpression {
    /// Create a new CommonInlineUserDefinedFunction expression.
    pub fn new(
        function_name: String,
        deterministic: bool,
        arguments: Vec<Expression>,
        python_udf: PythonUDFPayload,
    ) -> Self {
        CommonInlineUserDefinedFunctionExpression {
            function_name,
            deterministic,
            arguments,
            python_udf,
        }
    }

    /// Convert to a proto CommonInlineUserDefinedFunction message.
    pub fn to_proto(&self) -> proto::CommonInlineUserDefinedFunction {
        let mut proto = proto::CommonInlineUserDefinedFunction::default();
        proto.function_name = self.function_name.clone();
        proto.deterministic = self.deterministic;
        proto.arguments = self.arguments.iter().map(|expr| expr.to_proto()).collect();
        proto.is_distinct = false;
        proto.function = Some(
            proto::common_inline_user_defined_function::Function::PythonUdf(
                self.python_udf.to_proto(),
            ),
        );
        proto
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_python_udf_payload_to_proto() {
        let payload = PythonUDFPayload::new(
            DataType::Integer,
            eval_type::SQL_BATCHED_UDF,
            vec![1, 2, 3, 4, 5],
            "3.9".to_string(),
        );

        let proto = payload.to_proto();

        assert_eq!(proto.eval_type, 100); // SQL_BATCHED_UDF
        assert_eq!(proto.python_ver, "3.9");
        assert_eq!(proto.command.len(), 5);
        assert!(proto.output_type.is_some());
    }

    #[test]
    fn test_a_payload_of_a_pickler_pickles_its_command_each_time_it_is_built() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let picklings = Arc::new(AtomicUsize::new(0));
        let pickler = CommandPickler::new(Arc::clone(&picklings), |counted| {
            let n = counted.fetch_add(1, Ordering::SeqCst);
            Ok(format!("command {n}").into_bytes())
        });
        let payload = PythonUDFPayload::pickled_at_build(
            DataType::Long,
            eval_type::SQL_SCALAR_PANDAS_UDF,
            pickler.clone(),
            "3.11".to_string(),
        );
        assert_eq!(picklings.load(Ordering::SeqCst), 0);
        assert_eq!(payload.to_proto().command.as_ref(), b"command 0");
        assert_eq!(payload.to_proto().command.as_ref(), b"command 1");
        assert_eq!(payload.to_proto().eval_type, 200);
        assert!(payload.command.is_empty());
        // A copy shares the pickler, so it is equal; a payload of bytes is not.
        assert_eq!(payload.clone(), payload);
        let other = PythonUDFPayload::pickled_at_build(
            DataType::Long,
            eval_type::SQL_SCALAR_PANDAS_UDF,
            CommandPickler::new((), |_| Ok(Vec::new())),
            "3.11".to_string(),
        );
        assert_ne!(payload, other);
        assert_ne!(
            payload,
            PythonUDFPayload::new(DataType::Long, 200, Vec::new(), "3.11".to_string())
        );
    }

    /// A UDF of `x` whose pickler's object is `tag`.
    fn udf_of(tag: &'static str) -> CommonInlineUserDefinedFunctionExpression {
        let pickler = CommandPickler::new(tag, |tag| Ok(tag.as_bytes().to_vec()));
        CommonInlineUserDefinedFunctionExpression::new(
            tag.to_string(),
            true,
            vec![Expression::column("x")],
            PythonUDFPayload::pickled_at_build(DataType::Long, 100, pickler, "3.11".to_string()),
        )
    }

    fn udf_expression(tag: &'static str) -> Expression {
        crate::expression::ExpressionNode::CommonInlineUserDefinedFunction(Box::new(udf_of(tag)))
            .into()
    }

    /// The tags of `objects`, of picklers of `udf_of`, sorted.
    fn tags(objects: Vec<&(dyn std::any::Any + Send + Sync)>) -> Vec<&'static str> {
        let mut tags: Vec<&'static str> = objects
            .into_iter()
            .map(|object| *object.downcast_ref::<&'static str>().unwrap())
            .collect();
        tags.sort();
        tags
    }

    #[test]
    fn test_the_pickler_objects_of_a_holder_are_those_that_it_alone_holds() {
        use crate::plan::LogicalPlan;
        // Of an expression, below a function too; a node that a clone shares is not walked, so
        // that no object is counted twice.
        let a = udf_expression("a");
        assert_eq!(tags(a.exclusive_pickler_objects()), ["a"]);
        let shared = a.clone();
        assert!(a.exclusive_pickler_objects().is_empty());
        drop(shared);
        let both = Expression::function("struct", vec![udf_expression("b"), a]);
        assert_eq!(tags(both.exclusive_pickler_objects()), ["a", "b"]);
        // A pickler that a copy of the UDF shares is not either.
        let copied = udf_of("c");
        let copy = copied.clone();
        let holders = vec![PicklerHolder::Udf(&copied)];
        assert!(exclusive_pickler_objects(holders).is_empty());
        drop(copy);
        // Of a plan: of the expressions and functions of its nodes, below the inputs that it
        // alone holds, and of a UDTF.
        let input = Arc::new(LogicalPlan::Project {
            input: Arc::new(crate::plan::range(0, 1, 1)),
            columns: vec![crate::column::Column::new(udf_expression("project"))],
        });
        let mapped = LogicalPlan::MapPartitions {
            input: Arc::clone(&input),
            func: udf_of("map"),
            is_barrier: false,
            profile_id: None,
            profile_id_reader: None,
        };
        let holders = || vec![PicklerHolder::Plan(&mapped)];
        assert_eq!(tags(exclusive_pickler_objects(holders())), ["map"]);
        drop(input);
        assert_eq!(
            tags(exclusive_pickler_objects(holders())),
            ["map", "project"]
        );
        let udtf = LogicalPlan::CommonInlineUdtf {
            function_name: "t".to_string(),
            deterministic: false,
            arguments: vec![udf_expression("argument")],
            return_type: None,
            eval_type: eval_type::SQL_TABLE_UDF,
            command: Vec::new(),
            python_ver: "3.11".to_string(),
            pickler: Some(CommandPickler::new("handler", |_| Ok(Vec::new()))),
        };
        let holders = vec![PicklerHolder::Plan(&udtf)];
        assert_eq!(
            tags(exclusive_pickler_objects(holders)),
            ["argument", "handler"]
        );
    }

    /// The tag of the object of the pickler of the UDF `expression`.
    fn tag_of(expression: &Expression) -> &str {
        let crate::expression::ExpressionNode::CommonInlineUserDefinedFunction(udf) =
            expression.node()
        else {
            panic!("a UDF");
        };
        let pickler = udf.python_udf.pickler.as_ref().unwrap();
        pickler.object().downcast_ref::<&'static str>().unwrap()
    }

    /// The arguments of the function `expression`.
    fn arguments(expression: &Expression) -> &[Expression] {
        let crate::expression::ExpressionNode::UnresolvedFunction(function) = expression.node()
        else {
            panic!("a function");
        };
        &function.args
    }

    #[test]
    fn test_a_deep_copy_rebuilds_what_holds_the_copies_of_picklers_and_shares_the_rest() {
        use crate::plan::{LogicalPlan, PlanIds};
        // The picklers of an expression, each once, in the order of the build: a node that the
        // expression holds twice is visited once.
        let a = udf_expression("a");
        let plain = Expression::function("upper", vec![Expression::column("y")]);
        let expression = Expression::function(
            "struct",
            vec![a.clone(), udf_expression("b"), a.clone(), plain.clone()],
        );
        let picklers = expression.picklers();
        let names: Vec<&str> = picklers
            .iter()
            .map(|p| *p.object().downcast_ref::<&'static str>().unwrap())
            .collect();
        assert_eq!(names, ["a", "b"]);
        // Of a copy of `a` only: the node of `a` is rebuilt once, and shared by its two places, as
        // the memo of PySpark's deep copy shares the copy of an object; `b` and the plain
        // subtree are the nodes of the original.
        let mut copies = PicklerCopies::new();
        copies.insert(&picklers[0], CommandPickler::new("a'", |_| Ok(Vec::new())));
        let copied = expression.with_pickler_copies(&copies);
        let [first, b, second, copied_plain] = arguments(&copied) else {
            panic!("four arguments");
        };
        assert_eq!(
            (tag_of(first), tag_of(b), tag_of(second)),
            ("a'", "b", "a'")
        );
        assert_eq!(first.node_address(), second.node_address());
        assert_eq!(b.node_address(), arguments(&expression)[1].node_address());
        assert_eq!(copied_plain.node_address(), plain.node_address());
        assert_eq!(tag_of(&arguments(&expression)[0]), "a");
        // Without copies, the expression is shared as it is.
        let none = expression.with_pickler_copies(&PicklerCopies::new());
        assert_eq!(none.node_address(), expression.node_address());
        // A plan: the nodes above a copy are rebuilt, of the plan IDs of the nodes that they are
        // copies of, and the others shared.
        let ids = PlanIds::new();
        let input = Arc::new(LogicalPlan::Project {
            input: Arc::new(crate::plan::range(0, 1, 1)),
            columns: vec![crate::column::Column::new(udf_expression("c"))],
        });
        let plan = Arc::new(LogicalPlan::Filter {
            input: Arc::clone(&input),
            condition: crate::column::Column::new(Expression::column("x")),
        });
        let (input_id, plan_id) = (ids.id_of(&input), ids.id_of(&plan));
        let picklers = all_picklers(vec![PicklerHolder::Plan(&plan)]);
        let mut copies = PicklerCopies::new();
        copies.insert(&picklers[0], CommandPickler::new("c'", |_| Ok(Vec::new())));
        let rebuilt = PicklerRebuild::new(&copies, &[])
            .plan(&plan, Some(&ids))
            .expect("a copy below");
        assert!(!Arc::ptr_eq(&rebuilt, &plan));
        assert_eq!(ids.id_of(&rebuilt), plan_id);
        let LogicalPlan::Filter {
            input: copied_input,
            ..
        } = rebuilt.as_ref()
        else {
            panic!("a filter");
        };
        assert_eq!(ids.id_of(copied_input), input_id);
        let LogicalPlan::Project {
            input: range,
            columns,
        } = copied_input.as_ref()
        else {
            panic!("a project");
        };
        assert_eq!(tag_of(columns[0].expression()), "c'");
        let LogicalPlan::Project {
            input: original_range,
            ..
        } = input.as_ref()
        else {
            panic!("a project");
        };
        assert!(Arc::ptr_eq(range, original_range));
    }

    #[test]
    fn test_common_inline_udf_expression_to_proto() {
        let payload = PythonUDFPayload::new(
            DataType::String {
                collation: "UTF8_BINARY".to_string(),
            },
            eval_type::SQL_SCALAR_PANDAS_UDF,
            b"pickled_command_bytes".to_vec(),
            "3.11".to_string(),
        );

        let udf_expr = CommonInlineUserDefinedFunctionExpression::new(
            "my_udf".to_string(),
            true,
            vec![],
            payload,
        );

        let proto = udf_expr.to_proto();

        assert_eq!(proto.function_name, "my_udf");
        assert!(proto.deterministic);
        assert!(!proto.is_distinct);
        assert_eq!(proto.arguments.len(), 0);
        assert!(proto.function.is_some());

        // Verify the Python UDF is embedded
        if let Some(proto::common_inline_user_defined_function::Function::PythonUdf(py_udf)) =
            proto.function
        {
            assert_eq!(py_udf.eval_type, 200); // SQL_SCALAR_PANDAS_UDF
            assert_eq!(py_udf.python_ver, "3.11");
            assert_eq!(
                py_udf.command,
                bytes::Bytes::copy_from_slice(b"pickled_command_bytes")
            );
        } else {
            panic!("Expected Python UDF in CommonInlineUserDefinedFunction");
        }
    }

    #[test]
    fn test_eval_type_constants() {
        // Verify eval type constants match pyspark.util.PythonEvalType
        assert_eq!(eval_type::SQL_BATCHED_UDF, 100);
        assert_eq!(eval_type::SQL_ARROW_BATCHED_UDF, 101);
        assert_eq!(eval_type::SQL_SCALAR_PANDAS_UDF, 200);
        assert_eq!(eval_type::SQL_GROUPED_MAP_PANDAS_UDF, 201);
        assert_eq!(eval_type::SQL_GROUPED_AGG_PANDAS_UDF, 202);
        assert_eq!(eval_type::SQL_WINDOW_AGG_PANDAS_UDF, 203);
        assert_eq!(eval_type::SQL_SCALAR_PANDAS_ITER_UDF, 204);
        assert_eq!(eval_type::SQL_MAP_PANDAS_ITER_UDF, 205);
        assert_eq!(eval_type::SQL_COGROUPED_MAP_PANDAS_UDF, 206);
        assert_eq!(eval_type::SQL_MAP_ARROW_ITER_UDF, 207);
        assert_eq!(eval_type::SQL_GROUPED_MAP_ARROW_UDF, 209);
        assert_eq!(eval_type::SQL_COGROUPED_MAP_ARROW_UDF, 210);
        assert_eq!(eval_type::SQL_SCALAR_ARROW_UDF, 250);
        assert_eq!(eval_type::SQL_SCALAR_ARROW_ITER_UDF, 251);
        assert_eq!(eval_type::SQL_GROUPED_AGG_ARROW_UDF, 252);
        assert_eq!(eval_type::SQL_WINDOW_AGG_ARROW_UDF, 253);
        assert_eq!(eval_type::SQL_GROUPED_AGG_ARROW_ITER_UDF, 254);
        assert_eq!(eval_type::SQL_TABLE_UDF, 300);
        assert_eq!(eval_type::SQL_ARROW_TABLE_UDF, 301);
        assert_eq!(eval_type::SQL_ARROW_UDTF, 302);
    }
}
