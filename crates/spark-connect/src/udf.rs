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
pub struct CommandPickler(Arc<dyn Pickling>);

/// A handle of the same pickling, of which the check of the clones of a binding is told (see
/// [`crate::gc::check_clones_with`]).
impl Clone for CommandPickler {
    fn clone(&self) -> Self {
        crate::gc::cloned(self.address());
        CommandPickler(Arc::clone(&self.0))
    }
}

/// The object of a [`CommandPickler`] and its pickling.
trait Pickling: Send + Sync {
    fn pickle(&self) -> Result<Vec<u8>>;
    fn object(&self) -> &(dyn std::any::Any + Send + Sync);
    /// Whether the pickle is a `PythonUdf` message of the output type and the command of a UDF,
    /// or a `PythonUdtf` message of the return type and the command of a UDTF (see
    /// [`CommandPickler::with_output_type`]).
    fn gives_output_type(&self) -> bool;
}

struct PicklingOf<O, F> {
    object: O,
    pickle: F,
    gives_output_type: bool,
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

    fn gives_output_type(&self) -> bool {
        self.gives_output_type
    }
}

impl CommandPickler {
    /// The pickler of `object`, which `pickle` pickles: it gives the bytes of the command, or the
    /// error that the request raises.
    pub fn new<O: std::any::Any + Send + Sync>(
        object: O,
        pickle: impl Fn(&O) -> Result<Vec<u8>> + Send + Sync + 'static,
    ) -> Self {
        CommandPickler(Arc::new(PicklingOf {
            object,
            pickle,
            gives_output_type: false,
        }))
    }

    /// The pickler of the command of a Python UDF and of its output type, both of `object`, as
    /// PySpark's `PythonUDF.to_plan` converts the output type of the UDF and pickles it with the
    /// function each time it builds the plan, of the one object that it keeps: so a type that was
    /// changed in place after the UDF was applied is the type that the request sends and the
    /// function's command carries. `pickle` gives the bytes of a `PythonUdf` message of the
    /// `output_type` and the `command`, which [`PythonUDFPayload::to_proto`] takes, or the error
    /// that the request raises. Of a UDTF, as PySpark's `PythonUDTF.to_plan` converts the return
    /// type that it keeps and then pickles the handler class at each build, `pickle` gives a
    /// `PythonUdtf` message of the `return_type`, none for a UDTF of none, and the `command`,
    /// which the plan of the UDTF sends (`TableValuedFunction::udtf_pickled_at_build`). Not a
    /// stable API.
    #[doc(hidden)]
    pub fn with_output_type<O: std::any::Any + Send + Sync>(
        object: O,
        pickle: impl Fn(&O) -> Result<Vec<u8>> + Send + Sync + 'static,
    ) -> Self {
        CommandPickler(Arc::new(PicklingOf {
            object,
            pickle,
            gives_output_type: true,
        }))
    }

    /// The command as it pickles now.
    pub(crate) fn pickle(&self) -> Result<Vec<u8>> {
        self.0.pickle()
    }

    /// Whether this pickler gives the output type of its UDF, or the return type of its UDTF,
    /// with the command ([`CommandPickler::with_output_type`]).
    pub(crate) fn gives_output_type(&self) -> bool {
        self.0.gives_output_type()
    }

    /// The object of this pickler, which a deep copy of the UDF copies, as PySpark's deep copy of
    /// a UDF copies its function (see [`PicklerCopies`]). Not a stable API.
    #[doc(hidden)]
    pub fn object(&self) -> &(dyn std::any::Any + Send + Sync) {
        self.0.object()
    }

    /// The address of the pickling of this pickler, which its copies share.
    pub(crate) fn address(&self) -> usize {
        Arc::as_ptr(&self.0) as *const () as usize
    }

    /// The number of the references to the pickling of this pickler: of this pickler, of its
    /// copies and of any other holder of it.
    pub(crate) fn strong_count(&self) -> usize {
        Arc::strong_count(&self.0)
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

/// A plan, an expression, a subquery, a UDF, the pickler of a UDTF, or the plans of the
/// subqueries of an expression as it was sent, of which [`all_picklers`] visits the picklers.
pub(crate) enum PicklerHolder<'a> {
    Plan(&'a crate::plan::LogicalPlan),
    Expression(&'a Expression),
    Subquery(&'a crate::expression::SubqueryExpression),
    Udf(&'a CommonInlineUserDefinedFunctionExpression),
    Pickler(&'a CommandPickler),
    SentPlans(&'a crate::expression::SentSubqueries),
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
            PicklerHolder::Subquery(subquery) => subquery.push_pickler_holders(&mut below),
            PicklerHolder::Udf(udf) => {
                below.extend(udf.arguments.iter().map(PicklerHolder::Expression));
                below.extend(udf.python_udf.pickler.as_ref().map(PicklerHolder::Pickler));
            }
            PicklerHolder::Pickler(pickler) => {
                if seen.insert(pickler.address()) {
                    picklers.push(pickler.clone());
                }
            }
            PicklerHolder::SentPlans(plans) => {
                below.extend(plans.plans().map(|plan| PicklerHolder::Plan(plan)));
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
    /// in `ids` of the node that it is a copy of, and its root plan ID within the `WithRelations`
    /// that sends it with the plans of its subqueries, if it has them, as PySpark's copy of a plan
    /// keeps its `_plan_id` and its `_root_plan_id`.
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
                    PlanPart::Input(input) => match self.plan(input.arc(), ids) {
                        Some(rebuilt) => {
                            *input = rebuilt.into();
                            true
                        }
                        None => false,
                    },
                    PlanPart::ArcInput(input) => self.replace_plan(input, ids),
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
            if crate::plan::has_subquery_references(plan) {
                ids.assign_root(rebuilt, ids.root_id_of(plan));
            }
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
                    ExpressionPart::Pickler(pickler) => self.pickler(pickler),
                    ExpressionPart::Plan(plan, ids) => self.replace_plan(plan, Some(ids)),
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
        // By the name of the plan as it is built now, by which the copies are named (see
        // `LogicalPlan::observations`).
        let current = observation
            .as_ref()
            .and_then(crate::observation::Observation::name_of_plan)
            .and_then(Result::ok);
        let name = current.as_deref().unwrap_or(name);
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

/// The kind of the message that a hole of a relation or an expression as it was sent holds, which
/// its pickler gives at each request (see [`SentHole`]). Not a stable API.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SentHoleKind {
    /// A `PythonUdf`, whose output type and command the pickler gives, as a pickler of a UDF of
    /// the output type does ([`CommandPickler::with_output_type`]); its other fields stay.
    PythonUdf,
    /// A `PythonUdtf`, whose return type and command the pickler gives; its other fields stay.
    PythonUdtf,
    /// A `DataType`, which the pickler gives, as the reader of the schema of `DataFrame.to` does.
    DataType,
}

/// A hole of a relation or an expression as it was sent, such as that of a pickled DataFrame or
/// Column, which carries the functions of its UDFs rather than their commands, as PySpark's
/// pickle of a plan carries the plan objects: `marker` stands in it for the command of a UDF (or
/// the string of an unparsed type, of a `DataType`), and a request replaces the message of the
/// kind `kind` that holds it with the message that `pickler` gives when the request builds the
/// plan, as PySpark's unpickled plan pickles the functions of its UDFs at each request. Not a
/// stable API.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct SentHole {
    pub marker: Vec<u8>,
    pub kind: SentHoleKind,
    pub pickler: Option<CommandPickler>,
}

impl PartialEq for SentHole {
    fn eq(&self, other: &Self) -> bool {
        self.marker == other.marker && self.kind == other.kind && self.pickler == other.pickler
    }
}

/// The holes of a relation or an expression as it was sent, in the order in which the plan that
/// it is of built them, in which a request pickles them, as PySpark's builds its plan. Not a
/// stable API.
#[doc(hidden)]
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SentHoles(pub Vec<SentHole>);

impl SentHoles {
    /// Whether there are no holes.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The encoding of the relation or the expression of these holes, which `encode` gives, with
    /// each hole filled with the message that its pickler gives as a request builds the plan, in
    /// turn (see [`crate::expression::pickled_command`]); None of the first build of a request,
    /// which meets the picklers and gives no messages, so the encoding stays as it is, and is not
    /// made. The places of the markers are found in one pass of the encoding, a field is read only
    /// when its value holds one, a marker is a hole only as the whole value of a field, and the
    /// filled encoding is written once, of the lengths of the fields that change, computed first:
    /// a time of the length of the encoding, however many holes it has.
    pub(crate) fn fill(&self, encode: impl FnOnce() -> Vec<u8>) -> Result<Option<Vec<u8>>> {
        let fills: Vec<Vec<u8>> = self
            .0
            .iter()
            .map(|hole| {
                hole.pickler
                    .as_ref()
                    .map(crate::expression::pickled_command)
                    .unwrap_or_default()
            })
            .collect();
        let filled: std::collections::HashMap<&[u8], usize> = self
            .0
            .iter()
            .enumerate()
            .filter(|(index, _)| !fills[*index].is_empty())
            .map(|(index, hole)| (hole.marker.as_slice(), index))
            .collect();
        if filled.is_empty() {
            return Ok(None);
        }
        let encoded = encode();
        let places = marker_places(&encoded, &filled);
        let mut fill = HoleFill {
            holes: &self.0,
            fills: &fills,
            places: &places,
            found: vec![false; self.0.len()],
        };
        let outcome = fill.message(&encoded, 0)?;
        let missing = filled.values().any(|&index| !fill.found[index]);
        match outcome {
            // The message of a hole is the value of a field, so never the whole encoding.
            FillOutcome::Changed(edit @ FillEdit::Fields(..)) if !missing => {
                let mut out = Vec::with_capacity(edit.len());
                edit.write(&encoded, &mut out);
                Ok(Some(out))
            }
            _ => Err(missing_hole()),
        }
    }
}

/// The error of a relation or an expression as it was sent without the place of a hole.
fn missing_hole() -> spark_connect_core::error::SparkError {
    spark_connect_core::error::SparkError::connect_msg(
        "a relation or an expression as it was sent has no place of the command of a UDF of its \
         pickle",
    )
}

/// The stack that the fill of the holes of a relation or an expression as it was sent takes: its
/// encode, and the decode of the filled bytes, recurse for each level of its messages, of those of
/// its expressions too, of which a pickled object has at most the levels that the client bounds,
/// as the binding decodes such a message (frames of about 1 KiB in a release build, and more in a
/// debug one).
pub(crate) const SENT_FILL_STACK: usize = if cfg!(debug_assertions) {
    16 << 20
} else {
    4 << 20
};

/// The start of each marker of a hole ([`SentHoleMarkers`]).
const MARKER_PREFIX: &str = "spark-connect-rust-udf-";

/// The length of each marker of a hole: the prefix, the token of its maker (32 hexadecimal
/// digits), a dash and its number (20 decimal digits). All markers have this length, so none is
/// the prefix of another.
const MARKER_LEN: usize = MARKER_PREFIX.len() + 32 + 1 + 20;

/// The maker of the markers of the holes of one pickle ([`SentHole`]): each marker is of a token
/// of this maker, which no other has, and of its number, of a fixed width, so that no marker is
/// the prefix of another; ASCII, so that it stands for the bytes of a command and for a string
/// too. Not a stable API.
#[doc(hidden)]
pub struct SentHoleMarkers {
    token: String,
    next: u64,
}

impl Default for SentHoleMarkers {
    fn default() -> Self {
        SentHoleMarkers {
            token: uuid::Uuid::new_v4().simple().to_string(),
            next: 0,
        }
    }
}

impl SentHoleMarkers {
    /// A new maker of markers.
    pub fn new() -> Self {
        SentHoleMarkers::default()
    }

    /// The next marker, of no other hole.
    pub fn marker(&mut self) -> Vec<u8> {
        let marker = format!("{MARKER_PREFIX}{}-{:020}", self.token, self.next);
        debug_assert_eq!(marker.len(), MARKER_LEN);
        self.next += 1;
        marker.into_bytes()
    }
}

/// The places of the markers of `holes` in `encoded`, the holes by their markers: the offset of
/// each marker, of the start of the markers of holes, and the index of its hole, in order, found
/// in one pass.
fn marker_places(
    encoded: &[u8],
    holes: &std::collections::HashMap<&[u8], usize>,
) -> Vec<(usize, usize)> {
    let prefix = MARKER_PREFIX.as_bytes();
    let mut places = Vec::new();
    let mut at = 0;
    while let Some(found) = encoded[at..].iter().position(|&byte| byte == prefix[0]) {
        let offset = at + found;
        if let Some(&hole) = encoded
            .get(offset..offset + MARKER_LEN)
            .filter(|marker| marker.starts_with(prefix))
            .and_then(|marker| holes.get(marker))
        {
            places.push((offset, hole));
            at = offset + MARKER_LEN;
        } else {
            at = offset + 1;
        }
    }
    places
}

/// Whether the bytes `start..end` of an encoding of the marker `places` hold a marker whole.
fn holds_a_marker(places: &[(usize, usize)], start: usize, end: usize) -> bool {
    let first = places.partition_point(|&(offset, _)| offset < start);
    places
        .get(first)
        .is_some_and(|&(offset, _)| offset + MARKER_LEN <= end)
}

/// The hole whose marker is the bytes `start..end` of an encoding of the marker `places`, if
/// they are one whole.
fn marker_at(places: &[(usize, usize)], start: usize, end: usize) -> Option<usize> {
    if end - start != MARKER_LEN {
        return None;
    }
    let index = places
        .binary_search_by_key(&start, |&(offset, _)| offset)
        .ok()?;
    Some(places[index].1)
}

/// A field of an encoded message: its wire type, the end of its key, and the start and the end
/// of its value.
struct WireField {
    wire_type: u64,
    key_end: usize,
    value_start: usize,
    end: usize,
}

/// The field of `encoded`, read as a message, at `pos`; None when the bytes there are not one, as
/// of a group, which a proto3 file has none of, of the number 0 or of a value past the end.
fn wire_field(encoded: &[u8], pos: usize) -> Option<WireField> {
    let (key, key_end) = read_varint(encoded, pos)?;
    let wire_type = key & 7;
    let (value_start, end) = match wire_type {
        0 => read_varint(encoded, key_end).map(|(_, end)| (end, end))?,
        1 => (key_end, key_end + 8),
        2 => {
            let (length, start) = read_varint(encoded, key_end)?;
            (start, start.checked_add(usize::try_from(length).ok()?)?)
        }
        5 => (key_end, key_end + 4),
        _ => return None,
    };
    (key >> 3 != 0 && end <= encoded.len()).then_some(WireField {
        wire_type,
        key_end,
        value_start,
        end,
    })
}

/// Which of `markers` the message `encoded` holds as the whole value of a field, as the fill of
/// the holes of a relation or an expression as it was sent finds them ([`SentHoles`]): a marker
/// that is only a part of a value, such as of the bytes of a command, is no place of a hole. In
/// one pass of the encoding, as the fill. Not a stable API.
#[doc(hidden)]
pub fn held_markers(encoded: &[u8], markers: &[&[u8]]) -> Vec<bool> {
    let holes: std::collections::HashMap<&[u8], usize> = markers
        .iter()
        .enumerate()
        .map(|(index, marker)| (*marker, index))
        .collect();
    let places = marker_places(encoded, &holes);
    let mut found = vec![false; markers.len()];
    find_held_markers(encoded, 0, &places, &mut found);
    found
}

/// The fields of `encoded`, the bytes at `base` of an encoding of the marker `places`, read as a
/// message, of the wire type of length-delimited values, whose values hold a marker whole; None
/// when the bytes are not a message, as those of a string, which holds no hole then. The whole
/// message is read before any of its fields is.
fn fields_holding_markers(
    encoded: &[u8],
    base: usize,
    places: &[(usize, usize)],
) -> Option<Vec<WireField>> {
    let mut holding = Vec::new();
    let mut pos = 0;
    while pos < encoded.len() {
        let field = wire_field(encoded, pos)?;
        pos = field.end;
        if field.wire_type == 2
            && holds_a_marker(places, base + field.value_start, base + field.end)
        {
            holding.push(field);
        }
    }
    Some(holding)
}

/// The hole whose marker is the whole value of one of `fields`, fields of the bytes at `base` of
/// an encoding of the marker `places`, if one is.
fn hole_of_fields(fields: &[WireField], base: usize, places: &[(usize, usize)]) -> Option<usize> {
    fields
        .iter()
        .find_map(|field| marker_at(places, base + field.value_start, base + field.end))
}

/// Mark in `found` the holes of the markers of `places` that `encoded`, the bytes at `base` of the
/// encoding, read as a message, holds as the whole value of a field (see [`held_markers`]).
fn find_held_markers(encoded: &[u8], base: usize, places: &[(usize, usize)], found: &mut [bool]) {
    if !holds_a_marker(places, base, base + encoded.len()) {
        return;
    }
    let Some(fields) = fields_holding_markers(encoded, base, places) else {
        return;
    };
    if let Some(hole) = hole_of_fields(&fields, base, places) {
        found[hole] = true;
        return;
    }
    for field in fields {
        crate::expression::with_stack_for_a_level(|| {
            find_held_markers(
                &encoded[field.value_start..field.end],
                base + field.value_start,
                places,
                found,
            )
        });
    }
}

/// The fill of the holes of an encoding, of the places of their markers.
struct HoleFill<'a> {
    holes: &'a [SentHole],
    fills: &'a [Vec<u8>],
    places: &'a [(usize, usize)],
    /// Whether the marker of each hole was found as the value of a field.
    found: Vec<bool>,
}

/// What the fill of the holes does to a message.
enum FillOutcome {
    Unchanged,
    Changed(FillEdit),
    /// The message of a hole is the one above this one, as the `DataType` that holds the unparsed
    /// type of a marker is above the message of the unparsed type.
    Above(usize),
}

/// The change of an encoding that the fill makes.
enum FillEdit {
    /// The message of a hole, replaced by its message.
    Replaced(Vec<u8>),
    /// The fields of a message whose values change, in order, and the length of the encoding with
    /// the changes.
    Fields(Vec<FilledField>, usize),
}

/// A field of a message whose value changes: the offsets in the message of the end of its key, the
/// start of its value and its end, and the change of its value.
struct FilledField {
    key_end: usize,
    value_start: usize,
    end: usize,
    edit: FillEdit,
}

impl FillEdit {
    /// The length of the changed encoding.
    fn len(&self) -> usize {
        match self {
            FillEdit::Replaced(bytes) => bytes.len(),
            FillEdit::Fields(_, len) => *len,
        }
    }

    /// Write the changed encoding of `encoded`, the encoding that changes, into `out`.
    fn write(&self, encoded: &[u8], out: &mut Vec<u8>) {
        match self {
            FillEdit::Replaced(bytes) => out.extend_from_slice(bytes),
            FillEdit::Fields(fields, _) => {
                let mut at = 0;
                for field in fields {
                    out.extend_from_slice(&encoded[at..field.key_end]);
                    write_varint(field.edit.len() as u64, out);
                    let value = &encoded[field.value_start..field.end];
                    crate::expression::with_stack_for_a_level(|| field.edit.write(value, out));
                    at = field.end;
                }
                out.extend_from_slice(&encoded[at..]);
            }
        }
    }
}

impl HoleFill<'_> {
    /// What the fill does to `encoded`, the bytes at `base` of the encoding, read as a message: a
    /// field of the wire type of length-delimited values whose value is a marker whole makes this
    /// message the one that holds the hole, which is replaced (that of a `PythonUdf` or a
    /// `PythonUdtf`) or is the message below the one that is (the unparsed type of a `DataType`);
    /// else each field whose value holds a marker is read as a message in turn. An encoding that
    /// does not read as a message, such as that of a string, is unchanged.
    fn message(&mut self, encoded: &[u8], base: usize) -> Result<FillOutcome> {
        if !holds_a_marker(self.places, base, base + encoded.len()) {
            return Ok(FillOutcome::Unchanged);
        }
        let Some(holding) = fields_holding_markers(encoded, base, self.places) else {
            return Ok(FillOutcome::Unchanged);
        };
        if let Some(hole) = hole_of_fields(&holding, base, self.places) {
            // This message holds the marker of the hole as the value of a field.
            self.found[hole] = true;
            return Ok(match self.holes[hole].kind {
                SentHoleKind::PythonUdf | SentHoleKind::PythonUdtf => {
                    FillOutcome::Changed(FillEdit::Replaced(filled_message(
                        &self.holes[hole],
                        encoded,
                        &self.fills[hole],
                    )?))
                }
                SentHoleKind::DataType => FillOutcome::Above(hole),
            });
        }
        let mut fields = Vec::new();
        let mut len = encoded.len();
        for WireField {
            key_end,
            value_start,
            end,
            ..
        } in holding
        {
            let value = &encoded[value_start..end];
            match crate::expression::with_stack_for_a_level(|| {
                self.message(value, base + value_start)
            })? {
                FillOutcome::Unchanged => {}
                // This message is the one of the hole, which its fill replaces.
                FillOutcome::Above(hole) => {
                    return Ok(FillOutcome::Changed(FillEdit::Replaced(
                        self.fills[hole].clone(),
                    )));
                }
                FillOutcome::Changed(edit) => {
                    len = len - (end - key_end) + varint_len(edit.len() as u64) + edit.len();
                    fields.push(FilledField {
                        key_end,
                        value_start,
                        end,
                        edit,
                    });
                }
            }
        }
        Ok(if fields.is_empty() {
            FillOutcome::Unchanged
        } else {
            FillOutcome::Changed(FillEdit::Fields(fields, len))
        })
    }
}

/// The message of `hole`, `message` as the pickle carries it, with the type and the command of
/// `fill`, the message that the pickler of the hole gave: of a `PythonUdf` or a `PythonUdtf`, its
/// other fields stay as they are.
fn filled_message(hole: &SentHole, message: &[u8], fill: &[u8]) -> Result<Vec<u8>> {
    use prost::Message;
    use spark_connect_core::error::SparkError;
    let decode = |error: prost::DecodeError| SparkError::connect_msg(error.to_string());
    Ok(match hole.kind {
        SentHoleKind::PythonUdf => {
            let mut udf = proto::PythonUdf::decode(message).map_err(decode)?;
            let built = proto::PythonUdf::decode(fill).map_err(decode)?;
            if built.output_type.is_some() {
                udf.output_type = built.output_type;
            }
            udf.command = built.command;
            udf.encode_to_vec()
        }
        SentHoleKind::PythonUdtf => {
            let mut udtf = proto::PythonUdtf::decode(message).map_err(decode)?;
            let built = proto::PythonUdtf::decode(fill).map_err(decode)?;
            if built.return_type.is_some() {
                udtf.return_type = built.return_type;
            }
            udtf.command = built.command;
            udtf.encode_to_vec()
        }
        SentHoleKind::DataType => fill.to_vec(),
    })
}

/// The varint of `bytes` at `at`, and where it ends.
fn read_varint(bytes: &[u8], mut at: usize) -> Option<(u64, usize)> {
    let mut value = 0u64;
    let mut shift = 0;
    loop {
        let byte = *bytes.get(at)?;
        at += 1;
        if shift >= 64 {
            return None;
        }
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Some((value, at));
        }
        shift += 7;
    }
}

/// Append the varint of `value` to `out`.
fn write_varint(mut value: u64, out: &mut Vec<u8>) {
    while value >= 0x80 {
        out.push((value as u8) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

/// The number of bytes of the varint of `value`.
fn varint_len(value: u64) -> usize {
    let mut len = 1;
    let mut value = value >> 7;
    while value != 0 {
        len += 1;
        value >>= 7;
    }
    len
}

/// The command of a UDF or UDTF as a request builds it: that of `pickler`, which pickles it
/// then, or the bytes of `command`.
pub(crate) fn command_bytes(command: &[u8], pickler: Option<&CommandPickler>) -> bytes::Bytes {
    match pickler {
        Some(pickler) => bytes::Bytes::from(crate::expression::pickled_command(pickler)),
        None => bytes::Bytes::copy_from_slice(command),
    }
}

/// The return type and the command of a UDTF as a request builds its plan, of the `return_type`
/// and the `command` of the plan, or of its `pickler`. Of a pickler that gives the return type
/// with the command ([`CommandPickler::with_output_type`]), both are those of the `PythonUdtf`
/// message that it gives as the request builds the plan, as PySpark's `PythonUDTF.to_plan`
/// converts the return type and pickles the handler class then: so a type changed in place after
/// the call is the type that the request sends. `return_type` stands until it gives one, as in
/// the build of a request that meets the picklers first (see `crate::expression::pickled_command`).
/// Else `return_type`, and the command of [`command_bytes`].
pub(crate) fn udtf_return_type_and_command(
    return_type: Option<&DataType>,
    command: &[u8],
    pickler: Option<&CommandPickler>,
) -> (Option<proto::DataType>, bytes::Bytes) {
    match pickler.filter(|pickler| pickler.gives_output_type()) {
        Some(pickler) => {
            use prost::Message;
            let built = crate::expression::pickled_command(pickler);
            let built = proto::PythonUdtf::decode(built.as_slice()).unwrap_or_default();
            let return_type = built
                .return_type
                .or_else(|| return_type.map(DataType::to_proto));
            (return_type, built.command)
        }
        None => (
            return_type.map(DataType::to_proto),
            command_bytes(command, pickler),
        ),
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

    /// The payload of a Python function that is not a UDF, such as the function of a streaming
    /// `foreach` or `foreachBatch` sink: of its command and Python version, which PySpark sets in
    /// the `PythonUDF` of the function, and of no output type nor eval type, which it does not
    /// set. Its output type is [`DataType::Null`], which is not sent (see
    /// [`crate::streaming::DataStreamWriter::foreach_batch`]).
    pub fn function(command: Vec<u8>, python_ver: String) -> Self {
        PythonUDFPayload::new(DataType::Null, 0, command, python_ver)
    }

    /// Convert to a proto PythonUDF message. Of a pickler that gives the output type with the
    /// command ([`CommandPickler::with_output_type`]), both are those that it gives as the
    /// request builds the plan, as PySpark's `PythonUDF.to_plan` converts them then; the
    /// `output_type` of the payload stands until it gives one, as in the build of a request
    /// that meets the picklers first (see `crate::expression::pickled_command`).
    pub fn to_proto(&self) -> proto::PythonUdf {
        let mut proto = proto::PythonUdf::default();
        match self
            .pickler
            .as_ref()
            .filter(|pickler| pickler.gives_output_type())
        {
            Some(pickler) => {
                use prost::Message;
                let built = crate::expression::pickled_command(pickler);
                let built = proto::PythonUdf::decode(built.as_slice()).unwrap_or_default();
                proto.output_type = built
                    .output_type
                    .or_else(|| Some(self.output_type.to_proto()));
                proto.command = built.command;
            }
            None => {
                proto.output_type = Some(self.output_type.to_proto());
                proto.command = command_bytes(&self.command, self.pickler.as_ref());
            }
        }
        proto.eval_type = self.eval_type;
        proto.python_ver = self.python_ver.clone();
        proto
    }

    /// The `PythonUDF` of the function of a streaming `foreach` or `foreachBatch` sink, as
    /// PySpark's `DataStreamWriter` sets it: of the command and the Python version only, as the
    /// function is not a UDF (connect/streaming/readwriter.py).
    pub(crate) fn to_function_proto(&self) -> proto::PythonUdf {
        proto::PythonUdf {
            command: command_bytes(&self.command, self.pickler.as_ref()),
            python_ver: self.python_ver.clone(),
            ..Default::default()
        }
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

    #[test]
    fn test_a_pickler_with_the_output_type_gives_the_type_and_the_command_of_each_build() {
        use prost::Message;
        use std::sync::Mutex;
        // As PySpark's `PythonUDF.to_plan`, which converts the output type of the object that it
        // pickles with the function each time it builds the plan: the type that the object has
        // then is sent, not the type of the payload, which stands when the pickler gives none.
        let current = Arc::new(Mutex::new(DataType::Long));
        let pickler = CommandPickler::with_output_type(Arc::clone(&current), |current| {
            let output_type = current.lock().unwrap().clone();
            Ok(proto::PythonUdf {
                output_type: Some(output_type.to_proto()),
                command: bytes::Bytes::from(format!("command of {}", output_type.json())),
                ..Default::default()
            }
            .encode_to_vec())
        });
        let payload = PythonUDFPayload::pickled_at_build(
            DataType::Long,
            eval_type::SQL_BATCHED_UDF,
            pickler,
            "3.11".to_string(),
        );
        let built = payload.to_proto();
        assert_eq!(built.output_type, Some(DataType::Long.to_proto()));
        assert_eq!(built.command.as_ref(), b"command of \"long\"");
        assert_eq!((built.eval_type, built.python_ver.as_str()), (100, "3.11"));
        *current.lock().unwrap() = DataType::Double;
        let built = payload.to_proto();
        assert_eq!(built.output_type, Some(DataType::Double.to_proto()));
        assert_eq!(built.command.as_ref(), b"command of \"double\"");
        let silent = PythonUDFPayload::pickled_at_build(
            DataType::Integer,
            eval_type::SQL_BATCHED_UDF,
            CommandPickler::with_output_type((), |_| Ok(Vec::new())),
            "3.11".to_string(),
        );
        let built = silent.to_proto();
        assert_eq!(built.output_type, Some(DataType::Integer.to_proto()));
        assert!(built.command.is_empty());
    }

    #[test]
    fn test_a_udtf_pickler_with_the_return_type_gives_the_type_and_the_command_of_each_build() {
        use crate::plan::LogicalPlan;
        use prost::Message;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Mutex;
        // As PySpark's `PythonUDTF.to_plan`, which converts the return type of the object that the
        // call kept and pickles the handler class each time it builds the plan: the type that the
        // object has then is sent with the command, by a request, which meets the pickler in its
        // first build and takes its message in the second, and by the Rust API's `to_proto`; a
        // UDTF of no return type sends none.
        let current = Arc::new(Mutex::new(Some(DataType::Long)));
        let picklings = Arc::new(AtomicUsize::new(0));
        let counted = Arc::clone(&picklings);
        let pickler = CommandPickler::with_output_type(Arc::clone(&current), move |current| {
            counted.fetch_add(1, Ordering::SeqCst);
            let return_type = current.lock().unwrap().clone();
            let name = return_type
                .as_ref()
                .map_or("none".to_string(), DataType::json);
            Ok(proto::PythonUdtf {
                return_type: return_type.as_ref().map(DataType::to_proto),
                command: bytes::Bytes::from(format!("handler of {name}")),
                ..Default::default()
            }
            .encode_to_vec())
        });
        let udtf = |return_type: Option<DataType>, pickler: CommandPickler| {
            LogicalPlan::CommonInlineUdtf {
                function_name: "t".to_string(),
                deterministic: false,
                arguments: vec![Expression::column("x")],
                return_type,
                eval_type: eval_type::SQL_TABLE_UDF,
                command: Vec::new(),
                python_ver: "3.11".to_string(),
                pickler: Some(pickler),
            }
        };
        let python_udtf = |relation: proto::Relation| match relation.rel_type {
            Some(proto::relation::RelType::CommonInlineUserDefinedTableFunction(call)) => {
                match call.function {
                    Some(
                        proto::common_inline_user_defined_table_function::Function::PythonUdtf(
                            udtf,
                        ),
                    ) => udtf,
                    _ => panic!("expected a PythonUDTF"),
                }
            }
            _ => panic!("expected an inline UDTF"),
        };
        let plan = udtf(None, pickler.clone());
        let built = python_udtf(plan.try_to_proto().unwrap());
        assert_eq!(built.return_type, Some(DataType::Long.to_proto()));
        assert_eq!(built.command.as_ref(), b"handler of \"long\"");
        assert_eq!((built.eval_type, built.python_ver.as_str()), (300, "3.11"));
        assert_eq!(picklings.load(Ordering::SeqCst), 1);
        *current.lock().unwrap() = Some(DataType::Double);
        let built = python_udtf(plan.try_to_proto().unwrap());
        assert_eq!(built.return_type, Some(DataType::Double.to_proto()));
        assert_eq!(built.command.as_ref(), b"handler of \"double\"");
        *current.lock().unwrap() = Some(DataType::Integer);
        let built = python_udtf(plan.to_proto());
        assert_eq!(built.return_type, Some(DataType::Integer.to_proto()));
        assert_eq!(built.command.as_ref(), b"handler of \"integer\"");
        *current.lock().unwrap() = None;
        let built = python_udtf(plan.try_to_proto().unwrap());
        assert_eq!(built.return_type, None);
        assert_eq!(built.command.as_ref(), b"handler of none");
        assert_eq!(picklings.load(Ordering::SeqCst), 4);
        // The type of the plan stands where the pickler gives none; the error of a pickling is
        // that of the request.
        let silent = CommandPickler::with_output_type((), |_| Ok(Vec::new()));
        let built = python_udtf(udtf(Some(DataType::Short), silent).try_to_proto().unwrap());
        assert_eq!(built.return_type, Some(DataType::Short.to_proto()));
        assert!(built.command.is_empty());
        let failing = CommandPickler::with_output_type((), |_| {
            Err(spark_connect_core::error::SparkError::value_msg("no type"))
        });
        let error = udtf(None, failing).try_to_proto().unwrap_err();
        assert_eq!(error.message, "no type");
        // A pickler of the command only gives the command as it is, with the type of the plan.
        let command_only = CommandPickler::new((), |_| Ok(b"handler".to_vec()));
        let built = python_udtf(
            udtf(Some(DataType::Long), command_only)
                .try_to_proto()
                .unwrap(),
        );
        assert_eq!(built.return_type, Some(DataType::Long.to_proto()));
        assert_eq!(built.command.as_ref(), b"handler");
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

    /// The tags of the objects of the picklers of `udf_of` among the parts that `roots` reach,
    /// sorted, once `roots` are checked to give every reference to those parts (see `crate::gc`).
    #[track_caller]
    fn tags(roots: &[crate::gc::PartRef<'_>]) -> Vec<&'static str> {
        crate::gc::tests::assert_exact(roots);
        tags_below(roots)
    }

    /// The tags of the objects of the picklers of `udf_of` among the parts that `roots` reach,
    /// sorted.
    fn tags_below(roots: &[crate::gc::PartRef<'_>]) -> Vec<&'static str> {
        let mut tags: Vec<&'static str> = crate::gc::tests::references(roots)
            .values()
            .filter_map(|(_, part)| match part {
                crate::gc::Part::Pickler(pickler) => {
                    pickler.object().downcast_ref::<&'static str>().copied()
                }
                _ => None,
            })
            .collect();
        tags.sort();
        tags
    }

    #[test]
    fn test_the_parts_of_a_holder_reach_the_picklers_below_it() {
        use crate::gc::PartRef::{Expression as Of, Plan};
        use crate::plan::LogicalPlan;
        // Of an expression, below a function too, and of an expression that a clone shares.
        let a = udf_expression("a");
        assert_eq!(tags(&[Of(&a)]), ["a"]);
        let shared = a.clone();
        assert_eq!(tags(&[Of(&a), Of(&shared)]), ["a"]);
        drop(shared);
        let both = Expression::function("struct", vec![udf_expression("b"), a]);
        assert_eq!(tags(&[Of(&both)]), ["a", "b"]);
        // A pickler that a copy of the UDF shares, which another expression holds.
        let copied = udf_of("c");
        let of_udf = |udf: CommonInlineUserDefinedFunctionExpression| -> Expression {
            crate::expression::ExpressionNode::CommonInlineUserDefinedFunction(Box::new(udf)).into()
        };
        let (first, second) = (of_udf(copied.clone()), of_udf(copied));
        assert_eq!(tags(&[Of(&first), Of(&second)]), ["c"]);
        // Of a plan: of the expressions and functions of its nodes, of the inputs that it shares
        // with another holder too, and of a UDTF.
        let input = Arc::new(LogicalPlan::Project {
            input: Arc::new(crate::plan::range(0, 1, 1)).into(),
            columns: vec![crate::column::Column::new(udf_expression("project"))],
        });
        let mapped = Arc::new(LogicalPlan::MapPartitions {
            input: Arc::clone(&input).into(),
            func: udf_of("map"),
            is_barrier: false,
            profile_id: None,
            profile_id_reader: Some(CommandPickler::new("profile", |_| Ok(Vec::new()))),
        });
        assert_eq!(
            tags(&[Plan(&mapped), Plan(&input)]),
            ["map", "profile", "project"]
        );
        drop(input);
        assert_eq!(tags(&[Plan(&mapped)]), ["map", "profile", "project"]);
        let udtf = Arc::new(LogicalPlan::CommonInlineUdtf {
            function_name: "t".to_string(),
            deterministic: false,
            arguments: vec![udf_expression("argument")],
            return_type: None,
            eval_type: eval_type::SQL_TABLE_UDF,
            command: Vec::new(),
            python_ver: "3.11".to_string(),
            pickler: Some(CommandPickler::new("handler", |_| Ok(Vec::new()))),
        });
        assert_eq!(tags(&[Plan(&udtf)]), ["argument", "handler"]);
    }

    #[test]
    fn test_the_parts_of_the_objects_of_a_dataframe_are_every_reference_to_them() {
        use crate::column::Column;
        let spark = crate::session::SparkSession::builder()
            .remote("sc://localhost:15002")
            .get_or_create()
            .unwrap();
        let column = |tag| Column::new(udf_expression(tag));
        // A DataFrame, its grouped data and cogroups, its writers, its statistics, a table
        // argument, a merge builder and a window specification of Columns of UDFs, which share
        // the plan and each other's expressions.
        let df = spark.range(3).unwrap().select(vec![column("plan")]);
        assert_eq!(tags(&df.held_parts()), ["plan"]);
        let grouping = column("grouping");
        let grouped = df.group_by(vec![grouping.clone()]);
        let cogrouped = grouped.cogroup(&grouped);
        let writer = df.write();
        let v2 = df.write_to("t").partition_by(vec![column("partition")]);
        let stream = df.write_stream();
        let stat = df.stat();
        let table = crate::table_arg::TableArg::new(df.clone())
            .partition_by(vec![column("table")])
            .unwrap();
        let merge = df
            .merge_into("t", column("condition"))
            .when_matched(Some(column("matched")))
            .update(vec![("v", grouping.clone())]);
        let window = crate::window::WindowSpec::new()
            .partition_by(vec![grouping.clone()])
            .order_by(vec![column("order").desc()]);
        let all = [
            df.held_parts(),
            vec![crate::gc::PartRef::Expression(grouping.expression())],
            grouped.held_parts(),
            cogrouped.held_parts(),
            writer.held_parts(),
            v2.held_parts(),
            stream.held_parts(),
            stat.held_parts(),
            table.held_parts(),
            merge.held_parts(),
            window.held_parts(),
        ]
        .concat();
        assert_eq!(
            tags(&all),
            [
                "condition",
                "grouping",
                "matched",
                "order",
                "partition",
                "plan",
                "table"
            ]
        );
        drop(all);
        assert_eq!(tags_below(&stream.held_parts()), ["plan"]);
        assert_eq!(tags_below(&stream.dataframe().held_parts()), ["plan"]);
        assert_eq!(
            tags_below(&merge.held_parts()),
            ["condition", "grouping", "matched", "plan"]
        );
        // The parts that clearing drops are gone.
        let (mut v2, mut merge, mut grouped) = (v2, merge, grouped);
        v2.clear_plan();
        merge.clear_plan();
        grouped.clear_plan();
        assert!(tags_below(&v2.held_parts()).is_empty());
        assert!(tags_below(&merge.held_parts()).is_empty());
        assert!(tags_below(&grouped.held_parts()).is_empty());
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
            input: Arc::new(crate::plan::range(0, 1, 1)).into(),
            columns: vec![crate::column::Column::new(udf_expression("c"))],
        });
        let plan = Arc::new(LogicalPlan::Filter {
            input: Arc::clone(&input).into(),
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
        assert_eq!(ids.id_of(copied_input.arc()), input_id);
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
        assert!(Arc::ptr_eq(range.arc(), original_range.arc()));
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

    /// The relation of a projection of a UDF whose command is `command`, a sort of another of
    /// `other`, both of the eval type, the Python version and the includes of a UDF of a pickle,
    /// over a `to` of the unparsed type `unparsed`, of a padding between the fields.
    fn relation_of_holes(command: &[u8], other: &[u8], unparsed: &str) -> proto::Relation {
        use proto::expression::ExprType;
        use proto::relation::RelType;
        let udf = |command: &[u8], output_type: Option<proto::DataType>| proto::Expression {
            expr_type: Some(ExprType::CommonInlineUserDefinedFunction(
                proto::CommonInlineUserDefinedFunction {
                    function_name: "f".to_string(),
                    function: Some(
                        proto::common_inline_user_defined_function::Function::PythonUdf(
                            proto::PythonUdf {
                                output_type,
                                eval_type: 100,
                                command: bytes::Bytes::copy_from_slice(command),
                                python_ver: "3.11".to_string(),
                                additional_includes: vec!["kept".to_string()],
                            },
                        ),
                    ),
                    ..Default::default()
                },
            )),
            ..Default::default()
        };
        let to = proto::Relation {
            rel_type: Some(RelType::ToSchema(Box::new(proto::ToSchema {
                input: Some(Box::new(proto::Relation {
                    rel_type: Some(RelType::Range(proto::Range {
                        end: 3,
                        ..Default::default()
                    })),
                    ..Default::default()
                })),
                schema: Some(proto::DataType {
                    kind: Some(proto::data_type::Kind::Unparsed(
                        proto::data_type::Unparsed {
                            data_type_string: unparsed.to_string(),
                        },
                    )),
                }),
            }))),
            ..Default::default()
        };
        let sorted = proto::Relation {
            rel_type: Some(RelType::Sort(Box::new(proto::Sort {
                input: Some(Box::new(to)),
                order: vec![proto::expression::SortOrder {
                    child: Some(Box::new(udf(other, None))),
                    ..Default::default()
                }],
                ..Default::default()
            }))),
            ..Default::default()
        };
        proto::Relation {
            common: Some(proto::RelationCommon {
                plan_id: Some(7),
                ..Default::default()
            }),
            rel_type: Some(RelType::Project(Box::new(proto::Project {
                input: Some(Box::new(sorted)),
                expressions: vec![
                    udf(command, Some(DataType::Integer.to_proto())),
                    proto::Expression {
                        expr_type: Some(ExprType::ExpressionString(
                            proto::expression::ExpressionString {
                                expression: "x".repeat(300),
                            },
                        )),
                        ..Default::default()
                    },
                ],
            }))),
        }
    }

    #[test]
    fn test_holes_are_filled_with_the_messages_of_their_picklers_in_their_places() {
        // As PySpark's unpickled plan, which pickles the functions of its UDFs when it is built:
        // each hole of a relation as it was sent is the message that its pickler gives then, of
        // the type and the command of a PythonUdf, of any length, and of the eval type, the
        // Python version and the includes of the UDF of the pickle; or the DataType of a schema.
        // The fields that hold it take its length, and the rest of the relation stays.
        use prost::Message;
        let mut markers = SentHoleMarkers::new();
        let (first, second, third) = (markers.marker(), markers.marker(), markers.marker());
        assert!(first != second && second != third);
        let template = relation_of_holes(&first, &second, std::str::from_utf8(&third).unwrap());
        let long = vec![b'c'; 20_000];
        let udf_of = |command: Vec<u8>, output_type: Option<proto::DataType>| {
            CommandPickler::with_output_type((), move |_| {
                Ok(proto::PythonUdf {
                    output_type: output_type.clone(),
                    command: command.clone().into(),
                    ..Default::default()
                }
                .encode_to_vec())
            })
        };
        let long_type = Some(DataType::Long.to_proto());
        let schema = DataType::from_ddl("a long").unwrap().to_proto();
        let encoded_schema = schema.encode_to_vec();
        let holes = SentHoles(vec![
            SentHole {
                marker: first.clone(),
                kind: SentHoleKind::PythonUdf,
                pickler: Some(udf_of(long.clone(), long_type.clone())),
            },
            SentHole {
                marker: second.clone(),
                kind: SentHoleKind::PythonUdf,
                pickler: Some(udf_of(b"short".to_vec(), None)),
            },
            SentHole {
                marker: third.clone(),
                kind: SentHoleKind::DataType,
                pickler: Some(CommandPickler::new((), move |_| Ok(encoded_schema.clone()))),
            },
        ]);
        let filled = holes.fill(|| template.encode_to_vec()).unwrap().unwrap();
        let filled = proto::Relation::decode(filled.as_slice()).unwrap();
        let mut expected = relation_of_holes(&long, b"short", "");
        let Some(proto::relation::RelType::Project(project)) = expected.rel_type.as_mut() else {
            unreachable!()
        };
        // The type of the pickler, and of the UDF of the pickle where the pickler gives none.
        if let Some(proto::expression::ExprType::CommonInlineUserDefinedFunction(udf)) =
            project.expressions[0].expr_type.as_mut()
        {
            if let Some(proto::common_inline_user_defined_function::Function::PythonUdf(udf)) =
                udf.function.as_mut()
            {
                udf.output_type = long_type;
            }
        }
        let Some(proto::relation::RelType::Sort(sort)) =
            project.input.as_mut().unwrap().rel_type.as_mut()
        else {
            unreachable!()
        };
        let Some(proto::relation::RelType::ToSchema(to)) =
            sort.input.as_mut().unwrap().rel_type.as_mut()
        else {
            unreachable!()
        };
        to.schema = Some(schema);
        assert_eq!(filled, expected);

        // A hole whose marker the relation does not hold is an error, and so is a marker that no
        // message of its kind holds; a pickler that gives nothing leaves the hole as it is, as the
        // first build of a request.
        let lost = SentHoles(vec![SentHole {
            marker: markers.marker(),
            kind: SentHoleKind::PythonUdf,
            pickler: Some(udf_of(b"x".to_vec(), None)),
        }]);
        assert!(lost.fill(|| template.encode_to_vec()).is_err());
        let misplaced = SentHoles(vec![SentHole {
            marker: third,
            kind: SentHoleKind::PythonUdf,
            pickler: Some(udf_of(b"x".to_vec(), None)),
        }]);
        assert!(misplaced.fill(|| template.encode_to_vec()).is_err());
        // A pickler that gives nothing, as in the first build of a request: no encoding is made.
        let silent = SentHoles(vec![SentHole {
            marker: first,
            kind: SentHoleKind::PythonUdf,
            pickler: Some(CommandPickler::with_output_type((), |_| Ok(Vec::new()))),
        }]);
        assert!(silent
            .fill(|| unreachable!("no encoding of the first build"))
            .unwrap()
            .is_none());
        assert!(!holes.is_empty() && SentHoles::default().is_empty());
    }

    /// A projection of a UDF of each marker of `markers`, in their order, as a pickle builds the
    /// expressions of a select.
    fn projection_of(markers: &[Vec<u8>]) -> proto::Relation {
        use proto::expression::ExprType;
        proto::Relation {
            rel_type: Some(proto::relation::RelType::Project(Box::new(
                proto::Project {
                    input: None,
                    expressions: markers
                        .iter()
                        .map(|marker| {
                            proto::Expression {
                        expr_type: Some(ExprType::CommonInlineUserDefinedFunction(
                            proto::CommonInlineUserDefinedFunction {
                                function_name: "f".to_string(),
                                function: Some(
                                    proto::common_inline_user_defined_function::Function::PythonUdf(
                                        proto::PythonUdf {
                                            command: marker.clone().into(),
                                            ..Default::default()
                                        },
                                    ),
                                ),
                                ..Default::default()
                            },
                        )),
                        ..Default::default()
                    }
                        })
                        .collect(),
                },
            ))),
            ..Default::default()
        }
    }

    /// The commands of the UDFs of `relation`, a projection of [`projection_of`].
    fn commands_of(relation: &proto::Relation) -> Vec<Vec<u8>> {
        let Some(proto::relation::RelType::Project(project)) = relation.rel_type.as_ref() else {
            unreachable!()
        };
        project
            .expressions
            .iter()
            .map(|expression| match expression.expr_type.as_ref() {
                Some(proto::expression::ExprType::CommonInlineUserDefinedFunction(udf)) => {
                    match udf.function.as_ref() {
                        Some(proto::common_inline_user_defined_function::Function::PythonUdf(
                            udf,
                        )) => udf.command.to_vec(),
                        _ => unreachable!(),
                    }
                }
                _ => unreachable!(),
            })
            .collect()
    }

    #[test]
    fn test_markers_are_of_one_width_and_each_hole_is_filled_in_its_own_place() {
        // As PySpark's unpickled plan of many UDFs: each of the holes of 30 UDFs, of the markers 0
        // to 29 of one maker, none the prefix of another, takes the message of its own pickler,
        // of the holes in an order other than that of the encoding too, as the build of a group
        // map builds its initial state before its function, which it encodes first: here the
        // markers are encoded from the last, so that the markers 10 to 19 come before marker 1.
        use prost::Message;
        let mut markers = SentHoleMarkers::new();
        let made: Vec<Vec<u8>> = (0..30).map(|_| markers.marker()).collect();
        assert!(made.iter().all(|marker| marker.len() == MARKER_LEN));
        for (index, marker) in made.iter().enumerate() {
            assert!(made
                .iter()
                .enumerate()
                .all(|(other, of)| other == index || !of.starts_with(marker)));
        }
        let encoded_order: Vec<Vec<u8>> = made.iter().rev().cloned().collect();
        let template = projection_of(&encoded_order);
        let pickler = |index: usize| {
            CommandPickler::with_output_type((), move |_| {
                Ok(proto::PythonUdf {
                    command: format!("<{index}>").into_bytes().into(),
                    ..Default::default()
                }
                .encode_to_vec())
            })
        };
        // The holes in the order of the build, other than that of the encoding.
        let holes = SentHoles(
            made.iter()
                .enumerate()
                .map(|(index, marker)| SentHole {
                    marker: marker.clone(),
                    kind: SentHoleKind::PythonUdf,
                    pickler: Some(pickler(index)),
                })
                .collect(),
        );
        let filled = holes.fill(|| template.encode_to_vec()).unwrap().unwrap();
        let filled = proto::Relation::decode(filled.as_slice()).unwrap();
        let expected: Vec<Vec<u8>> = (0..30)
            .rev()
            .map(|i| format!("<{i}>").into_bytes())
            .collect();
        assert_eq!(commands_of(&filled), expected);
    }

    #[test]
    fn test_the_fill_of_many_holes_reads_the_encoding_once() {
        // The fill reads the encoding once, however many holes it has, as PySpark's unpickled plan
        // builds its relation once: 4,000 holes of 4,000 UDFs of names of 1 KB, an encoding of
        // about 4.4 MB, which a fill of one read of the encoding for each hole reads 4,000 times
        // (over 17 GB), are filled in a time of one read. The bound is of the slowest build.
        use prost::Message;
        let mut markers = SentHoleMarkers::new();
        let made: Vec<Vec<u8>> = (0..4000).map(|_| markers.marker()).collect();
        let mut template = projection_of(&made);
        let Some(proto::relation::RelType::Project(project)) = template.rel_type.as_mut() else {
            unreachable!()
        };
        for expression in project.expressions.iter_mut() {
            if let Some(proto::expression::ExprType::CommonInlineUserDefinedFunction(udf)) =
                expression.expr_type.as_mut()
            {
                udf.function_name = "f".repeat(1000);
            }
        }
        let holes = SentHoles(
            made.iter()
                .map(|marker| SentHole {
                    marker: marker.clone(),
                    kind: SentHoleKind::PythonUdf,
                    pickler: Some(CommandPickler::with_output_type((), |_| {
                        Ok(proto::PythonUdf {
                            command: b"<filled>".to_vec().into(),
                            ..Default::default()
                        }
                        .encode_to_vec())
                    })),
                })
                .collect(),
        );
        let started = std::time::Instant::now();
        let filled = holes.fill(|| template.encode_to_vec()).unwrap().unwrap();
        let took = started.elapsed();
        assert!(
            took < std::time::Duration::from_secs(5),
            "the fill took {took:?}"
        );
        let filled = proto::Relation::decode(filled.as_slice()).unwrap();
        let commands = commands_of(&filled);
        assert!(commands.len() == 4000 && commands.iter().all(|command| command == b"<filled>"));
    }

    #[test]
    fn test_a_marker_is_a_hole_only_as_the_whole_value_of_a_field() {
        // A marker that is only a part of the value of a field, as of the bytes of a command, is no
        // place of its hole: of no other place, the fill fails, rather than splice into the bytes.
        use prost::Message;
        let mut markers = SentHoleMarkers::new();
        let marker = markers.marker();
        let mut within = b"<".to_vec();
        within.extend_from_slice(&marker);
        within.extend_from_slice(b">");
        let template = projection_of(&[within.clone()]);
        let hole = SentHoles(vec![SentHole {
            marker: marker.clone(),
            kind: SentHoleKind::PythonUdf,
            pickler: Some(CommandPickler::with_output_type((), |_| {
                Ok(proto::PythonUdf {
                    command: b"<filled>".to_vec().into(),
                    ..Default::default()
                }
                .encode_to_vec())
            })),
        }]);
        assert!(hole.fill(|| template.encode_to_vec()).is_err());
        // With the marker whole beside it, that place is filled, and the bytes stay as they are.
        let template = projection_of(&[within.clone(), marker]);
        let filled = hole.fill(|| template.encode_to_vec()).unwrap().unwrap();
        let filled = proto::Relation::decode(filled.as_slice()).unwrap();
        assert_eq!(commands_of(&filled), [within, b"<filled>".to_vec()]);
    }

    #[test]
    fn test_the_varints_of_the_lengths_of_the_fields_that_hold_a_hole() {
        for value in [
            0u64,
            1,
            127,
            128,
            16_383,
            16_384,
            u64::from(u32::MAX),
            u64::MAX,
        ] {
            let mut out = Vec::new();
            write_varint(value, &mut out);
            assert_eq!(out.len(), varint_len(value), "{value}");
            assert_eq!(read_varint(&out, 0), Some((value, out.len())), "{value}");
        }
        // A truncated varint, and one longer than 64 bits.
        assert_eq!(read_varint(&[0x80], 0), None);
        assert_eq!(read_varint(&[0xff; 11], 0), None);
        // A message of fields of the fixed widths and of a varint before the field of a hole, whose
        // bytes stay as they are; and one of a group, which Spark Connect does not use, which
        // holds no hole.
        use prost::Message;
        let mut markers = SentHoleMarkers::new();
        let marker = markers.marker();
        let udf = proto::PythonUdf {
            command: marker.clone().into(),
            ..Default::default()
        }
        .encode_to_vec();
        let mut message = vec![
            0x09, 1, 1, 1, 1, 1, 1, 1, 1, 0x15, 2, 2, 2, 2, 0x18, 0x96, 0x01,
        ];
        message.push(0x22);
        write_varint(udf.len() as u64, &mut message);
        message.extend_from_slice(&udf);
        let hole = |marker: &[u8]| {
            SentHoles(vec![SentHole {
                marker: marker.to_vec(),
                kind: SentHoleKind::PythonUdf,
                pickler: Some(CommandPickler::with_output_type((), |_| {
                    Ok(proto::PythonUdf {
                        command: b"<filled>".to_vec().into(),
                        ..Default::default()
                    }
                    .encode_to_vec())
                })),
            }])
        };
        let filled = hole(&marker).fill(|| message.clone()).unwrap().unwrap();
        assert_eq!(filled[..17], message[..17]);
        let udf = proto::PythonUdf::decode(&filled[19..]).unwrap();
        assert_eq!(udf.command.as_ref(), b"<filled>");
        let mut grouped = vec![0x0b];
        grouped.extend_from_slice(&message[17..]);
        assert!(hole(&marker).fill(|| grouped.clone()).is_err());
        // The message of a hole is the value of a field, never the whole encoding.
        let whole = proto::PythonUdf {
            command: marker.clone().into(),
            ..Default::default()
        }
        .encode_to_vec();
        assert!(hole(&marker).fill(|| whole.clone()).is_err());
        assert_eq!(held_markers(&whole, &[&marker]), [true]);
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
