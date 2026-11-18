import importlib
import sys
from decimal import Decimal

import pytest
from _local_data_conversion_probe import LocalDataConversionProbe
from pyspark.errors import PySparkValueError
from pyspark.sql import types as T
from row_conversion_cases import conversion_result

# The rows, or the values of an array, of the part of the conversion that a case observes: steps
# enough for another thread to wait for the GIL at its checkpoints, every 1,024 steps, for about
# a tenth of a second at least. The parts that take each step faster take more.
_COUNT = 262144


def _deep(levels=80):
    """A value nested deeper than inference takes, whose RecursionError inference raises in
    native code: a failure that runs no Python code, so that no thread runs after it."""
    value = []
    for _ in range(levels):
        value = [value]
    return value


class _Held:
    """Whether the conversion is in a part, by `test` of the references to `objects` beyond those
    that they have when armed, right before the conversion."""

    def __init__(self, test, *objects):
        self.test = test
        self.objects = objects
        self.bases = None

    def counts(self):
        # Read the same way on arming and after, as each way holds its own references.
        return [sys.getrefcount(obj) for obj in self.objects]

    def arm(self):
        self.bases = self.counts()

    def __call__(self):
        return self.test(*[count - base for count, base in zip(self.counts(), self.bases)])


class _OneHeld:
    """Whether exactly one of `objects` has one reference more than the others, as the value that
    a conversion takes has. The counts are read in one C call, without a switch of threads."""

    def __init__(self, objects):
        self.objects = objects

    def arm(self):
        pass

    def __call__(self):
        counts = list(map(sys.getrefcount, self.objects))
        top = max(counts)
        return top == min(counts) + 1 and counts.count(top) == 1


def _case(layout, count):
    """The data and schema of a case of `count` rows or values; whether the conversion is in the
    part that the case observes, and whether it still holds the references that it holds until it
    ends; and the error that ends the conversion right after that part, of a value that fails
    without running Python code, so that the part is the last one that the data reaches. Most rows
    are the same objects, of which the conversion holds a reference for each row that it took; the
    bounds leave room for a reference that it keeps of its first row. A failure that the
    conversion checks the class of is normalized, which briefly releases the GIL in PyO3: the part
    that it ends is then observed by a value that is held only while it is converted."""
    text = f"text {id(object())}"
    number = 1000 + id(object()) % 1000
    payload = bytes(8)
    strings = T.StructType([T.StructField("c", T.BinaryType()), T.StructField("s", T.StringType())])
    if layout == "collected rows":
        row = (payload, text)
        # The first row is short, which taking the rows apart fails at once.
        collected = 8 * count
        data = [(payload,)] + [row] * (collected - 1)
        in_part = _Held(lambda rows: 0 < rows < collected - 1, row)
        return data, strings, in_part, _Held(lambda rows: rows > 0, row), PySparkValueError
    if layout == "rows taken apart":
        row = (payload, text)
        count *= 2
        # The first value that the converters take is no binary.
        data = [(0, text)] + [row] * (count - 1)
        in_part = _Held(lambda rows, texts: rows >= count - 1 and 0 < texts < count, row, text)
        return data, strings, in_part, _Held(lambda rows: rows > 0, row), AssertionError
    if layout == "converted columns":
        row = (number, text, payload)
        # The last value that the converters take, of the last column, is no binary.
        data = [row] * (count - 1) + [(number, text, 0)]
        schema = T.StructType([
            T.StructField("a", T.LongType()), T.StructField("s", T.StringType()),
            T.StructField("c", T.BinaryType()),
        ])
        held = _Held(lambda texts: texts >= count, text)
        return data, schema, held, held, AssertionError
    if layout == "inferred rows":
        # Each row holds a text of its own, which inference holds while it types the row: of
        # three steps a row, its own and those of its two values, the checkpoints fall on them.
        texts = [f"text {i}" for i in range(count - 1)]
        data = [(text, None) for text in texts] + [("last", _deep())]
        held = _Held(lambda first: first > 0, data[0])
        return data, ["s", "n"], _OneHeld(texts), held, RecursionError
    if layout == "sorted dicts":
        # The sorted copy of each dict refers to its values; the first one fails inference.
        data = [{"s": text, "a": _deep()}] + [{"s": text, "a": number}] * (count - 1)
        in_part = _Held(lambda texts: 0 < texts < count, text)
        return data, None, in_part, _Held(lambda texts: texts > 0, text), RecursionError
    if layout == "array":
        value = Decimal("1.5")
        data = [([value] * (count - 1) + [0],)]
        schema = T.StructType([T.StructField("c", T.ArrayType(T.DecimalType(10, 2)))])
        held = _Held(lambda row: row > 0, data[0])
        return data, schema, _Held(lambda values: values >= 1, value), held, AssertionError
    if layout == "inferred array":
        value = {"k": 1.5}
        count *= 2
        data = [([value] * (count - 1) + [_deep()],)]
        held = _Held(lambda row: row > 0, data[0])
        return data, ["c"], _Held(lambda values: values >= 1, value), held, RecursionError
    pd = pytest.importorskip("pandas")
    if layout == "pandas strings":
        # The exact strings of the columns of objects of a plain frame of at least 10,000 rows,
        # of a struct of strings, which `_pandas_string_ipc` checks, then encodes, a step a
        # value: the value that it takes, of the same text in every cell, is referred to once
        # more. The last cell, a lone surrogate, which UTF-8 cannot encode, ends the encoding,
        # and the general conversion that follows raises for it, without reading the cells in
        # Python code. The frame is held until the conversion ends.
        rows = max(4 * count, 10000)
        frame = pd.DataFrame({
            "s": pd.Series([text] * rows, dtype=object),
            "t": pd.Series([text] * (rows - 1) + ["\ud800"], dtype=object),
        })
        schema = T.StructType([T.StructField(name, T.StringType()) for name in ("s", "t")])
        in_part = _Held(lambda texts: texts > 0, text)
        return frame, schema, in_part, _Held(lambda frames: frames > 0, frame), PySparkValueError
    assert layout == "pandas rows"
    # Each row and each cell is a step: of four cells a row, four of five checkpoints fall on
    # cells.
    count //= 2
    names = ["s", "t", "u", "v"]
    columns = [[f"{name} {i}" for i in range(count)] for name in names]
    last = object()
    columns[-1][-1] = last
    # Columns of objects keep the cells as given, which pandas 3 would otherwise keep as its `str`
    # dtype, of no reference to them. The row path converts the frame without PyArrow
    # (`_without_pyarrow`).
    frame = pd.DataFrame({
        name: pd.Series(column, dtype=object) for name, column in zip(names, columns)
    })
    schema = T.StructType([T.StructField(name, T.StringType()) for name in names])
    # The rows of the frame refer to its cells until the conversion ends. The last cell, which
    # fails, is not one of those that `_OneHeld` compares, of which this would hold one.
    cells = [cell for column in columns for cell in column if cell is not last]
    return frame, schema, _OneHeld(cells), _Held(lambda cell: cell > 0, last), TypeError


def _without_pyarrow(monkeypatch):
    """Make createDataFrame convert pandas frames with the native row path, which converts them
    without PyArrow: with it, PySpark's conversion of each column to Arrow converts them."""
    # The module of the native binding, which earlier import-hook tests can leave another object
    # than the attribute of the package.
    helper = importlib.import_module("pyspark.sql._pandas_input")
    monkeypatch.setattr(helper, "pyarrow_available", lambda: False)


class TestLocalDataConversionInterrupts:
    @pytest.mark.parametrize("layout", [
        "collected rows", "rows taken apart", "converted columns", "inferred rows",
        "sorted dicts", "array", "inferred array", "pandas rows", "pandas strings",
    ])
    @pytest.mark.parametrize("cancel", [False, True])
    def test_python_thread_runs_before_native_conversion_finishes(
        self, mock_connect, monkeypatch, layout, cancel,
    ):
        # As PySpark's createDataFrame, whose conversion of local data is Python code: in each
        # part of the native conversion, another thread runs, and a signal interrupts it, before
        # the part finishes.
        spark, service = mock_connect
        if layout == "pandas rows":
            _without_pyarrow(monkeypatch)
        # The first conversion of a process caches the Python classes and modules that it
        # reads, which runs Python code: a conversion of a few rows caches them first.
        data, schema, _, _, failure = _case(layout, 8)
        with pytest.raises(failure):
            spark.createDataFrame(data, schema)
        # The other thread takes the GIL at the checkpoints that the operating system schedules
        # it at, of which only some fall in the part: a conversion in which it read the part at
        # none is run again, a few times. With no checkpoint, it reads it in none.
        for _ in range(5):
            data, schema, in_part, held, failure = _case(layout, _COUNT)
            in_part.arm()
            held.arm()
            probe = LocalDataConversionProbe(in_part, held, cancel=cancel)
            interrupted = False
            with probe:
                try:
                    with pytest.raises(failure):
                        probe.convert(lambda: spark.createDataFrame(data, schema))
                except KeyboardInterrupt:
                    interrupted = True
            assert not probe.failures, probe.failures
            if "progress" in probe.observed:
                break
        progress = probe.observed.get("progress")
        assert progress is not None, probe.observed
        assert progress["captured_at"] < probe.observed["conversion_ended_at"], probe.observed
        assert interrupted is cancel, probe.observed
        if cancel:
            assert probe.observed.get("signal_in_conversion"), probe.observed
            # The session takes another conversion and execution.
            spark.createDataFrame([(1, "x")], ["a", "s"])
            conversion_result(service, "rows", 2, "unused")
            assert [tuple(row) for row in spark.range(2).collect()] == [(0,), (1,)]
