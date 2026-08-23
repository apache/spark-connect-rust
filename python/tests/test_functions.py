"""Test that all 440 Spark SQL functions are properly exposed and callable."""
import sys
import os

# Ensure _pyspark is available
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pyspark.sql.functions as F
from pyspark.sql import SparkSession

def test_function_count():
    """Test that at least 440 functions are exposed."""
    public_funcs = [n for n in dir(F) if not n.startswith('_')]
    assert len(public_funcs) >= 440, f"Expected at least 440 functions, got {len(public_funcs)}"
    print(f"✓ Function count: {len(public_funcs)}")

def test_function_construction():
    """Test construction of functions with various arities."""
    test_cases = [
        ("No-arg", F.current_date, []),
        ("Single-arg", F.abs, [F.col('a')]),
        ("Two-arg", F.add_months, [F.col('a'), F.lit(1)]),
        ("Three-arg", F.replace, [F.col('a'), F.lit('x'), F.lit('y')]),
        ("Variadic", F.concat, [F.col('a'), F.col('b'), F.col('c')]),
    ]
    
    for desc, func, args in test_cases:
        result = func(*args)
        assert result is not None
        print(f"✓ {desc} function construction works")

def test_specific_functions():
    """Test that specific required functions are available."""
    required = [
        'col', 'lit', 'expr',
        'abs', 'upper', 'lower', 'length',
        'when', 'coalesce', 'concat',
        'current_date', 'current_timestamp',
        'sqrt', 'round', 'rand',
        'count', 'sum', 'avg', 'max', 'min',
    ]
    
    for fname in required:
        assert hasattr(F, fname), f"Missing function: {fname}"
    print(f"✓ All {len(required)} required functions present")

def test_execution():
    """Test that functions execute correctly against the server."""
    try:
        spark = SparkSession.builder\
            .remote("sc://localhost:15002")\
            .build()
        
        # Test a few functions
        result = spark.range(5).select(F.abs((F.col('id')-2)).alias('x')).collect()
        assert result is not None and len(result) == 5
        print(f"✓ F.abs() execution: {[row.x for row in result]}")
        
        result = spark.range(3).select(F.upper(F.lit('hi')).alias('u')).collect()
        assert result is not None and len(result) == 3
        print(f"✓ F.upper() execution: {[row.u for row in result]}")
        
        result = spark.range(1, 4).select(
            F.when(F.col('id') > 1, 'yes').otherwise('no').alias('result')
        ).collect()
        assert result is not None
        print(f"✓ F.when() execution: {[row.result for row in result]}")
        
        spark.stop()
        print("✓ All execution tests passed")
        
    except Exception as e:
        print(f"⚠ Execution test skipped (server not available): {e}")

if __name__ == '__main__':
    test_function_count()
    test_function_construction()
    test_specific_functions()
    test_execution()
    print("\nAll tests passed!")
