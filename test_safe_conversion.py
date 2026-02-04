"""
Quick test of safe value conversion
"""
import pandas as pd
import numpy as np

# Test safe_int function
def safe_int(value, default=0):
    """Safely convert value to int, handling None and NaN"""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

# Test safe_float function
def safe_float(value, default=0.0):
    """Safely convert value to float, handling None and NaN"""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

# Test cases
test_cases = [
    None,
    np.nan,
    float('nan'),
    0,
    100,
    "50",
    50.5,
    "-",
    ""
]

print("Testing safe_int():")
for val in test_cases:
    result = safe_int(val)
    print(f"  safe_int({repr(val):20}) = {result}")

print("\nTesting safe_float():")
for val in test_cases:
    result = safe_float(val)
    print(f"  safe_float({repr(val):20}) = {result}")

print("\n✅ All conversions handled safely!")
