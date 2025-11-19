# Method Mapping Summary

This document provides a comprehensive summary of the method mapping analysis performed on `check_data_consistency.py`.

## Overview

The analysis identified and categorized **335 test methods** across the DataConsistencyChecker codebase. Each method has been mapped to its corresponding category based on the `tests_definitions` structure.

## Files Generated

1. **method_mapping.json** (50.7 KB)
   - Complete JSON mapping of all methods to their categories
   - Includes start line, end line, and method signature for each method
   - Organized by category for easy navigation

2. **build_method_mapping.py**
   - Python script used to generate the mapping
   - Can be re-run if the codebase changes

## Summary Statistics

### Total Methods: 335

### Methods by Category:

| Category              | Total Methods | Check Methods | Generate Methods | Notes |
|-----------------------|---------------|---------------|------------------|-------|
| **base_tests**        | 18            | 9             | 9                | Base patterns for any column type |
| **numeric_tests**     | 118           | 59            | 59               | Numeric column patterns |
| **date_tests**        | 36            | 18            | 18               | Date/datetime patterns |
| **string_tests**      | 118           | 59            | 59               | String/text patterns |
| **binary_tests**      | 24            | 12            | 12               | Binary column patterns |
| **multi_column_tests**| 18            | 9             | 9                | Multi-column relationships |
| **uncategorized**     | 3             | 3             | 0                | Helper methods |

### Total Categorized Test Methods: 332
### Helper Methods (not direct tests): 3

## Category Details

### Base Tests (18 methods)
Tests applicable to any column type, including:
- Missing values detection
- Rare values identification
- Unique value patterns
- Predictive patterns using decision trees
- Matched/opposite missing patterns between columns
- Same values across columns

**Key Methods:**
- `__check_missing` / `__generate_missing` (Lines 4034-4062 / 4020-4033)
- `__check_rare_values` / `__generate_rare_values` (Lines 4076-4123 / 4063-4075)
- `__check_unique_values` / `__generate_unique_values` (Lines 4134-4178 / 4124-4133)
- `__check_matched_missing` / `__generate_matched_missing` (Lines 4499-4568 / 4480-4498)

### Numeric Tests (118 methods)
Comprehensive tests for numeric columns:
- Value range tests (positive, negative, zero, magnitude)
- Decimal patterns and rounding
- Column ordering and trends
- Two-column relationships (larger, similar, correlated, constant operations)
- Three-column relationships (sums, products, ratios, differences)
- Multi-column aggregations (sum, mean, min, max of columns)
- Machine learning predictions (decision trees, linear regression)

**Key Methods:**
- `__check_positive_values` / `__generate_positive_values` (Lines 4959-4970 / 4950-4958)
- `__check_constant_sum` / `__generate_constant_sum` (Lines 6640-6698 / 6630-6639)
- `__check_dt_regressor` / `__generate_dt_regressor` (Lines 9119-9255 / 9091-9118)

### Date Tests (36 methods)
Date and datetime specific patterns:
- Early/late date detection
- Day of week, day of month, month patterns
- Hour and minute patterns
- Date gaps and relationships
- Date-numeric combinations

**Key Methods:**
- `__check_early_dates` / `__generate_early_dates` (Lines 9722-9743 / 9697-9721)
- `__check_constant_date_gap` / `__generate_constant_date_gap` (Lines 10089-10105 / 10060-10088)
- `__check_large_given_date` / `__generate_large_given_date` (Lines 10455-10544 / 10439-10454)

### String Tests (118 methods)
Text and character pattern analysis:
- Character patterns (alpha, numeric, special characters)
- Case patterns (uppercase, lowercase)
- Whitespace patterns
- Word patterns and frequency
- String similarity and relationships
- Prefix, suffix, and containment
- String-numeric combinations

**Key Methods:**
- `__check_chars_pattern` / `__generate_chars_pattern` (Lines 12855-12914 / 12838-12854)
- `__check_a_prefix_of_b` / `__generate_a_prefix_of_b` (Lines 14504-14560 / 14490-14503)
- `__check_dt_classifier` / `__generate_dt_classifier` (Lines 15956-16067 / 15928-15955)

### Binary Tests (24 methods)
Binary column patterns:
- Binary same/opposite values
- Binary implications
- Logical operations (AND, OR, XOR)
- Binary-numeric relationships
- Binary-string relationships

**Key Methods:**
- `__check_binary_same` / `__generate_binary_same` (Lines 10661-10718 / 10649-10660)
- `__check_binary_and` / `__generate_binary_and` (Lines 10886-10983 / 10874-10885)
- `__check_binary_matches_values` / `__generate_binary_matches_values` (Lines 11412-11524 / 11401-11411)

### Multi-Column Tests (18 methods)
Tests spanning multiple columns:
- Three+ column relationships
- Row-level aggregations
- Unique sets of values
- Value counts per row

**Key Methods:**
- `__check_c_is_a_or_b` / `__generate_c_is_a_or_b` (Lines 16253-16349 / 16068-16252)
- `__check_missing_values_per_row` / `__generate_missing_values_per_row` (Lines 16608-16617 / 16590-16607)
- `__check_unique_sets_values` / `__generate_unique_sets_values` (Lines 16531-16589 / 16510-16530)

### Uncategorized (Helper Methods)
These are internal helper methods not directly registered as tests:
- `__check_grouped_strings_column` (Lines 13313-13413) - Helper for grouped strings
- `__check_sum_exact` (Lines 7718-7722) - Helper for exact sum calculations
- `__check_two_cols_larger` (Lines 6076-6217) - Helper for column comparison

## JSON Structure

The `method_mapping.json` file has the following structure:

```json
{
  "base_tests": {
    "__check_missing": {
      "start_line": 4034,
      "end_line": 4062,
      "signature": "def __check_missing(self, test_id):"
    },
    ...
  },
  "numeric_tests": { ... },
  "date_tests": { ... },
  "string_tests": { ... },
  "binary_tests": { ... },
  "multi_column_tests": { ... },
  "uncategorized": { ... }
}
```

## Usage

This mapping can be used to:
1. **Extract methods into separate mixin files** - Use the line numbers to extract each method
2. **Understand method organization** - See which methods belong to which category
3. **Refactor the codebase** - Systematically reorganize code into logical modules
4. **Generate documentation** - Create category-specific documentation
5. **Track coverage** - Ensure all test methods are properly categorized

## Next Steps

Based on this mapping, you can now:
1. Create mixin classes for each category (e.g., `BaseTestsMixin`, `NumericTestsMixin`)
2. Extract methods from `check_data_consistency.py` into separate files
3. Update imports and class structure
4. Verify all tests still work correctly
5. Update documentation to reflect the new structure

## File Locations

- **Mapping File:** `/home/user/DataConsistencyChecker/method_mapping.json`
- **Summary:** `/home/user/DataConsistencyChecker/METHOD_MAPPING_SUMMARY.md`
- **Builder Script:** `/home/user/DataConsistencyChecker/build_method_mapping.py`
- **Source File:** `/home/user/DataConsistencyChecker/check_data_consistency.py`

## Validation

All methods have been validated to ensure:
- ✓ Each method has start_line, end_line, and signature fields
- ✓ All line ranges are valid (start < end)
- ✓ Total methods match between categories and source file
- ✓ Each check method has a corresponding generate method (with 3 exceptions for helper methods)

---

Generated on: 2025-11-19
Total Analysis Time: Complete
Source File Size: 885.8 KB (too large to read at once)
Methods Analyzed: 335
