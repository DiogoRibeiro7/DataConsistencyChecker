"""
Test Registry Module for DataConsistencyChecker

This module defines constants and structures for organizing the 164 tests
performed by DataConsistencyChecker. Tests are categorized by the types
of columns they operate on and registered in a central dictionary.

Constants:
    TEST_DEFN_* - Indices for accessing test definition tuples

Test Definition Tuple Structure:
    (short_desc, description, test_func, gen_func, shortlist, implemented, fast, code)
"""

# Constants used to define the tests. The tests are defined in a dictionary.
# These specify the indexes of the elements of the dictionary values.
TEST_DEFN_SHORT_DESC = 0    # Element 0: Short description of the test (for progress display)
TEST_DEFN_DESC = 1          # Element 1: Full description of the test
TEST_DEFN_FUNC = 2          # Element 2: Function used to execute the test
TEST_DEFN_GEN_FUNC = 3      # Element 3: Function used to generate demo data for the test
TEST_DEFN_SHORTLIST = 4     # Element 4: Include in shortlist (patterns without exceptions)
TEST_DEFN_IMPLEMENTED = 5   # Element 5: Whether the test is implemented
TEST_DEFN_FAST = 6          # Element 6: Fast even with thousands of columns
TEST_DEFN_CODE = 7          # Element 7: Assumes values may represent code or ID values


class TestRegistry:
    """
    Central registry for all tests in DataConsistencyChecker.

    This class manages test definitions organized by category and provides
    methods to retrieve, filter, and register tests.
    """

    def __init__(self):
        """Initialize an empty test registry."""
        self._tests = {}

    def register_test(self, test_id, short_desc, description, test_func, gen_func,
                     shortlist=False, implemented=True, fast=True, code=False):
        """
        Register a new test in the registry.

        Args:
            test_id (str): Unique identifier for the test
            short_desc (str): Short description for progress display
            description (str): Full description of what the test does
            test_func (callable): Function that executes the test
            gen_func (callable): Function that generates synthetic data for the test
            shortlist (bool): Whether to include in default pattern listings
            implemented (bool): Whether the test is fully implemented
            fast (bool): Whether the test runs fast even with many columns
            code (bool): Whether the test assumes code/ID values
        """
        self._tests[test_id] = (
            short_desc,
            description,
            test_func,
            gen_func,
            shortlist,
            implemented,
            fast,
            code
        )

    def register_tests(self, test_dict):
        """
        Register multiple tests from a dictionary.

        Args:
            test_dict (dict): Dictionary mapping test_id to test definition tuples
        """
        self._tests.update(test_dict)

    def get_test(self, test_id):
        """
        Get a test definition by ID.

        Args:
            test_id (str): Test identifier

        Returns:
            tuple: Test definition tuple or None if not found
        """
        return self._tests.get(test_id)

    def get_all_tests(self):
        """
        Get all registered tests.

        Returns:
            dict: Dictionary of all test definitions
        """
        return self._tests.copy()

    def get_test_ids(self, implemented_only=False, fast_only=False, code_only=False):
        """
        Get list of test IDs with optional filtering.

        Args:
            implemented_only (bool): Only return implemented tests
            fast_only (bool): Only return fast tests
            code_only (bool): Only return code/ID tests

        Returns:
            list: Filtered list of test IDs
        """
        test_ids = []
        for test_id, test_def in self._tests.items():
            if implemented_only and not test_def[TEST_DEFN_IMPLEMENTED]:
                continue
            if fast_only and not test_def[TEST_DEFN_FAST]:
                continue
            if code_only and not test_def[TEST_DEFN_CODE]:
                continue
            test_ids.append(test_id)
        return test_ids

    def get_tests_by_category(self):
        """
        Organize tests by their categories.

        Returns:
            dict: Tests organized by category
        """
        categories = {
            'single_any': [],
            'pair_any': [],
            'single_numeric': [],
            'pair_numeric': [],
            'numeric_mixed': [],
            'triple_numeric': [],
            'single_date': [],
            'pair_date': [],
            'date_numeric': [],
            'binary_pair': [],
            'binary_sets': [],
            'binary_numeric': [],
            'binary_triple': [],
            'binary_string_numeric': [],
            'binary_multi_numeric': [],
            'single_string': [],
            'pair_string': [],
            'string_numeric_triple': [],
            'string_binary_numeric': [],
        }
        # This would be populated based on test metadata
        return categories

    def __len__(self):
        """Return the number of registered tests."""
        return len(self._tests)

    def __contains__(self, test_id):
        """Check if a test is registered."""
        return test_id in self._tests

    def __iter__(self):
        """Iterate over test IDs."""
        return iter(self._tests)

    def items(self):
        """Return test items (id, definition) pairs."""
        return self._tests.items()

    def keys(self):
        """Return test IDs."""
        return self._tests.keys()

    def values(self):
        """Return test definitions."""
        return self._tests.values()
