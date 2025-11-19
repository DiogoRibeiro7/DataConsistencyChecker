#!/usr/bin/env python3
"""
Script to build a complete mapping of all test methods in check_data_consistency.py
to their categories based on the tests_definitions structure.
"""

import json
import re
from pathlib import Path


def parse_method_definitions(file_path):
    """Extract all __check_* and __generate_* methods with their line numbers and signatures."""
    methods = {}

    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Find all method definitions
    for i, line in enumerate(lines, start=1):
        # Match methods that start with __check_, __generate_, check_, or generate_
        match = re.match(r'^\s+def (__check_|__generate_|check_|generate_)(\w+)\((.*?)\):', line)
        if match:
            prefix = match.group(1)
            method_name = match.group(2)
            full_name = f"{prefix}{method_name}"
            params = match.group(3)

            # Only include methods that match our pattern
            if full_name.startswith('__check_') or full_name.startswith('__generate_') or \
               (full_name.startswith('check_') and 'mathed' in full_name or 'matched' in full_name) or \
               (full_name.startswith('generate_') and 'mathed' in full_name or 'matched' in full_name):
                methods[full_name] = {
                    'start_line': i,
                    'signature': f"def {full_name}({params}):",
                    'end_line': None  # Will be filled later
                }

    # Now find the end lines for each method
    method_starts = sorted([(v['start_line'], k) for k, v in methods.items()])

    # For each method, find where it ends (next method or class definition)
    for idx, (start_line, method_name) in enumerate(method_starts):
        # Find the next method or class definition
        if idx < len(method_starts) - 1:
            next_start = method_starts[idx + 1][0]
            # The method ends at the line before the next method
            methods[method_name]['end_line'] = next_start - 1
        else:
            # Last method - find the end of file or next class
            # Read through to find next class or end of file
            end_line = start_line
            for i in range(start_line, len(lines) + 1):
                if i < len(lines):
                    # Check if we hit a class definition or end of indentation
                    if re.match(r'^class\s+', lines[i-1]):
                        end_line = i - 1
                        break
                else:
                    end_line = len(lines)
            methods[method_name]['end_line'] = end_line

    return methods


def load_test_definitions():
    """Load all test definition files and extract method references."""
    base_path = Path('/home/user/DataConsistencyChecker/tests_definitions')

    categories = {
        'base_tests': base_path / 'base_tests.py',
        'numeric_tests': base_path / 'numeric_tests.py',
        'date_tests': base_path / 'date_tests.py',
        'string_tests': base_path / 'string_tests.py',
        'binary_tests': base_path / 'binary_tests.py',
        'multi_column_tests': base_path / 'multi_column_tests.py',
    }

    method_to_category = {}

    for category, file_path in categories.items():
        with open(file_path, 'r') as f:
            content = f.read()

        # Find all method references in the form:
        # checker._DataConsistencyChecker__check_xxx or
        # checker._DataConsistencyChecker__generate_xxx or
        # checker.check_xxx or checker.generate_xxx

        # Pattern for private methods (__check_, __generate_)
        private_pattern = r'checker\._DataConsistencyChecker(__(?:check|generate)_\w+)'
        for match in re.finditer(private_pattern, content):
            method_name = match.group(1)
            method_to_category[method_name] = category

        # Pattern for public methods (check_, generate_)
        public_pattern = r'checker\.((?:check|generate)_\w+)'
        for match in re.finditer(public_pattern, content):
            method_name = match.group(1)
            # Only include if it matches our specific patterns
            if 'mathed' in method_name or 'matched' in method_name:
                method_to_category[method_name] = category

    return method_to_category


def build_mapping():
    """Build the complete mapping of methods to categories."""
    print("Parsing method definitions from check_data_consistency.py...")
    methods = parse_method_definitions('/home/user/DataConsistencyChecker/check_data_consistency.py')
    print(f"Found {len(methods)} methods")

    print("\nLoading test definitions...")
    method_to_category = load_test_definitions()
    print(f"Found {len(method_to_category)} method references in test definitions")

    # Build the final mapping organized by category
    mapping = {
        'base_tests': {},
        'numeric_tests': {},
        'date_tests': {},
        'string_tests': {},
        'binary_tests': {},
        'multi_column_tests': {},
        'uncategorized': {}
    }

    for method_name, method_info in methods.items():
        category = method_to_category.get(method_name, 'uncategorized')
        mapping[category][method_name] = method_info

    # Print summary statistics
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total methods found: {len(methods)}")
    print(f"\nMethods per category:")
    for category in ['base_tests', 'numeric_tests', 'date_tests', 'string_tests',
                     'binary_tests', 'multi_column_tests', 'uncategorized']:
        count = len(mapping[category])
        print(f"  {category:25s}: {count:3d} methods")

    return mapping


def main():
    mapping = build_mapping()

    output_file = '/home/user/DataConsistencyChecker/method_mapping.json'
    print(f"\nWriting mapping to {output_file}...")

    with open(output_file, 'w') as f:
        json.dump(mapping, f, indent=2)

    print(f"\nMapping file created successfully!")
    print(f"File location: {output_file}")

    # Print some examples
    print("\n" + "="*70)
    print("SAMPLE METHODS FROM EACH CATEGORY")
    print("="*70)
    for category in ['base_tests', 'numeric_tests', 'date_tests']:
        methods = list(mapping[category].keys())[:3]
        if methods:
            print(f"\n{category}:")
            for method in methods:
                info = mapping[category][method]
                print(f"  {method:40s} Lines {info['start_line']:5d}-{info['end_line']:5d}")


if __name__ == '__main__':
    main()
