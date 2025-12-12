#!/usr/bin/env python
"""Test script for target combination feature."""

import sys
import os

# Add the deepem module to the path
sys.path.insert(0, os.path.dirname(__file__))

from deepem.data.dataset.zettaset import parse_target_combination
from deepem.data.dataset.multi_zettaset import parse_target_combination as parse_target_combination_multi

def test_parse_target_combination():
    """Test the parse_target_combination function."""

    # Test single target
    result = parse_target_combination("mye")
    assert result == ["mye"], f"Expected ['mye'], got {result}"
    print("✓ Single target: 'mye' -> ['mye']")

    # Test two targets with spaces
    result = parse_target_combination("mye + ecs")
    assert result == ["mye", "ecs"], f"Expected ['mye', 'ecs'], got {result}"
    print("✓ Two targets with spaces: 'mye + ecs' -> ['mye', 'ecs']")

    # Test two targets without spaces
    result = parse_target_combination("mye+ecs")
    assert result == ["mye", "ecs"], f"Expected ['mye', 'ecs'], got {result}"
    print("✓ Two targets without spaces: 'mye+ecs' -> ['mye', 'ecs']")

    # Test three targets
    result = parse_target_combination("a + b + c")
    assert result == ["a", "b", "c"], f"Expected ['a', 'b', 'c'], got {result}"
    print("✓ Three targets: 'a + b + c' -> ['a', 'b', 'c']")

    # Test with extra whitespace
    result = parse_target_combination("  mye  +  ecs  ")
    assert result == ["mye", "ecs"], f"Expected ['mye', 'ecs'], got {result}"
    print("✓ Extra whitespace: '  mye  +  ecs  ' -> ['mye', 'ecs']")

    # Verify multi_zettaset has the same function
    result = parse_target_combination_multi("mye + ecs")
    assert result == ["mye", "ecs"], f"Expected ['mye', 'ecs'], got {result}"
    print("✓ multi_zettaset.parse_target_combination works correctly")

    print("\n✅ All tests passed!")

if __name__ == "__main__":
    test_parse_target_combination()
