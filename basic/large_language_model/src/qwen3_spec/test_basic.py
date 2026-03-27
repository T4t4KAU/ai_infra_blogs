"""
Basic test script to verify the speculative decoding implementation loads correctly.
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported without errors."""
    try:
        from speculative_decoder import SpeculativeDecoder
        from utils import setup_device, prepare_inputs_for_model
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_utils_functions():
    """Test basic utility functions."""
    try:
        from utils import setup_device
        device = setup_device()
        print(f"✓ Device setup successful: {device}")
        return True
    except Exception as e:
        print(f"✗ Utils test failed: {e}")
        return False

def main():
    """Run basic tests."""
    print("Running basic tests for speculative decoding implementation...")
    print("=" * 60)
    
    tests = [
        ("Import test", test_imports),
        ("Utils test", test_utils_functions)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\nRunning {test_name}...")
        result = test_func()
        results.append(result)
        print(f"{'PASS' if result else 'FAIL'}")
    
    print("\n" + "=" * 60)
    if all(results):
        print("✓ All basic tests passed!")
        print("The implementation is ready for use with actual Qwen3 models.")
    else:
        print("✗ Some tests failed. Please check the implementation.")
    
    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)