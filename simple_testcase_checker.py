#!/usr/bin/env python3
"""
Simple script to check if a function has a corresponding test case file.
Usage: python simple_testcase_checker.py
"""

import os
import re
from pathlib import Path

def check_function_has_testcase(function_code: str, search_directory: str = ".") -> bool:
    """
    Check if a function has a corresponding test case file.
    
    Args:
        function_code: The C# function code to check
        search_directory: Directory to search for test files
    
    Returns:
        True if test case exists, False otherwise 
    """
    try:
        # Extract function name from the code
        function_name = extract_function_name(function_code)
        if not function_name:
            print("Could not extract function name from code")
            return False
        
        print(f"Looking for test cases for function: {function_name}")
        
        # Find all .cs files in the directory
        cs_files = []
        for root, dirs, files in os.walk(search_directory):
            for file in files:
                if file.endswith('.cs'):
                    cs_files.append(os.path.join(root, file))
        
        print(f"Found {len(cs_files)} .cs files in directory")
        
        # Look for test files
        test_files = []
        for cs_file in cs_files:
            filename = os.path.basename(cs_file)
            
            # Check if filename suggests it's a test file
            if any(keyword in filename.lower() for keyword in ['test', 'spec']):
                # Check if function name appears in the file
                try:
                    with open(cs_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        if function_name.lower() in content.lower():
                            test_files.append(cs_file)
                            print(f"Found potential test file: {cs_file}")
                except Exception as e:
                    continue
        
        return len(test_files) > 0
        
    except Exception as e:
        print(f"Error: {e}")
        return False

def extract_function_name(code: str) -> str:
    """Extract function name from C# code."""
    # Common C# function patterns
    patterns = [
        r'public\s+(?:static\s+)?(?:void|int|string|bool|float|double|object)\s+(\w+)\s*\(',
        r'private\s+(?:static\s+)?(?:void|int|string|bool|float|double|object)\s+(\w+)\s*\(',
        r'protected\s+(?:static\s+)?(?:void|int|string|bool|float|double|object)\s+(\w+)\s*\(',
        r'public\s+virtual\s+(?:void|int|string|bool|float|double|object)\s+(\w+)\s*\(',
        r'public\s+override\s+(?:void|int|string|bool|float|double|object)\s+(\w+)\s*\(',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, code, re.MULTILINE)
        if match:
            return match.group(1)
    
    return None

def main():
    """Main function."""
    print("Function Test Case Checker")
    print("=" * 30)
    
    # Example function from your ai-code.py
    example_function = """public void updateBrightness(float brightness)
    {
        volume.profile.TryGetSettings(out colorGradingLayer);
        colorGradingLayer.enabled.value = true;
        colorGradingLayer.brightness.value = brightness;   
    }"""
    
    print("Checking example function:")
    print(example_function)
    print()
    
    has_testcase = check_function_has_testcase(example_function)
    print(f"Result: {'Test case exists' if has_testcase else 'No test case found'}")
    
    print("\n" + "=" * 30)
    print("Enter your own function code to check:")
    print("(Type 'quit' to exit)")
    
    while True:
        print("\nFunction code:")
        user_input = input("> ")
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        
        if user_input.strip():
            has_testcase = check_function_has_testcase(user_input)
            print(f"Result: {'Test case exists' if has_testcase else 'No test case found'}")

if __name__ == "__main__":
    main()
