#!/usr/bin/env python3
"""
Simple function to check if a C# function has a test case in data/repos directory.
Returns True/False only.
"""

import os
import re

def has_testcase(function_code: str) -> bool:
    """
    Check if a C# function has a corresponding test case file.
    
    Args:
        function_code: The C# function code to check
    
    Returns:
        True if test case exists, False otherwise
    """
    try:
        # Extract function name
        function_name = extract_function_name(function_code)
        if not function_name:
            return False
        
        # Check data/repos directory
        repos_dir = "data/repos"
        if not os.path.exists(repos_dir):
            return False
        
        # Search for test files
        for root, dirs, files in os.walk(repos_dir):
            for file in files:
                if file.endswith('.cs') and any(keyword in file.lower() for keyword in ['test', 'spec']):
                    try:
                        with open(os.path.join(root, file), 'r', encoding='utf-8', errors='ignore') as f:
                            if function_name.lower() in f.read().lower():
                                return True
                    except:
                        continue
        
        return False
        
    except:
        return False

def has_testcase_by_name(function_name: str) -> bool:
    """
    Check if a function (by name) has a corresponding test case file.
    
    Args:
        function_name: The name of the function to check
    
    Returns:
        True if test case exists, False otherwise
    """
    try:
        # Check data/repos directory
        repos_dir = "data/repos"
        if not os.path.exists(repos_dir):
            return False
        
        # Search for test files
        for root, dirs, files in os.walk(repos_dir):
            for file in files:
                if file.endswith('.cs') and any(keyword in file.lower() for keyword in ['test', 'spec']):
                    try:
                        with open(os.path.join(root, file), 'r', encoding='utf-8', errors='ignore') as f:
                            if function_name.lower() in f.read().lower():
                                return True
                    except:
                        continue
        
        return False
        
    except:
        return False

def extract_function_name(code: str) -> str:
    """Extract function name from C# code."""
    patterns = [
        r'public\s+(?:static\s+)?(?:void|int|string|bool|float|double|object)\s+(\w+)\s*\(',
        r'private\s+(?:static\s+)?(?:void|int|string|bool|float|double|object)\s+(\w+)\s*\(',
        r'protected\s+(?:static\s+)?(?:void|int|string|bool|float|double|object)\s+(\w+)\s*\(',
        r'public\s+virtual\s+(?:void|int|string|bool|float|double|object)\s+(\w+)\s*\(',
        r'public\s+override\s+(?:void|int|string|bool|float|double|object)\s+(\w+)\s*\(',
        r'public\s+async\s+(?:Task|Task<.*?>|void)\s+(\w+)\s*\(',
        r'private\s+async\s+(?:Task|Task<.*?>|void)\s+(\w+)\s*\(',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, code, re.MULTILINE)
        if match:
            return match.group(1)
    
    return None

# Example usage
if __name__ == "__main__":
    print("Function Test Case Checker")
    print("=" * 30)
    
    # Ask for function name first
    function_name = input("Enter the function name: ").strip()
    
    if not function_name:
        print("No function name provided. Exiting.")
        exit()
    
    print(f"Checking for test case for function: {function_name}")
    
    # Check if test case exists
    result = has_testcase_by_name(function_name)
    print(f"Result: {result}")
