#!/usr/bin/env python3
"""
Test compilation for a single test file
"""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from improved_automated_test_generator import ImprovedAutomatedTestGenerator

def test_single_file(test_file_path):
    """Test compilation of a single test file"""
    print("="*60)
    print("TESTING COMPILATION OF SINGLE TEST FILE")
    print("="*60)
    print(f"\nTest file: {test_file_path}\n")
    
    # Check if file exists
    test_path = Path(test_file_path)
    if not test_path.exists():
        print(f"[ERROR] Test file not found: {test_file_path}")
        return False
    
    print(f"[INFO] File exists: {test_path.absolute()}")
    print(f"[INFO] File size: {test_path.stat().st_size} bytes\n")
    
    # Initialize generator
    generator = ImprovedAutomatedTestGenerator()
    
    # Compile the test file
    print("="*60)
    print("COMPILING TEST FILE...")
    print("="*60)
    print()
    
    result = generator.compile_test_file(str(test_path.absolute()))
    
    # Display results
    print("\n" + "="*60)
    print("COMPILATION RESULTS")
    print("="*60)
    print()
    
    print(f"Success: {result['success']}")
    print(f"Compiled: {result['compiled']}")
    print(f"Build Tool: {result['build_tool']}")
    print(f"Project File: {result['project_file']}")
    print(f"Errors: {len(result['errors'])}")
    print(f"Warnings: {len(result['warnings'])}")
    print()
    
    if result['compiled']:
        print("[OK] [SUCCESS] Test file compiled successfully!")
        
        if result['warnings']:
            print(f"\n[WARNING] Warnings ({len(result['warnings'])}):")
            for i, warning in enumerate(result['warnings'][:10], 1):
                print(f"   {i}. {warning}")
            if len(result['warnings']) > 10:
                print(f"   ... and {len(result['warnings']) - 10} more warnings")
    else:
        print("[ERROR] [FAILED] Test file compilation failed!")
        
        if result['errors']:
            print(f"\n[ERROR] Errors ({len(result['errors'])}):")
            for i, error in enumerate(result['errors'][:15], 1):
                print(f"   {i}. {error}")
            if len(result['errors']) > 15:
                print(f"   ... and {len(result['errors']) - 15} more errors")
    
    if result['output']:
        print(f"\n[INFO] Build Output (last 500 characters):")
        print("-" * 60)
        output_preview = result['output'][-500:] if len(result['output']) > 500 else result['output']
        print(output_preview)
        print("-" * 60)
    
    print("\n" + "="*60)
    
    return result['compiled']

def main():
    """Main function"""
    # Default test file path
    script_dir = Path(__file__).parent
    default_test_file = script_dir / "data" / "repos" / "11-VRapp_VR-project" / "The RegnAnt" / "Assets" / "Tests" / "AntWalkingInGrassTest.cs"
    
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    else:
        test_file = str(default_test_file)
    
    success = test_single_file(test_file)
    
    if success:
        print("\n[OK] Compilation successful! The test file is ready to use.")
        sys.exit(0)
    else:
        print("\n[ERROR] Compilation failed. Please check the errors above.")
        sys.exit(1)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

