#!/usr/bin/env python3
"""
Example usage of the End-to-End Test Generation Orchestrator

This script demonstrates how to use the run_e2e_test_generation.py orchestrator
for fully automated test case generation.
"""

import os
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from run_e2e_test_generation import E2ETestGenerator

def example_1_simple_function():
    """Example 1: Generate test for a simple function by name."""
    print("="*60)
    print("EXAMPLE 1: Simple Function by Name")
    print("="*60)
    
    # Initialize the generator
    generator = E2ETestGenerator(
        embeddings_path='data/embeddings',
        allow_inferred_tests=True,
        infer_threshold=1.2
    )
    
    # Target function (just the name)
    target_function = "CalculateDistance"
    
    # Run the full pipeline
    results = generator.run_full_pipeline(
        target_function=target_function,
        top_k=3,
        output_path="examples/CalculateDistanceTest.cs"
    )
    
    if results['success']:
        print(f"✅ Generated test for {target_function}")
        print(f"📁 Saved to: {results['output_file']}")
    else:
        print(f"❌ Failed: {results['error']}")

def example_2_complex_function():
    """Example 2: Generate test for a complex function with full code."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Complex Function with Full Code")
    print("="*60)
    
    # Initialize the generator
    generator = E2ETestGenerator(
        embeddings_path='data/embeddings',
        allow_inferred_tests=True,
        infer_threshold=1.2
    )
    
    # Target function with full code
    target_function = """
    public class VRController : MonoBehaviour
    {
        public void HandleGrab(Transform objectToGrab)
        {
            if (objectToGrab != null)
            {
                objectToGrab.SetParent(this.transform);
                objectToGrab.localPosition = Vector3.zero;
                Debug.Log("Object grabbed successfully");
            }
        }
    }
    """
    
    # Run the full pipeline
    results = generator.run_full_pipeline(
        target_function=target_function,
        top_k=3,
        output_path="examples/VRControllerTest.cs"
    )
    
    if results['success']:
        print(f"✅ Generated test for VRController.HandleGrab")
        print(f"📁 Saved to: {results['output_file']}")
    else:
        print(f"❌ Failed: {results['error']}")

def example_3_batch_processing():
    """Example 3: Process multiple functions in batch."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Batch Processing Multiple Functions")
    print("="*60)
    
    # Initialize the generator
    generator = E2ETestGenerator(
        embeddings_path='data/embeddings',
        allow_inferred_tests=True,
        infer_threshold=1.2
    )
    
    # List of functions to process
    functions_to_test = [
        "MovePlayer",
        "RotateObject", 
        "CheckCollision",
        "PlaySound",
        "UpdateUI"
    ]
    
    results_summary = []
    
    for i, func_name in enumerate(functions_to_test, 1):
        print(f"\n🔄 Processing function {i}/{len(functions_to_test)}: {func_name}")
        
        try:
            results = generator.run_full_pipeline(
                target_function=func_name,
                top_k=3,
                output_path=f"examples/batch/{func_name}Test.cs"
            )
            
            if results['success']:
                results_summary.append({
                    'function': func_name,
                    'status': 'success',
                    'output_file': results['output_file'],
                    'reference_count': results['reference_count']
                })
                print(f"✅ Success: {func_name}")
            else:
                results_summary.append({
                    'function': func_name,
                    'status': 'failed',
                    'error': results['error']
                })
                print(f"❌ Failed: {func_name} - {results['error']}")
                
        except Exception as e:
            results_summary.append({
                'function': func_name,
                'status': 'error',
                'error': str(e)
            })
            print(f"❌ Error: {func_name} - {e}")
    
    # Print summary
    print("\n" + "="*60)
    print("BATCH PROCESSING SUMMARY")
    print("="*60)
    successful = [r for r in results_summary if r['status'] == 'success']
    failed = [r for r in results_summary if r['status'] != 'success']
    
    print(f"✅ Successful: {len(successful)}/{len(functions_to_test)}")
    print(f"❌ Failed: {len(failed)}/{len(functions_to_test)}")
    
    if successful:
        print("\n📁 Generated files:")
        for result in successful:
            print(f"  - {result['output_file']} ({result['reference_count']} references)")
    
    if failed:
        print("\n❌ Failed functions:")
        for result in failed:
            print(f"  - {result['function']}: {result['error']}")

def example_4_custom_configuration():
    """Example 4: Using custom configuration options."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Custom Configuration")
    print("="*60)
    
    # Initialize with custom configuration
    generator = E2ETestGenerator(
        embeddings_path='data/embeddings',
        allow_inferred_tests=False,  # Disable inferred tests
        infer_threshold=2.0  # Higher threshold for stricter matching
    )
    
    target_function = "ValidateInput"
    
    # Run with custom parameters
    results = generator.run_full_pipeline(
        target_function=target_function,
        top_k=5,  # Get more references
        output_path="examples/custom/ValidateInputTest.cs"
    )
    
    if results['success']:
        print(f"✅ Generated test with custom configuration")
        print(f"📁 Saved to: {results['output_file']}")
        print(f"🔧 Used {results['reference_count']} reference functions")
    else:
        print(f"❌ Failed: {results['error']}")

def main():
    """Run all examples."""
    print("🚀 End-to-End Test Generation Examples")
    print("="*60)
    
    # Create examples directory
    Path("examples").mkdir(exist_ok=True)
    Path("examples/batch").mkdir(exist_ok=True)
    Path("examples/custom").mkdir(exist_ok=True)
    
    try:
        # Run examples
        example_1_simple_function()
        example_2_complex_function()
        example_3_batch_processing()
        example_4_custom_configuration()
        
        print("\n🎉 All examples completed!")
        print("\n📋 Next Steps:")
        print("1. Check the generated test files in the 'examples' directory")
        print("2. Review the test cases and modify as needed")
        print("3. Integrate the tests into your project")
        print("4. Run the tests to verify they work correctly")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

