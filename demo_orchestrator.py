#!/usr/bin/env python3
"""
Demo script for the End-to-End Test Generation Orchestrator

This script demonstrates the key capabilities of the new orchestrator
without requiring a full setup of the RAG system.
"""

import os
import sys
from pathlib import Path

def demo_basic_usage():
    """Demonstrate basic usage of the orchestrator."""
    print("🚀 End-to-End Test Generation Orchestrator Demo")
    print("="*60)
    
    print("\n📋 What this orchestrator does:")
    print("1. Takes a target function (by name or full code)")
    print("2. Automatically finds top-3 similar functions with existing tests")
    print("3. Uses AI to generate appropriate test cases")
    print("4. Intelligently decides whether to create new test files or append to existing ones")
    
    print("\n🔧 Key Features:")
    print("✅ Fully automated - no manual reference selection needed")
    print("✅ Intelligent prompt selection (prompt-1.txt vs prompt-2.txt)")
    print("✅ Batch processing support")
    print("✅ Comprehensive error handling and logging")
    print("✅ Configurable parameters")
    
    print("\n📝 Example Usage:")
    print("# Command line usage:")
    print("python run_e2e_test_generation.py --target 'CalculateDistance' --top-k 3")
    print("python run_e2e_test_generation.py --target 'public void HandleGrab(Transform obj) { ... }' --output 'MyTest.cs'")
    
    print("\n# Programmatic usage:")
    print("""
from run_e2e_test_generation import E2ETestGenerator

generator = E2ETestGenerator()
results = generator.run_full_pipeline(
    target_function="MyFunction",
    top_k=3,
    output_path="MyTest.cs"
)
""")
    
    print("\n🎯 Use Cases:")
    print("• Generate tests for new VR functions")
    print("• Batch process multiple functions")
    print("• Learn from existing test patterns")
    print("• Automate test case creation")
    print("• Improve code coverage")

def demo_workflow():
    """Demonstrate the complete workflow."""
    print("\n" + "="*60)
    print("🔄 Complete Workflow Demonstration")
    print("="*60)
    
    workflow_steps = [
        ("1. Input", "Target function (name or code)"),
        ("2. Search", "Find top-3 similar functions with tests using semantic search"),
        ("3. Analyze", "Determine if target function already has test file"),
        ("4. Select", "Choose appropriate prompt template (prompt-1.txt or prompt-2.txt)"),
        ("5. Generate", "Use AI to generate test case based on references"),
        ("6. Save", "Save generated test to appropriate location"),
        ("7. Report", "Provide detailed results and metadata")
    ]
    
    for step, description in workflow_steps:
        print(f"\n{step}: {description}")
    
    print("\n💡 The orchestrator handles all these steps automatically!")

def demo_configuration():
    """Demonstrate configuration options."""
    print("\n" + "="*60)
    print("⚙️ Configuration Options")
    print("="*60)
    
    config_options = [
        ("--target, -t", "Target function code or name (required)"),
        ("--top-k, -k", "Number of reference functions (default: 3)"),
        ("--output, -o", "Output path for generated test file"),
        ("--embeddings-path", "Path to embeddings directory (default: data/embeddings)"),
        ("--allow-inferred-tests", "Allow inferred tests when no direct tests found"),
        ("--infer-threshold", "Threshold for test inference confidence (default: 1.2)"),
        ("--verbose, -v", "Enable verbose logging")
    ]
    
    print("\nCommand Line Options:")
    for option, description in config_options:
        print(f"  {option:<25} {description}")
    
    print("\nProgrammatic Configuration:")
    print("""
generator = E2ETestGenerator(
    embeddings_path='data/embeddings',
    allow_inferred_tests=True,
    infer_threshold=1.2
)
""")

def demo_examples():
    """Show example scenarios."""
    print("\n" + "="*60)
    print("📚 Example Scenarios")
    print("="*60)
    
    scenarios = [
        {
            "title": "Simple Function by Name",
            "input": "CalculateDistance",
            "description": "Generate test for a function identified by name only"
        },
        {
            "title": "Complex Function with Code",
            "input": "public void HandleGrab(Transform objectToGrab) { ... }",
            "description": "Generate test for a function with full implementation"
        },
        {
            "title": "VR Controller Method",
            "input": "public void UpdateVRPosition(Vector3 newPosition)",
            "description": "Generate test for VR-specific functionality"
        },
        {
            "title": "Batch Processing",
            "input": "['MovePlayer', 'RotateObject', 'CheckCollision']",
            "description": "Process multiple functions in a single run"
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}. {scenario['title']}")
        print(f"   Input: {scenario['input']}")
        print(f"   Description: {scenario['description']}")

def demo_integration():
    """Show integration with existing system."""
    print("\n" + "="*60)
    print("🔗 Integration with Existing System")
    print("="*60)
    
    print("\nThe orchestrator integrates with existing components:")
    print("• query_system.py - For semantic search and reference discovery")
    print("• test_method_generator.py - For AI-powered test generation")
    print("• prompt-1.txt - For appending to existing test files")
    print("• prompt-2.txt - For creating new test files")
    print("• tree_sitter_extractor.py - For function analysis")
    
    print("\nDependencies:")
    print("• OpenAI API for test generation")
    print("• FAISS index for semantic search")
    print("• SentenceTransformers for embeddings")
    print("• Tree-sitter for C# parsing")

def main():
    """Run the complete demo."""
    try:
        demo_basic_usage()
        demo_workflow()
        demo_configuration()
        demo_examples()
        demo_integration()
        
        print("\n" + "="*60)
        print("🎉 Demo Complete!")
        print("="*60)
        print("\n📋 Next Steps:")
        print("1. Set up your OpenAI API key in test_method_generator.py")
        print("2. Ensure your embeddings are built (run the full pipeline)")
        print("3. Try the orchestrator with your own functions:")
        print("   python run_e2e_test_generation.py --target 'YourFunction'")
        print("4. Check out example_usage.py for more detailed examples")
        
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

