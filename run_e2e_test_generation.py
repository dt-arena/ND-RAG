#!/usr/bin/env python3
"""
End-to-End Test Case Generation Orchestrator

This script automates the complete test case generation process by:
1. Taking a target function as input
2. Automatically querying the semantic search system for top-3 references
3. Automatically generating test cases using the appropriate prompt template
4. Intelligently deciding whether to create new test files or append to existing ones
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import re

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from query_system import SemanticQuerySystem
from test_method_generator import (
    check_existing_test_file, 
    extract_function_name,
    process_reference_functions,
    build_multi_reference_prompt
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('e2e_test_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class E2ETestGenerator:
    """End-to-end test case generation orchestrator."""
    
    def __init__(self, embeddings_path: str = 'data/embeddings', 
                 allow_inferred_tests: bool = True,
                 infer_threshold: float = 1.2):
        """
        Initialize the E2E test generator.
        
        Args:
            embeddings_path: Path to embeddings directory
            allow_inferred_tests: Whether to allow inferred tests
            infer_threshold: Threshold for test inference confidence
        """
        self.embeddings_path = embeddings_path
        self.allow_inferred_tests = allow_inferred_tests
        self.infer_threshold = infer_threshold
        
        # Load prompt templates
        self.prompt_template_1 = self._load_prompt_template('prompt-1.txt')
        self.prompt_template_2 = self._load_prompt_template('prompt-2.txt')
        
        # Initialize query system
        try:
            self.query_system = SemanticQuerySystem(
                embeddings_path=embeddings_path,
                infer_tests=allow_inferred_tests,
                infer_threshold=infer_threshold
            )
            logger.info("✅ Query system initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize query system: {e}")
            raise
    
    def _load_prompt_template(self, filename: str) -> str:
        """Load prompt template from file."""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"❌ Prompt template file not found: {filename}")
            raise
        except Exception as e:
            logger.error(f"❌ Error loading prompt template {filename}: {e}")
            raise
    
    def identify_target_function(self, target_input: str) -> str:
        """
        Identify and extract the target function from input.
        
        Args:
            target_input: Function code or function name
            
        Returns:
            Cleaned function code
        """
        # If it's just a function name, we'll need to find the actual code
        if self._is_function_name_only(target_input):
            logger.info(f"🔍 Function name detected: {target_input}")
            # For now, return as-is. In a full implementation, you might want to
            # search for the actual function code in the codebase
            return target_input
        
        # Clean and validate function code
        cleaned_function = self._clean_function_code(target_input)
        logger.info("✅ Target function identified and cleaned")
        return cleaned_function
    
    def _is_function_name_only(self, text: str) -> bool:
        """Check if input is just a function name (not full code)."""
        # Simple heuristic: if it doesn't contain common code patterns
        code_indicators = ['{', '}', ';', 'public', 'private', 'void', 'return']
        return not any(indicator in text for indicator in code_indicators)
    
    def _clean_function_code(self, code: str) -> str:
        """Clean and validate function code."""
        # Remove extra whitespace and normalize
        cleaned = re.sub(r'\s+', ' ', code.strip())
        return cleaned
    
    def get_reference_functions(self, target_function: str, top_k: int = 3) -> List[Dict]:
        """
        Automatically get top-k reference functions using semantic search.
        
        Args:
            target_function: Target function code or name
            top_k: Number of reference functions to retrieve
            
        Returns:
            List of reference function dictionaries
        """
        logger.info(f"🔍 Searching for top-{top_k} reference functions...")
        
        try:
            # Use the query system to find similar functions with tests
            results = self.query_system.query(
                query=target_function,
                top_k=top_k,
                only_with_tests=True,
                require_tests_strict=False
            )
            
            if not results:
                logger.warning("⚠️ No reference functions found with tests")
                return []
            
            # Process results into reference format
            reference_functions = []
            for i, result in enumerate(results):
                if result.get('test_source') and result.get('function_source'):
                    ref_dict = {
                        'function': result['function_source'],
                        'test': result['test_source'],
                        'repo_name': result.get('repo_name', 'unknown'),
                        'match_score': result.get('hybrid_score', 0.0),
                        'match_type': result.get('match_type', 'unknown')
                    }
                    reference_functions.append(ref_dict)
                    logger.info(f"✅ Reference {i+1}: {result.get('function_name', 'Unknown')} "
                              f"(score: {ref_dict['match_score']:.3f})")
            
            logger.info(f"✅ Found {len(reference_functions)} reference functions")
            return reference_functions
            
        except Exception as e:
            logger.error(f"❌ Error getting reference functions: {e}")
            return []
    
    def determine_generation_strategy(self, target_function: str) -> Tuple[str, str]:
        """
        Determine whether to create new test file or append to existing one.
        
        Args:
            target_function: Target function code
            
        Returns:
            Tuple of (prompt_template, generation_type)
        """
        logger.info("🔍 Checking if target function already has test file...")
        
        try:
            has_existing_test = check_existing_test_file(target_function)
            
            if has_existing_test:
                logger.info("✅ Found existing test file. Using prompt-1.txt...")
                return self.prompt_template_1, "test method"
            else:
                logger.info("❌ No existing test file. Using prompt-2.txt...")
                return self.prompt_template_2, "complete test file"
                
        except Exception as e:
            logger.warning(f"⚠️ Error checking existing test file: {e}")
            logger.info("🔄 Defaulting to complete test file generation...")
            return self.prompt_template_2, "complete test file"
    
    def generate_test_case(self, target_function: str, reference_functions: List[Dict], 
                          prompt_template: str) -> str:
        """
        Generate test case using OpenAI API.
        
        Args:
            target_function: Target function code
            reference_functions: List of reference function dictionaries
            prompt_template: Prompt template to use
            
        Returns:
            Generated test case
        """
        logger.info("🤖 Generating test case with OpenAI...")
        
        try:
            import openai
            
            # Process reference functions
            processed_references = process_reference_functions([
                f"{ref['function']}\n\nTestcase:\n{ref['test']}" 
                for ref in reference_functions
            ])
            
            if not processed_references:
                raise ValueError("No valid reference functions to process")
            
            # Build prompt with multiple references
            filled_prompt = build_multi_reference_prompt(
                target_function, 
                processed_references, 
                prompt_template
            )
            
            # Generate test case
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": filled_prompt}],
                temperature=0.3,
                max_tokens=2000
            )
            
            generated_testcase = response.choices[0].message.content.strip()
            logger.info("✅ Test case generated successfully")
            return generated_testcase
            
        except Exception as e:
            logger.error(f"❌ Error generating test case: {e}")
            raise
    
    def save_test_case(self, test_case: str, target_function: str, 
                      generation_type: str, output_path: Optional[str] = None) -> str:
        """
        Save generated test case to file.
        
        Args:
            test_case: Generated test case code
            target_function: Target function code
            generation_type: Type of generation (test method or complete file)
            output_path: Optional output path
            
        Returns:
            Path to saved file
        """
        try:
            # Extract function name for filename
            function_name = extract_function_name(target_function)
            if not function_name:
                function_name = "GeneratedTest"
            
            # Determine output path
            if output_path:
                output_file = Path(output_path)
            else:
                if generation_type == "complete test file":
                    output_file = Path(f"generated_tests/{function_name}Test.cs")
                else:
                    output_file = Path(f"generated_tests/{function_name}TestMethod.cs")
            
            # Create output directory if it doesn't exist
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save test case
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(test_case)
            
            logger.info(f"✅ Test case saved to: {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"❌ Error saving test case: {e}")
            raise
    
    def run_full_pipeline(self, target_function: str, top_k: int = 3, 
                         output_path: Optional[str] = None) -> Dict:
        """
        Run the complete end-to-end test generation pipeline.
        
        Args:
            target_function: Target function code or name
            top_k: Number of reference functions to retrieve
            output_path: Optional output path for generated test
            
        Returns:
            Dictionary with results and metadata
        """
        logger.info("🚀 Starting end-to-end test generation pipeline...")
        
        try:
            # Step 1: Identify target function
            logger.info("Step 1: Identifying target function...")
            cleaned_target = self.identify_target_function(target_function)
            
            # Step 2: Get reference functions
            logger.info("Step 2: Getting reference functions...")
            reference_functions = self.get_reference_functions(cleaned_target, top_k)
            
            if not reference_functions:
                raise ValueError("No reference functions found. Cannot generate test case.")
            
            # Step 3: Determine generation strategy
            logger.info("Step 3: Determining generation strategy...")
            prompt_template, generation_type = self.determine_generation_strategy(cleaned_target)
            
            # Step 4: Generate test case
            logger.info("Step 4: Generating test case...")
            test_case = self.generate_test_case(cleaned_target, reference_functions, prompt_template)
            
            # Step 5: Save test case
            logger.info("Step 5: Saving test case...")
            output_file = self.save_test_case(test_case, cleaned_target, generation_type, output_path)
            
            # Prepare results
            results = {
                'success': True,
                'target_function': cleaned_target,
                'reference_count': len(reference_functions),
                'generation_type': generation_type,
                'output_file': output_file,
                'test_case': test_case,
                'references': reference_functions
            }
            
            logger.info("🎉 End-to-end test generation completed successfully!")
            return results
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'target_function': target_function
            }


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description='End-to-End Test Case Generation')
    parser.add_argument('--target', '-t', type=str, required=True,
                       help='Target function code or name')
    parser.add_argument('--top-k', '-k', type=int, default=3,
                       help='Number of reference functions to retrieve (default: 3)')
    parser.add_argument('--output', '-o', type=str,
                       help='Output path for generated test file')
    parser.add_argument('--embeddings-path', default='data/embeddings',
                       help='Path to embeddings directory')
    parser.add_argument('--allow-inferred-tests', action='store_true',
                       help='Allow inferred tests when no direct tests found')
    parser.add_argument('--infer-threshold', type=float, default=1.2,
                       help='Threshold for test inference confidence')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize generator
        generator = E2ETestGenerator(
            embeddings_path=args.embeddings_path,
            allow_inferred_tests=args.allow_inferred_tests,
            infer_threshold=args.infer_threshold
        )
        
        # Run pipeline
        results = generator.run_full_pipeline(
            target_function=args.target,
            top_k=args.top_k,
            output_path=args.output
        )
        
        if results['success']:
            print("\n" + "="*60)
            print("🎉 TEST GENERATION SUCCESSFUL!")
            print("="*60)
            print(f"📁 Output file: {results['output_file']}")
            print(f"🔧 Generation type: {results['generation_type']}")
            print(f"📊 Reference functions used: {results['reference_count']}")
            print("\n📋 Generated Test Case:")
            print("-"*40)
            print(results['test_case'])
            print("-"*40)
        else:
            print(f"\n❌ TEST GENERATION FAILED: {results['error']}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

