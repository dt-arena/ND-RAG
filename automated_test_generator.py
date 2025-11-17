#!/usr/bin/env python3
"""
Automated Test Case Generation with RAG Integration

This script provides a fully automated test case generation process that:
1. Takes a target function as input
2. Uses RAG (Retrieval-Augmented Generation) to automatically find reference functions
3. Generates a backbone class with minimal structure
4. Generates test cases using prompt-1 (always)
5. Saves both backbone class and test case
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

# Import RAG system components
from query_system import SemanticQuerySystem
from test_method_generator import (
    extract_function_name,
    process_reference_functions,
    build_multi_reference_prompt
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('automated_test_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AutomatedTestGenerator:
    """Automated test case generation with RAG integration."""
    
    def __init__(self, embeddings_path: str = 'data/embeddings'):
        """
        Initialize the automated test generator.
        
        Args:
            embeddings_path: Path to embeddings directory
        """
        self.embeddings_path = embeddings_path
        
        # Load prompt templates
        self.prompt_template_1 = self._load_prompt_template('prompt-1.txt')
        self.prompt_template_3 = self._load_prompt_template('prompt-3.txt')
        
        # Initialize RAG query system
        try:
            self.query_system = SemanticQuerySystem(embeddings_path=embeddings_path)
            logger.info("✅ RAG query system initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize RAG query system: {e}")
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
    
    def load_target_function_from_json(self, function_name: str) -> Optional[Dict]:
        """
        Load target function from untested_functions.json by function name.
        
        Args:
            function_name: Name of the function to load
            
        Returns:
            Function data dictionary or None if not found
        """
        try:
            with open('data/untested/untested_functions.json', 'r', encoding='utf-8') as f:
                functions = json.load(f)
            
            for func in functions:
                if func.get('function_name', '').lower() == function_name.lower():
                    logger.info(f"✅ Found target function: {function_name}")
                    return func
            
            logger.warning(f"⚠️ Function '{function_name}' not found in untested_functions.json")
            return None
            
        except Exception as e:
            logger.error(f"❌ Error loading function from JSON: {e}")
            return None
    
    def get_reference_functions_rag(self, target_function: str, top_k: int = 3) -> List[Dict]:
        """
        Automatically get reference functions using RAG semantic search.
        
        Args:
            target_function: Target function code or name
            top_k: Number of reference functions to retrieve
            
        Returns:
            List of reference function dictionaries
        """
        logger.info(f"🔍 Using RAG to find top-{top_k} reference functions...")
        
        try:
            # Use RAG system to find similar functions with tests
            results = self.query_system.query(
                query=target_function,
                top_k=top_k,
                only_with_tests=True
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
                        'match_score': result.get('match_score', 0.0),
                        'match_type': result.get('match_type', 'unknown')
                    }
                    reference_functions.append(ref_dict)
                    logger.info(f"✅ Reference {i+1}: {result.get('function_name', 'Unknown')} "
                              f"(score: {ref_dict['match_score']:.3f})")
            
            logger.info(f"✅ Found {len(reference_functions)} reference functions via RAG")
            return reference_functions
            
        except Exception as e:
            logger.error(f"❌ Error getting reference functions via RAG: {e}")
            return []
    
    def generate_backbone_class(self, target_function: str) -> str:
        """
        Generate backbone class using prompt-3.
        
        Args:
            target_function: Target function code
            
        Returns:
            Generated backbone class code
        """
        logger.info("🏗️ Generating backbone class...")
        
        try:
            import openai
            
            # Set up OpenAI API key
            openai.api_key = ""
            
            # Create backbone prompt
            backbone_prompt = self.prompt_template_3.replace("{TARGET_FUNCTION}", target_function)
            
            # Generate backbone class
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": backbone_prompt}],
                temperature=0.3,
                max_tokens=1000
            )
            
            generated_backbone = response.choices[0].message.content.strip()
            logger.info("✅ Backbone class generated successfully")
            return generated_backbone
            
        except Exception as e:
            logger.error(f"❌ Error generating backbone class: {e}")
            raise
    
    def generate_test_case(self, target_function: str, reference_functions: List[Dict]) -> str:
        """
        Generate test case using prompt-1 (always use prompt-1).
        
        Args:
            target_function: Target function code
            reference_functions: List of reference function dictionaries
            
        Returns:
            Generated test case code
        """
        logger.info("🧪 Generating test case using prompt-1...")
        
        try:
            import openai
            
            # Set up OpenAI API key
            openai.api_key = ""
            
            # Process reference functions
            processed_references = process_reference_functions([
                f"{ref['function']}\n\nTestcase:\n{ref['test']}" 
                for ref in reference_functions
            ])
            
            if not processed_references:
                raise ValueError("No valid reference functions to process")
            
            # Build prompt with multiple references using prompt-1
            filled_prompt = build_multi_reference_prompt(
                target_function, 
                processed_references, 
                self.prompt_template_1
            )
            
            # Generate test case
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": filled_prompt}],
                temperature=0.3,
                max_tokens=2000
            )
            
            generated_testcase = response.choices[0].message.content.strip()
            logger.info("✅ Test case generated successfully using prompt-1")
            return generated_testcase
            
        except Exception as e:
            logger.error(f"❌ Error generating test case: {e}")
            raise
    
    def save_generated_files(self, backbone_class: str, test_case: str, 
                           target_function: str, output_dir: str = "generated_tests") -> Dict[str, str]:
        """
        Save generated backbone class and test case to files.
        
        Args:
            backbone_class: Generated backbone class code
            test_case: Generated test case code
            target_function: Target function code
            output_dir: Output directory
            
        Returns:
            Dictionary with file paths
        """
        try:
            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Extract function name for filenames
            function_name = extract_function_name(target_function)
            if not function_name:
                function_name = "Generated"
            
            # Save backbone class
            backbone_file = output_path / f"{function_name}Backbone.cs"
            with open(backbone_file, 'w', encoding='utf-8') as f:
                f.write(backbone_class)
            
            # Save test case
            test_file = output_path / f"{function_name}Test.cs"
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(test_case)
            
            file_paths = {
                'backbone_file': str(backbone_file),
                'test_file': str(test_file)
            }
            
            logger.info(f"✅ Files saved:")
            logger.info(f"   Backbone: {backbone_file}")
            logger.info(f"   Test: {test_file}")
            
            return file_paths
            
        except Exception as e:
            logger.error(f"❌ Error saving files: {e}")
            raise
    
    def run_automated_pipeline(self, target_function: str, top_k: int = 3, 
                             output_dir: str = "generated_tests") -> Dict:
        """
        Run the complete automated test generation pipeline.
        
        Args:
            target_function: Target function code or name
            top_k: Number of reference functions to retrieve
            output_dir: Output directory for generated files
            
        Returns:
            Dictionary with results and metadata
        """
        logger.info("🚀 Starting automated test generation pipeline...")
        
        try:
            # Step 1: Get reference functions using RAG
            logger.info("Step 1: Finding reference functions using RAG...")
            reference_functions = self.get_reference_functions_rag(target_function, top_k)
            
            if not reference_functions:
                raise ValueError("No reference functions found via RAG. Cannot generate test case.")
            
            # Step 2: Generate backbone class
            logger.info("Step 2: Generating backbone class...")
            backbone_class = self.generate_backbone_class(target_function)
            
            # Step 3: Generate test case using prompt-1
            logger.info("Step 3: Generating test case using prompt-1...")
            test_case = self.generate_test_case(target_function, reference_functions)
            
            # Step 4: Save generated files
            logger.info("Step 4: Saving generated files...")
            file_paths = self.save_generated_files(backbone_class, test_case, target_function, output_dir)
            
            # Prepare results
            results = {
                'success': True,
                'target_function': target_function,
                'reference_count': len(reference_functions),
                'backbone_class': backbone_class,
                'test_case': test_case,
                'file_paths': file_paths,
                'references': reference_functions
            }
            
            logger.info("🎉 Automated test generation completed successfully!")
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
    parser = argparse.ArgumentParser(description='Automated Test Case Generation with RAG')
    parser.add_argument('--target', '-t', type=str, required=True,
                       help='Target function code, name, or function name from JSON')
    parser.add_argument('--top-k', '-k', type=int, default=3,
                       help='Number of reference functions to retrieve (default: 3)')
    parser.add_argument('--output-dir', '-o', type=str, default='generated_tests',
                       help='Output directory for generated files')
    parser.add_argument('--embeddings-path', default='data/embeddings',
                       help='Path to embeddings directory')
    parser.add_argument('--from-json', action='store_true',
                       help='Load target function from untested_functions.json by name')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize generator
        generator = AutomatedTestGenerator(embeddings_path=args.embeddings_path)
        
        # Load target function
        if args.from_json:
            # Load from JSON file
            func_data = generator.load_target_function_from_json(args.target)
            if not func_data:
                print(f"❌ Function '{args.target}' not found in untested_functions.json")
                sys.exit(1)
            target_function = func_data['function_source']
            print(f"✅ Loaded function '{args.target}' from JSON")
        else:
            target_function = args.target
        
        # Run automated pipeline
        results = generator.run_automated_pipeline(
            target_function=target_function,
            top_k=args.top_k,
            output_dir=args.output_dir
        )
        
        if results['success']:
            print("\n" + "="*60)
            print("🎉 AUTOMATED TEST GENERATION SUCCESSFUL!")
            print("="*60)
            print(f"📁 Backbone file: {results['file_paths']['backbone_file']}")
            print(f"📁 Test file: {results['file_paths']['test_file']}")
            print(f"📊 Reference functions used: {results['reference_count']}")
            print("\n🏗️ Generated Backbone Class:")
            print("-"*40)
            print(results['backbone_class'])
            print("-"*40)
            print("\n🧪 Generated Test Case:")
            print("-"*40)
            print(results['test_case'])
            print("-"*40)
        else:
            print(f"\n❌ AUTOMATED TEST GENERATION FAILED: {results['error']}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
