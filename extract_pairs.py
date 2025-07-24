import ast
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

class FunctionTestExtractor:

    # FunctionTestExtractor class to extract function-test pairs from Python files.
    # It scans through Python files in specified repositories, identifies functions and their corresponding test cases,
    # and saves the pairs in a structured format.

    def __init__(self):
        """Initialize the function-test pair extractor."""
        self.repos_dir = Path("data/repos")
        self.pairs_dir = Path("data/pairs")
        self.pairs_dir.mkdir(parents=True, exist_ok=True)

    # Check if a file is a test file based on naming conventions.
    # This method checks if the file name starts with 'test_' or ends with '_test.py',
    # or contains 'test' in its name (case insensitive).
    def is_test_file(self, file_path: Path) -> bool:
        return any([
            file_path.name.startswith('test_'),
            file_path.name.endswith('_test.py'),
            'test' in file_path.name.lower()
        ])

    # Get the source code of a function from its AST node.
    # This method uses the ast module to extract the source code segment of a function definition.
    # It takes an AST node representing a function and the original source code as input,
    # and returns the source code of the function as a string.
    def get_function_source(self, node: ast.FunctionDef, source: str) -> str:
        return ast.get_source_segment(source, node)

    # Find a test function that corresponds to a given function name.
    # This method searches through a list of test files to find a test function
    # that matches the name of a given function. It checks both the function name
    # and the docstring of the test function to see if it references the function.
    # If a matching test function is found, it returns the source code of that test function.
    # If no matching test function is found, it returns None.
    def find_test_for_function(self, func_name: str, test_files: List[Path]) -> Optional[str]:
        """Find a test function that corresponds to the given function name."""
        for test_file in test_files:
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    source = f.read()
                    tree = ast.parse(source)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            # Check if test function name or docstring references the function
                            if (func_name in node.name or 
                                (node.body and isinstance(node.body[0], ast.Expr) and 
                                 isinstance(node.body[0].value, ast.Str) and 
                                 func_name in node.body[0].value.s)):
                                return self.get_function_source(node, source)
            except Exception as e:
                print(f"Error processing test file {test_file}: {str(e)}")
                continue
        return None

    # Extract function-test pairs from a source file.
    # This method reads a Python source file, parses it into an AST,    
    # and walks through the AST to find function definitions.
    # For each function, it retrieves the source code and looks for a corresponding test function
    # in the provided list of test files. If a matching test function is found,
    # it creates a dictionary containing the function name, its source code, the test source code
    def extract_pairs_from_file(self, file_path: Path, test_files: List[Path]) -> List[Dict]:
        """Extract function-test pairs from a source file."""
        pairs = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
                tree = ast.parse(source)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        func_source = self.get_function_source(node, source)
                        test_source = self.find_test_for_function(node.name, test_files)
                        
                        if test_source:
                            pairs.append({
                                'function_name': node.name,
                                'function_source': func_source,
                                'test_source': test_source,
                                'source_file': str(file_path)
                            })
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
        
        return pairs

    # Extract function-test pairs from all Python files in a repository.
    # This method processes a single repository, collects all Python files,
    # and extracts function-test pairs from each source file.
    # It uses the `tqdm` library to show a progress bar for the processing of each repository.
    def process_repository(self, repo_path: Path) -> List[Dict]:
        """Process a single repository to extract function-test pairs."""
        all_pairs = []
        source_files = []
        test_files = []
        
        # Collect all Python files
        for file_path in repo_path.rglob("*.py"):
            if self.is_test_file(file_path):
                test_files.append(file_path)
            else:
                source_files.append(file_path)
        
        # Process each source file
        for source_file in tqdm(source_files, desc=f"Processing {repo_path.name}"):
            pairs = self.extract_pairs_from_file(source_file, test_files)
            all_pairs.extend(pairs)
        
        return all_pairs

    # Extract function-test pairs from all repositories.
    # This method iterates through all repositories in the specified directory,
    # processes each repository to extract function-test pairs,
    # and saves the results in a JSON file.
    def extract_all(self):
        """Extract function-test pairs from all repositories."""
        all_pairs = []
        
        # Process each repository
        for repo_path in self.repos_dir.iterdir():
            if repo_path.is_dir() and not repo_path.name.startswith('.'):
                print(f"\nProcessing repository: {repo_path.name}")
                pairs = self.process_repository(repo_path)
                all_pairs.extend(pairs)
        
        # Save all pairs
        output_path = self.pairs_dir / "function_test_pairs.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_pairs, f, indent=2)
        
        print(f"\nExtracted {len(all_pairs)} function-test pairs.")
        print(f"Results saved to {output_path}")

if __name__ == "__main__":
    extractor = FunctionTestExtractor()
    extractor.extract_all() 