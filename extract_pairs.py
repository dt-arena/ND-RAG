import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from tree_sitter_extractor import TreeSitterExtractor

class FunctionTestExtractor:
    """
    FunctionTestExtractor class to extract function-test pairs from C# files using tree-sitter.
    It scans through C# files in specified repositories, identifies functions and their corresponding test cases,
    and saves the pairs in a structured format.
    """

    def __init__(self):
        """Initialize the function-test pair extractor with tree-sitter only."""
        self.repos_dir = Path("data/repos")
        self.pairs_dir = Path("data/pairs")
        self.pairs_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tree-sitter extractor
        try:
            self.tree_sitter_extractor = TreeSitterExtractor()
            print("Tree-sitter extractor initialized successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize tree-sitter extractor: {e}. Tree-sitter is required.")

    # Check if a file is a test file based on naming conventions.
    # This method checks if the file name starts with 'test_' or ends with '_test.cs',
    # or contains 'test' in its name (case insensitive).
    def is_test_file(self, file_path: Path) -> bool:
        return any([
            file_path.name.startswith('test_'),
            file_path.name.endswith('_test.cs'),
            file_path.name.endswith('Tests.cs'),
            file_path.name.endswith('Test.cs'),
            'test' in file_path.name.lower()
        ])


    def extract_pairs_from_file(self, file_path: Path, test_files: List[Path]) -> List[Dict]:
        """Extract function-test pairs from a C# source file using tree-sitter."""
        return self.tree_sitter_extractor.extract_pairs_from_file(file_path, test_files)
    

    # Extract function-test pairs from all C# files in a repository.
    def process_repository(self, repo_path: Path) -> List[Dict]:
        """Process a single repository to extract function-test pairs."""
        all_pairs = []
        source_files = []
        test_files = []
        
        # Collect all C# files
        for file_path in repo_path.rglob("*.cs"):
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
        output_filename = "function_test_pairs_treesitter.json"
        output_path = self.pairs_dir / output_filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_pairs, f, indent=2)
        
        print(f"\nExtracted {len(all_pairs)} function-test pairs using tree-sitter.")
        print(f"Results saved to {output_path}")

if __name__ == "__main__":
    extractor = FunctionTestExtractor()
    extractor.extract_all() 