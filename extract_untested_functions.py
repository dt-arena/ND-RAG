import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from tree_sitter_extractor import TreeSitterExtractor

class UntestedFunctionExtractor:
    """
    UntestedFunctionExtractor class to extract functions that don't have corresponding test case methods.
    It scans through C# files in specified repositories, identifies functions and checks if they have 
    corresponding test cases, then saves the untested functions in a structured format.
    """

    def __init__(self):
        """Initialize the untested function extractor with tree-sitter."""
        self.repos_dir = Path("data/repos")
        self.untested_dir = Path("data/untested")
        self.untested_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tree-sitter extractor
        try:
            self.tree_sitter_extractor = TreeSitterExtractor()
            print("Tree-sitter extractor initialized successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize tree-sitter extractor: {e}. Tree-sitter is required.")

    def is_test_file(self, file_path: Path) -> bool:
        """Check if a file is a test file based on naming conventions."""
        return any([
            file_path.name.startswith('test_'),
            file_path.name.endswith('_test.cs'),
            file_path.name.endswith('Tests.cs'),
            file_path.name.endswith('Test.cs'),
            'test' in file_path.name.lower()
        ])

    def has_corresponding_test(self, func_name: str, test_files: List[Path]) -> bool:
        """
        Check if a function has a corresponding test method.
        Returns True if a test is found, False otherwise.
        """
        for test_file in test_files:
            try:
                tree = self.tree_sitter_extractor.parse_csharp_file(test_file)
                if not tree:
                    continue
                
                with open(test_file, 'r', encoding='utf-8') as f:
                    source = f.read()
                
                test_methods = self.tree_sitter_extractor.extract_test_methods_from_tree(tree, source)
                
                for test_method in test_methods:
                    test_method_name = test_method['name']
                    # Check if test method name references the function
                    if (func_name.lower() in test_method_name.lower() or 
                        test_method_name.lower().startswith('test') or
                        test_method_name.lower().endswith('test') or
                        test_method_name.lower().startswith(func_name.lower()) or
                        test_method_name.lower().endswith(func_name.lower())):
                        return True
                        
            except Exception as e:
                print(f"Error processing test file {test_file}: {str(e)}")
                continue
        return False

    def extract_untested_functions_from_file(self, file_path: Path, test_files: List[Path]) -> List[Dict]:
        """Extract functions that don't have corresponding test methods from a C# source file."""
        untested_functions = []
        
        try:
            tree = self.tree_sitter_extractor.parse_csharp_file(file_path)
            if not tree:
                print(f"Could not parse {file_path} with tree-sitter")
                return untested_functions
            
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
            
            methods = self.tree_sitter_extractor.extract_methods_from_tree(tree, source)
            
            for method in methods:
                method_name = method['name']
                method_source = method['source']
                
                # Skip constructors, destructors, and property accessors
                if (method_name.startswith('__') or 
                    method_name in ['get', 'set', 'add', 'remove'] or
                    len(method_name) < 2):
                    continue
                
                # Check if this function has a corresponding test
                has_test = self.has_corresponding_test(method_name, test_files)
                
                # Only include functions that don't have tests
                if not has_test:
                    untested_functions.append({
                        'function_name': method_name,
                        'function_source': method_source,
                        'source_file': str(file_path),
                        'modifiers': method['modifiers'],
                        'return_type': method['return_type'],
                        'parameters': method['parameters'],
                        'start_line': method['start_line'],
                        'end_line': method['end_line'],
                        'has_test': False,
                        'test_status': 'untested'
                    })
                    
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
        
        return untested_functions

    def process_repository(self, repo_path: Path) -> List[Dict]:
        """Process a single repository to extract untested functions."""
        all_untested_functions = []
        source_files = []
        test_files = []
        
        # Collect all C# files
        for file_path in repo_path.rglob("*.cs"):
            if self.is_test_file(file_path):
                test_files.append(file_path)
            else:
                source_files.append(file_path)
        
        print(f"Found {len(source_files)} source files and {len(test_files)} test files in {repo_path.name}")
        
        # Process each source file
        for source_file in tqdm(source_files, desc=f"Processing {repo_path.name}"):
            untested_functions = self.extract_untested_functions_from_file(source_file, test_files)
            all_untested_functions.extend(untested_functions)
        
        return all_untested_functions

    def extract_all_untested_functions(self):
        """Extract all untested functions from all repositories."""
        all_untested_functions = []
        
        # Process each repository
        for repo_path in self.repos_dir.iterdir():
            if repo_path.is_dir() and not repo_path.name.startswith('.'):
                print(f"\nProcessing repository: {repo_path.name}")
                untested_functions = self.process_repository(repo_path)
                all_untested_functions.extend(untested_functions)
                print(f"Found {len(untested_functions)} untested functions in {repo_path.name}")
        
        # Save all untested functions
        output_filename = "untested_functions.json"
        output_path = self.untested_dir / output_filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_untested_functions, f, indent=2)
        
        print(f"\nExtracted {len(all_untested_functions)} untested functions.")
        print(f"Results saved to {output_path}")
        
        # Generate summary statistics
        self.generate_summary_statistics(all_untested_functions)

    def generate_summary_statistics(self, untested_functions: List[Dict]):
        """Generate and save summary statistics about untested functions."""
        if not untested_functions:
            print("No untested functions found.")
            return
        
        # Group by repository
        repo_stats = {}
        for func in untested_functions:
            repo_name = Path(func['source_file']).parts[-3] if len(Path(func['source_file']).parts) > 2 else "unknown"
            if repo_name not in repo_stats:
                repo_stats[repo_name] = 0
            repo_stats[repo_name] += 1
        
        # Group by return type
        return_type_stats = {}
        for func in untested_functions:
            return_type = func.get('return_type', 'unknown')
            if return_type not in return_type_stats:
                return_type_stats[return_type] = 0
            return_type_stats[return_type] += 1
        
        # Group by modifiers
        modifier_stats = {}
        for func in untested_functions:
            modifiers = func.get('modifiers', [])
            modifier_key = ', '.join(modifiers) if modifiers else 'none'
            if modifier_key not in modifier_stats:
                modifier_stats[modifier_key] = 0
            modifier_stats[modifier_key] += 1
        
        # Create summary
        summary = {
            'total_untested_functions': len(untested_functions),
            'repositories_processed': len(repo_stats),
            'functions_by_repository': repo_stats,
            'functions_by_return_type': return_type_stats,
            'functions_by_modifiers': modifier_stats,
            'top_untested_repositories': sorted(repo_stats.items(), key=lambda x: x[1], reverse=True)[:10]
        }
        
        # Save summary
        summary_path = self.untested_dir / "untested_functions_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nSummary statistics saved to {summary_path}")
        print(f"Total untested functions: {summary['total_untested_functions']}")
        print(f"Repositories processed: {summary['repositories_processed']}")
        print("\nTop repositories with untested functions:")
        for repo, count in summary['top_untested_repositories'][:5]:
            print(f"  {repo}: {count} untested functions")

    def extract_untested_functions_from_single_repo(self, repo_name: str):
        """Extract untested functions from a single repository."""
        repo_path = self.repos_dir / repo_name
        if not repo_path.exists():
            print(f"Repository {repo_name} not found in {self.repos_dir}")
            return
        
        print(f"Processing single repository: {repo_name}")
        untested_functions = self.process_repository(repo_path)
        
        # Save results for this repository
        output_filename = f"untested_functions_{repo_name}.json"
        output_path = self.untested_dir / output_filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(untested_functions, f, indent=2)
        
        print(f"Found {len(untested_functions)} untested functions in {repo_name}")
        print(f"Results saved to {output_path}")
        
        return untested_functions

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract functions without corresponding test methods')
    parser.add_argument('--repo', type=str, help='Process a single repository by name')
    parser.add_argument('--all', action='store_true', help='Process all repositories')
    
    args = parser.parse_args()
    
    extractor = UntestedFunctionExtractor()
    
    if args.repo:
        extractor.extract_untested_functions_from_single_repo(args.repo)
    elif args.all:
        extractor.extract_all_untested_functions()
    else:
        print("Please specify --repo <repo_name> or --all")
        print("Example: python extract_untested_functions.py --all")
        print("Example: python extract_untested_functions.py --repo my-repo")
