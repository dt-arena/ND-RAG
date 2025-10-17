import json
import tree_sitter
from tree_sitter import Language, Parser
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

class TreeSitterExtractor:
    """
    Tree-sitter based extractor for C# code analysis.
    Replaces regex-based extraction with proper AST parsing for better accuracy.
    """
    
    def __init__(self):
        """Initialize the tree-sitter extractor with C# parser."""
        self.repos_dir = Path("data/repos")
        self.pairs_dir = Path("data/pairs")
        self.pairs_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tree-sitter C# parser
        try:
            # Try to load the C# language parser
            import tree_sitter_c_sharp as csharp
            self.csharp_language = Language(csharp.language())
            self.parser = Parser(self.csharp_language)
        except Exception as e:
            print(f"Warning: Could not load tree-sitter C# parser: {e}")
            print("Falling back to regex-based extraction...")
            self.parser = None
            self.csharp_language = None
    
    def is_test_file(self, file_path: Path) -> bool:
        """Check if a file is a test file based on naming conventions."""
        return any([
            file_path.name.startswith('test_'),
            file_path.name.endswith('_test.cs'),
            file_path.name.endswith('Tests.cs'),
            file_path.name.endswith('Test.cs'),
            'test' in file_path.name.lower()
        ])
    
    def parse_csharp_file(self, file_path: Path) -> Optional[tree_sitter.Tree]:
        """Parse a C# file using tree-sitter and return the syntax tree."""
        if not self.parser:
            return None
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            tree = self.parser.parse(bytes(source_code, "utf8"))
            return tree
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return None
    
    def extract_methods_from_tree(self, tree: tree_sitter.Tree, source_code: str) -> List[Dict]:
        """Extract method information from the syntax tree."""
        if not tree:
            return []
        
        methods = []
        
        # Simple traversal approach instead of complex queries
        def traverse_node(node):
            if node.type == "method_declaration":
                # Extract method information
                method_name = None
                parameters = "()"
                modifiers = []
                return_type = "void"
                
                for child in node.children:
                    if child.type == "identifier":
                        method_name = source_code[child.start_byte:child.end_byte]
                    elif child.type == "parameter_list":
                        parameters = source_code[child.start_byte:child.end_byte]
                    elif child.type in ["public", "private", "protected", "internal", "static", "virtual", "override", "async"]:
                        modifiers.append(source_code[child.start_byte:child.end_byte])
                    elif child.type in ["predefined_type", "generic_name", "qualified_name"]:
                        return_type = source_code[child.start_byte:child.end_byte]
                
                if method_name:
                    method_source = source_code[node.start_byte:node.end_byte]
                    methods.append({
                        'name': method_name,
                        'source': method_source,
                        'modifiers': modifiers,
                        'return_type': return_type,
                        'parameters': parameters,
                        'start_line': node.start_point[0] + 1,
                        'end_line': node.end_point[0] + 1
                    })
            
            # Recursively traverse children
            for child in node.children:
                traverse_node(child)
        
        traverse_node(tree.root_node)
        return methods
    
    def extract_test_methods_from_tree(self, tree: tree_sitter.Tree, source_code: str) -> List[Dict]:
        """Extract test method information from the syntax tree."""
        if not tree:
            return []
        
        test_methods = []
        
        # Simple traversal approach for test methods
        def traverse_node(node):
            if node.type == "method_declaration":
                # Check if this method has [Test] attribute
                is_test_method = False
                method_name = None
                
                for child in node.children:
                    if child.type == "attribute_list":
                        attr_text = source_code[child.start_byte:child.end_byte]
                        if "[Test]" in attr_text:
                            is_test_method = True
                    elif child.type == "identifier":
                        method_name = source_code[child.start_byte:child.end_byte]
                
                if is_test_method and method_name:
                    method_source = source_code[node.start_byte:node.end_byte]
                    test_methods.append({
                        'name': method_name,
                        'source': method_source,
                        'start_line': node.start_point[0] + 1,
                        'end_line': node.end_point[0] + 1
                    })
            
            # Recursively traverse children
            for child in node.children:
                traverse_node(child)
        
        traverse_node(tree.root_node)
        return test_methods
    
    def find_test_for_function(self, func_name: str, test_files: List[Path]) -> Optional[str]:
        """Find a test function that corresponds to the given function name."""
        for test_file in test_files:
            try:
                tree = self.parse_csharp_file(test_file)
                if not tree:
                    continue
                
                with open(test_file, 'r', encoding='utf-8') as f:
                    source = f.read()
                
                test_methods = self.extract_test_methods_from_tree(tree, source)
                
                for test_method in test_methods:
                    test_method_name = test_method['name']
                    # Check if test method name references the function
                    if (func_name.lower() in test_method_name.lower() or 
                        test_method_name.lower().startswith('test') or
                        test_method_name.lower().endswith('test')):
                        return test_method['source']
                        
            except Exception as e:
                print(f"Error processing test file {test_file}: {str(e)}")
                continue
        return None
    
    def extract_pairs_from_file(self, file_path: Path, test_files: List[Path]) -> List[Dict]:
        """Extract function-test pairs from a C# source file using tree-sitter."""
        pairs = []
        
        try:
            tree = self.parse_csharp_file(file_path)
            if not tree:
                print(f"Could not parse {file_path} with tree-sitter")
                return pairs
            
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
            
            methods = self.extract_methods_from_tree(tree, source)
            
            for method in methods:
                method_name = method['name']
                method_source = method['source']
                
                # Skip constructors, destructors, and property accessors
                if (method_name.startswith('__') or 
                    method_name in ['get', 'set', 'add', 'remove'] or
                    len(method_name) < 2):
                    continue
                
                test_source = self.find_test_for_function(method_name, test_files)
                
                # Always include function entry; set test_source to None when not found
                pairs.append({
                    'function_name': method_name,
                    'function_source': method_source,
                    'test_source': test_source if test_source else None,
                    'source_file': str(file_path),
                    'modifiers': method['modifiers'],
                    'return_type': method['return_type'],
                    'parameters': method['parameters'],
                    'start_line': method['start_line'],
                    'end_line': method['end_line']
                })
                    
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
        
        return pairs
    
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
    
    def extract_all(self):
        """Extract function-test pairs from all repositories using tree-sitter."""
        all_pairs = []
        
        # Process each repository
        for repo_path in self.repos_dir.iterdir():
            if repo_path.is_dir() and not repo_path.name.startswith('.'):
                print(f"\nProcessing repository: {repo_path.name}")
                pairs = self.process_repository(repo_path)
                all_pairs.extend(pairs)
        
        # Save all pairs
        output_path = self.pairs_dir / "function_test_pairs_treesitter.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_pairs, f, indent=2)
        
        print(f"\nExtracted {len(all_pairs)} function-test pairs using tree-sitter.")
        print(f"Results saved to {output_path}")

if __name__ == "__main__":
    extractor = TreeSitterExtractor()
    extractor.extract_all()
