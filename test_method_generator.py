import openai
import re
from pathlib import Path
from tree_sitter_extractor import TreeSitterExtractor

openai.api_key = "" 

# Load prompt templates
with open("prompt-1.txt", "r") as f:
    prompt_template_1 = f.read()

with open("prompt-2.txt", "r") as f:
    prompt_template_2 = f.read()

with open("prompt-3.txt", "r") as f:
    prompt_template_3 = f.read()

def check_existing_test_file(target_function: str) -> bool:
    """Check if the target function already has a test file."""
    try:
        extractor = TreeSitterExtractor()
        function_name = extract_function_name(target_function)
        if not function_name:
            return False
        
        repos_path = Path("data/repos")
        if not repos_path.exists():
            return False
        
        for repo_path in repos_path.iterdir():
            if not repo_path.is_dir() or repo_path.name.startswith('.'):
                continue
                
            for file_path in repo_path.rglob("*.cs"):
                if extractor.is_test_file(file_path) and has_test_for_function(function_name, file_path, extractor):
                    return True
                    
    except Exception:
        pass
    
    return False

def extract_function_name(function_code: str) -> str:
    """Extract function name from function code."""
    patterns = [
        r'public\s+\w+\s+(\w+)\s*\(',
        r'private\s+\w+\s+(\w+)\s*\(',
        r'protected\s+\w+\s+(\w+)\s*\(',
        r'void\s+(\w+)\s*\(',
        r'(\w+)\s*\([^)]*\)\s*{',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, function_code)
        if match:
            return match.group(1)
    return ""

def has_test_for_function(function_name: str, test_file: Path, extractor: TreeSitterExtractor) -> bool:
    """Check if a test file contains a test for the given function."""
    try:
        tree = extractor.parse_csharp_file(test_file)
        if not tree:
            return False
        
        with open(test_file, 'r', encoding='utf-8') as f:
            source = f.read()
        
        test_methods = extractor.extract_test_methods_from_tree(tree, source)
        
        for test_method in test_methods:
            test_method_name = test_method['name'].lower()
            func_name_lower = function_name.lower()
            if (func_name_lower in test_method_name or 
                test_method_name.startswith('test') or
                test_method_name.endswith('test') or
                test_method_name.startswith(func_name_lower) or
                test_method_name.endswith(func_name_lower)):
                return True
    except Exception:
        pass
    
    return False

def get_user_input():
    """Get function inputs from user."""
    print("Function Test Case Generator")
    print("=" * 40)
    
    print("Enter the target function (end with blank line):")
    target_function_lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        target_function_lines.append(line)
    
    print("\nHow many reference functions do you want to provide? (1-5)")
    while True:
        try:
            num_references = int(input("Number of references: "))
            if 1 <= num_references <= 5:
                break
            else:
                print("Please enter a number between 1 and 5")
        except ValueError:
            print("Please enter a valid number")
    
    reference_functions = []
    for i in range(num_references):
        print(f"\n--- Reference Function {i+1} ---")
        print("Enter the reference function with test case (end with blank line):")
        print("Format: Function code, then 'Testcase:', then test case")
        reference_lines = []
        while True:
            line = input()
            if line.strip() == "":
                break
            reference_lines.append(line)
        reference_functions.append("\n".join(reference_lines))
    
    return "\n".join(target_function_lines), reference_functions

def process_reference_functions(reference_functions):
    """Process multiple reference functions and combine them."""
    processed_references = []
    
    for ref_func in reference_functions:
        if not ref_func.strip():
            continue
            
        parts = ref_func.split("Testcase:")
        if len(parts) != 2:
            print(f"Warning: Skipping reference function without 'Testcase:' separator")
            continue
        
        function_code = parts[0].strip()
        test_code = parts[1].strip()
        
        processed_references.append({
            'function': function_code,
            'test': test_code
        })
    
    return processed_references

def build_multi_reference_prompt(target_function, reference_functions, prompt_template):
    """Build prompt with multiple reference functions."""
    # Combine all reference functions and tests
    all_reference_functions = []
    all_reference_tests = []
    
    for ref in reference_functions:
        all_reference_functions.append(ref['function'])
        all_reference_tests.append(ref['test'])
    
    # Create combined reference text
    combined_reference_function = "\n\n--- REFERENCE FUNCTION ---\n".join(all_reference_functions)
    combined_reference_test = "\n\n--- REFERENCE TEST ---\n".join(all_reference_tests)
    
    # Fill the prompt template
    filled_prompt = prompt_template.replace("{TARGET_FUNCTION}", target_function)
    filled_prompt = filled_prompt.replace("{REFERENCE_FUNCTION}", combined_reference_function)
    filled_prompt = filled_prompt.replace("{REFERENCE_TEST_CASE}", combined_reference_test)
    
    return filled_prompt

def main():
    """Main function to run the test case generator."""
    target_function, reference_functions = get_user_input()
    
    if not target_function.strip() or not reference_functions:
        print("Error: Both target function and at least one reference function are required.")
        return

    # Process reference functions
    processed_references = process_reference_functions(reference_functions)
    if not processed_references:
        print("Error: No valid reference functions provided.")
        return
    
    print(f"\n✅ Processing {len(processed_references)} reference function(s)...")
    
    # Step 1: Generate backbone class
    print("\n🏗️ Step 1: Generating backbone class...")
    backbone_prompt = prompt_template_3.replace("{TARGET_FUNCTION}", target_function)
    
    backbone_response = openai.chat.completions.create(
        model="gpt-4o-mini", 
        messages=[
            {"role": "user", "content": backbone_prompt}
        ],
        temperature=0.3,
        max_tokens=1000
    )
    
    generated_backbone = backbone_response.choices[0].message.content.strip()
    print("Generated Backbone Class:")
    print("=" * 50)
    print(generated_backbone)
    
    # Step 2: Generate test case using prompt-1 (always use prompt-1)
    print("\n🧪 Step 2: Generating test case...")
    print("Using prompt-1.txt for test generation...")
    
    # Build prompt with multiple references
    filled_prompt = build_multi_reference_prompt(target_function, processed_references, prompt_template_1)
    
    print(f"\n🤖 Generating test method with {len(processed_references)} reference(s)...")
    response = openai.chat.completions.create(
        model="gpt-4o-mini", 
        messages=[
            {"role": "user", "content": filled_prompt}
        ],
        temperature=0.3,
        max_tokens=2000  # Increased for multiple references
    )
    
    generated_testcase = response.choices[0].message.content.strip()
    print(f"\nGenerated Test Method:")
    print("=" * 50)
    print(generated_testcase)
    
    print("\n📋 Next Steps:")
    print("1. Save the backbone class as a new .cs file in your project")
    print("2. Save the test method in your test directory")
    print("3. Make sure your project has NUnit testing framework installed")
    print("4. Add the necessary Unity Test Framework packages if using Unity")
    print("5. Run the test to verify it works correctly")

if __name__ == "__main__":
    main()
