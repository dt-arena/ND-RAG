import openai
import os

openai.api_key = ""

# Load prompt template
with open("prompt-1.txt", "r") as f:
    prompt_template = f.read()

def get_user_input():
    """Get function inputs from user."""
    print("Function Test Case Generator")
    print("=" * 40)
    
    # Get target function
    print("Enter the target function (the one that needs a test case):")
    print("(You can paste multiple lines, end with a blank line)")
    target_function_lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        target_function_lines.append(line)
    
    target_function = "\n".join(target_function_lines)
    
    # Get reference function with test case
    print("\nEnter the reference function with its test case:")
    print("(Format: Function code, then 'Testcase:', then test case)")
    print("(End with a blank line)")
    reference_lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        reference_lines.append(line)
    
    reference_with_testcase = "\n".join(reference_lines)
    
    return target_function, reference_with_testcase

def main():
    """Main function to run the test case generator."""
    # Get user input
    target_function, reference_with_testcase = get_user_input()
    
    if not target_function.strip() or not reference_with_testcase.strip():
        print("Error: Both target function and reference function are required.")
        return

    # Parse reference function and test case
    reference_parts = reference_with_testcase.split("Testcase:")
    if len(reference_parts) != 2:
        print("Error: Reference function must include 'Testcase:' separator.")
        return
    
    reference_function = reference_parts[0].strip()
    reference_testcase = reference_parts[1].strip()
    
    print(f"\nUsing prompt-1.txt to generate test method")
    
    # Fill the prompt template
    filled_prompt = prompt_template.replace("{TARGET_FUNCTION}", target_function)
    filled_prompt = filled_prompt.replace("{REFERENCE_FUNCTION}", reference_function)
    filled_prompt = filled_prompt.replace("{REFERENCE_TEST_CASE}", reference_testcase)
    
    # Generate test case using LLM
    print("\nGenerating test case...")
    response = openai.chat.completions.create(
        model="gpt-4o-mini", 
        messages=[
            {"role": "user", "content": filled_prompt}
        ],
        temperature=0.3,
        max_tokens=1500
    )
    
    generated_testcase = response.choices[0].message.content.strip()
    print("\nGenerated Test Case:")
    print("=" * 50)
    print(generated_testcase)

if __name__ == "__main__":
    main()
