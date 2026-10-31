import re

def extract_func_regex(generated_text, func_name):
    '''Extract function using regex - from def name(:) to end of useful code.'''
    # Pattern: def func_name(params): followed by anything until next top-level def or end
    pattern = rf'def {func_name}\([^)]*\):.*'
    match = re.search(pattern, generated_text, re.DOTALL)
    if match:
        # Now extend to capture the body - look for the end
        start = match.start()
        func_def = match.group()
        # Try to find where the function body ends
        # Look for '## -*-' or a top-level 'def ' after this
        remaining = generated_text[start:]
        # Find next top-level def
        next_def = re.search(r'^\n\s*def \w+', remaining, re.MULTILINE)
        if next_def:
            # Truncate before the next def
            func_def = remaining[:next_def.start()].rstrip()
        else:
            # Also truncate at ## -*- marker
            marker = remaining.find('## -*-')
            if marker >= 0:
                func_def = remaining[:marker].rstrip()
        return func_def.strip()
    return None

# Test with fibonacci
fib_text = 'def fibonacci(n):\n        return stdin.get("0")\n\n## -*- coding: utf-13\n\n#     else:\n#             "success": "'
result = extract_func_regex(fib_text, 'fibonacci')
print(f'Fibonacci result: {repr(result)}')

# Test with quicksort
qs_text = 'def quicksort(arr):\n\t\t\t\treturn 0;\n\t\t\t}\n\t\t}\n\n\t\tif (array == null)\n\t\t{\n\t\t\tif (array == null)'
result = extract_func_regex(qs_text, 'quicksort')
print(f'Quicksort result: {repr(result)}')