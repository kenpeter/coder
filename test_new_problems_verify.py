#!/usr/bin/env /home/kenpeter/work/coder/.venv/bin/python3
"""Verify Max Subarray and Valid Parentheses functions from model."""
import subprocess
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained('bigcode/starcoder2-7b', clean_up_tokenization_spaces=False)

# Max Subarray test
max_subarray_prompt = "def max_subarray(nums):\n    if not nums:\n        return 0\n    max_current = max_global = nums[0]\n    for num in nums[1:]:\n        max_current = max(num, max_current + num)\n        max_global = max(max_global, max_current)\n    return max_global"

result = subprocess.run(
    ["/usr/bin/python3", "/home/kenpeter/work/coder/infer.py",
     "--prompt", max_subarray_prompt,
     "--max-new-tokens", "32",
     "--temperature", "0.2",
     "--top-p", "0.95",
     "--device", "cuda"],
    capture_output=True, text=True, timeout=120
)
output = result.stdout + result.stderr
sep = "=" * 70
idx = output.find(sep)
if idx == -1:
    lines = output.split("\n")
    for line in lines:
        if "=" * 70 in line:
            sep = line
            break
    idx = output.find(sep) if sep else -1
after_sep = output[idx + len(sep):].strip() if idx >= 0 else None

print("Max Subarray raw output after separator:")
print(after_sep[:500] if after_sep else "None")

# Extract function
if after_sep:
    idx2 = after_sep.find("def max_subarray")
    if idx2 >= 0:
        remaining = after_sep[idx2:]
        lines = remaining.split("\n")
        func_lines = []
        for i, line in enumerate(lines):
            if i > 0 and line.strip() == "" and func_lines and not func_lines[-1].strip().startswith("def "):
                break
            if line.strip().startswith("def ") and i > 0:
                prev_non_empty = [l.strip() for l in lines[:i] if l.strip()]
                if prev_non_empty and not all(l == "" for l in lines[i-2:i]):
                    break
            func_lines.append(line)
        if func_lines:
            extracted = "\n".join(func_lines).rstrip()
            print("\nExtracted function:")
            print(extracted)
            
            ns = {}
            exec(extracted, ns)
            fn = ns["max_subarray"]
            
            test_cases = [
                ([-2,1,-3,4,-1,2,1,-5,3], 6),
                ([1], 1),
                ([5,4,-1,7,8], 23),
                ([-1,-2,-3], -1),
                ([2,-1,2], 3),
            ]
            all_pass = True
            for inp, expected in test_cases:
                result = fn(list(inp))
                status = "PASS" if result == expected else "FAIL"
                if result != expected:
                    all_pass = False
                print(f"max_subarray({inp}) = {result}, expected {expected}: {status}")
            
            print(f"\nMax Subarray ALL PASS: {all_pass}")

# Valid Parentheses test
vp_prompt = "def valid_parentheses(s):\n    stack = []\n    mapping = {')': '(', '}': '{', ']': '['}\n    for char in s:\n        if char in mapping.values():\n            stack.append(char)\n        elif char in mapping.keys():\n            if not stack or stack[-1] != mapping[char]:\n                return False\n            stack.pop()\n        else:\n            return False\n    return not stack"

result2 = subprocess.run(
    ["/usr/bin/python3", "/home/kenpeter/work/coder/infer.py",
     "--prompt", vp_prompt,
     "--max-new-tokens", "32",
     "--temperature", "0.2",
     "--top-p", "0.95",
     "--device", "cuda"],
    capture_output=True, text=True, timeout=120
)
output2 = result2.stdout + result2.stderr
sep2 = "=" * 70
idx2 = output2.find(sep2)
if idx2 == -1:
    lines2 = output2.split("\n")
    for line in lines2:
        if "=" * 70 in line:
            sep2 = line
            break
    idx2 = output2.find(sep2) if sep2 else -1
after_sep2 = output2[idx2 + len(sep2):].strip() if idx2 >= 0 else None

print("\n\nValid Parentheses raw output after separator:")
print(after_sep2[:500] if after_sep2 else "None")

# Extract function
if after_sep2:
    idx3 = after_sep2.find("def valid_parentheses")
    if idx3 >= 0:
        remaining2 = after_sep2[idx3:]
        lines2 = remaining2.split("\n")
        func_lines2 = []
        for i, line in enumerate(lines2):
            if i > 0 and line.strip() == "" and func_lines2 and not func_lines2[-1].strip().startswith("def "):
                break
            if line.strip().startswith("def ") and i > 0:
                prev_non_empty = [l.strip() for l in lines2[:i] if l.strip()]
                if prev_non_empty and not all(l == "" for l in lines2[i-2:i]):
                    break
            func_lines2.append(line)
        if func_lines2:
            extracted2 = "\n".join(func_lines2).rstrip()
            print("\n\nExtracted function:")
            print(extracted2)
            
            ns2 = {}
            exec(extracted2, ns2)
            fn2 = ns2["valid_parentheses"]
            
            test_cases2 = [
                ("()", True),
                ("()[]{}", True),
                ("(]", False),
                ("([)]", False),
                ("{[]}", True),
            ]
            all_pass2 = True
            for inp, expected in test_cases2:
                result = fn2(inp)
                status = "PASS" if result == expected else "FAIL"
                if result != expected:
                    all_pass2 = False
                print(f"valid_parentheses({repr(inp)}) = {result}, expected {expected}: {status}")
            
            print(f"\nValid Parentheses ALL PASS: {all_pass2}")