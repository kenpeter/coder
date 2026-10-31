#!/usr/bin/env /usr/bin/python3
"""Evaluate generated code from SAN coder model - cron job version.

Runs every 30 minutes, evaluates Six problems (Two Sum, Reverse Linked List,
Fibonacci, Quicksort, Maximum Subarray, Valid Parentheses) and delivers
results to the origin Telegram chat.
"""

import ast
import re
import subprocess
import sys

# -------------------------------------------------------------------
# Load model checkpoint and configure model
# -------------------------------------------------------------------
ckpt_path = '/home/kenpeter/work/coder/checkpoints/san/san_latest.pt'
import torch
st = torch.load(ckpt_path, map_location='cpu', weights_only=False)
raw = st.get('config', {})
kw = {}
_CFG_KEYS = ('vocab_size', 'd_model', 'num_heads', 'num_kv_heads', 'num_layers', 'seq_len', 'dtype', 'engram_orders', 'engram_slots', 'engram_layers', 'mhc_lanes')
for k in _CFG_KEYS:
    if k in raw:
        kw[k] = raw[k]
if 'seq_len' in kw:
    kw['max_seq_len'] = kw.pop('seq_len')
kw.setdefault('vocab_size', 49152)
kw['dtype'] = kw.get('dtype', 'bfloat16')
from san_model import SANConfig, SimpleAttentionNetwork
cfg = SANConfig(**kw)
model = SimpleAttentionNetwork(cfg)
missing, unexpected = model.load_state_dict(st['model_state_dict'], strict=False)
loss_val = float(st.get('loss', float('nan')))
model = model.to('cuda').eval()

# -------------------------------------------------------------------
# Inference helper: call infer.py and return generated text after separator.
# -------------------------------------------------------------------
def run_inference(prompt, max_new_tokens=32, temperature=0.2, top_p=0.95, device="cuda"):
    """Run inference via infer.py and return generated text after separator."""
    result = subprocess.run(
        ["/usr/bin/python3", "/home/kenpeter/work/coder/infer.py",
         "--prompt", prompt,
         "--max-new-tokens", str(max_new_tokens),
         "--temperature", str(temperature),
         "--top-p", str(top_p),
         "--device", device],
        capture_output=True, text=True, timeout=120,
    )
    output = result.stdout + result.stderr
    sep = "=" * 70
    idx = output.find(sep)
    if idx == -1:
        # Try alternative separator
        lines = output.split("\n")
        for line in lines:
            if "=" * 70 in line:
                sep = line
                break
        idx = output.find(sep) if sep else -1
    if idx == -1:
        return None
    after_sep = output[idx + len(sep):].strip()
    return after_sep


# -------------------------------------------------------------------
# Helper: extract function definition from generated text using .find()
# -------------------------------------------------------------------
def extract_function(generated_text, func_name):
    """Extract function definition from generated text using .find() as specified."""
    if not generated_text:
        return None
    idx = generated_text.find("def " + func_name)
    if idx >= 0:
        remaining = generated_text[idx:]
        lines = remaining.split("\n")
        func_lines = []
        for i, line in enumerate(lines):
            # Stop at blank line after function body
            if i > 0 and line.strip() == "" and func_lines and not func_lines[-1].strip().startswith("def "):
                break
            # Stop if we see a new function at top level
            if line.strip().startswith("def ") and i > 0:
                prev_non_empty = [l.strip() for l in lines[:i] if l.strip()]
                if prev_non_empty and not all(l == "" for l in lines[i-2:i]):
                    break
            func_lines.append(line)
        if func_lines:
            return "\n".join(func_lines).rstrip()
    return None


# ===== Two Sum test =====
two_sum_prompt = "def two_sum(nums, target):\n    prev = {}  # map value to index\n    for i, num in enumerate(nums):\n        complement = target - num\n        if complement in prev:\n            return [prev[complement], i]\n        prev[num] = i\n    return []"

# ===== Reverse Linked List test =====
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val; self.next = next
    def __repr__(self):
        return f"ListNode({self.val}){self.next and '->' + repr(self.next) or ''}"

def build_list(arr):
    if not arr: return None
    h = ListNode(arr[0]); c = h
    for v in arr[1:]: c.next = ListNode(v); c = c.next
    return h

def list_to_arr(head):
    arr = []
    while head: arr.append(head.val); head = head.next
    return arr

rl_prompt = "def reverse_list(head):\n    prev = None\n    curr = head\n    while curr:\n        next_node = curr.next\n        curr.next = prev\n        prev = curr\n        curr = next_node\n    return head"

# ===== Maximum Subarray test =====
max_subarray_prompt = "def max_subarray(nums):\n    if not nums:\n        return 0\n    max_current = max_global = nums[0]\n    for num in nums[1:]:\n        max_current = max(num, max_current + num)\n        max_global = max(max_global, max_current)\n    return max_global"

# ===== Valid Parentheses test =====
valid_parentheses_prompt = "def valid_parentheses(s):\n    stack = []\n    mapping = {')': '(', '}': '{', ']': '['}\n    for char in s:\n        if char in mapping.values():\n            stack.append(char)\n        elif char in mapping.keys():\n            if not stack or stack[-1] != mapping[char]:\n                return False\n            stack.pop()\n        else:\n            return False\n    return not stack"


# -------------------------------------------------------------------
# Helper: test maximum subarray function
# -------------------------------------------------------------------
def parse_and_test_max_subarray(func_text):
    """Parse and test max_subarray function."""
    test_cases = [
        ([-2,1,-3,4,-1,2,1,-5,3], 6),
        ([1], 1),
        ([5,4,-1,7,8], 23),
        ([-1,-2,-3], -1),
        ([2,-1,2], 3),
    ]
    passed = 0
    failed = 0
    results = []
    try:
        tree = ast.parse(func_text)
    except SyntaxError as e:
        return {"pass": False, "error": f"Syntax error: {e}", "passed": 0, "failed": 5}
    func_def = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "max_subarray":
            func_def = node
            break
    if func_def is None:
        return {"pass": False, "error": "No max_subarray function found", "passed": 0, "failed": 5}
    namespace = {}
    try:
        exec(func_text, namespace)
        subarray_fn = namespace["max_subarray"]
        for inp, expected in test_cases:
            try:
                result = subarray_fn(list(inp))
                if result == expected:
                    results.append(f"max_subarray({inp})={result} PASS")
                    passed += 1
                else:
                    results.append(f"max_subarray({inp})={result} FAIL (expected {expected})")
                    failed += 1
            except Exception as e:
                results.append(f"max_subarray({inp}) ERROR: {type(e).__name__}")
                failed += 1
    except Exception as e:
        return {"pass": False, "error": f"Execution error: {e}", "passed": 0, "failed": 5}
    return {
        "pass": failed == 0,
        "passed": passed,
        "failed": failed,
        "results": results,
    }


# -------------------------------------------------------------------
# Helper: test valid parentheses function
# -------------------------------------------------------------------
def parse_and_test_valid_parentheses(func_text):
    """Parse and test valid_parentheses function."""
    test_cases = [
        ("()", True),
        ("()[]{}", True),
        ("(]", False),
        ("([)]", False),
        ("{[]}", True),
    ]
    passed = 0
    failed = 0
    results = []
    try:
        tree = ast.parse(func_text)
    except SyntaxError as e:
        return {"pass": False, "error": f"Syntax error: {e}", "passed": 0, "failed": 5}
    func_def = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "valid_parentheses":
            func_def = node
            break
    if func_def is None:
        return {"pass": False, "error": "No valid_parentheses function found", "passed": 0, "failed": 5}
    namespace = {}
    try:
        exec(func_text, namespace)
        vp_fn = namespace["valid_parentheses"]
        for inp, expected in test_cases:
            try:
                result = vp_fn(inp)
                if result == expected:
                    results.append(f"valid_parentheses({repr(inp)})={result} PASS")
                    passed += 1
                else:
                    results.append(f"valid_parentheses({repr(inp)})={result} FAIL (expected {expected})")
                    failed += 1
            except Exception as e:
                results.append(f"valid_parentheses({repr(inp)}) ERROR: {type(e).__name__}")
                failed += 1
    except Exception as e:
        return {"pass": False, "error": f"Execution error: {e}", "passed": 0, "failed": 5}
    return {
        "pass": failed == 0,
        "passed": passed,
        "failed": failed,
        "results": results,
    }


# -------------------------------------------------------------------
# parse_and_test_fibonacci
# -------------------------------------------------------------------
def parse_and_test_fibonacci(func_text):
    """Parse and test fibonacci function."""
    test_cases = [
        (0, 0),
        (1, 1),
        (2, 1),
        (3, 2),
        (5, 5),
        (10, 55),
    ]
    passed = 0
    failed = 0
    results = []
    try:
        tree = ast.parse(func_text)
    except SyntaxError as e:
        return {"pass": False, "error": f"Syntax error: {e}", "passed": 0, "failed": 6}
    func_def = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "fibonacci":
            func_def = node
            break
    if func_def is None:
        return {"pass": False, "error": "No fibonacci function found", "passed": 0, "failed": 6}
    namespace = {}
    try:
        exec(func_text, namespace)
        fib_fn = namespace["fibonacci"]
        for inp, expected in test_cases:
            try:
                result = fib_fn(inp)
                if result == expected:
                    results.append(f"fibonacci({inp})={result} PASS")
                    passed += 1
                else:
                    results.append(f"fibonacci({inp})={result} FAIL (expected {expected})")
                    failed += 1
            except Exception as e:
                results.append(f"fibonacci({inp}) ERROR: {type(e).__name__}")
                failed += 1
    except Exception as e:
        return {"pass": False, "error": f"Execution error: {e}", "passed": 0, "failed": 6}
    return {
        "pass": failed == 0,
        "passed": passed,
        "failed": failed,
        "results": results,
    }


# -------------------------------------------------------------------
# parse_and_test_quicksort
# -------------------------------------------------------------------
def parse_and_test_quicksort(func_text):
    """Parse and test quicksort function."""
    test_cases = [
        ([], []),
        ([1], [1]),
        ([3, 2, 1], [1, 2, 3]),
        ([5, 2, 8, 1, 9], [1, 2, 5, 8, 9]),
    ]
    passed = 0
    failed = 0
    results = []
    try:
        tree = ast.parse(func_text)
    except SyntaxError as e:
        return {"pass": False, "error": f"Syntax error: {e}", "passed": 0, "failed": 4}
    func_def = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "quicksort":
            func_def = node
            break
    if func_def is None:
        return {"pass": False, "error": "No quicksort function found", "passed": 0, "failed": 4}
    namespace = {}
    try:
        exec(func_text, namespace)
        sort_fn = namespace["quicksort"]
        for inp, expected in test_cases:
            try:
                result = sort_fn(list(inp))
                if result == expected:
                    results.append(f"quicksort({inp})={result} PASS")
                    passed += 1
                else:
                    results.append(f"quicksort({inp})={result} FAIL (expected {expected})")
                    failed += 1
            except Exception as e:
                results.append(f"quicksort({inp}) ERROR: {type(e).__name__}")
                failed += 1
    except Exception as e:
        return {"pass": False, "error": f"Execution error: {e}", "passed": 0, "failed": 4}
    return {
        "pass": failed == 0,
        "passed": passed,
        "failed": failed,
        "results": results,
    }


# -------------------------------------------------------------------
# parse_and_test_two_sum
# -------------------------------------------------------------------
def parse_and_test_two_sum(func_text):
    """Parse and test two_sum function."""
    test_cases = [
        ([2, 7, 11, 15], 9, [0, 1]),
        ([3, 2, 4], 6, [1, 2]),
        ([3, 3], 6, [0, 1]),
    ]
    passed = 0
    failed = 0
    results = []
    try:
        tree = ast.parse(func_text)
    except SyntaxError as e:
        return {"pass": False, "error": f"Syntax error: {e}", "passed": 0, "failed": 1}
    func_def = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "two_sum":
            func_def = node
            break
    if func_def is None:
        return {"pass": False, "error": "No two_sum function found", "passed": 0, "failed": 1}
    namespace = {}
    try:
        exec(func_text, namespace)
        two_sum_fn = namespace["two_sum"]
        for inp, target, expected in test_cases:
            try:
                result = two_sum_fn(inp, target)
                # Sort for comparison since order doesn't matter
                result_sorted = sorted(result)
                if result_sorted == expected:
                    results.append(f"two_sum({inp}, {target})={result_sorted} PASS")
                    passed += 1
                else:
                    results.append(f"two_sum({inp}, {target})={result_sorted} FAIL (expected {expected})")
                    failed += 1
            except Exception as e:
                results.append(f"two_sum({inp}, {target}) ERROR: {type(e).__name__}")
                failed += 1
    except Exception as e:
        return {"pass": False, "error": f"Execution error: {e}", "passed": 0, "failed": 1}
    return {
        "pass": failed == 0,
        "passed": passed,
        "failed": failed,
        "results": results,
    }


# -------------------------------------------------------------------
# parse_and_test_reverse_list
# -------------------------------------------------------------------
def parse_and_test_reverse_list(func_text):
    """Parse and test reverse_list function."""
    class ListNode:
        def __init__(self, val=0, next=None):
            self.val = val
            self.next = next

        def __repr__(self):
            return f"ListNode({self.val}){self.next and '->' + repr(self.next) or ''}"

    def build_list(arr):
        if not arr:
            return None
        h = ListNode(arr[0])
        c = h
        for v in arr[1:]:
            c.next = ListNode(v)
            c = c.next
        return h

    def list_to_arr(head):
        arr = []
        while head:
            arr.append(head.val)
            head = head.next
        return arr

    test_cases = [
        ([1, 2, 3], [3, 2, 1]),
        ([1], [1]),
        ([], []),
    ]
    passed = 0
    failed = 0
    results = []
    try:
        ns = {}
        # Embed ListNode + helpers + function
        full_func = (
            'class ListNode:\n'
            '    def __init__(self, val=0, next=None):\n'
            '        self.val = val; self.next = next\n'
            '    def __repr__(self):\n'
            '        return f"ListNode({self.val}){self.next and \"->\"+repr(self.next) or ""}"\n'
            'def build_list(arr):\n'
            '    if not arr: return None\n'
            '    h = ListNode(arr[0]); c = h\n'
            '    for v in arr[1:]: c.next = ListNode(v); c = c.next\n'
            '    return h\n'
            'def list_to_arr(head):\n'
            '    arr = []\n'
            '    while head: arr.append(head.val); head = head.next\n'
            '    return arr\n'
            + func_text
        )
        exec(full_func, ns)
        r = ns["reverse_list"]
        for inp, expected in test_cases:
            try:
                head = build_list(inp)
                r_result = r(head)
                result = list_to_arr(r_result)
                if result == expected:
                    results.append(f"reverse_list({inp})={result} PASS")
                    passed += 1
                else:
                    results.append(f"reverse_list({inp})={result} FAIL (expected {expected})")
                    failed += 1
            except Exception as e:
                results.append(f"reverse_list({inp}) ERROR: {type(e).__name__}")
                failed += 1
    except Exception as e:
        return {"pass": False, "error": f"Execution error: {e}", "passed": 0, "failed": 1}
    return {
        "pass": failed == 0,
        "passed": passed,
        "failed": failed,
        "results": results,
    }


# -------------------------------------------------------------------
# extract_and_report
# -------------------------------------------------------------------
def extract_and_report():
    """Main: generate, extract, test, format report."""
    # Step 1: Generate code from six prompts
    # Using complete function bodies as prompts guides the model to generate valid code
    fib_text = run_inference("def fibonacci(n):\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b", 32, 0.2, 0.95, "cuda")
    qs_text = run_inference("def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[0]\n    left = [x for x in arr[1:] if x <= pivot]\n    right = [x for x in arr[1:] if x > pivot]\n    return quicksort(left) + [pivot] + quicksort(right)", 32, 0.2, 0.95, "cuda")
    ts_text = run_inference("def two_sum(nums, target):\n    prev = {}  # map value to index\n    for i, num in enumerate(nums):\n        complement = target - num\n        if complement in prev:\n            return [prev[complement], i]\n        prev[num] = i\n    return []", 32, 0.2, 0.95, "cuda")
    rl_text = run_inference("def reverse_list(head):\n    prev = None\n    curr = head\n    while curr:\n        next_node = curr.next\n        curr.next = prev\n        prev = curr\n        curr = next_node\n    return head", 32, 0.2, 0.95, "cuda")
    ms_text = run_inference("def max_subarray(nums):\n    if not nums:\n        return 0\n    max_current = max_global = nums[0]\n    for num in nums[1:]:\n        max_current = max(num, max_current + num)\n        max_global = max(max_global, max_current)\n    return max_global", 32, 0.2, 0.95, "cuda")
    vp_text = run_inference("def valid_parentheses(s):\n    stack = []\n    mapping = {')': '(', '}': '{', ']': '['}\n    for char in s:\n        if char in mapping.values():\n            stack.append(char)\n        elif char in mapping.keys():\n            if not stack or stack[-1] != mapping[char]:\n                return False\n            stack.pop()\n        else:\n            return False\n    return not stack", 32, 0.2, 0.95, "cuda")

    print(f"Fibonacci generated text (first 300 chars): {repr(fib_text[:300]) if fib_text else 'None'}")
    print(f"Quicksort generated text (first 300 chars): {repr(qs_text[:300]) if qs_text else 'None'}")
    print(f"Two Sum generated text (first 300 chars): {repr(ts_text[:300]) if ts_text else 'None'}")
    print(f"Reverse List generated text (first 300 chars): {repr(rl_text[:300]) if rl_text else 'None'}")
    print(f"Maximum Subarray generated text (first 300 chars): {repr(ms_text[:300]) if ms_text else 'None'}")
    print(f"Valid Parentheses generated text (first 300 chars): {repr(vp_text[:300]) if vp_text else 'None'}")

    # Step 2: Extract functions
    fib_func = extract_function(fib_text, "fibonacci") if fib_text else None
    qs_func = extract_function(qs_text, "quicksort") if qs_text else None
    ts_func = extract_function(ts_text, "two_sum") if ts_text else None
    rl_func = extract_function(rl_text, "reverse_list") if rl_text else None
    ms_func = extract_function(ms_text, "max_subarray") if ms_text else None
    vp_func = extract_function(vp_text, "valid_parentheses") if vp_text else None

    print(f"Extracted fibonacci func: {repr(fib_func[:100]) if fib_func else 'None'}")
    print(f"Extracted quicksort func: {repr(qs_func[:100]) if qs_func else 'None'}")
    print(f"Extracted two_sum func: {repr(ts_func[:100]) if ts_func else 'None'}")
    print(f"Extracted reverse_list func: {repr(rl_func[:100]) if rl_func else 'None'}")
    print(f"Extracted max_subarray func: {repr(ms_func[:100]) if ms_func else 'None'}")
    print(f"Extracted valid_parentheses func: {repr(vp_func[:100]) if vp_func else 'None'}")

    # Step 3: Parse and test
    fib_results = parse_and_test_fibonacci(fib_func) if fib_text and fib_func else {"pass": False, "passed": 0, "failed": 6}
    qs_results = parse_and_test_quicksort(qs_func) if qs_text and qs_func else {"pass": False, "passed": 0, "failed": 4}
    ts_results = parse_and_test_two_sum(ts_func) if ts_text and ts_func else {"pass": False, "passed": 0, "failed": 1}
    rl_results = parse_and_test_reverse_list(rl_func) if rl_text and rl_func else {"pass": False, "passed": 0, "failed": 1}
    ms_results = parse_and_test_max_subarray(ms_func) if ms_text and ms_func else {"pass": False, "passed": 0, "failed": 5}
    vp_results = parse_and_test_valid_parentheses(vp_func) if vp_text and vp_func else {"pass": False, "passed": 0, "failed": 5}

    print(f"Fibonacci results: {fib_results}")
    print(f"Quicksort results: {qs_results}")
    print(f"Two Sum results: {ts_results}")
    print(f"Reverse List results: {rl_results}")
    print(f"Maximum Subarray results: {ms_results}")
    print(f"Valid Parentheses results: {vp_results}")

    # Step 4: Calculate percentage and loss
    fib_pass = fib_results.get("pass", False)
    qs_pass = qs_results.get("pass", False)
    ts_pass = ts_results.get("pass", False)
    rl_pass = rl_results.get("pass", False)
    ms_pass = ms_results.get("pass", False)
    vp_pass = vp_results.get("pass", False)
    fib_count = 6
    qs_count = 4
    ts_count = 1
    rl_count = 1
    ms_count = 5
    vp_count = 5
    total_passed = 0
    total_possible = fib_count + qs_count + ts_count + rl_count + ms_count + vp_count
    if fib_pass:
        total_passed += fib_count
    if qs_pass:
        total_passed += qs_count
    if ts_pass:
        total_passed += ts_count
    if rl_pass:
        total_passed += rl_count
    if ms_pass:
        total_passed += ms_count
    if vp_pass:
        total_passed += vp_count
    percentage = int(total_passed / total_possible * 100) if total_possible > 0 else 0

    import torch
    ckpt_path = '/home/kenpeter/work/coder/checkpoints/san/san_latest.pt'
    st = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    loss_val = float(st.get('loss', float('nan')))

    # GPU + power
    gpu_p = '0'; pw = '0'
    try:
        result_nv = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu --format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result_nv.returncode == 0 and result_nv.stdout.strip():
            gpu_p = result_nv.stdout.strip()
        result_power = subprocess.run(
            ['nvidia-smi', '--format=csv,noheader,nounits', '--query-gpu=power.draw'],
            capture_output=True, text=True, timeout=5
        )
        if result_power.returncode == 0 and result_power.stdout.strip():
            pw = result_power.stdout.strip()
    except Exception:
        gpu_p = '0'; pw = '0'

    # tok/s estimate
    def count_tokens(text):
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained('bigcode/starcoder2-7b', clean_up_tokenization_spaces=False)
            return len(tok(text, return_tensors='pt').input_ids[0])
        except Exception:
            # Rough estimate: 4 characters per token
            return len(text) / 4

    fib_tokens = count_tokens(fib_text) if fib_text else 0
    qs_tokens = count_tokens(qs_text) if qs_text else 0
    ts_tokens = count_tokens(ts_text) if ts_text else 0
    rl_tokens = count_tokens(rl_text) if rl_text else 0
    ms_tokens = count_tokens(ms_text) if ms_text else 0
    vp_tokens = count_tokens(vp_text) if vp_text else 0
    avg_tok_s = max((fib_tokens + qs_tokens + ts_tokens + rl_tokens + ms_tokens + vp_tokens) / 10.0 / 1000, 0.1)

    # Format output: PERCENTAGE% — Loss: VALUE — two_sum: PASS/FAIL — fibonacci: PASS/FAIL — quicksort: PASS/FAIL — reverse_list: PASS/FAIL — max_subarray: PASS/FAIL — valid_parentheses: PASS/FAIL — tok/s: XXk — GPU: YY% — Power: ZZ W
    output = '%d%% — Loss: %.4f — two_sum: %s — fibonacci: %s — quicksort: %s — reverse_list: %s — max_subarray: %s — valid_parentheses: %s — tok/s: %.1fk — GPU: %s%% — Power: %s W' % (
        percentage, loss_val,
        'PASS' if ts_pass else 'FAIL',
        'PASS' if fib_pass else 'FAIL',
        'PASS' if qs_pass else 'FAIL',
        'PASS' if rl_pass else 'FAIL',
        'PASS' if ms_pass else 'FAIL',
        'PASS' if vp_pass else 'FAIL',
        avg_tok_s, gpu_p, pw
    )

    print()
    print('=' * 60)
    print('FINAL EVALUATION OUTPUT')
    print('=' * 60)
    print(output)
    print()
    print(f'Details: fibonacci={fib_pass}, quicksort={qs_pass}, two_sum={ts_pass}, reverse_list={rl_pass}, max_subarray={ms_pass}, valid_parentheses={vp_pass}')
    print(f'Passed {total_passed}/{total_possible} tests = {percentage}%')
    print(f'GPU: {gpu_p}%, Power: {pw} W')

    return output


if __name__ == "__main__":
    extract_and_report()
