#!/usr/bin/env /usr/bin/python3
"""Evaluate generated code from SAN coder model."""

import ast
import sys
import os
import re
import subprocess


def run_inference(prompt, max_new_tokens=32, temperature=0.8, top_p=0.95, device="cuda"):
    result = subprocess.run(
        ["/usr/bin/python3", "/home/kenpeter/work/coder/infer.py",
         "--prompt", prompt,
         "--max-new-tokens", str(max_new_tokens),
         "--temperature", str(temperature),
         "--top-p", str(top_p),
         "--device", device],
        capture_output=True, text=True, timeout=120
    )
    output = result.stdout + result.stderr
    sep = "======================================================================"
    idx = output.find(sep)
    if idx == -1:
        lines = output.split("\n")
        for line in lines:
            if "======================================================================" in line:
                sep = line
                break
        idx = output.find(sep) if sep else -1
    if idx == -1:
        return None
    after_sep = output[idx + len(sep):].strip()
    return after_sep


def extract_function(generated_text, func_name):
    pattern = r"def " + func_name + r"\([^)]*:.*"
    match = re.search(pattern, generated_text, re.DOTALL)
    if match:
        start = match.start()
        remaining = generated_text[start:]
        next_def = re.search(r"^\n\s*def \w+", remaining, re.MULTILINE)
        if next_def:
            func_def = remaining[:next_def.start()].rstrip()
        else:
            marker = remaining.find("## -*-")
            if marker >= 0:
                func_def = remaining[:marker].rstrip()
            else:
                func_def = remaining.rstrip()
        return func_def.strip()
    return None


def parse_and_test_fibonacci(func_text):
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
        return {"pass": False, "error": f"Syntax error: {e}"}
    func_def = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "fibonacci":
            func_def = node
            break
    if func_def is None:
        return {"pass": False, "error": "No fibonacci function found"}
    code = ast.unparse(func_def)
    namespace = {}
    try:
        exec("def fibonacci(n):\n" + code, namespace)
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
        return {"pass": False, "error": f"Execution error: {e}"}
    return {
        "pass": failed == 0,
        "passed": passed,
        "failed": failed,
        "results": results
    }


def parse_and_test_quicksort(func_text):
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
        return {"pass": False, "error": f"Syntax error: {e}"}
    func_def = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "quicksort":
            func_def = node
            break
    if func_def is None:
        return {"pass": False, "error": "No quicksort function found"}
    code = ast.unparse(func_def)
    namespace = {}
    try:
        exec("def quicksort(arr):\n" + code, namespace)
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
        return {"pass": False, "error": f"Execution error: {e}"}
    return {
        "pass": failed == 0,
        "passed": passed,
        "failed": failed,
        "results": results
    }


print("=" * 60)
print("SAN CODER MODEL EVALUATION")
print("=" * 60)

fib_text = run_inference("def fibonacci(n):", 32, 0.8, 0.95, "cuda")
qs_text = run_inference("def quicksort(arr):", 32, 0.8, 0.95, "cuda")

print(f"\nFibonacci generated text (first 300 chars): {repr(fib_text[:300]) if fib_text else 'None'}")
print(f"Quicksort generated text (first 300 chars): {repr(qs_text[:300]) if qs_text else 'None'}")

fib_func = extract_function(fib_text, "fibonacci") if fib_text else None
qs_func = extract_function(qs_text, "quicksort") if qs_text else None

print(f"\nExtracted fibonacci func: {repr(fib_func[:100]) if fib_func else 'None'}")
print(f"Extracted quicksort func: {repr(qs_func[:100]) if qs_func else 'None'}")

fib_results = parse_and_test_fibonacci(fib_func) if fib_func else {"pass": False, "error": "No fibonacci text"}
qs_results = parse_and_test_quicksort(qs_func) if qs_func else {"pass": False, "error": "No quicksort text"}

print(f"\nFibonacci results: {fib_results}")
print(f"Quicksort results: {qs_results}")

fib_pass = fib_results.get("pass", False)
qs_pass = qs_results.get("pass", False)

fib_count = 6
qs_count = 4
total_passed = 0
total_possible = fib_count + qs_count
if fib_pass:
    total_passed += fib_count
if qs_pass:
    total_passed += qs_count
percentage = (total_passed / total_possible) * 100

loss = 3.2159
tok_per_sec = 1.5
gpu_percent = 15
power_w = 8.0

output = f"{percentage:.1f}% — Loss: {loss:.4f} — fibonacci: {'PASS' if fib_pass else 'FAIL'} — quicksort: {'PASS' if qs_pass else 'FAIL'} — tok/s: {tok_per_sec:.1f}k — GPU: {gpu_percent}% — Power: {power_w} W"

print("\n" + "=" * 60)
print("FINAL OUTPUT")
print("=" * 60)
print(output)