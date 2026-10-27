"""Diagnostic micro-eval: 20 canonical one-function tasks.

Unlike HumanEval, the prompt opens the function body ("def f(...):\\n    ") so
the model cannot dodge the task by starting a new top-level def. This measures
whether the checkpoint can emit correct short Python functions, separately from
its zero-shot HumanEval behaviour.
"""
import subprocess
import sys
import tempfile

import torch

from infer import load_model, generate

TASKS = [
    ("add", "a, b", "return a + b", "assert add(2, 3) == 5\nassert add(-1, 1) == 0"),
    ("square", "x", "return x * x", "assert square(4) == 16\nassert square(-3) == 9"),
    ("is_even", "n", "return n % 2 == 0", "assert is_even(4)\nassert not is_even(7)"),
    ("max_of_two", "a, b", "return a if a > b else b", "assert max_of_two(3, 8) == 8\nassert max_of_two(9, 2) == 9"),
    ("sum_list", "xs", "return sum(xs)", "assert sum_list([1, 2, 3]) == 6\nassert sum_list([]) == 0"),
    ("reverse_string", "s", "return s[::-1]", "assert reverse_string('abc') == 'cba'"),
    ("count_vowels", "s", "return sum(c in 'aeiou' for c in s.lower())", "assert count_vowels('hello') == 2"),
    ("factorial", "n", "return 1 if n <= 1 else n * factorial(n - 1)", "assert factorial(5) == 120\nassert factorial(0) == 1"),
    ("fib", "n", "return n if n < 2 else fib(n - 1) + fib(n - 2)", "assert fib(7) == 13\nassert fib(0) == 0"),
    ("abs_val", "x", "return abs(x)", "assert abs_val(-5) == 5\nassert abs_val(2) == 2"),
    ("min_list", "xs", "return min(xs)", "assert min_list([4, 1, 9]) == 1"),
    ("c_to_f", "c", "return c * 9 / 5 + 32", "assert c_to_f(0) == 32\nassert c_to_f(100) == 212"),
    ("is_palindrome", "s", "return s == s[::-1]", "assert is_palindrome('racecar')\nassert not is_palindrome('abc')"),
    ("power", "base, exp", "return base ** exp", "assert power(2, 10) == 1024"),
    ("first_char", "s", "return s[0]", "assert first_char('xyz') == 'x'"),
    ("last_char", "s", "return s[-1]", "assert last_char('xyz') == 'z'"),
    ("list_len", "xs", "return len(xs)", "assert list_len([1, 2, 3]) == 3"),
    ("multiply", "a, b", "return a * b", "assert multiply(6, 7) == 42"),
    ("floor_div", "a, b", "return a // b", "assert floor_div(7, 2) == 3"),
    ("sign", "x", "return (x > 0) - (x < 0)", "assert sign(5) == 1\nassert sign(-5) == -1\nassert sign(0) == 0"),
]

STOPS = ("\ndef ", "\nclass ", "\nif __name__")


def run(program, timeout=5):
    with tempfile.TemporaryDirectory() as d:
        p = d + "/t.py"
        open(p, "w").write(program)
        try:
            r = subprocess.run([sys.executable, p], capture_output=True, text=True, timeout=timeout)
            return r.returncode == 0, (r.stdout + r.stderr)[-300:]
        except subprocess.TimeoutExpired:
            return False, "timeout"


def main():
    m, tok, st = load_model(device="cuda")
    passed = 0
    for name, args, ref, tests in TASKS:
        prompt = f"def {name}({args}):\n    "
        torch.manual_seed(0)
        out = generate(m, tok, prompt, max_new_tokens=64, temperature=0,
                       stop_strings=STOPS, device="cuda")[0]
        program = prompt + out + "\n\n" + tests + "\n"
        ok, log = run(program)
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'}  {name:<15} {out.strip()[:90]!r}")
        if not ok:
            print(f"      {log.splitlines()[-1] if log.splitlines() else ''}")
    print(f"\nmicro pass@1: {passed}/{len(TASKS)} = {passed/len(TASKS):.1%}")
    print(f"step {st.get('step')}")


if __name__ == "__main__":
    main()
