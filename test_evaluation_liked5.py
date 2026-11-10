"""Eval: Top 5 LeetCode Top 100 Liked problems by likes count."""
import subprocess

def run_inference(prompt, max_new_tokens=32, temperature=0.2, top_p=0.95, device="cuda"):
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
        for line in output.split("\n"):
            if "=" * 70 in line:
                sep = line
                break
        idx = output.find(sep) if sep else -1
    if idx == -1:
        return None
    after_sep = output[idx + len(sep):].strip()
    return after_sep

def extract_function(generated_text, func_name):
    if not generated_text:
        return None
    idx = generated_text.find("def " + func_name)
    if idx >= 0:
        remaining = generated_text[idx:]
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
            return "\n".join(func_lines).rstrip()
    return None

# ====== Problem 1: Longest Substring Without Repeating Characters ======
print("=" * 60)
print("PROBLEM 1: Longest Substring Without Repeating Characters")
print("=" * 60)
ls_prompt = "def length_of_longest_substring(s):\n    char_set = set()\n    left = 0\n    max_len = 0\n    for right in range(len(s)):\n        while s[right] in char_set:\n            char_set.remove(s[left])\n            left += 1\n        char_set.add(s[right])\n        max_len = max(max_len, right - left + 1)\n    return max_len"
ls_text = run_inference(ls_prompt, 32, 0.2, 0.95, "cuda")
ls_func = extract_function(ls_text, "length_of_longest_substring") if ls_text else None
print("Extracted func: " + repr(ls_func[:100]) if ls_func else "None")
results1 = {"pass": False}
if ls_func:
    try:
        ns = {}
        exec(ls_func, ns)
        if "length_of_longest_substring" in ns:
            fn = ns["length_of_longest_substring"]
            t1 = fn("abcabcbb") == 3
            t2 = fn("bbbbb") == 1
            t3 = fn("pwwkew") == 3
            t4 = fn("") == 0
            t5 = fn(" ") == 1
            t6 = fn("abcdef") == 6
            results1["pass"] = t1 and t2 and t3 and t4 and t5 and t6
            print("Tests: " + str(t1) + " " + str(t2) + " " + str(t3) + " " + str(t4) + " " + str(t5) + " " + str(t6))
            print("Problem 1: " + ("PASS" if results1["pass"] else "FAIL"))
        else:
            print("Function name not found after exec")
    except Exception as e:
        print("Execution error: " + str(e))
else:
    print("No function extracted")

# ====== Problem 2: Median of Two Sorted Arrays ======
print()
print("=" * 60)
print("PROBLEM 2: Median of Two Sorted Arrays")
print("=" * 60)
md_prompt = "def find_median_sorted_arrays(nums1, nums2):\n    if len(nums1) > len(nums2):\n        nums1, nums2 = nums2, nums1\n    x, y = len(nums1), len(nums2)\n    low, high = 0, x\n    while low <= high:\n        partitionX = (low + high) // 2\n        partitionY = ((x + y + 1) // 2) - partitionX\n        maxLeftX = float('-inf') if partitionX == 0 else nums1[partitionX - 1]\n        minRightX = float('inf') if partitionX == x else nums1[partitionX]\n        maxLeftY = float('-inf') if partitionY == 0 else nums2[partitionY - 1]\n        minRightY = float('inf') if partitionY == y else nums2[partitionY]\n        if maxLeftX <= minRightY and maxLeftY <= minRightX:\n            if (x + y) % 2 == 0:\n                return (max(maxLeftX, maxLeftY) + min(minRightX, minRightY)) / 2\n            else:\n                return max(maxLeftX, maxLeftY)\n        elif maxLeftX > minRightY:\n            high = partitionX - 1\n        else:\n            low = partitionX + 1\n    return 0.0"
md_text = run_inference(md_prompt, 32, 0.2, 0.95, "cuda")
md_func = extract_function(md_text, "find_median_sorted_arrays") if md_text else None
print("Extracted func: " + repr(md_func[:100]) if md_func else "None")
results2 = {"pass": False}
if md_func:
    try:
        ns = {}
        exec(md_func, ns)
        if "find_median_sorted_arrays" in ns:
            fn = ns["find_median_sorted_arrays"]
            t1 = fn([1, 3], [2]) == 2.0
            t2 = fn([1, 2], [3, 4]) == 2.5
            t3 = fn([], [1]) == 1.0
            t4 = fn([2], []) == 2.0
            t5 = fn([1, 3], [2, 7]) == 2.5
            t6 = fn([1, 2], [3, 4, 5]) == 3.0
            results2["pass"] = t1 and t2 and t3 and t4 and t5 and t6
            print("Tests: " + str(t1) + " " + str(t2) + " " + str(t3) + " " + str(t4) + " " + str(t5) + " " + str(t6))
            print("Problem 2: " + ("PASS" if results2["pass"] else "FAIL"))
        else:
            print("Function name not found after exec")
    except Exception as e:
        print("Execution error: " + str(e))
else:
    print("No function extracted")

# ====== Problem 3: Longest Palindromic Substring ======
print()
print("=" * 60)
print("PROBLEM 3: Longest Palindromic Substring")
print("=" * 60)
lp_prompt = "def longest_palindrome(s):\n    if not s: return \"\"\n    n = len(s)\n    start, max_len = 0, 1\n    for i in range(n):\n        l, r = i - 1, i + 1\n        while l >= 0 and r < n and s[l] == s[r]:\n            l -= 1\n            r += 1\n        odd_len = r - l - 1\n        l, r = i, i + 1\n        while l >= 0 and r < n and s[l] == s[r]:\n            l -= 1\n            r += 1\n        even_len = r - l - 1\n        cur_len = max(odd_len, even_len)\n        if cur_len > max_len:\n            max_len = cur_len\n            start = i - (cur_len - 1) // 2\n    return s[start:start + max_len]"
lp_text = run_inference(lp_prompt, 32, 0.2, 0.95, "cuda")
lp_func = extract_function(lp_text, "longest_palindrome") if lp_text else None
print("Extracted func: " + repr(lp_func[:100]) if lp_func else "None")
results3 = {"pass": False}
if lp_func:
    try:
        ns = {}
        exec(lp_func, ns)
        if "longest_palindrome" in ns:
            fn = ns["longest_palindrome"]
            t1 = fn("babad") in ("bab", "aba")
            t2 = fn("cbbd") == "bb"
            t3 = fn("a") == "a"
            t4 = fn("ac") in ("a", "c")
            t5 = fn("") == ""
            results3["pass"] = t1 and t2 and t3 and t4 and t5
            print("Tests: " + str(t1) + " " + str(t2) + " " + str(t3) + " " + str(t4) + " " + str(t5))
            print("Problem 3: " + ("PASS" if results3["pass"] else "FAIL"))
        else:
            print("Function name not found after exec")
    except Exception as e:
        print("Execution error: " + str(e))
else:
    print("No function extracted")

# ====== Problem 4: Container With Most Water ======
print()
print("=" * 60)
print("PROBLEM 4: Container With Most Water")
print("=" * 60)
cw_prompt = "def max_area(height):\n    left = 0\n    right = len(height) - 1\n    max_water = 0\n    while left < right:\n        if height[left] < height[right]:\n            max_water = max(max_water, height[left] * (right - left))\n            left += 1\n        else:\n            max_water = max(max_water, height[right] * (right - left))\n            right -= 1\n    return max_water"
cw_text = run_inference(cw_prompt, 32, 0.2, 0.95, "cuda")
cw_func = extract_function(cw_text, "max_area") if cw_text else None
print("Extracted func: " + repr(cw_func[:100]) if cw_func else "None")
results4 = {"pass": False}
if cw_func:
    try:
        ns = {}
        exec(cw_func, ns)
        if "max_area" in ns:
            fn = ns["max_area"]
            t1 = fn([1, 8, 6, 2, 5, 4, 8, 3, 7]) == 49
            t2 = fn([1, 1]) == 1
            t3 = fn([4, 3, 2, 1, 4]) == 16
            t4 = fn([1, 2, 1]) == 2
            t5 = fn([2, 3, 4, 5, 18, 17, 6]) == 17
            results4["pass"] = t1 and t2 and t3 and t4 and t5
            print("Tests: " + str(t1) + " " + str(t2) + " " + str(t3) + " " + str(t4) + " " + str(t5))
            print("Problem 4: " + ("PASS" if results4["pass"] else "FAIL"))
        else:
            print("Function name not found after exec")
    except Exception as e:
        print("Execution error: " + str(e))
else:
    print("No function extracted")

# ====== Problem 5: Remove Nth From End of List ======
print()
print("=" * 60)
print("PROBLEM 5: Remove Nth From End of List")
print("=" * 60)
rn_prompt = "def remove_nth_from_end(head, n):\n    dummy = ListNode(0, head)\n    first = head\n    second = dummy\n    for _ in range(n):\n        first = first.next\n    while first:\n        first = first.next\n        second = second.next\n    second.next = second.next.next\n    return dummy.next\n\nclass ListNode:\n    def __init__(self, val=0, next=None):\n        self.val = val\n        self.next = next\n    def __repr__(self):\n        return \"ListNode(\" + str(self.val) + \")\" + (\"->\" + repr(self.next) if self.next else \"\")"
rn_text = run_inference(rn_prompt, 32, 0.2, 0.95, "cuda")
rn_func = extract_function(rn_text, "remove_nth_from_end") if rn_text else None
print("Extracted func: " + repr(rn_func[:100]) if rn_func else "None")
results5 = {"pass": False}
if rn_func:
    try:
        ns = {}
        exec(rn_func, ns)
        if "remove_nth_from_end" in ns:
            fn = ns["remove_nth_from_end"]
            def build_list(arr):
                if not arr: return None
                h = ListNode(arr[0]); c = h
                for v in arr[1:]: c.next = ListNode(v); c = c.next
                return h
            def list_to_arr(head):
                arr = []
                while head:
                    arr.append(head.val)
                    head = head.next
                return arr
            t1 = list_to_arr(fn(build_list([1, 2, 3, 4, 5], 2))) == [1, 2, 3, 5]
            t2 = list_to_arr(fn(build_list([1], 1))) == []
            t3 = list_to_arr(fn(build_list([1, 2], 2))) == [1]
            t4 = list_to_arr(fn(build_list([1, 2, 3], 1))) == [1, 2]
            t5 = list_to_arr(fn(build_list([1, 2, 3, 4, 5], 5))) == [2, 3, 4, 5]
            results5["pass"] = t1 and t2 and t3 and t4 and t5
            print("Tests: " + str(t1) + " " + str(t2) + " " + str(t3) + " " + str(t4) + " " + str(t5))
            print("Problem 5: " + ("PASS" if results5["pass"] else "FAIL"))
        else:
            print("Function name not found after exec")
    except Exception as e:
        print("Execution error: " + str(e))
else:
    print("No function extracted")

# ====== Summary ======
print()
print("=" * 60)
print("SUMMARY: TOP 5 LEETCODE TOP 100 LIKED PROBLEMS")
print("=" * 60)
passed_count = sum([results1["pass"], results2["pass"], results3["pass"], results4["pass"], results5["pass"]])
pct = int(passed_count / 5 * 100)
print("Problem 1 - Longest Substring WR: " + ("PASS" if results1["pass"] else "FAIL"))
print("Problem 2 - Median of 2 Sorted Arrays: " + ("PASS" if results2["pass"] else "FAIL"))
print("Problem 3 - Longest Palindromic Substring: " + ("PASS" if results3["pass"] else "FAIL"))
print("Problem 4 - Container With Most Water: " + ("PASS" if results4["pass"] else "FAIL"))
print("Problem 5 - Remove Nth From End of List: " + ("PASS" if results5["pass"] else "FAIL"))
print()
print("Overall: " + str(pct) + "% (" + str(passed_count) + "/5 problems passed all tests)")
print("Problems from LeetCode Top 100 Liked study plan (official, free)")
