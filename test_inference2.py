#!/usr/bin/env /usr/bin/python3
"""Test max_subarray and valid_parentheses inference"""
import subprocess
import ast
import torch
from transformers import AutoTokenizer

# Load model
ckpt_path = '/home/kenpeter/work/coder/checkpoints/san/san_latest.pt'
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
model.load_state_dict(st['model_state_dict'], strict=False)
model = model.to('cuda').eval()
tok = AutoTokenizer.from_pretrained('bigcode/starcoder2-7b', clean_up_tokenization_spaces=False)

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

# Test max_subarray
ms_prompt = "def max_subarray(nums):\n    if not nums:\n        return 0\n    max_current = max_global = nums[0]\n    for num in nums[1:]:\n        max_current = max(num, max_current + num)\n        max_global = max(max_global, max_current)\n    return max_global"
ms_text = run_inference(ms_prompt, 32, 0.2, 0.95, "cuda")
print(f"Max subarray text (first 200): {repr(ms_text[:200]) if ms_text else 'None'}")

ms_func = extract_function(ms_text, "max_subarray") if ms_text else None
print(f"Extracted max_subarray: {repr(ms_func[:100]) if ms_func else 'None'}")

# Test valid_parentheses
vp_prompt = "def valid_parentheses(s):\n    stack = []\n    mapping = {')': '(', '}': '{', ']': '['}\n    for char in s:\n        if char in mapping.values():\n            stack.append(char)\n        elif char in mapping.keys():\n            if not stack or stack[-1] != mapping[char]:\n                return False\n            stack.pop()\n        else:\n            return False\n    return not stack"
vp_text = run_inference(vp_prompt, 32, 0.2, 0.95, "cuda")
print(f"VP text (first 200): {repr(vp_text[:200]) if vp_text else 'None'}")

vp_func = extract_function(vp_text, "valid_parentheses") if vp_text else None
print(f"Extracted valid_parentheses: {repr(vp_func[:100]) if vp_func else 'None'}")