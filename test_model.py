import sys
sys.path.insert(0, '/home/kenpeter/work/coder')

# Load the model
from infer import load_model
model, tok, st = load_model('/home/kenpeter/work/coder/checkpoints/san/san_latest.pt')
print(f'Model: step {st.get("step")}, loss {float(st.get("loss", "unknown"))}')
print(f'Params: {model.count_parameters()/1e6:.1f}M, Layers: {model.cfg.num_layers}')

# Try generating code
prompt = 'def two_sum(nums, target):\n    prev = {}\n    for i, num in enumerate(nums):\n        complement = target - num\n        if complement in prev:\n            return [prev[complement], i]\n        prev[num] = i\n    return []'

# Use the generate function from infer
import torch
model.eval()

# Tokenize the prompt
from transformers import AutoTokenizer as HFTokenizer
hf_tok = HFTokenizer.from_pretrained('bigcode/starcoder2-7b', clean_up_tokenization_spaces=False)

# Generate
print(f'Prompt: {prompt[:50]}...')
