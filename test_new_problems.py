#!/usr/bin/env /usr/bin/python3
"""Test new LeetCode problems: Maximum Subarray and Valid Parentheses"""
import ast
import subprocess
import sys
from transformers import AutoTokenizer

# -------------------------------------------------------------------
# Load model checkpoint and configure model
# -------------------------------------------------------------------
ckpt_path = '/home/kenpeter/work/coder/checkpoints/san/san_latest.pt'
st = torch.load(ckpt_path, map_location='cpu', weights_only=False) if 'torch' in sys.modules else None
print("Need to import torch first")