#!/usr/bin/env python3
import os
import time

# Set HF token for authenticated downloads
HF_TOKEN = "REDACTED"
if HF_TOKEN:
    os.environ['HF_TOKEN'] = HF_TOKEN
    print(f"HF_TOKEN set (length: {len(HF_TOKEN)})")
else:
    print("No HF_TOKEN provided")

from huggingface_hub import hf_hub_download
import os

REPO = "google/gemma-4-12B-it"
CDIR = "/home/kenpeter/work/coder/"

start_time = time.time()
print(f"Starting download from: {REPO}")
print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Download tokenizer first (quick check)
print("Downloading tokenizer.json...")
try:
    path = hf_hub_download(
        repo_id=REPO,
        filename="tokenizer.json",
        local_dir=CDIR,
        resume_download=True,
        token=os.environ.get('HF_TOKEN'),
    )
    size = os.path.getsize(path)
    print(f"✅ tokenizer.json: {size} bytes ({size/1024:.1f} KB)")
    tok_time = time.time() - start_time
    print(f"Tokenizer download took {tok_time:.1f} seconds")
except Exception as e:
    print(f"❌ tokenizer.json failed: {type(e).__name__}: {str(e)[:200]}")
    tok_time = 0

# Download model.safetensors (the main weights - large file)
print("Downloading model.safetensors (12B params, ~6-8 GB)...")
print(f"SSD has ~686 GiB free, should be sufficient.")
print("This may take 5-15 minutes depending on download speed.")

try:
    path = hf_hub_download(
        repo_id=REPO,
        filename="model.safetensors",
        local_dir=CDIR,
        resume_download=True,
        token=os.environ.get('HF_TOKEN'),
    )
    size = os.path.getsize(path)
    elapsed = time.time() - start_time
    print(f"✅ model.safetensors: {size} bytes ({size/1024/1024/1024:.2f} GB)")
    print(f"Download completed in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
except Exception as e:
    elapsed = time.time() - start_time
    print(f"❌ model.safetensors failed: {type(e).__name__}: {str(e)[:200]}")
    print(f"Download attempted for {elapsed:.1f} seconds")
    print("(May need more time or token may not have sufficient permissions)")