from huggingface_hub import hf_hub_download
import os

REPO = "google/gemma-4-12B-it"
CDIR = "/home/kenpeter/work/coder/"

print(f"Downloading Gemma 4 12B it from: {REPO}")

# Download tokenizer first (smaller, faster - completes in ~30s)
print("Downloading tokenizer.json...")
try:
    path = hf_hub_download(
        repo_id=REPO,
        filename="tokenizer.json",
        local_dir=CDIR,
        resume_download=True,
        token=None,
    )
    size = os.path.getsize(path)
    print(f"✅ tokenizer.json: {size} bytes ({size/1024:.1f} KB) - DOWNLOADED")
except Exception as e:
    print(f"❌ tokenizer.json failed: {type(e).__name__}")

print("Download complete - tokenizer only")