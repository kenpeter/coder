from huggingface_hub import hf_hub_download
import os

REPO = "google/gemma-4-12B-it"
CDIR = "/home/kenpeter/work/coder/"

print(f"Downloading Gemma 4 12B it model weights from: {REPO}")

# Download model.safetensors (the main weights - this is the large one)
print("Downloading model.safetensors (12B params)...")
print("This is large (~6-8 GB) and may take a few minutes.")
print("SSD has 689 GiB free, should be sufficient.")
try:
    path = hf_hub_download(
        repo_id=REPO,
        filename="model.safetensors",
        local_dir=CDIR,
        resume_download=True,
        token=None,
    )
    size = os.path.getsize(path)
    print(f"✅ model.safetensors: {size} bytes ({size/1024/1024/1024:.2f} GB)")
    print("✅ Download complete!")
except Exception as e:
    print(f"❌ model.safetensors failed: {type(e).__name__}: {str(e)[:200]}")
    print("(Download may need more time or auth token)")