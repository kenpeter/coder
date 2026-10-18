import sys
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer

root = Path(sys.argv[1])
tok = AutoTokenizer.from_pretrained("bigcode/starcoder2-7b", clean_up_tokenization_spaces=False)
eos = tok.eos_token_id
print("eos id:", eos, "vocab:", len(tok))

grand = 0
for d in sorted(root.glob("_shards_code_*")):
    files = sorted(d.glob("*.bin"))
    tot = 0
    for f in files:
        arr = np.memmap(f, dtype=np.uint16, mode="r")
        mx = int(arr.max()) if arr.size else -1
        assert mx < 49152, f"BAD max id {mx} in {f}"
        tot += arr.size
    grand += tot
    print(f"{d.name}: {len(files)} shard(s), {tot:,} tokens, files={[f.name for f in files]}")
    if files:
        arr = np.memmap(files[0], dtype=np.uint16, mode="r")
        head = arr[:80].tolist()
        print("   first 80 ids:", head[:20], "...")
        print("   decode head:", repr(tok.decode(head)))
        # show an EOS boundary
        idx = int(np.argmax(np.asarray(arr[:5000]) == eos)) if arr.size else -1
        print("   first EOS index in first 5000:", idx)

print("GRAND TOTAL tokens:", f"{grand:,}")
print("OK: all ids < 49152, dtype uint16")
