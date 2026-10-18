import os, glob, pyarrow.parquet as pq
from collections import Counter, defaultdict
d="/mnt/file_drive/data/ultradata"
files=sorted(glob.glob(os.path.join(d,"*.parquet")))[:50]
print("files", len(files))
total_rows=0
total_chars=0
by_ds=defaultdict(lambda: {"files":0,"rows":0,"chars":0})
for f in files:
    pf=pq.ParquetFile(f)
    n=pf.metadata.num_rows
    t=pq.read_table(f, columns=["content"])
    chars=sum(len(x) if isinstance(x,str) else 0 for x in t.column("content").to_pylist())
    key=os.path.basename(f).split("-part")[0]
    by_ds[key]["files"]+=1
    by_ds[key]["rows"]+=n
    by_ds[key]["chars"]+=chars
    total_rows+=n
    total_chars+=chars
print("total rows", total_rows)
print("total chars", total_chars)
print("est tokens (chars/4)", total_chars//4)
print("\nper dataset:")
for k,v in sorted(by_ds.items()):
    print(f"{k}: files={v['files']} rows={v['rows']:,} chars={v['chars']:,} est_tok={v['chars']//4:,}")
# disk
import shutil
print("\ndisk free /home/kenpeter/work:", shutil.disk_usage("/home/kenpeter/work").free//1e9, "GB")
print("disk free /mnt/file_drive:", shutil.disk_usage("/mnt/file_drive").free//1e9, "GB")
PY