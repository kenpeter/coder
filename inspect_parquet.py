import os, glob
import pyarrow.parquet as pq
d="/mnt/file_drive/data/ultradata"
files=sorted(glob.glob(os.path.join(d,"*.parquet")))
print("num files", len(files))
f=files[0]
print("sample file", f)
pf=pq.ParquetFile(f)
print("rows", pf.metadata.num_rows)
print("schema:", pf.schema_arrow)
print("columns:", pf.schema_arrow.names)
t=pq.read_table(f).slice(0,3)
print(t.to_pandas().to_string())
total=sum(os.path.getsize(p) for p in files)
print("total bytes", total)
from collections import Counter
prefixes=Counter()
for p in files:
    name=os.path.basename(p)
    key=name.split("-part")[0]
    prefixes[key]+=1
print("num distinct datasets", len(prefixes))
for k,v in sorted(prefixes.items()):
    print(k, v)
