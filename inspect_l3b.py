import pyarrow.parquet as pq, os, glob
f=sorted(glob.glob("/mnt/file_drive/data/ultradata/UltraData-Code-L3-*.parquet"))[0]
t=pq.read_table(f)
print("columns:", t.column_names)
for c in t.column_names:
    v=t.column(c)[0].as_py()
    s=str(v)
    print(f"\n--- {c} (len={len(s)}) ---")
    print(s[:500])