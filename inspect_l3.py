import os, glob, pyarrow.parquet as pq
d="/mnt/file_drive/data/ultradata"
files=sorted(glob.glob(os.path.join(d,"UltraData-Code-L3-*.parquet")))
print("num L3 files", len(files))
f=files[0]
print("sample", f)
pf=pq.ParquetFile(f)
print("rows", pf.metadata.num_rows)
t=pq.read_table(f)
print("schema:", t.schema)
print("columns:", t.column_names)
print(t.slice(0,2).to_pandas().to_string())