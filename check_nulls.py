import glob, os, pyarrow.parquet as pq
files=sorted(glob.glob('/mnt/file_drive/data/ultradata/UltraData-Code-L3-*.parquet'))
# sample evenly
idx=[0, len(files)//4, len(files)//2, 3*len(files)//4, len(files)-1]
for i in idx:
    f=files[i]
    t=pq.read_table(f)
    print('===', os.path.basename(f), 'rows', t.num_rows)
    for c in t.column_names:
        col=t.column(c)
        nulls=col.null_count
        # empty string count for string cols
        empties=0
        try:
            arr=col.to_pylist()
            empties=sum(1 for v in arr if v is None or (isinstance(v,str) and v.strip()==''))
        except Exception as e:
            empties='ERR '+str(e)[:50]
        print(f'  {c}: nulls={nulls} empties={empties}')
