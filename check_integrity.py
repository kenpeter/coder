import glob, os, pyarrow.parquet as pq
files=sorted(glob.glob('/mnt/file_drive/data/ultradata/UltraData-Code-L3-*.parquet'))
bad=[]
for f in files:
    try:
        pf=pq.ParquetFile(f)
        t=pf.read_row_group(0) if pf.metadata.num_row_groups>0 else None
        # read last few rows via slice
        tab=pq.read_table(f).slice(pf.metadata.num_rows-1,1)
        # ensure uuid present
        assert tab.column('uuid')[0].as_py() is not None
    except Exception as e:
        bad.append((os.path.basename(f), str(e)[:300]))
print('files', len(files), 'bad', len(bad))
for b in bad[:30]:
    print('BAD', b)
