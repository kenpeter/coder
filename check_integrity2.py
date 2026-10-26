import os, glob, pyarrow.parquet as pq, json
files=sorted(glob.glob('/mnt/file_drive/data/ultradata/UltraData-Code-L3-*.parquet'))
bad=[]
for idx,f in enumerate(files):
    try:
        # read only uuid column, first 5 rows, to force footer+column decode
        t=pq.read_table(f, columns=['uuid']).slice(0,5)
        assert t.num_rows>=1
        # also read last row
        pf=pq.ParquetFile(f)
        t2=pq.read_table(f, columns=['uuid']).slice(pf.metadata.num_rows-1,1)
        assert t2.num_rows==1
    except Exception as e:
        bad.append((os.path.basename(f), str(e)[:300]))
    if idx%50==0:
        print('progress', idx, 'bad', len(bad), flush=True)
print('TOTAL files', len(files), 'bad', len(bad))
for b in bad[:30]: print('BAD',b)
with open('/home/kenpeter/work/coder/l3_bad_integrity.json','w') as fh:
    json.dump(bad,fh)
