import os, glob, pyarrow.parquet as pq, json
files=sorted(glob.glob('/mnt/file_drive/data/ultradata/UltraData-Code-L3-*.parquet'))
agg={'nulls':{c:0 for c in ['uuid','task','analysis','solution','test']},
     'empties':{c:0 for c in ['uuid','task','analysis','solution','test']},
     'rows':0}
bad=[]
for idx,f in enumerate(files):
    try:
        t=pq.read_table(f, columns=['uuid','task','analysis','solution','test'])
        agg['rows']+=t.num_rows
        for c in ['uuid','task','analysis','solution','test']:
            col=t.column(c)
            agg['nulls'][c]+=col.null_count
            for v in col.to_pylist():
                if v is None or (isinstance(v,str) and v.strip()==''):
                    agg['empties'][c]+=1
    except Exception as e:
        bad.append((os.path.basename(f), str(e)[:200]))
    if idx%25==0 or idx==len(files)-1:
        print('progress', idx, 'rows', agg['rows'], 'bad', len(bad), flush=True)
print('DONE files', len(files), 'rows', agg['rows'], 'bad', len(bad))
print('nulls', agg['nulls'])
print('empties', agg['empties'])
with open('/home/kenpeter/work/coder/l3_data_quality.json','w') as fh:
    json.dump({'rows':agg['rows'],'nulls':agg['nulls'],'empties':agg['empties'],'bad':bad},fh)
print('SAVED')
