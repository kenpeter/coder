import glob, os, pyarrow.parquet as pq
from collections import Counter
files=sorted(glob.glob('/mnt/file_drive/data/ultradata/UltraData-Code-L3-*.parquet'))
cf=Counter(); fcf=Counter()
lens_content=[]; lens_full=[]; same=0; diff=0
n=0
for f in files:
    t=pq.read_table(f, columns=['content','full_content','content_format','full_content_format'])
    for c in t.column('content_format').to_pylist():
        cf[c]+=1
    for c in t.column('full_content_format').to_pylist():
        fcf[c]+=1
    # sample lengths
    for a,b in zip(t.column('content').to_pylist(), t.column('full_content').to_pylist()):
        if a==b: same+=1
        else: diff+=1
    n+=t.num_rows
    if n>2000000: 
        break
print('rows sampled', n)
print('content_format', cf)
print('full_content_format', fcf)
print('content==full_content count', same, 'diff', diff)
