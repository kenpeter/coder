import os, json, glob
exp=json.load(open('/home/kenpeter/work/coder/l3_expected.json'))
# build map flat name -> expected size
expmap={}
for p,sz in exp:
    flat=p.split('/')[-1]
    expmap[flat]=sz
files=sorted(glob.glob('/mnt/file_drive/data/ultradata/UltraData-Code-L3-*.parquet'))
mism=[]
missing=[]
for f in files:
    flat=os.path.basename(f)
    have=os.path.getsize(f)
    want=expmap.get(flat)
    if want is None:
        missing.append((flat,'no expected'))
    elif have!=want:
        mism.append((flat,have,want))
# expected files not present locally
local=set(os.path.basename(f) for f in files)
for flat in expmap:
    if flat not in local:
        missing.append((flat,'not local'))
print('local L3 files', len(files), 'expected', len(expmap))
print('size mismatches', len(mism))
for m in mism[:30]: print('MISMATCH',m)
print('missing/unmapped', len(missing))
for m in missing[:30]: print('MISSING',m)
# save mismatch list
import json
with open('/home/kenpeter/work/coder/l3_mismatches.json','w') as fh:
    json.dump({'mismatches':mism,'missing':missing},fh)
