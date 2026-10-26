import os, json
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError
api=HfApi()
# list_repo_tree returns RepoFile/RepoFolder; filter by type
entries=api.list_repo_tree(repo_id='openbmb/UltraData-Code', repo_type='dataset', path_in_repo='data', recursive=True)
l3=[]
for e in entries:
    t=type(e).__name__
    if 'File' in t:
        p=getattr(e,'rfilename',None) or getattr(e,'path',None)
        if p and 'L3' in p and p.endswith('.parquet'):
            l3.append((p, getattr(e,'size',None)))
print('num L3 files', len(l3))
print('sample', l3[:3])
with open('/home/kenpeter/work/coder/l3_expected.json','w') as f:
    json.dump(l3,f)
# also L2 count
l2=[e for e in entries if 'File' in type(e).__name__ and getattr(e,'rfilename',getattr(e,'path','')) and 'L2' in (getattr(e,'rfilename',None) or getattr(e,'path',''))]
print('num L2 files', len(l2))
