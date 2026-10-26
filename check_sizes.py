import os, glob
files=sorted(glob.glob('/mnt/file_drive/data/ultradata/UltraData-Code-L3-*.parquet'))
sizes=[os.path.getsize(f) for f in files]
print('num', len(files))
print('min', min(sizes), 'max', max(sizes), 'avg', sum(sizes)//len(sizes))
small=[(os.path.basename(f),s) for f,s in zip(files,sizes) if s < 500*1024*1024]
print('small (<500MB):', len(small))
for s in small[:20]:
    print(s)
