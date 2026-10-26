import shutil, os
p = "/home/kenpeter/work/data_ssd"
if os.path.exists(p):
    shutil.rmtree(p)
    print("removed", p)
else:
    print("not found", p)
