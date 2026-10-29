"""Host compatibility shim (Python 3.14 defaults to forkserver).

x-small's loader was written for Python 3.11, where DataLoader workers fork and
share the dataset index copy-on-write. On 3.14 the forkserver pickles a private
copy into every worker (~10 GB x 8). Restore fork so the identical training code
keeps x-small's memory behaviour. No model/training logic is changed here.
"""
import multiprocessing as _mp

try:
    _mp.set_start_method("fork", force=True)
except (RuntimeError, ValueError):
    pass
