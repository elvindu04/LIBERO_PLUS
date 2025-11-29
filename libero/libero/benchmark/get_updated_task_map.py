import os
from pprint import pprint

task_suite = "libero_10_train"

path = os.path.join("/home/whu/LIBERO_PLUS/libero/libero/bddl_files/", task_suite)

files = os.listdir(path)
out = []
for f in files:
    out.append(f[:-5])

pprint(out)