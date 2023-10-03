import numpy as np


arr = np.loadtxt("/home/user/red_globe_2021_09_06_all_images/Outer_13to25/output_det.txt",
                 delimiter=",", dtype=str)

print(len(np.unique(arr[:, 1])))
