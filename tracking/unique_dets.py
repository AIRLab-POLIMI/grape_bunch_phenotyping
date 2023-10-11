import numpy as np


arr = np.loadtxt("/home/user/Red_Globe_2022_90deg/red_globe_2022_90deg_all_images/Outer_13to25/sort_output_det.txt",
                 delimiter=",", dtype=str)

print(len(np.unique(arr[:, 1])))
