import numpy as np
from matplotlib import pyplot as plt

from plot.analysis_tools import linear_regression

col_num = [10 * (i + 1) for i in range(10)]
col_num_time = [55, 118.9, 192.4, 279.8, 371.9, 499.6, 578.7, 665.7, 874.1, 993.2]

# regress
w = linear_regression(np.array(col_num).reshape((len(col_num), 1)), np.array(col_num_time).reshape((len(col_num), 1)))

# plot
plt.plot(col_num, col_num_time)
plt.grid(True)
plt.xlabel("Column Number")
plt.ylabel("OPE Processing Time (sec)")
plt.title("OPE Processing Time v.s. Column Number")
plt.show()
