import numpy as np
from matplotlib import pyplot as plt

from plot.analysis_tools import linear_regression

sample_num = [5000 + 5000 * i for i in range(10)]
sample_num_time = [197.3, 371.9, 564.4, 714.3, 925.4, 1076.8, 1249.5, 1420.9, 1611.6, 1778.3]

# regress
w = linear_regression(np.array(sample_num).reshape((len(sample_num), 1)), np.array(sample_num_time).reshape((len(sample_num), 1)))

# plot
plt.plot(sample_num, sample_num_time)
plt.grid(True)
plt.xlabel("Sample Number")
plt.ylabel("OPE Processing Time (sec)")
plt.title("OPE Processing Time v.s. Sample Number")
plt.show()
