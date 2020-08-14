import numpy as np

from algorithm.order_preserving_encryption import OrderPreservingEncryption
from test.data_gen import DataGen

# data gen
length = 1000
data_gen = DataGen(mode=0)
input_data = data_gen.generate(length)

# parameters
max_bin_num = 10
min_bin_volume = 3
approximation_index = 1
precision = 0.99
distribution_distance_metric = "KLDivergence"

# encrypt
ope = OrderPreservingEncryption(max_bin_num, min_bin_volume, approximation_index,
                                input_data,
                                precision,
                                distribution_distance_metric)
output_data = ope.encrypt(input_data)
input_data_again = ope.decrypt(output_data)

error_vector = np.array(input_data) - np.array(input_data_again)
error = np.linalg.norm(error_vector)

pass
