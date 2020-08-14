import random
import sys

import numpy as np

ABSOLUTE_MIN = sys.float_info.min
ABSOLUTE_MAX = sys.float_info.max
ZERO = 0.0


class OrderPreservingEncryption(object):
    def __init__(self, max_bin_num, min_bin_volume, approximation_index,
                 input_data,
                 precision=0.99,
                 distribution_distance_metric="KLDivergence",
                 exponent_set=(1, 3, 5, -1, -3, 5),
                 ciphertext_magnitude_level=1):
        """

        :param min_bin_volume: the least number of samples in each bin
        :param max_bin_num: maximum bin number
        :param approximation_index: fall in [0, 1],
                                    Z quantile sub-bins will be created to discretize the continuous uniform
                                    distribution used in gain computation, where Z = 10 ** (approximation_index + 1)
        :param input_data:
        :param precision: stop when a target distribution is close enough to the delta distribution
        :param distribution_distance_metric: distance measurement on how close a target distribution is to the uniform
                distribution, supports "KLDivergence" and "Total Variation Difference"
        :param exponent_set: the set of potential exponents from which one will be uniformly chosen at random
        :param ciphertext_magnitude_level: falls in (0, 1]
        """
        self.max_bin_num = max_bin_num
        self.min_bin_volume = min_bin_volume

        self.approximation_index = approximation_index
        self.sub_bin_num = 10 ** (self.approximation_index + 1)  # falls in [10, 100]
        self.gain_upper_bound = np.log(self.sub_bin_num)  # occurs when a bin's data are delta distributed

        self.precision = precision

        self.distribution_distance_metric = distribution_distance_metric

        self.exponent_set = exponent_set

        self.ciphertext_magnitude_level = ciphertext_magnitude_level

        self.plain_bin_split_points = self._plaintext_bin_growth(input_data)                 # List[double]
        self.cipher_bin_split_points = self._ciphertext_bin_construction(input_data)         # List[double]
        self.transform_params = self._transformation()                                       # List[(slope, intercept)]

    def encrypt(self, input_data):
        output_data = []
        for val in input_data:
            bin_index = OrderPreservingEncryption.binary_search(val, self.plain_bin_split_points)
            slope = self.transform_params[bin_index][0]
            intercept = self.transform_params[bin_index][1]
            output_val = slope * val + intercept
            output_data.append(output_val)
        return output_data

    def decrypt(self, output_data):
        input_data = []
        for output_val in output_data:
            bin_index = OrderPreservingEncryption.binary_search(output_val, self.cipher_bin_split_points)
            slope = self.transform_params[bin_index][0]
            intercept = self.transform_params[bin_index][1]
            val = (output_val - intercept) / slope
            input_data.append(val)
        return input_data

    def _transformation(self):
        transform_params = []
        for bin_index in range(len(self.plain_bin_split_points) - 1):
            try:
                slope = (self.cipher_bin_split_points[bin_index + 1] - self.cipher_bin_split_points[bin_index]) / \
                        (self.plain_bin_split_points[bin_index + 1] - self.plain_bin_split_points[bin_index])
            except ZeroDivisionError:
                slope = random.random()
            intercept = self.cipher_bin_split_points[bin_index] - slope * self.plain_bin_split_points[bin_index]
            transform_params.append((slope, intercept))
        return transform_params

    def _ciphertext_bin_construction(self, input_data):
        # get bin number
        bin_num = len(self.plain_bin_split_points) - 1

        # find min and max
        plain_min = min(input_data)
        plain_max = max(input_data)
        plain_max_absolute = max((np.abs(plain_min), np.abs(plain_max)))

        # generate exponent
        exponent = self._generate_exponent()

        # compute normal distribution parameters
        normal_mean = self.ciphertext_magnitude_level / plain_max_absolute
        normal_std_dev = normal_mean / 0.4

        # generate cipherspace bounds
        cipher_min, cipher_max = self._generate_cipherspace_bounds(
            normal_mean, normal_std_dev, plain_min, plain_max, exponent)

        # compute quantile split points
        bin_split_points = self._compute_quantile_split_points(cipher_min, cipher_max, bin_num)

        return bin_split_points

    def _compute_quantile_split_points(self, cipher_min, cipher_max, bin_num):
        interval = (cipher_max - cipher_min) / bin_num
        bin_split_points = [cipher_min]
        for i in range(bin_num):
            bin_split_points.append(bin_split_points[i] + interval)
        return bin_split_points

    def _generate_cipherspace_bounds(self, normal_mean, normal_std_dev, plain_min, plain_max, exponent):
        slope = np.random.normal(normal_mean, normal_std_dev)
        intercept = np.random.normal(1.5 * normal_std_dev, 1)
        cipher_bounds = (random.random() if plain_min == 0 else slope * plain_min ** exponent + intercept,
                         random.random() if plain_max == 0 else slope * plain_max ** exponent + intercept)
        return min(cipher_bounds), max(cipher_bounds)

    def _generate_exponent(self):
        random_index = random.randint(0, len(self.exponent_set) - 1)
        return self.exponent_set[random_index]

    def _plaintext_bin_growth(self, input_data):
        """

        :param input_data: List[double] or ndarray
        :return:
        """
        # find min and max
        min_value = min(input_data)
        max_value = max(input_data)

        # compute epsilon
        epsilon = 0.1 * random.random() * (max_value - min_value)

        # init cache
        bin_split_points = self._init_bin_split_points(min_value, max_value, epsilon)   # List[double]
        bin_num_info = self._init_bin_num_info(bin_split_points)        # int
        data_with_bin_assignment = self._init_data_with_bin_assignment(
            input_data, bin_split_points)                               # [[val, bin_index]]
        bin_volume_info = self._init_bin_volume_info(data_with_bin_assignment)  # List[int]
        bin_convergence_info = self._init_bin_convergence_info(bin_volume_info)
        bin_gain_info = []
        new_split_bin_indices = None

        # recursively grow bins
        partition_count = 1                             # count the number of partitions performed on all bins
        while not bin_convergence_info:
            partition_count += 1
            max_gain_value = ZERO
            max_gain_bin_index = None

            for bin_index in range(bin_num_info):
                if bin_volume_info[bin_index] > self.min_bin_volume:
                    if new_split_bin_indices is None:
                        # init all gains
                        gain = self._compute_gain(data_with_bin_assignment, bin_index, partition_count)
                        bin_gain_info.append(gain)
                    elif bin_index in new_split_bin_indices:
                        # compute the gain of the bins newly arising
                        gain = self._compute_gain(data_with_bin_assignment, bin_index, partition_count)
                        bin_gain_info[bin_index] = gain
                    else:
                        pass        # for those bins unchanged, unnecessary to compute their gains
                else:
                    if new_split_bin_indices is None:
                        # only in effect when initializing bin_gain_info
                        bin_gain_info.append(ZERO)
                    else:
                        # bin already convergent due to insufficient samples
                        bin_gain_info[bin_index] = ZERO

                if max_gain_value < np.abs(bin_gain_info[bin_index]) < self.precision * self.gain_upper_bound:
                    max_gain_value = np.abs(bin_gain_info[bin_index])
                    max_gain_bin_index = bin_index
            print("iteration = {}".format(partition_count))
            print("gain info = {}".format(bin_gain_info))
            bin_num_info, bin_convergence_info = self._update_all_bin_infos(
                data_with_bin_assignment,
                bin_split_points,
                bin_volume_info,
                bin_gain_info,
                max_gain_bin_index)
            new_split_bin_indices = (max_gain_bin_index, max_gain_bin_index + 1)

        return bin_split_points

    def _update_all_bin_infos(self,
                              data_with_bin_assignment,
                              bin_split_points,
                              bin_volume_info,
                              bin_gain_info,
                              max_gain_bin_index):
        if max_gain_bin_index is None:
            bin_num_info = self._update_bin_num_info(bin_split_points)
            bin_convergence_info = True
            return bin_num_info, bin_convergence_info
        new_split_point = self._update_bin_split_points(bin_split_points, max_gain_bin_index)
        bin_num_info = self._update_bin_num_info(bin_split_points)
        self._update_sample_assignment(data_with_bin_assignment, max_gain_bin_index, new_split_point)
        self._update_bin_volume_info(data_with_bin_assignment, bin_volume_info, max_gain_bin_index)
        bin_convergence_info = self._update_bin_convergence_info(bin_volume_info)
        self._update_bin_gain_info(bin_gain_info, max_gain_bin_index)
        return bin_num_info, bin_convergence_info

    def _update_bin_gain_info(self, bin_gain_info, max_gain_bin_index):
        bin_gain_info[max_gain_bin_index] = None
        bin_gain_info.insert(max_gain_bin_index, None)

    def _update_bin_convergence_info(self, bin_volume_info):
        if len(bin_volume_info) > self.max_bin_num:
            return True
        for volume in bin_volume_info:
            if volume > self.min_bin_volume:
                return False
        return True

    def _update_bin_volume_info(self, data_with_bin_assignment, bin_volume_info, max_gain_bin_index):
        new_volume = 0
        for _, val_bin_index in data_with_bin_assignment:
            if val_bin_index == max_gain_bin_index:
                new_volume += 1
        next_volume = bin_volume_info[max_gain_bin_index] - new_volume

        bin_volume_info[max_gain_bin_index] = next_volume
        bin_volume_info.insert(max_gain_bin_index, new_volume)

    def _update_sample_assignment(self, data_with_bin_assignment, max_gain_bin_index, new_split_point):
        for i in range(len(data_with_bin_assignment)):
            val = data_with_bin_assignment[i][0]
            val_bin_index = data_with_bin_assignment[i][1]
            if val_bin_index == max_gain_bin_index:
                if val >= new_split_point:
                    data_with_bin_assignment[i][1] += 1
            elif val_bin_index > max_gain_bin_index:
                data_with_bin_assignment[i][1] += 1

    def _update_bin_num_info(self, bin_split_points):
        return len(bin_split_points) - 1

    def _update_bin_split_points(self, bin_split_points, max_gain_bin_index):
        new_split_point = (bin_split_points[max_gain_bin_index] + bin_split_points[max_gain_bin_index + 1]) / 2
        bin_split_points.insert(max_gain_bin_index + 1, new_split_point)
        return new_split_point

    def _compute_gain(self, data_with_bin_assignment, bin_index, partition_count):
        # compute bin max and min
        bin_min_value = ABSOLUTE_MAX
        bin_max_value = ABSOLUTE_MIN
        for val, val_bin_index in data_with_bin_assignment:
            if val_bin_index == bin_index:
                if val < bin_min_value:
                    bin_min_value = val
                if val > bin_max_value:
                    bin_max_value = val

        # parameter setup
        interval = (bin_max_value - bin_min_value) / self.sub_bin_num
        if interval == 0:
            return self.gain_upper_bound

        # discretize the distribution of the input data
        sub_bin_cache = {}          # {sub_bin_index: count}
        for sub_bin_index in range(self.sub_bin_num):       # init
            sub_bin_cache[sub_bin_index] = 0
        for val, val_bin_index in data_with_bin_assignment:
            if val_bin_index == bin_index:
                sub_bin_index = min(self.sub_bin_num - 1,
                                    int((val - bin_min_value) / interval))
                sub_bin_cache[sub_bin_index] += 1

        # introduce the uniform distribution
        uniform_probability = 1 / self.sub_bin_num

        # compute the gain
        total_sample_num = sum(sub_bin_cache.values())
        gain = 0
        if self.distribution_distance_metric == "KLDivergence":
            for sub_bin_sample_num in sub_bin_cache.values():
                if sub_bin_sample_num != 0:
                    sub_bin_probability = sub_bin_sample_num / total_sample_num
                    gain += sub_bin_probability * np.log(sub_bin_probability / uniform_probability)
        elif self.distribution_distance_metric == "TotalVariationDifference":
            for sub_bin_sample_num in sub_bin_cache.values():
                sub_bin_probability = sub_bin_sample_num / total_sample_num
                gain += np.abs(sub_bin_probability - uniform_probability)
            gain = 0.5 * gain
        else:
            raise TypeError("distance metric not supported")

        return gain * 0.5 ** partition_count

    def _init_bin_convergence_info(self, bin_volume_info):
        if len(bin_volume_info) - 1 > self.max_bin_num:
            return True
        if len(bin_volume_info) < self.max_bin_num:
            for volume in bin_volume_info:
                if volume > self.min_bin_volume:
                    return False
        return True

    def _init_bin_volume_info(self, data_with_bin_assignment):
        bin_volume_info = [0 for _ in range(2)]
        for val, bin_index in data_with_bin_assignment:
            bin_volume_info[bin_index] += 1
        return bin_volume_info

    def _init_data_with_bin_assignment(self, input_data, bin_split_points):
        data_with_bin_assignment = []
        for val in input_data:
            bin_index = OrderPreservingEncryption.binary_search(val, bin_split_points)
            data_with_bin_assignment.append([val, bin_index])
        return data_with_bin_assignment

    def _init_bin_num_info(self, bin_split_points):
        return len(bin_split_points) - 1

    def _init_bin_split_points(self, min_value, max_value, epsilon):
        bin_split_points = [min_value - epsilon, (max_value + min_value) / 2, max_value]
        return bin_split_points

    @staticmethod
    def binary_search(val, arr):
        """

        :param val:
        :param arr:
        :return: the index at which the val should be inserted to the sorted array
        """
        low = 0
        high = len(arr) - 1
        while high - low > 1:
            mid = (high + low) // 2
            if val >= arr[mid]:
                low = mid
            else:
                high = mid
        return low
