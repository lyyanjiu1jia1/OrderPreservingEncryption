import random

import numpy as np


class DataGen(object):
    def __init__(self, mode=0):
        """

        :param mode: 0 for standard normally distributed array
        """
        self.mode = mode

    def generate(self, length):
        """

        :param length: int
        :return:
        """
        if self.mode == 0:
            # all gaussian
            random_list = np.random.normal(size=length)
            return random_list.tolist()
        elif self.mode == 1:
            # all bernoulli
            random_list = []
            for i in range(length):
                random_list.append(1.0 if random.random() < 0.1 else 0.0)
            return random_list
        elif self.mode == 2:
            # gaussian-bernoulli mixed
            random_list = np.random.normal(size=length // 2).tolist()
            for i in range(length - length // 2):
                random_list.append(1.0 if random.random() < 0.1 else 0.0)
            return random_list
        elif self.mode == 3:
            # fixed and all zeros
            random_list = [0 for _ in range(length)]
            return random_list
        else:
            raise TypeError("mode not supported")
