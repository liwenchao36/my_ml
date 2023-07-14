# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time  : 2023/7/14 21:07
# @Author: Lee Wen-tsao
# @E-mail: liwenchao36@163.com
# @Software:
import numpy as np


class IntegerCategoricalPSO:
    def __init__(self, num_pop: int, dim: int, num_values: int, iteration=300):
        self.num_pop = num_pop
        self.dim = dim
        self.num_values = num_values
        self.iteration = iteration

        self.pop = None
        self.velocity = None
        pass

    def initialization(self):
        """ Initialization"""
        pop = np.random.rand(self.num_pop, self.dim, self.num_values)
        pop /= np.sum(pop, axis=1, keepdims=True)
        self.pop = pop

        velocity = np.zeros((self.num_pop, self.dim, self.num_values))
        self.velocity = velocity

    def predict(self):
        for i in range(self.iteration):
            pass
        pass


def test():
    ICPSO = IntegerCategoricalPSO(4, 7, 3)
    ICPSO.initialization()


if __name__ == "__main__":
    test()