# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time  : 2023/7/14 21:07
# @Author: Lee Wen-tsao
# @E-mail: liwenchao36@163.com
# @Software:
import numpy as np


class IntegerCategoricalPSO:
    def __init__(self, pop: int, dim: int, num_states: int, max_iter=300):
        self.pop = pop
        self.dim = dim
        self.num_states = num_states
        self.max_iter = max_iter

        self.position = None
        self.velocity = None
        pass

    def initialization(self):
        """ Initialization"""
        position = np.random.rand(self.pop, self.dim, self.num_states)
        position /= np.sum(position, axis=1, keepdims=True)
        self.position = position

        velocity = np.zeros((self.pop, self.dim, self.num_states))
        self.velocity = velocity

    def predict(self):
        weight = 1
        for i in range(self.max_iter):
            weight = 0.98 * weight

            # update velocity
            self.velocity[i, :, :] = weight * self.velocity[i, :, :] + np.random.rand() * \
                                     (pBest - self.position[i, :, :]) + (gBest - self.position[i, :, :])

            # update position
            self.position[i, :, :] = self.velocity[i, :, :] + self.position[i, :, :]

            # deal boundary
            self.position[i, :, :] = np.clip(self.position[i, :, :], 0, 1)

            pass
        pass


def test():
    ICPSO = IntegerCategoricalPSO(4, 7, 3)
    ICPSO.initialization()


if __name__ == "__main__":
    test()