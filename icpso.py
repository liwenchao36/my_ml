# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time  : 2023/7/14 21:07
# @Author: Lee Wen-tsao
# @E-mail: liwenchao36@163.com
# @Software:
import numpy as np


class IntegerCategoricalPSO:
    def __init__(self, num_pop: int, dim: int, num_states: int, max_iter=300):
        self.num_pop = num_pop
        self.dim = dim
        self.num_states = num_states
        self.max_iter = max_iter
        self.fscore = np.zeros(num_pop)

        self.position = None
        self.velocity = None

        self.pBestSample = None
        self.gBestSample = None
        pass

    @staticmethod
    def fitness(self, sample):
        return 0

    def setting_best_vector(self, sample, position):
        epson = np.random.rand()
        for i in range(position.shape[1]):
            for j in range(position.shape[0]):
                if j == sample[i]:
                    position[i, j] = position[i, j] + (1 - epson) * np.sum(position[i, np.where(j != sample[i])])
                else:
                    position[i, j] = epson * position[i, j]
        return position

    def initialization(self):
        """ Initialization"""
        position = np.random.rand(self.num_pop, self.dim, self.num_states)
        position /= np.sum(position, axis=1, keepdims=True)
        self.position = position

        velocity = np.zeros((self.num_pop, self.dim, self.num_states))
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
            self.position[i, :, :] /= np.sum(self.position[i, :, :], axis=1, keepdims=True)

            # sampling
            sample = np.argmax(self.position[i, :, :])

            # evaluate
            self.fscore[i] = self.fitness(sample)

            if f_pBest > self.fscore[i]:
                f_pBest = self.fscore[i]
                pBest = self.setting_best_vector(sample, self.position[i, :, :])
                self.pBestSample = np.argmax(pBest)
                if f_gBest > self.fscore[i]:
                    f_gBest = self.fscore[i]
                    gBest = self.setting_best_vector(sample, self.position[i, :, :])
                    self.gBestSample = np.argmax(gBest)
        pass


def test():
    ICPSO = IntegerCategoricalPSO(4, 7, 3)
    ICPSO.initialization()


if __name__ == "__main__":
    test()
