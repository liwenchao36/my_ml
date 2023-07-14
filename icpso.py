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

        self.f_pBest = None
        self.f_gBest = np.inf

        self.gBest = None
        self.pBest = None
        pass

    @staticmethod
    def fitness(sample):
        return 0

    @staticmethod
    def setting_best_vector(sample, position):
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
        velocity = np.zeros((self.num_pop, self.dim, self.num_states))
        self.velocity = velocity

        position = np.random.rand(self.num_pop, self.dim, self.num_states)
        position /= np.sum(position, axis=1, keepdims=True)

        for i in range(self.num_pop):
            sample = np.argmax(position[i, :, :])
            self.fscore[i] = self.fitness(sample)

            self.f_pBest = self.fscore[i]
            self.pBest = position

    def predict(self):
        weight = 1
        for i in range(self.max_iter):
            weight = 0.98 * weight

            # update velocity
            self.velocity[i, :, :] = weight * self.velocity[i, :, :] + np.random.rand() * \
                                     (self.pBest - self.position[i, :, :]) + (self.gBest - self.position[i, :, :])

            # update position
            self.position[i, :, :] = self.velocity[i, :, :] + self.position[i, :, :]

            # deal boundary
            self.position[i, :, :] = np.clip(self.position[i, :, :], 0, 1)
            self.position[i, :, :] /= np.sum(self.position[i, :, :], axis=1, keepdims=True)

            # sampling
            sample = np.argmax(self.position[i, :, :])

            # evaluate
            self.fscore[i] = self.fitness(sample)

            if self.f_pBest > self.fscore[i]:
                self.f_pBest = self.fscore[i]
                self.pBest = self.setting_best_vector(sample, self.position[i, :, :])
                if self.f_gBest > self.fscore[i]:
                    self.f_gBest = self.fscore[i]
                    self.gBest = self.setting_best_vector(sample, self.position[i, :, :])
                    self.gBestSample = np.argmax(self.gBest)
        pass


def test():
    ICPSO = IntegerCategoricalPSO(4, 7, 3)
    ICPSO.initialization()


if __name__ == "__main__":
    test()
