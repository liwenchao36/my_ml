# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time  : 2023/7/14 21:07
# @Author: Lee Wen-tsao
# @E-mail: liwenchao36@163.com
# @Software: python
from src.utils.fitness import fitness
import numpy as np


class IntegerCategoricalPSO:
    """ Integer and categorical particle swarm optimization.

    This new PSO algorithm,which we call Integer and Categorical PSO,incorporates ideas from Estimation of
    Distribution Algorithms (EDAs)in that particles represent probability distributions rather than solution
    values, and the PSO update modifies the probability distributions.

    Attributes
    ----------
    n_particles: int
        The number of population.
    dim: int
        dimension.
    num_states: int
        The number of state.
    max_iter: int
        The number of iteration.
    verbose： bool
        Detailed display of iteration results

    """

    def __init__(self, n_particles, dim, num_states, max_iter=300, verbose=True):
        self.n_particles = n_particles
        self.dim = dim
        self.num_states = num_states
        self.max_iter = max_iter

        self.positions = np.random.rand(self.n_particles, self.dim, self.num_states)
        self.velocities = np.zeros((n_particles, dim, num_states))
        self.f_scores = np.zeros(n_particles)

        self.p_best = np.empty((self.n_particles, self.dim, self.num_states))
        self.g_best = np.empty((self.dim, self.num_states))

        self.f_p_best = np.empty(self.n_particles)
        self.f_g_best = np.inf

        self.g_best_sample = None
        self.verbose = verbose
        pass

    def initialize_population(self):
        """ Initialize Population.

        In the initialization population method, random positions are generated for each particle in the population.
        These positions are then normalized to represent probability distributions, ensuring that the sum of
        probabilities for each particle adds up to 1.

        """
        self.positions /= np.sum(self.positions, axis=1, keepdims=True)
        self.p_best = self.positions

        for i in range(self.n_particles):
            self.f_p_best[i] = self.f_scores[i]
            sample = np.argmax(self.positions[i, :, :])
            self.f_scores[i] = fitness(sample)

            if self.f_g_best > self.f_scores[i]:
                self.f_g_best = self.f_scores[i]
                self.g_best = self.positions[i, :, :]
                self.g_best_sample = sample

    def optimize(self):
        self.initialize_population()

        for j in range(self.max_iter):
            for i in range(self.n_particles):
                # Update velocity and position
                self.velocities[i] = 0.729 * self.velocities[i] + 1.49618 * (self.p_best[i] - self.positions[i]) \
                                     + 1.49618 * (self.g_best - self.positions[i])
                self.positions[i] += self.velocities[i] + self.positions[i]

                # Deal with boundaries
                self.positions[i] = np.clip(self.positions[i], 0, 1)
                self.positions[i] /= np.sum(self.positions[i], axis=1, keepdims=True)

                # Sampling and evaluation
                sample = np.argmax(self.positions[i])
                self.f_scores[i] = fitness(sample)

                # Update personal best and global best
                if self.f_p_best > self.f_scores[i]:
                    self.f_p_best = self.f_scores[i]
                    self.p_best[i] = self.positions[i]
                    # self.p_best = self.setting_best_vector(sample, self.positions[i])

                    if self.f_g_best > self.f_scores[i]:
                        self.f_g_best = self.f_scores[i]
                        self.g_best = self.setting_best_vector(sample, self.positions[i])
                        self.g_best_sample = np.argmax(self.g_best)
                pass

            if self.verbose:
                print(f"iteration{j} : f_g_best{self.f_g_best}")
                pass

    @staticmethod
    def setting_best_vector(sample, position):
        """ Setting the Best vector.

        Parameters
        ----------
        sample: ndarray
            Its distributions are sampled to create a candidate solution.
        position: ndarray
            The position for a particle in ICPSO is a set of probability distributions.

        """
        epson = 0.95
        for i in range(position.shape[1]):
            for j in range(position.shape[0]):
                if j == sample[i]:
                    position[i, j] = position[i, j] + (1 - epson) * np.sum(position[i, np.where(j != sample[i])])
                else:
                    position[i, j] = epson * position[i, j]
        return position


def test():
    pso = IntegerCategoricalPSO(4, 7, 3)


if __name__ == "__main__":
    test()
