from math import log

import numpy as np

from perlin import perlin
from tools import normalize, get_coords, sigmoid, mse, mae, corr2

from math import atan2
from numpy import pi
from scipy.stats import wrapcauchy
from tools import *
import sys
import copy
import numpy as np
import random
MAX_ATTEMPTS_BANDIT_DISTANCE = 100

DESIRED_MEAN = 0.5


INF = float("inf")

BANDIT_DISTANCE_EPSILON = .05


class ContextualBandit():
    def __init__(self, n_arms=1, complexity=10, dims=2, reset=True, bernoulli=True):

        self.k = n_arms
        self.dims = dims
        self.complexity = complexity

        self.cached_contexts = None
        self.cached_values = None
        self.cached_rewards = None
        self.last_contexts = None
        self.cache_id = -1
        self.bernoulli = bernoulli
        if reset:
            self.reset()

    def reset(self, base=None):
        raise NotImplementedError()

    def cache_contexts(self, t, cache_id):

        if self.cached_contexts is None or len(self.cached_contexts) != t:
            self.cached_contexts = np.random.uniform(
                0, 1, size=(t, self.k, self.dims))
            self.cached_values = self.get(self.cached_contexts.reshape(
                (t * self.k, self.dims))).reshape((t, self.k))
            self.cached_rewards = self.sample(
                self.cached_values).reshape((t, self.k))
            self.cache_id = cache_id

        return self.cached_contexts

    def observe_contexts(self, center=None, spread=None, kn=None, cache_index=None):
        if cache_index is not None:
            self.contexts = self.cached_contexts[cache_index]
            self.action_values = self.cached_values[cache_index]
            self.optimal_value = np.max(self.action_values)
            return self.contexts

        if n is None:
            n = self.k
        if center is None:
            center = np.ones(self.dims) / 2
            spread = np.ones(self.dims)

        self.contexts = np.random.uniform(
            center - spread / 2, center + spread / 2, size=(n, self.dims))

        self.contexts[self.contexts > 1] = self.contexts[self.contexts > 1] - 1
        self.contexts[self.contexts < 0] = self.contexts[self.contexts < 0] + 1
        self.action_values = self.get(self.contexts)
        self.optimal_value = np.max(self.action_values)

        return self.contexts

    def get(self, contexts):
        raise NotImplementedError()

    def pull(self, action, cache_index=None):
        if cache_index is not None:
            return self.cached_rewards[cache_index][action], action == np.argmax(self.cached_values[cache_index])
        if self.bernoulli:
            return np.random.uniform() < self.action_values[action], action == np.argmax(self.action_values)
        else:
            return self.action_values[action], action == np.argmax(self.action_values)

    def sample(self, values=None, cache_index=None):
        if cache_index is not None:
            return self.cached_rewards[cache_index]
        if values is None:
            values = self.action_values
        if self.bernoulli:
            return np.random.uniform(size=np.shape(values)) < values
        else:
            return values

    def max_distance(self):
        return PerlinBandit.DISTANCE_METRIC(self.value_landscape < .5,
                                            self.value_landscape)   # np.mean(np.abs(((1-self.value_landscape) - self.value_landscape)**2))

    @property
    def value_landscape(self):
        if self._value_landscape is None:
            grid_data = get_coords(self.dims, self.complexity, self.precision)
            self._value_landscape = self.get(grid_data, override=True)
        return self._value_landscape

    def correlation(self, other):
        return corr2(other.value_landscape, self.value_landscape)

    def distance(self, other):
        return PerlinBandit.DISTANCE_METRIC(other.value_landscape,
                                            self.value_landscape)  # np.mean(np.abs((other.value_landscape - self.value_landscape)**2))

    @property
    def expected_reward(self):
        if self.cached_contexts is None:
            return np.mean(self.grid_data)
        return np.mean(self.cached_rewards)


class SuperBandit():
    def __init__(self, n_arms=1, complexity=2, precision=200, reset=True, invert=False, smooth_function=None):
        self.dims = 2
        self.k = n_arms
        self.cache_id = -1
        self.bernoulli = True
        self.precision = precision
        self.invert = invert
        self.smooth_function = smooth_function

        self.cached_contexts = None
        self.cached_values = None
        self.cached_rewards = None
        self.last_contexts = None

        self.complexity = complexity
        self._value_landscapes = None
        self.sub_bandits = [PerlinBandit(complexity=complexity,
                                         precision=precision,
                                         smooth_function=smooth_function) for _ in range(n_arms)]
        if reset:
            self.reset()

    @property
    def value_landscapes(self):
        if self._value_landscapes is None:
            grid_data = get_coords(self.dims, self.complexity, self.precision)
            self._value_landscapes = np.array(
                [s.get(grid_data, override=True) for s in self.sub_bandits])

        return self._value_landscapes

    def max_distance(self):
        return np.mean([s.max_distance() for s in self.sub_bandits])

    def correlation(self, other):
        return np.mean([corr2(o.value_landscape, s.value_landscape) for o, s in zip(other.sub_bandits, self.sub_bandits)])

    def distance(self, other):
        return np.mean([PerlinBandit.DISTANCE_METRIC(o.value_landscape, s.value_landscape) for o, s in zip(other.sub_bandits, self.sub_bandits)])

    @property
    def grid_data(self):
        return self.value_landscapes

    @property
    def expected_reward(self):
        if self.cached_contexts is None:
            return np.mean(self.grid_data)
        return np.mean(self.cached_rewards)

    def summary(self):
        return ", ".join([s.summary() for s in self.sub_bandits])

    @staticmethod
    def from_bandit(bandit, desired_distance, enforce_distance=False):
        prior_bandit = SuperBandit(bandit.k, bandit.complexity, bandit.precision, reset=False,
                                   smooth_function=bandit.smooth_function)
        for i in range(len(bandit.sub_bandits)):
            prior_bandit.sub_bandits[i] = bandit.sub_bandits[i].from_bandit(
                bandit.sub_bandits[i], desired_distance=desired_distance)
        prior_bandit.value_landscapes
        return prior_bandit

    def reset(self, angles=None, p=None, noise=0, source_bandit=None):
        [s.reset(angles if angles is None else angles[i], p if p is None else p[i], noise if noise == 0 else noise[i],
                 source_bandit if source_bandit is None else source_bandit[i]) for i, s in enumerate(self.sub_bandits)]
        self._value_landscapes = None
        self.cached_contexts = None
        a = self.value_landscapes

    def get(self, contexts, override=True):
        if len(np.shape(contexts)) != 3:
            if len(contexts) != self.k:
                contexts = np.repeat(
                    contexts[:, np.newaxis, :], self.k, axis=1)
            else:
                contexts = np.array(contexts).reshape((1, self.k, self.dims))
        assert np.shape(contexts)[1] == len(self.sub_bandits)

        return np.array([s.get(contexts[:, i], override) for i, s in enumerate(self.sub_bandits)])

    def observe_contexts(self, center=None, spread=None, n=None, cache_index=None):
        if cache_index is not None:
            self.contexts = self.cached_contexts[cache_index]
            self.action_values = self.cached_values[cache_index]
            self.optimal_value = np.max(self.action_values)
            return self.contexts

        if n is None:
            n = self.k
        if center is None:
            center = np.ones(self.dims) / 2
            spread = np.ones(self.dims)

        self.contexts = np.repeat(np.random.uniform(
            center - spread / 2, center + spread / 2, size=(1, self.dims)), n, axis=0)
        assert np.shape(self.contexts) == (n, self.dims)

        self.contexts[self.contexts > 1] = self.contexts[self.contexts > 1] - 1
        self.contexts[self.contexts < 0] = self.contexts[self.contexts < 0] + 1
        self.action_values = self.get(self.contexts)
        self.optimal_value = np.max(self.action_values)

        return self.contexts

    def cache_contexts(self, t, cache_id):

        if self.cached_contexts is None or len(self.cached_contexts) != t:
            self.cached_contexts = np.repeat(np.random.uniform(
                0, 1, size=(t, 1, self.dims)), self.k, axis=1)

            self.cached_values = self.get(self.cached_contexts).T
            assert np.shape(self.cached_values) == (t, self.k)
            self.cached_rewards = self.sample(self.cached_values)
            # print("cached:",np.shape(self.cached_rewards))
            assert np.shape(self.cached_rewards) == (t, self.k)
            self.cache_id = cache_id

        return self.cached_contexts

    def pull(self, action, cache_index=None):
        if cache_index is not None:
            return self.cached_rewards[cache_index][action], action == np.argmax(self.cached_values[cache_index])
        if self.bernoulli:
            return np.random.uniform() < self.action_values[action], action == np.argmax(self.action_values)
        else:
            return self.action_values[action], action == np.argmax(self.action_values)

    def sample(self, values=None, cache_index=None):
        if cache_index is not None:
            return self.cached_rewards[cache_index]
        if values is None:
            values = self.action_values
        if self.bernoulli:
            return np.random.uniform(size=np.shape(values)) < values
        else:
            return values


class PerlinBandit(ContextualBandit):
    DISTANCE_METRIC = mse

    def __init__(self, n_arms=1, complexity=2, precision=200, reset=True,  smooth_function=None):
        self.precision = precision
        self.grid_data = None
        self._value_landscape = None
        self.smooth_function = smooth_function
        super().__init__(n_arms, complexity, reset=reset)

    def summary(self):
        return str((np.mean(self.grid_data), "angles", self.angles[:3], "p", self.p[:3]))

    def max_distance(self):
        return PerlinBandit.DISTANCE_METRIC(1 - self.value_landscape,
                                            self.value_landscape)

    @staticmethod
    def from_bandit(bandit, desired_distance):

        best_bandit = None
        best_goal = INF

        for attempts in range(MAX_ATTEMPTS_BANDIT_DISTANCE):
            prior_bandit = PerlinBandit(bandit.k, bandit.complexity, bandit.precision, reset=False,
                                        smooth_function=bandit.smooth_function)

            angles = np.copy(bandit.angles)

            if desired_distance > .5:
                angles = angles + np.pi
                angles[angles > np.pi] = angles[angles > np.pi] - 2 * np.pi
                desired_distance = 1 - desired_distance
            prior_bandit.reset(angles, bandit.p, noise=desired_distance*2)

            max_distance = bandit.max_distance()
            scaled_distance = min(1, bandit.distance(
                prior_bandit) / max_distance)
            goal = np.abs(scaled_distance - desired_distance)

            if goal < best_goal:
                best_bandit, best_goal = prior_bandit, goal

            if goal <= BANDIT_DISTANCE_EPSILON:
                break
        else:
            prior_bandit = best_bandit
        return prior_bandit

    def reset(self, angles=None, p=None, noise=0, source_bandit=None):
        lin = np.linspace(0, self.complexity,
                          self.precision + 1, endpoint=True)
        x, y = np.meshgrid(lin, lin)

        self.grid_data, self.angles, self.p, self.unsmoothed = perlin(
            x, y, angles=angles, pre_p=p, noise=noise)

        if self.smooth_function == 'sigmoid':
            self.grid_data = sigmoid(1 * self.grid_data)
        if self.smooth_function == 'strongsigmoid':
            self.grid_data = sigmoid(10 * self.grid_data)
        self.grid_data = normalize(self.grid_data, offset=np.min(
            self.grid_data), scale=np.ptp(self.grid_data))

        self._value_landscape = None
        self.cached_contexts = None

    def get(self, contexts, override=True):
        if np.array_equal(contexts, self.last_contexts) and not override:
            return self.last_values

        indices = (np.array(contexts) * self.precision).astype(int).T
        # print("indices:",indices)
        try:
            data = self.grid_data[tuple(indices)]
        except:
            print(contexts, indices)
            raise
        self.last_values = data
        self.last_contexts = contexts

        return data
