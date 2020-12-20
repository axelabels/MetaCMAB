import numpy as np
from math import log

from tools import arr2str, softmax
import agent as Agent


class Policy(object):

    def __init__(self, b=1):
        self.b = b

        self.key = 'value'

    def __str__(self):
        return 'generic policy'

    def probabilities(self, agent, contexts):
        a = agent.value_estimates(contexts)
        return softmax(a*self.b)

    def choose(self, agent, contexts, greedy=False):

        try:
            self.pi = self.probabilities(agent, contexts)
        except:
            self.pi[:] = self.probabilities(agent, contexts)
        assert len(np.shape(self.pi)) == 1, "probabilities must be a vector, but is {} for agent {} and contexts {}".format(
            np.shape(self.pi), agent, contexts)
        if greedy:
            max_v = np.max(self.pi)
            check = np.where(self.pi == max_v)[0]
            assert len(check) > 0, (self.pi, agent.mu)
            self.pi[:] = 0
            self.pi[check] = 1 / len(check)
        action = np.searchsorted(np.cumsum(self.pi), np.random.rand(1))[0]
        assert action < agent.bandit.k, (self, agent, self.pi,  [
            e.policy.pi for e in agent.experts] if issubclass(type(agent), Agent.Collective) else None)

        for arm, r in enumerate(agent.reward_history):
            if type(r) != list:
                break
            if len(r) < 1:
                return arm
        return action


class EpsilonGreedyPolicy(Policy):
    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.key = 'value'
        self._probabilities = None

    def __str__(self):
        return 'eps'.format(self.epsilon)

    def probabilities(self, agent, contexts):
        if self._probabilities is None:
            self._probabilities = np.empty(agent.bandit.k)

        self._probabilities.fill(self.epsilon / agent.bandit.k)
        v = agent.value_estimates(contexts)
        max_v = v.max()
        check = np.where(v == max_v)[0]
        try:
            assert len(check) > 0, (str(agent), v, agent.t,
                                    agent.value_estimates(contexts))
            self._probabilities[check] += (1 - self.epsilon) / len(check)
            if len(check) == len(self._probabilities):
                self._probabilities[:] = 1/len(check)
        except:
            print("v", v)
            print(max_v)
            print(check)
            raise
        self.pi = self._probabilities
        assert 0.9 < np.sum(self.pi) <= 1.1, (v, check, max_v,
                                              self.epsilon, (1-self.epsilon)/len(check))
        if 'LinUCB' in str(agent) and agent.shared_model and agent.shared_meta:
            assert np.std(self._probabilities) == 0, (self._probabilities, v)
        return self._probabilities


class ProbabilityGreedyPolicy(Policy):
    def __init__(self, epsilon=0):
        self.epsilon = epsilon
        self.datas = []
        self.key = 'probability'

    def __str__(self):
        return 'PGP'.format(self.epsilon)

    def probabilities(self, agent, contexts):
        probabilities = np.empty(agent.bandit.k)
        probabilities.fill(self.epsilon / agent.bandit.k)
        pi = agent.probabilities(contexts)

        self.agent_pi = pi
        max_v = pi.max()
        check = np.where(np.array(pi) == max_v)[0]

        assert len(check) > 0, (agent, pi)
        probabilities[check] += (1 - self.epsilon) / len(check)

        return probabilities


class GPUCBPolicy(Policy):

    def __init__(self, beta=100):
        self.beta = beta

    def __str__(self):
        return 'GPUCB'

    def probabilities(self, agent, contexts):

        self.q = agent.ucb_values(contexts)
        self.q = self.q - np.max(self.q)

        self._probabilities = softmax(1*self.q)

        self.pi = self._probabilities
        return self._probabilities
