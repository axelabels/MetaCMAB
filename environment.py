
from scipy.stats import gmean
import traceback
import warnings
import sys
from math import log
from agent import Collective,   MetaCMAB
from expert import KernelUCB
from tools import *
from bandit import *
import os


def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

    log = file if hasattr(file, 'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(
        message, category, filename, lineno, line))


warnings.showwarning = warn_with_traceback


def expert_runner(agent, bandit, results, trials, seed, marker=''):
    agent.bandit = bandit
    np.random.seed(seed)
    bandit.reset()

    np.random.seed(seed)
    agent.reset()

    np.random.seed(seed)
    bandit.cache_contexts(trials, seed)
    agent.bandit.observe_contexts()

    np.random.seed(seed)

    for t in (range(trials)):

        bandit.observe_contexts(cache_index=t)

        if issubclass(type(bandit), SuperBandit):
            action = agent.choose(bandit.contexts[0], greedy=True)
        else:
            action = agent.choose(bandit.contexts, greedy=True)
        sampled_rewards = bandit.sample(cache_index=t)
        arm_values = bandit.action_values
        reward = sampled_rewards[action]

        agent.observe(reward, action, bandit.contexts, compute=True)

        results[0][t] = reward

        results[2][t] = sampled_rewards[np.argmax(
            arm_values)]  # best expected arm

        results[5][t] = sampled_rewards[np.argmin(
            arm_values)]  # worst expected arm

    return results, None


def runner(agent, bandit, n_prior_max, n_bias_max, results, trials, seed,  max_distance=0, marker=''):
    if not issubclass(type(agent), Collective):
        return expert_runner(agent, bandit, results, trials, seed)
    agent.bandit = bandit
    np.random.seed(seed)
    bandit.reset()

    np.random.seed(seed)
    agent.reset()

    for e in agent.full_experts:
        e.honest = agent.use_true_confidence
        e.bandit = bandit

    np.random.seed(seed + 1)
    if not agent.experts[0].is_prepared(bandit.cache_id):

        agent.prior_play(n_bias_max, bandit,
                         max_distance=max_distance, spread=KernelUCB.MAX_SPREAD)

    np.random.seed(seed)
    bandit.cache_contexts(trials, seed)
    agent.bandit.observe_contexts()

    experts_rewards = np.zeros((len(agent.experts), trials))
    np.random.seed(seed)

    recomputed = np.zeros(len(agent.full_experts))
    for i, e in enumerate(agent.full_experts):
        e.expert_id = i
        np.random.seed(seed * (i + 1) % int(2 ** 31))

        recomputed[i] = e.cache_predictions(bandit, trials)

        np.random.seed(seed * (i + 1) % int(2 ** 31))

        e.cache_confidences(bandit, agent.choice_only, agent.value_mode, agent.confidence_noise,
                            agent.scale_first, agent.use_regret_confidence)
    order = list(enumerate(agent.full_experts))
    np.random.shuffle(order)

    expert_perfs = [e.total_expected_reward for e in agent.full_experts]
    expert_indices = sorted(np.argsort(expert_perfs)[-len(agent.experts):])
    if agent.ablate:
        agent.experts = [agent.full_experts[ie] for ie in expert_indices]
    else:
        agent.experts = agent.full_experts

    np.random.seed(seed)

    for t in range(trials):

        bandit.observe_contexts(cache_index=t)
        action = agent.choose(t)
        sampled_rewards = bandit.sample(cache_index=t)
        arm_values = bandit.action_values
        reward = sampled_rewards[action]

        expert_values = []
        for e, expert in enumerate(agent.experts):
            choice = np.argmax(expert.probabilities(t))
            experts_rewards[e][t] = sampled_rewards[choice]
            expert_values.append(arm_values[choice])

        agent.observe(reward, action, bandit.contexts)

        results[0][t] = reward

        results[2][t] = sampled_rewards[np.argmax(
            arm_values)]  # best expected arm
        # best expected expert at time t
        results[3][t] = experts_rewards[np.argmax(expert_values), t]

        results[5][t] = sampled_rewards[np.argmin(
            arm_values)]  # worst expected arm
        # best expected expert at time t
        results[6][t] = experts_rewards[np.argmin(expert_values), t]

    expert_cum_rewards = np.sum(experts_rewards, axis=1)
    # best overall expert
    results[1][:] = (experts_rewards[np.argmax(expert_cum_rewards)])
    # worst overall expert
    results[4][:] = (experts_rewards[np.argmin(expert_cum_rewards)])

    return results, experts_rewards


class Environment(object):
    def __init__(self, bandit, agents, seed=0, label='Multi-Armed Bandit', n_prior_max=0, n_bias_max=0,
                 prior_bandit=None, max_distance=0):

        self.bandit = bandit
        self.agents = agents
        self.seed = seed
        self.label = label
        self.n_prior_max = n_prior_max
        self.n_bias_max = n_bias_max
        self.prior_bandit = prior_bandit or bandit
        self.max_distance = max_distance
        self.seeds = []

    def run(self, trials=100, experiments=1,  marker=''):
        np.random.seed(self.seed)
        self.seeds = [np.random.randint(2 ** 31 - 1)
                      for _ in range(experiments)]
        results = np.zeros((7, trials, len(self.agents), experiments))
        for agent in self.agents:
            try:
                expert_rewards = np.zeros(
                    (len(agent.experts), trials, experiments))
                break
            except:
                pass
        else:
            expert_rewards = None

        for k in range(experiments):
            for i, agent in enumerate(self.agents):
                if i > 0 and issubclass(type(agent), Collective):  # reuse experts
                    agent.experts_gen = self.agents[0].experts
                    agent.experts = self.agents[0].experts

                results[:, :, i, k], partial_expert_rewards = runner(agent, self.bandit, self.n_prior_max,
                                                                     self.n_bias_max, results[:,
                                                                                              :, i, k],
                                                                     trials, int(
                                                                         self.seeds[k]),
                                                                     self.max_distance, marker=marker)
                if partial_expert_rewards is not None:
                    expert_rewards[:, :, k] = partial_expert_rewards

        return results, expert_rewards
