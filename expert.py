from __future__ import print_function
from tools import *
from bandit import SuperBandit
from sklearn.metrics.pairwise import *
from sklearn.gaussian_process.kernels import *

from math import ceil

from scipy.stats import gmean


EXPECTED_STD_REWARD = np.float64(.5)
EXPECTED_MEAN_REWARD = np.float64(.5)

MAX_SLICE_SIZE = 100000
RANDOM_CONFIDENCE = .5


class Agent(object):
    """
    An Agent is able to take one of a set of actions at each time step. The
    action is chosen using a strategy based on the history of prior actions
    and outcome observations.
    """

    CONFIGURATION = None

    def __init__(self, bandit, policy, name=None):
        self.bandit = bandit
        self.policy = policy
        self.confidence_noise = 0
        self.center = None

        self.t = 0

        self.cache_id = None
        self.prior_bandit = None

        self.reward_history = []
        self.context_history = []

        self.honest = False
        self.name = name
        self.cached_predictions = None
        self.cached_confidences = None
        self.cached_probabilities = None
        self.cached_votes = None
        self.confidence = np.zeros(bandit.k)
        self.mu = np.zeros(bandit.k)

    def predict_normalized(self, contexts):
        mu = np.random.uniform(size=len(contexts))
        sigma = np.zeros(len(contexts))

        return mu, sigma

    def value_estimates(self, contexts, return_std=False):

        if isinstance(contexts, int):
            assert self.is_prepared()

            self.mu, self.sigma = self.cached_predictions[contexts]
        else:
            self.mu, self.sigma = np.random.uniform(
                size=self.bandit.k), np.zeros(self.bandit.k)
        if return_std:
            return self.mu, self.sigma
        return self.mu

    def is_prepared(self, cache_id=None):
        if cache_id is None:
            return self.cache_id == self.bandit.cache_id
        return self.cache_id == cache_id

    def set_name(self, name):
        self.name = name

    def __str__(self):
        if self.name is not None:
            return self.name
        return '{}'.format(str(self.policy))

    def probabilities(self, contexts):
        return self.policy.probabilities(self, contexts)

    def prior_play(self, steps=0, bandit=None, spread=None, noise=None, exploratory=True):
        pass

    def reset(self):
        self.t = 0
        self.reward_history = []
        self.context_history = []
        self.cache_id = None

    def choose(self, contexts, greedy=False):
        return self.policy.choose(self, contexts, greedy=greedy)

    def observe(self, reward, arm, contexts):

        self.t += 1

        self.context_history.append(contexts[arm])

        self.reward_history.append(float(reward))

    def update_confidence(self, contexts=None, value_mode=False):
        if isinstance(contexts, int):
            assert self.is_prepared()
            self.confidence = self.cached_confidences[contexts]
            return self.confidence

        pred_val, uncertainty = self.value_estimates(contexts, return_std=True)

        if self.honest:
            true_v = np.clip(self.bandit.action_values, 0, 1)

            self.confidence = (np.ones(self.bandit.k) -
                               np.abs(true_v - self.mu))

        else:
            self.confidence = np.clip(1 - uncertainty, 1e-6, 1)

            self.confidence = .5 + self.confidence / 2
        if not value_mode:
            self.confidence[:] = np.product(
                self.confidence ** (1 / len(self.confidence)))

        self.confidence = np.clip(self.confidence, 1e-8, 1 - 1e-8)

        return self.confidence

    def cache_predictions(self, bandit, trials):
        recomputed = False
        if not self.is_prepared(bandit.cache_id) or len(self.cached_predictions) < trials:
            self.cached_predictions = np.array(self.predict_normalized(bandit.cached_contexts.reshape(
                (trials, bandit.k, bandit.dims))))

            if issubclass(type(bandit), SuperBandit):
                self.cached_predictions = np.moveaxis(
                    self.cached_predictions, 2, 0)
            else:
                self.cached_predictions = np.moveaxis(
                    self.cached_predictions, 0, 1)

            self.cached_predictions[:, 0] = scale(
                self.cached_predictions[:, 0])

            self.cache_id = bandit.cache_id

            self.cached_probabilities = softmax(
                self.cached_predictions[:, 0], axis=1)
            recomputed = True
        elif len(self.cached_predictions) > trials:
            self.cached_predictions = self.cached_predictions[:trials]
            self.cached_probabilities = self.cached_probabilities[:trials]

        return recomputed

    def cache_confidences(self, bandit, choice_only, value_mode, noise, scale_first, use_regret_confidence):

        values, uncertainties = self.cached_predictions[:,
                                                        0], self.cached_predictions[:, 1]

        self.cached_votes = np.zeros_like(self.cached_probabilities)
        self.cached_votes[np.arange(len(self.cached_votes)), np.argmax(
            self.cached_probabilities, axis=1)] = 1

        if choice_only:
            self.cached_probabilities = self.cached_votes

        self.expected_rewards = np.sum(
            self.cached_probabilities * bandit.cached_rewards, axis=1)
        self.total_expected_reward = np.sum(
            np.sum(self.cached_votes * bandit.cached_rewards, axis=1))
        if self.honest:
            if use_regret_confidence:
                self.cached_confidences = np.zeros_like(values)
                self.cached_confidences[:] = np.sum(
                    self.cached_probabilities * bandit.cached_rewards, axis=1)[:, np.newaxis]
                uniform_performance = bandit.expected_reward
                best_performance = np.mean(
                    np.max(bandit.cached_rewards, axis=1))

                worst_performance = np.mean(
                    np.min(bandit.cached_rewards, axis=1))
            else:  # prediction error confidence

                self.cached_confidences = 1 - \
                    (values - bandit.cached_rewards)**2

                uniform_performance = np.mean(
                    1 - np.abs(bandit.cached_rewards - bandit.expected_reward))
                best_performance = 1
                worst_performance = np.mean(
                    1 - np.abs(bandit.cached_rewards - (bandit.cached_rewards < 0).astype(float)))
                assert worst_performance

            if not value_mode:
                try:
                    self.cached_confidences[:, :] = gmean(np.clip(self.cached_confidences, 1e-6, 1), axis=1)[:,
                                                                                                             np.newaxis]
                except:
                    print(self.cached_confidences, use_regret_confidence)
                    raise
            self.cached_confidences[:] = np.mean(self.cached_confidences)

            if worst_performance < uniform_performance < best_performance:

                self.cached_confidences[:] = normalize(self.cached_confidences, offset=worst_performance,
                                                       scale=best_performance - worst_performance) ** (
                    np.log(RANDOM_CONFIDENCE) / np.log(
                        normalize(uniform_performance, offset=worst_performance,
                                  scale=best_performance - worst_performance)))

            if noise > 0:
                noisy_conf = np.random.beta(
                    1 + np.mean(self.cached_confidences) / noise,
                    1 + (1 - np.mean(self.cached_confidences)) / noise)

                self.cached_confidences[:] = noisy_conf
            else:
                np.random.uniform()
        else:
            self.cached_confidences = 1 - np.tanh(uncertainties)
            assert (0 <= self.cached_confidences).all() and (
                self.cached_confidences <= 1).all()

            if scale_first:
                self.cached_confidences = self.cached_confidences * \
                    (1 - RANDOM_CONFIDENCE) + RANDOM_CONFIDENCE
            if not value_mode:
                self.cached_confidences[:, :] = gmean(
                    np.clip(self.cached_confidences, 1e-6, 1), axis=1)[:, np.newaxis]
            if not scale_first:
                self.cached_confidences = self.cached_confidences * \
                    (1 - RANDOM_CONFIDENCE) + RANDOM_CONFIDENCE


class KernelUCB(Agent):

    MAX_SPREAD = 1
    EXPLORATORY = True
    KERNELS = [RationalQuadratic(alpha=.1), PairwiseKernel(metric='laplacian')]

    def __init__(self, bandit, policy, gamma=.1, beta=5, kernel=None):
        super().__init__(bandit, policy)

        self.rew_mu = self.rew_std = self.con_mu = self.con_std = 0

        self.gamma = gamma
        self.beta = beta

        self.set_kernel(kernel or RationalQuadratic(
            length_scale=(self.bandit.dims ** .5), alpha=0.1))

        self.reset()

    def set_kernel(self, kernel):
        self.kernel = kernel

    def reset(self):
        super().reset()

        self.learning = 1
        self.reward_history = ([[] for _ in range(self.k)])
        self.context_history = ([[] for _ in range(self.k)])
        self.k_inv = ([None for _ in range(self.k)])
        self.rew_mu = self.rew_std = self.con_mu = self.con_std = 0
        self.cache_id = None

    def choose(self, contexts, greedy=False):
        return self.policy.choose(self, contexts, greedy=greedy)

    @property
    def k(self):
        return 1 if not self.bandit_is_super else self.bandit.k

    @property
    def bandit_is_super(self):
        return issubclass(type(self.bandit), SuperBandit)

    def prior_play(self, steps=0, bandit=None, spread=None, noise=None):

        self.prior_bandit = bandit
        self.center = np.random.uniform(0, 1, size=bandit.dims)
        if KernelUCB.EXPLORATORY:
            for _ in range(steps):
                contexts = bandit.observe_contexts(self.center, spread)

                action = self.choose(contexts, greedy=True)

                reward, _ = bandit.pull(action)
                self.observe(reward, action, contexts, compute=True)
        else:
            if self.bandit_is_super:
                for _ in range(steps):

                    observed = bandit.observe_contexts(self.center, spread)
                    samp = bandit.sample().astype(float)
                    for arm in range(self.k):
                        self.context_history[arm].append(observed[arm])
                        self.reward_history[arm].append(samp[arm, 0])

                for arm in range(self.k):
                    self.k_inv[arm] = self.updated_model(
                        self.context_history[arm], self.reward_history[arm])

            else:
                self.context_history[0] += list(bandit.observe_contexts(
                    self.center, spread, steps))

                self.reward_history[0] += list(
                    bandit.sample().astype(float))

                self.k_inv[0] = self.updated_model()
        self.learning = 0

    def observe(self, reward, arm, contexts, compute=True):

        self.t += 1
        self.context_history[arm*self.bandit_is_super].append(contexts[arm])

        self.reward_history[arm*self.bandit_is_super].append(float(reward))
        if compute:
            self.k_inv[arm *
                       self.bandit_is_super] = self.updated_model(arm=arm*self.bandit_is_super)

    def probabilities(self, contexts):

        if isinstance(contexts, int):
            assert self.is_prepared(
            ), "When an int is given as context the expert should be prepared in advance"
            self._probabilities = self.cached_probabilities[contexts]
        else:
            self._probabilities = self.cached_probabilities(contexts)

        return self._probabilities

    def value_estimates(self, contexts, return_std=False, arm=None):

        if isinstance(contexts, int):
            assert self.is_prepared(
            ), "When an int is given as context the expert should be prepared in advance"

            self.mu, self.sigma = self.cached_predictions[contexts]
        else:
            self.mu, self.sigma = self.predict_normalized(contexts, arm=arm)

        if return_std:
            return self.mu, self.sigma
        return self.mu

    def ucb_values(self, contexts=None, armwise=False):
        if self.bandit_is_super:
            mus = []
            sigmas = []
            if type(contexts) is int:
                contexts = [contexts]*self.bandit.k
            for a, ctx in enumerate(contexts):
                mu, sigma = self.value_estimates(
                    ctx, return_std=True, arm=a)
                mus.append(mu[0])
                sigmas.append(sigma[0])
            mu = np.array(mus)
            sigma = np.array(sigmas)

        else:
            mu, sigma = self.value_estimates(contexts, return_std=True)

        return mu + self.learning * sigma * np.sqrt(self.beta)

    @property
    def advice_type(self):
        return "value"

    @property
    def choice_only(self):
        return False

    def updated_model(self, contexts=None, arm=None, mn=None, std=None):

        if contexts is None:
            if arm is not None:
                contexts = self.context_history[arm*self.bandit_is_super]
            else:
                contexts = self.context_history[0]

        contexts = normalize(contexts, offset=mn, scale=std, axis=0)

        try:
            return np.linalg.inv(self.kernel(contexts) + np.identity(len(contexts)) * self.gamma)
        except:
            return np.linalg.inv(self.kernel(contexts, contexts) + np.identity(len(contexts)) * self.gamma)

    def predict_normalized(self, contexts, arm=None, slice_size=None):
        shape = None
        if arm is None:
            if self.bandit_is_super:
                mus = []
                sigmas = []
                if len(np.shape(contexts)) == 2:
                    contexts = np.repeat(
                        contexts[:, np.newaxis, :], self.bandit.k, axis=1)
                elif np.shape(contexts) == (self.bandit.dims,):

                    for a in range(self.bandit.k):
                        mu, sigma = self.predict_normalized(
                            contexts[np.newaxis, :], arm=a)

                        mus.append(mu[0])
                        sigmas.append(sigma[0])
                    mus = np.array(mus)
                    sigmas = np.array(sigmas)

                    return mus, sigmas

                mus = []
                sigmas = []
                for a in range(self.bandit.k):
                    mu, sigma = self.predict_normalized(contexts[:, a], arm=a)

                    mus.append(mu)
                    sigmas.append(sigma)
                mus = np.array(mus)
                sigmas = np.array(sigmas)
                # print("mus2",mus)
                return mus, sigmas
            else:
                arm = 0
                shape = np.shape(contexts)[:-1]
                contexts = np.reshape(contexts, (-1, self.bandit.dims))
        if len(np.shape(contexts)) == 1:
            mu = np.zeros(1)+.5
            sigma = np.ones(1)
        else:
            mu = np.zeros(len(contexts)) + .5
            sigma = np.ones(len(contexts))

        if len(self.context_history[arm]) <= 1:
            return mu, sigma

        con_mu = np.mean(self.context_history[arm], axis=0)
        con_std = np.std(self.context_history[arm], axis=0)

        context_history = normalize(
            self.context_history[arm], offset=con_mu, scale=con_std)
        normalized_contexts = normalize(contexts, offset=con_mu, scale=con_std)

        reward_history = normalize(
            self.reward_history[arm], offset=EXPECTED_MEAN_REWARD, scale=EXPECTED_STD_REWARD)

        # Intermediary results below can occupy a lot of space if contexts is large, compute results by slices
        slice_size = min(MAX_SLICE_SIZE, len(normalized_contexts))
        for slice_index in range(ceil(len(normalized_contexts) / slice_size)):
            lo = slice_index * slice_size
            hi = (slice_index + 1) * slice_size
            k_x = self.kernel(normalized_contexts[lo:hi], context_history)

            k_x_Kinv = k_x.dot(self.k_inv[arm])

            mu[lo:hi] = rescale(k_x_Kinv.dot(reward_history),
                                EXPECTED_MEAN_REWARD, EXPECTED_STD_REWARD)

            sigma[lo:hi] = np.sqrt(np.maximum(0, 1 - (k_x_Kinv*k_x).sum(-1)))

        if shape is not None:
            mu = np.reshape(mu, shape)
            sigma = np.reshape(sigma, shape)

        return mu, sigma

    def __str__(self):
        if self.name is not None:
            return self.name
        return "kernel"+str(self.bandit_is_super)


class TransparentExpert(KernelUCB):
    def updated_model(self, contexts=None, mn=None, std=None):
        pass

    def predict_normalized(self, contexts, slice_size=None):
        mu = self.prior_bandit.get(contexts)
        sigma = np.zeros_like(mu)
        return mu, sigma
