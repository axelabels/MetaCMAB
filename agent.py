
from expert import *

EXPECTED_AVG_REWARD = .5
RANDOM_CONFIDENCE = .5


class Collective(Agent):
    def __init__(self, bandit, policy, experts_gen, choice_only=False, value_mode=None, name=None, confidence_type="",
                 confidence_noise=None, ablate=False):
        super(Collective, self).__init__(bandit, policy)

        self.ablate = ablate
        self.confidence_noise = confidence_noise
        self.confidence_type = confidence_type
        self.enforce_distance = True
        self.spread_distance = False
        self.experts_gen = experts_gen
        self.full_experts = self.experts_gen if not callable(
            self.experts_gen) else self.experts_gen()
        self.experts = self.full_experts if not self.ablate else self.full_experts[::2]

        self.choice_only = choice_only
        self.value_mode = value_mode

        self.bandit = bandit
        self.k = self.bandit.k
        self.n = len(self.experts)
        self.name = name
        self.advice = np.zeros((len(self.experts), self.bandit.k))
        self._value_estimates = np.zeros(self.k)
        self._probabilities = np.zeros(self.k)
        self.confidences = np.ones((len(self.experts), self.bandit.k))

    @property
    def advice_type(self):
        return 'decision' if self.choice_only else 'value' if self.value_mode else 'prob'

    @property
    def use_true_confidence(self):
        return 'alg' not in self.confidence_type

    @property
    def ignore_confidence(self):
        return 'none' in self.confidence_type

    @property
    def use_regret_confidence(self):
        return 'error' not in self.confidence_type

    @property
    def scale_first(self):
        return "post" not in self.confidence_type

    @property
    def info_str(self):

        info_str = ""
        if self.ignore_confidence:
            info_str += "nocon"
            return info_str

        if not self.scale_first:
            info_str += "l"
        if not self.use_true_confidence:
            info_str += "fcon"

        if self.value_mode and self.use_regret_confidence and self.use_true_confidence:
            info_str += "valalt"

        return info_str

    @property
    def base_str(self):
        return str(self.policy) + "Mj"

    @property
    def ablate_str(self):
        return "_top" if self.ablate else ""

    def __str__(self):
        if self.name is not None:
            if hasattr(self, 'confidence_type'):
                assert self.name == self.base_str + self.info_str, (
                    self.name, self.base_str + self.info_str, self.confidence_type)
            return self.name + self.ablate_str
        return self.base_str + self.info_str+self.ablate_str

    def set_name(self, name):
        self.name = name

    def observe(self, reward, arm, contexts):
        self.last_action = arm
        self.notify_experts(reward, arm, contexts)
        self.t += 1

    def prior_play(self, bias_steps, base_bandit=None, max_distance=0, spread=0):

        spreadalt = Agent.CONFIGURATION == 'spreadalt'
        clusterspreadalt = Agent.CONFIGURATION == 'clusterspreadalt'

        for i, e in list(enumerate(self.full_experts)):
            e.index = i

            window_width = min(1-max_distance, max_distance)
            if spreadalt:
                agent_pos = i / (len(self.full_experts) - 1)
                desired_distance = agent_pos*2*window_width + max_distance - window_width
            elif clusterspreadalt:
                cluster_id = i//(len(self.full_experts)//2)

                desired_distance = max_distance + \
                    (cluster_id*window_width) - (1-cluster_id)*window_width
            else:
                desired_distance = max_distance

            assert 0 <= desired_distance <= 1, (
                desired_distance, max_distance, Agent.CONFIGURATION)
            if clusterspreadalt:

                if i == 0 or i == len(self.full_experts)//2:
                    cluster_bandit = base_bandit.from_bandit(
                        base_bandit, desired_distance=desired_distance, enforce_distance=True)

                prior_bandit = cluster_bandit.from_bandit(
                    cluster_bandit, desired_distance=0.05, enforce_distance=True)
            else:
                prior_bandit = base_bandit.from_bandit(
                    base_bandit, desired_distance=desired_distance, enforce_distance=self.enforce_distance)

            e.prior_play(steps=bias_steps, bandit=prior_bandit, spread=spread)

    def update_advice(self, contexts):
        if self.value_mode:
            self.advice = np.array([e.value_estimates(contexts)
                                    for e in self.experts])
        elif self.choice_only:
            self.advice.fill(0)
            choices = [np.argmax(e.probabilities(contexts))
                       for e in self.experts]
            self.advice[np.arange(len(self.experts)), choices] = 1
        else:
            self.advice = np.array([e.probabilities(contexts)
                                    for e in self.experts])
        self.sync_confidences(contexts)

    def sync_confidences(self, contexts):
        for i, e in enumerate(self.experts):
            if self.ignore_confidence:
                e.confidence[:] = 1
                self.confidences[i][:] = 1
            else:
                self.confidences[i] = e.update_confidence(
                    contexts, self.value_mode)

    def probabilities(self, contexts):

        votes = [e.choose(contexts) for e in self.experts]

        self._probabilities = np.bincount(
            votes, minlength=self.k) / len(self.experts)

        assert 0 <= np.sum(self._probabilities) <= 1

        return self._probabilities

    def notify_experts(self, reward, arm, contexts):

        for ex in self.experts:
            ex.t += 1

    def reset(self):
        super().reset()

        self.full_experts = self.experts_gen if not callable(
            self.experts_gen) else self.experts_gen()

        for e in self.full_experts:
            if not e.is_prepared(self.bandit.cache_id):
                e.reset()

    def value_estimates(self, contexts):
        return self.probabilities(contexts)


class MetaMAB(Collective):

    def __init__(self, bandit, policy, experts, choice_only=False, value_mode=True,
                 name=None, confidence_type="", confidence_noise=None, ablate=False, add_random=False):

        super().__init__(bandit, policy,
                         experts, choice_only, name=name, confidence_type=confidence_type,
                         confidence_noise=confidence_noise, ablate=ablate)
        self.crop = 1
        self.gamma = None
        self.prefix = 'W'
        self.add_random = add_random
        self.value_mode = value_mode
        self.initialize_w()

    def reset(self):
        super().reset()
        self.initialize_w()

    def initialize_w(self):
        self.reward_history = []
        self.context_history = []
        self.chosen_expert = -1
        self.betas = np.ones(len(self.experts)+self.add_random)
        self.alphas = np.ones(len(self.experts)+self.add_random)
        self.chosen_expert = np.random.randint(
            len(self.experts)+self.add_random)

    def get_weights(self, contexts):
        if self.add_random and len(self.confidences) == len(self.experts):
            random_confidence = np.zeros(
                self.bandit.k)+(1 if self.ignore_confidence else RANDOM_CONFIDENCE)
            self.confidences = np.vstack((self.confidences, random_confidence))
        if self.ignore_confidence:
            self.confidences[:] = RANDOM_CONFIDENCE
        conf_weight = 0 if self.ignore_confidence else 100
        conf_alphas = self.confidences*conf_weight + self.alphas[:, None]
        conf_betas = (1 - self.confidences)*conf_weight + self.betas[:, None]

        expert_values = np.random.beta(conf_alphas, conf_betas)
        expert_values[:, :] = expert_values[:, 0:1]

        w = np.zeros_like(self.confidences)

        self.chosen_expert = randargmax(expert_values, axis=0)

        self.chosen_expert[:] = self.chosen_expert[0]
        w[self.chosen_expert, :] = 1

        return w

    def probabilities(self, contexts):
        self.update_advice(contexts)

        if self.add_random and len(self.advice) == len(self.experts):
            random_advice = np.random.uniform(size=self.bandit.k)
            random_advice /= np.sum(random_advice)
            self.advice = np.vstack([self.advice, random_advice])

        self._probabilities = np.sum(
            self.get_weights(contexts) * self.advice, axis=0)

        return self._probabilities

    def value_estimates(self, contexts):
        self.update_advice(contexts)

        if self.add_random:
            random_advice = np.random.uniform(size=self.bandit.k)
            self.advice = np.vstack([self.advice, random_advice])

        self._value_estimates = np.sum(
            self.get_weights(contexts) * (self.advice - EXPECTED_AVG_REWARD), axis=0)

        return self._value_estimates

    def observe(self, reward, arm, contexts):
        self.last_action = arm

        self.alphas[self.chosen_expert] += reward
        self.betas[self.chosen_expert] += 1 - reward

        self.notify_experts(reward, arm, contexts)

        self.t += 1

    @property
    def base_str(self):
        return str(self.policy) + ("V" if True or not self.ignore_confidence else "") + "TS" + str(2) + ("randddd" if self.add_random else "")


class Average(MetaMAB):
    def __init__(self, bandit, policy, experts, choice_only=False, name=None, value_mode=False, confidence_type="",
                 confidence_noise=None, ablate=False):
        super(Average, self).__init__(bandit, policy,
                                      experts, choice_only=choice_only, name=name,
                                      value_mode=value_mode, confidence_type=confidence_type,
                                      confidence_noise=confidence_noise, ablate=ablate)

    @property
    def base_str(self):
        return ("Av" if self.value_mode else 'Mj')

    def observe(self, reward, arm, contexts):
        self.last_action = arm

        self.notify_experts(reward, arm, contexts)

        self.t += 1

    def get_weights(self, contexts):
        w = np.clip(self.confidences, 1e-6, 1 - 1e-6)

        w = np.log(w / (1 - w))
        if self.value_mode:
            # set arm weight to 1 if all experts have the same confidence
            w[:, np.std(w, axis=0) == 0] = 1
            w /= np.sum(np.abs(w), axis=0)

        return w


class MetaCMAB(MetaMAB):
    def __init__(self, bandit, policy, experts, choice_only=False, beta=1,  name=None,
                 value_mode=False, add_b=True, confidence_type="", confidence_noise=None, ablate=False, alpha=1, early=False, shared_model=False, shared_meta=False):

        self.add_b = add_b
        super().__init__(bandit, policy,
                         experts, choice_only=choice_only, name=name, confidence_type=confidence_type,
                         confidence_noise=confidence_noise, ablate=ablate)

        self.value_mode = value_mode
        self.early = early
        self._model = None
        self.alpha = alpha
        self.meta_contexts = None
        self.beta = beta
        self.context_history = [[] for _ in range(self.k)]
        self.reward_history = [[] for _ in range(self.k)]
        self.shared_model = shared_model
        self.shared_meta = shared_meta
        self.shared = issubclass(type(bandit), SuperBandit)
        self.k = 1 if self.shared_model else self.bandit.k

    @property
    def model(self):
        if self._model is None:
            self.context_dimension = np.size(self.meta_contexts[0])

            self._model = self._init_model({})

        return self._model

    def _init_model(self, model):
        model['A'] = [np.identity(self.context_dimension)
                      * self.alpha for _ in range(self.k)]
        model['A_inv'] = [np.identity(
            self.context_dimension)/self.alpha for _ in range(self.k)]
        model['b'] = [np.zeros((self.context_dimension, 1))
                      for _ in range(self.k)]
        model['theta'] = np.array(
            [np.zeros((self.context_dimension, 1)) for _ in range(self.k)])
        return model

    def get_values(self, contexts, return_std=True):

        estimated_rewards = (contexts*self.model['theta'][:, :, 0]).sum(-1)

        if return_std:
            uncertainties = np.sqrt(
                ((contexts[:, :, np.newaxis]*self.model['A_inv']).sum(axis=1)*contexts).sum(-1))
            assert np.size(uncertainties) == self.bandit.k, uncertainties

            return estimated_rewards, uncertainties
        else:
            return estimated_rewards

    def initialize_w(self):
        self._model = None

        self.context_history = [[] for _ in range(self.k)]
        self.reward_history = [[] for _ in range(self.k)]

    @property
    def base_str(self):
        return ('LinUCB' + ("_alph{}".format(self.alpha) if self.alpha != 1 else '')
                + ("_beta{}".format(self.beta) if self.beta != 1 else '')+('_earl'+str(self.early)
                                                                           if self.early is not False else '')+('shared' if self.shared_model else '')
                + ('monocontext' if self.shared_meta else ''))

    def value_estimates(self, contexts):

        self.update_advice(contexts)

        offset_advice = self.advice - \
            (EXPECTED_AVG_REWARD if self.value_mode else 1/self.bandit.k)
        if self.shared_meta:
            offset_advice = offset_advice.flatten()
            offset_advice = np.repeat(
                offset_advice[np.newaxis, :], self.bandit.k, axis=0).T
        self.meta_contexts = np.concatenate(
            (offset_advice, self.confidences, np.ones((1, self.bandit.k))), axis=0).T

        mu, sigma = self.get_values(self.meta_contexts)
        return mu + sigma*self.beta

    def reset(self):
        super(MetaCMAB, self).reset()

    def observe(self, reward, arm, contexts):

        model_arm = arm*(not self.shared_model)
        assert len(self.reward_history) > model_arm, (str(
            self), self.shared_model)
        action_context = np.reshape(self.meta_contexts[arm], (-1, 1))

        self.reward_history[model_arm].append(reward)
        self.context_history[model_arm].append(self.meta_contexts[arm])

        self.model['A'] += action_context.dot(action_context.T)
        self.model['A_inv'][model_arm] = SMInv(
            self.model['A_inv'][model_arm], action_context, action_context)
        self.model['b'][model_arm] += (reward -
                                       EXPECTED_AVG_REWARD) * action_context

        self.model['theta'][model_arm] = self.model['A_inv'][model_arm].dot(
            self.model['b'][model_arm])

        self.notify_experts(reward, arm, contexts)
        self.t += 1
