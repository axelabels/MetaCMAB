#!/usr/bin/env python
# -*- coding: utf-8 -*-
from ast import literal_eval
import random
import shutil
from itertools import product, combinations
from optparse import OptionParser
import numpy as np

from agent import *
from bandit import *
from environment import *
from policy import *

parser = OptionParser()

parser.add_option(
    "--log_id",
    dest="log_id",
    default=None,
    help="log file identifier")


parser.add_option(
    "--spread",
    dest="spread",
    default=1,
    help="Ratio of the context spread covered by prior training", type=float)
parser.add_option(
    "--override",
    dest="override",
    default=False, action='store_true',
    help="override previous results for this configuration")

parser.add_option(
    "--trials",
    dest="trials",
    default=1000,
    help="number of trials", type=int)

parser.add_option(
    "--gamma",
    dest="gamma",
    default=.1,
    help="gamma parameter for KernelUCB experts", type=float)

parser.add_option(
    "--beta",
    dest="beta",
    default=100,
    help="exploration strength", type=float)

parser.add_option(
    "--experiments",
    dest="experiments",
    default=1,
    help="number of experiments", type=int)
parser.add_option(
    "--prior",
    dest="prior",
    default=100,
    help="prior rounds", type=int)


parser.add_option(
    "--seed",
    dest="seed",
    default='1',
    help="seed", type=str)

parser.add_option(
    "--marker",
    dest="marker",
    default='2d',
    help="marker", type=str)

parser.add_option(
    "--complexity",
    dest="complexity",
    default=4,
    help="complexity", type=int)
parser.add_option(
    "--dimensions",
    dest="dimensions",
    default=2,
    help="dimensions", type=int)

parser.add_option(
    "--exploratory",
    dest="exploratory",
    default=False,
    action='store_true',
    help="exploratory prior")

parser.add_option(
    "--clean",
    dest="clean",
    default=False,
    action='store_true',
    help="clean")

parser.add_option(
    "--activation",
    dest="activation",
    default='strongsigmoid')

parser.add_option(
    "--kernel",
    dest="kernel",
    default='rational')
parser.add_option(
    "--metric",
    dest="metric",
    default='mse')
parser.add_option(
    "--expert_config",
    dest="expert_config",
    default='normal')

Result = namedtuple('Result', ('prior', 'bias', 'std',
                               'A', 'N', 'tp', 'str', 'seed', 'scale'))


def save_results():
    import os

    def save_result(k, result):

        simple_filename = str(k.prior) + "_" + str(k.bias) + "_" + \
            str(k.std) + "_" + str(k.scale) + "_" + str(k.seed)

        if not os.path.exists(simple_dir):
            os.makedirs(simple_dir)

        with open((simple_dir + simple_filename + ".txt"), 'a') as filepath:
            n_slices = 50
            slices = []
            l = n_slices / log(options.trials // 2)
            for sl in range(n_slices):
                limit = int(np.round(np.exp((sl + 1) / l)) - 1)
                slices.append(limit)

            for sl in range(n_slices):
                limit = int(np.round(np.exp((sl + 1) / l)) - 1)
                slices.append(options.trials // 2 + limit)
            slices = list(np.linspace(0, options.trials - 1,
                                      num=n_slices + 1, dtype=int))[1:]

            # [s for s in slices if s<200 or ((s%100==0 or s==140) and s<400) or s%200==0]
            slices = np.array([15, 30, 60, 125, 250, 500, 1000])-1
            slices = (slices/1000*options.trials).astype(int)
            for i, limit in enumerate(slices):
                s = np.cumsum(result[0], axis=0)[limit]
                best_expert, best_oracle, best_expert_oracle = result[1]

                best_oracle = np.mean(
                    np.cumsum(best_oracle, axis=0)[limit])
                best_expert_oracle = np.mean(
                    np.cumsum(best_expert_oracle, axis=0)[limit])
                worst_expert, worst_oracle, worst_expert_oracle = result[2]
                worst_oracle = np.mean(
                    np.cumsum(worst_oracle, axis=0)[limit])
                worst_expert_oracle = np.mean(
                    np.cumsum(worst_expert_oracle, axis=0)[limit])
                tt = np.cumsum(worst_expert, axis=0)[limit]
                t = np.cumsum(best_expert, axis=0)[limit]
                if i > 0:
                    c = np.mean(result[0][slices[i-1]:limit + 1], axis=0)
                else:
                    c = np.mean(result[0][:limit + 1], axis=0)

                s = "{0:.4f} ± {1:.4f}".format(np.mean(s), np.std(s))
                tt = "{0:.4f} ± {1:.4f} / {2} / {3}".format(
                    np.mean(tt), np.std(tt), worst_expert_oracle, worst_oracle)
                t = "{0:.4f} ± {1:.4f} / {2} / {3}".format(
                    np.mean(t), np.std(t), best_expert_oracle, best_oracle)
                c = "{0:.4f} ± {1:.4f}".format(np.mean(c), np.std(c))
                slice_str = "sl" + \
                            "{0:04d}".format(limit + 1) + "_" + k.str
                if 'expert_' in slice_str and '_ablate' in slice_str and not ('expert_0' in slice_str or 'expert_'+str(k.N//2-1) in slice_str):
                    continue
                if 'expert_' in slice_str and '_ablate' not in slice_str and not ('expert_0' in slice_str or 'expert_'+str(k.N-1) in slice_str):
                    continue
                print("tab", k.seed, k.A, k.N, k.prior, k.bias, k.std, k.scale,
                      k.tp, slice_str, c, s, tt, t, sep=';', file=filepath)
        with open((simple_dir + simple_filename + ".txt"), 'r') as filepath:
            assert ';' in filepath.read()

        with open(completed_dir + "/done_{}.txt".format(seed), 'a') as f:
            if 'expert' not in str(k):
                print(str(k), file=f)

    for i, k in (list(enumerate(sorted(results.keys(), key=lambda k: (k.A, k.N, k.prior, k.bias, k.tp, k.str))))):
        if str(k) not in old_result_set:
            save_result(k, results[k])
            old_result_set.add(str(k))


if __name__ == '__main__':
    (options, args) = parser.parse_args()
    KernelUCB.MAX_SPREAD = options.spread
    n_trials = options.trials
    n_experiments = options.experiments
    options.prior = options.prior

    max_prior_rounds = 0
    Agent.CONFIGURATION = options.expert_config
    try:
        main_seed = int(options.seed)
        seeds = (main_seed,)
    except:
        try:
            seeds = literal_eval(options.seed)
        except:
            assert ":" in options.seed, "Unknown seed format"
            start, end = options.seed.split(":")
            start = int(start)
            end = int(end)
            seeds = tuple(range(start, end + 1))
    main_bar = seeds
    dimensions = options.dimensions

    log_id = options.log_id
    if log_id is None:
        log_id = "{}{}_{}_{}_dim{}_{}exploratory_{}_{}_{}_{}_{}".format(options.marker, ('' if options.prior == 100 else options.prior), options.expert_config, options.activation, dimensions,
                                                                        ("" if options.exploratory else "not "),
                                                                        options.kernel, options.gamma, options.beta,
                                                                        options.metric, options.complexity)

    work_path = 'results/'

    completed_dir = work_path + '{}ddata/'.format(log_id)
    simple_dir = work_path + '{}_sdata/'.format(log_id)

    if options.clean and (os.path.exists(completed_dir) or os.path.exists(simple_dir)):
        answer = input(
            "Are you sure you want to delete {} and {} ? y/n".format(completed_dir, simple_dir))
        if answer == "y":
            if os.path.exists(completed_dir):
                shutil.rmtree(completed_dir)
            if os.path.exists(simple_dir):
                shutil.rmtree(simple_dir)
        elif answer != "n":
            print("expected y/n")
            exit()

    for seed in main_bar:

        def gen_experts(n, bandit):

            from agent import KernelUCB,  Agent, TransparentExpert
            Agent.CONFIGURATION = options.expert_config

            from policy import GPUCBPolicy
            import numpy as np

            def gen_expert(bandit, i):
                experts = [KernelUCB(bandit, GPUCBPolicy(
                ), gamma=options.gamma, beta=options.beta)]
                # experts = [TransparentExpert(bandit, GreedyPolicy())]
                np.random.shuffle(experts)
                return experts[i % len(experts)]

            def generator(n, bandit):
                return [gen_expert(bandit, i) for i in range(n)]

            return lambda: generator(n, bandit)

        if options.metric == 'mse':
            PerlinBandit.DISTANCE_METRIC = mse
        elif options.metric == 'mae':
            PerlinBandit.DISTANCE_METRIC = mae

        def populate_results():
            from collections import namedtuple
            import os

            if not os.path.exists(completed_dir):
                os.makedirs(completed_dir)
            result_set = set()

            for filename in os.listdir(completed_dir):
                with open(completed_dir + filename, 'r') as donefile:
                    for line in donefile:
                        if 'expert' not in line:
                            result_set.add(line.replace(".pkl", "").strip())

            return result_set

        old_result_set = populate_results()

        confidence_noises = [0, .1, 1, 10]
        arm_set = [4, 32][1:]
        expert_set = [4, 32][1:]
        prior_noise_levels = np.arange(11)/10

        Agent.CONFIGURATION = options.expert_config
        KernelUCB.EXPLORATORY = options.exploratory

        base_kernels = []
        if 'rational' in options.kernel:
            base_kernels += [RationalQuadratic(length_scale=1, alpha=0.1)]
        if 'RBF' in options.kernel:
            base_kernels += [RBF(length_scale=1)]
        if 'linear' in options.kernel:
            base_kernels += [DotProduct()]

        complexities = [options.complexity]

        it = sorted(list(
            product(arm_set, expert_set, prior_noise_levels,  complexities)),
            key=lambda l: (l[0] * l[1], max(l[0], l[1]), l[1], l[2]))

        for n_arms, n_experts, prior_noise,  complexity in it:

            print("a {}, e {}, b {}, s {}, d {}".format(
                n_arms, n_experts, options.prior, seed, prior_noise))

            KernelUCB.KERNELS = list(base_kernels)

            for ablate in (False, True):
                results = {}

                epsilon = 0.05

                if 'super' in options.marker:
                    bandit = SuperBandit(
                        n_arms, complexity=complexity, smooth_function=options.activation)
                else:
                    bandit = PerlinBandit(
                        n_arms, complexity=complexity, smooth_function=options.activation)

                epsGreedyPolicy = EpsilonGreedyPolicy(epsilon)
                probGreedyPolicy = ProbabilityGreedyPolicy(epsilon)

                experts_generator = gen_experts(
                    n_experts, bandit)

                agents = []

                for value_mode, confidence_noise in product((True, False), confidence_noises):

                    for conf_type in ("none", "reward",):

                        if confidence_noise != confidence_noises[0] and conf_type == "none":
                            continue

                        agents += [Average(bandit, epsGreedyPolicy if value_mode else probGreedyPolicy,
                                           experts_generator,
                                           value_mode=value_mode,
                                           confidence_type=conf_type,
                                           confidence_noise=confidence_noise, ablate=ablate)]

                        agents += [MetaCMAB(bandit, epsGreedyPolicy,
                                            experts_generator,
                                            value_mode=value_mode,
                                            confidence_type=conf_type,
                                            confidence_noise=confidence_noise,
                                            beta=1/n_arms,
                                            ablate=ablate,
                                            shared_model=True, shared_meta=False)]

                        agents += [MetaMAB(bandit, epsGreedyPolicy if value_mode else probGreedyPolicy,
                                           experts_generator,
                                           value_mode=value_mode,
                                           confidence_type=conf_type,
                                           confidence_noise=confidence_noise, ablate=ablate, add_random=True)]

                agents = [a for a in agents if str(Result(prior=max_prior_rounds, bias=options.prior, A=n_arms,
                                                          N=n_experts, std=a.confidence_noise,
                                                          tp=a.advice_type,
                                                          str=str(a), seed=seed,
                                                          scale=prior_noise)) not in old_result_set]
                agents = sorted(
                    agents, key=lambda a: not issubclass(type(a), Collective))

                if len(agents) > 0:
                    env = Environment(bandit, agents, seed=seed,
                                      n_bias_max=options.prior, prior_bandit=bandit,
                                      max_distance=prior_noise)

                    (scores, best_expert, best_oracle, best_expert_oracle, worst_expert, worst_oracle,
                     worst_expert_oracle), expert_rewards = env.run(n_trials, n_experiments,  marker=log_id)

                    scores = np.array(scores)

                    for a in range(len(agents[:])):
                        s = scores[:, a]
                        o = best_expert[:, a], best_oracle[:,
                                                           a], best_expert_oracle[:, a]
                        w = worst_expert[:, a], worst_oracle[:,
                                                             a], worst_expert_oracle[:, a]
                        result_key = Result(prior=max_prior_rounds, bias=options.prior, A=n_arms, N=n_experts,
                                            std=agents[a].confidence_noise,
                                            tp=agents[a].advice_type,
                                            str=str(agents[a]), seed=seed, scale=prior_noise)
                        results[result_key] = (s, o, w)

                    if expert_rewards is not None:
                        sorted_scores = np.zeros_like(expert_rewards)

                        for s_i in range(n_experiments):
                            sorted_scores[:, :, s_i] = sorted(
                                expert_rewards[:, :, s_i], key=np.sum)

                        meaned_scores = np.mean(sorted_scores, axis=2)

                        for a, agent in enumerate(agents):
                            if not issubclass(type(agent), Collective):
                                continue
                            for e, expert in enumerate(agent.experts):
                                s = meaned_scores[e]
                                o = best_expert[:, a], best_oracle[:,
                                                                   a], best_expert_oracle[:, a]
                                w = worst_expert[:, a], worst_oracle[:,
                                                                     a], worst_expert_oracle[:, a]
                                result_key = Result(prior=max_prior_rounds, bias=options.prior, A=n_arms, N=n_experts,
                                                    std=agents[a].confidence_noise,
                                                    tp=agent.advice_type,
                                                    str='expert_{}{}'.format(e, '_ablate' if ablate else ''), seed=seed,
                                                    scale=prior_noise)
                                results[result_key] = (s, o, w)

                save_results()
