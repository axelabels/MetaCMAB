"""Perlin generation adapted from https://github.com/marceloprates/Numpy-Generative-Art"""
from itertools import product
import numpy as np
import math
from scipy.stats import wrapcauchy
from tools import *


def lerp(a, b, x):
    return (a + x * (b - a))


def smootheststep(t):
    return t**7*(1716 - 9009*t + 20020*t**2 - 24024 * t**3 + 16380*t**4 - 6006 * t**5 + 924*t**6)


def gradient(h, x, y, vectors=None):
    if vectors is None:
        vectors = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
    g = vectors[h % len(vectors)]

    assert g.shape[-1] == 2
    return g[:, :, 0] * x + g[:, :, 1] * y


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def perlin(x, y, seed=None, angles=None, pre_p=None, noise=0, angle_count=1000):

    if noise >= 1:
        noise = 1 - 1e-10
    PS = 256
    if seed is not None:
        np.random.seed(seed)
    p = np.arange(PS, dtype=int)
    np.random.shuffle(p)
    p = np.stack([p, p]).flatten()
    # coordinates of the top-left
    xi = x.astype(int)
    yi = y.astype(int)
    # internal coordinates
    xf = x - xi
    yf = y - yi

    u = smootheststep(xf)
    v = smootheststep(yf)
    if angles is not None:
        angles = np.copy(angles)
        p = np.copy(pre_p)
    else:
        angles = np.random.uniform(-np.pi, np.pi, size=angle_count)
    if noise > 0:
        angle_noise = wrapcauchy.rvs(c=1 - noise, size=angles.shape)
        angles = angles + angle_noise

    xy = np.rollaxis(np.array(angle_to_coords(angles, 2)), 0, 2)

    vectors = xy
    # noise components
    n00 = gradient(p[p[xi] + yi], xf, yf, vectors=vectors)
    n01 = gradient(p[p[xi] + yi + 1], xf, yf - 1, vectors=vectors)
    n11 = gradient(p[p[xi + 1] + yi + 1], xf - 1, yf - 1, vectors=vectors)
    n10 = gradient(p[p[xi + 1] + yi], xf - 1, yf, vectors=vectors)
    # combine noises
    x1 = lerp(n00, n10, u)
    x2 = lerp(n01, n11, u)

    return lerp(x1, x2, v), angles, p, None
