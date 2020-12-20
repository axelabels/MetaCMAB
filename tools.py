from scipy import stats
import math
import re
import numpy as np
import random
from scipy.special import softmax


def SMInv(Ainv, u, v, alpha=1):
    u = u.reshape((len(u), 1))
    v = v.reshape((len(v), 1))
    return Ainv - np.dot(Ainv, np.dot(np.dot(u, v.T), Ainv)) / (alpha + np.dot(v.T, np.dot(Ainv, u)))


def to_str(a):

    if hasattr(a, '__iter__'):
        string = "["
        for sub_a in a:
            string += to_str(sub_a)+","
        string = string[:-1]+"]\n"

    elif type(a) == str:
        string = a
    else:
        string = "{:.3f}".format(a)

    return string


def randargmax(b, **kw):
    return np.argmax(np.random.random(b.shape) * (b == b.max()), **kw)


def arr2str(array):
    """Converts an array into a one line string

    Arguments:
        array {array} -- Array to convert

    Returns:
        str -- The string representation
    """
    return re.sub(r'\s+', ' ',
                  str(array).replace('\r', '').replace('\n', '').replace(
                      "array", "").replace("\t", " "))


def empty(*args, **kwargs):
    pass


def angle_to_coords(a, dims):
    if dims == 2:
        return [np.cos(a), np.sin(a)]
    else:
        raise NotImplementedError()


def coords_to_angle(v):
    if len(v) == 2:
        return atan2(v[1], v[0])
    else:
        raise NotImplementedError()


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def scale(a):
    return normalize(a, offset=np.min(a), scale=np.ptp(a))


def normalize(a, offset=None, scale=None, axis=None):
    a = np.asarray(a)
    if offset is None:
        offset = np.mean(a, axis=axis)
    if scale is None:
        scale = np.std(a, axis=axis)
    if type(scale) in (float, np.float64):
        if scale == 0:
            scale = 1
    else:
        scale[scale == 0] = 1

    return (a - offset) / scale


def mae(a, b):
    return np.mean(np.abs(a-b))


def mse(a, b):
    return np.mean((a-b)**2)


def rescale(a, mu, std):
    a = np.asarray(a)
    return a * std + mu


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_bandit_coords(bandit, flatten=True):
    return get_coords(bandit.dims, bandit.complexity, bandit.precision, flatten=flatten)


def get_coords(dims=2, complexity=3, precision=100, flatten=True):
    lin = np.linspace(0, complexity, int(
        (precision)**(2/dims)) + 1, endpoint=True)
    coords = np.array(np.meshgrid(*(lin for _ in range(dims)))).T / complexity
    if flatten:
        coords = coords.reshape((-1, dims))
    return coords


def random_choice_2d(a, size, replace):
    indices = np.random.choice(
        len(a), size=np.product(size[:-1]), replace=replace)
    return a[indices].reshape(size)


def mean2(x):
    y = np.sum(x) / np.size(x)
    return y


def corr2(a, b):
    a = a - mean2(a)
    b = b - mean2(b)
    r = (a*b).sum() / math.sqrt((a*a).sum() * (b*b).sum())
    return r


def truncnorm(l, u, loc, scale):
    lower, upper = l, u
    mu, sigma = loc, scale
    X = stats.truncnorm(
        (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    N = stats.norm(loc=mu, scale=sigma)
    return X, N


def expected_regret(a, truth, K, PULLS=10000):
    a = a.flatten()
    truth = truth.flatten()
    if PULLS is None:
        PULLS = len(a)//K
    indices = np.random.choice(len(a), replace=True, size=PULLS*K)
    a_values = a[indices].reshape((-1, K))

    choices = np.argmax(a_values, axis=1)
    b_values = truth[indices].reshape((-1, K))

    pulled_values = b_values[np.arange(PULLS), choices]
    regrets = np.max(b_values, axis=1)-pulled_values
    assert len(regrets) == PULLS
    return np.mean(regrets)
