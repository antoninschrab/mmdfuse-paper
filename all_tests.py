from mmdfuse import mmdfuse
from mmd_median import mmd_median
from mmd_split import mmd_split
from mmdagg import mmdagg
from agginc import agginc
from other_tests.met_scf import met, scf
from other_tests.deep_mmd_not_image import deep_mmd_not_image
from other_tests.deep_mmd_image import deep_mmd_image
from goodpoints.ctt import ctt, actt
from kernel import compute_bandwidths
from utils import HiddenPrints
import numpy as np

def mmdfuse_test(X, Y, key, seed): 
    return int(mmdfuse(X, Y, key))


def mmd_median_test(X, Y, key, seed): 
    return int(mmd_median(X, Y, key))


def mmd_split_test(X, Y, key, seed): 
    return int(mmd_split(X, Y, key))


def mmdagg_test(X, Y, key, seed): 
    return int(mmdagg(X, Y))


def mmdagg_test_permutation(X, Y, key, seed): 
    return int(mmdagg(X, Y, permutations_same_sample_size=True))


def mmdagginc_test(X, Y, key, seed): 
    return int(agginc("mmd", X, Y))


def deep_mmd_test(X, Y, key, seed, n_epochs=1000):
    return deep_mmd_not_image(X, Y, n_epochs=n_epochs)


def deep_mmd_image_test(X, Y, key, seed, n_epochs=1000):
    X = X.reshape((X.shape[0], 3, 64, 64))
    Y = Y.reshape((Y.shape[0], 3, 64, 64))
    return deep_mmd_image(X, Y, n_epochs=n_epochs)


def met_test(X, Y, key, seed): 
    with HiddenPrints():
        output = met(X, Y, seed)
    return output


def scf_test(X, Y, key, seed): 
    with HiddenPrints():
        output = scf(X, Y, seed)
    return output


def ctt_test(X, Y, key, seed): 
    lam = compute_bandwidths(X, Y, "l2", 1, only_median=True)
    X = np.asarray(X).astype('double')
    Y = np.asarray(Y).astype('double')
    lam = np.asarray(lam).astype('double')
    return ctt(X, Y, g=4, lam=lam, null_seed=seed, statistic_seed=seed).rejects


def actt_test(X, Y, key, seed): 
    lam = compute_bandwidths(X, Y, "l2", 10, only_median=False)
    weights = np.array([1/len(lam),] * len(lam)).astype('double')
    X = np.asarray(X).astype('double')
    Y = np.asarray(Y).astype('double')
    lam = np.asarray(lam).astype('double')
    with HiddenPrints():
        output = actt(X, Y, g=4, lam=lam, weights=weights, null_seed=seed, statistic_seed=seed).rejects
    return output
