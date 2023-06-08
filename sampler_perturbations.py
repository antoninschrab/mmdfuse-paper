"""
The code in this file is based on the following
https://github.com/antoninschrab/mmdagg-paper/blob/master/sampling.py
which is under the MIT License.
"""


import numpy as np
import jax.numpy as jnp
import itertools


def G(x):
    """
    Function G defined in Section 5.4 of our paper.
    input: x: real number
    output: G(x): real number
    """
    if -1 < x and x < -0.5:
        return np.exp(-1 / (1 - (4 * x + 3) ** 2))
    if -0.5 < x and x < 0:
        return - np.exp(-1 / ( 1 - (4 * x + 1) ** 2))   
    return 0    
    

def f_theta(x, p, s, perturbation_multiplier=1, seed=None):
    """
    Function f_theta defined in in Section 5.4 (Eq. (17)) of our paper.
    inputs: x: (d,) array (point in R^d)
            p: non-negative integer (number of perturbations)
            s: positive number (smoothness parameter of Sobolev ball (Eq. (1))
            perturbation_multiplier: positive number (c_d in Eq. (17))
            seed: integer random seed (samples theta in Eq. (17))
    output: real number f_theta(x) 
    """
    x = np.atleast_1d(x)
    d = x.shape[0]
    assert perturbation_multiplier * p ** (-s) * np.exp(-d) <= 1, "density is negative"
    np.random.seed(seed)
    theta = np.random.choice([-1,1], p ** d)
    output = 0
    I = list(itertools.product([i+1 for i in range(p)], repeat=d))  # set {1,...,p}^d
    for i in range(len(I)):
        j = I[i]
        output += theta[i] * np.prod([G(x[r] * p - j[r]) for r in range(d)])
    output *= p ** (-s) * perturbation_multiplier
    if np.min(x) >= 0 and np.max(x) <= 1:
        output += 1
    np.random.seed(None)
    return output
    

def f0(x):
    """
    Probability density function of multi-dimensional uniform distribution.
    input: array
    output: probability density function evaluated at the input
    """
    output = 0
    if np.min(x) >= 0 and np.max(x) <= 1:
        output += 1
    return output


def rejection_sampler(seed, density, d, density_max, number_samples, x_min, x_max):
    """
    Sample from density using a rejection sampler.
    inputs: seed: integer random seed
            density: probability density function
            d: dimension of input of the density
            density_max: maximum of the density
            number_samples: number of samples
            x_min: density is 0 on (-\infty,x_min)^d
            x_max: density is 0 on (x_max,\infty)^d
    output: number_samples samples from density sampled from [x_min, x_max]^d
    """
    samples = []
    count = 0
    rs = np.random.RandomState(seed)
    while count < number_samples:
        x = rs.uniform(x_min, x_max, d)
        y = rs.uniform(0, density_max)
        if y <= density(x):
            count += 1
            samples.append(x)
    return np.array(samples)
       

def f_theta_sampler(
    f_theta_seed, sampling_seed, number_samples, p, s, perturbation_multiplier, d
):
    """
    Sample from the probability density function f_theta.
    inputs: f_theta_seed: integer random seed for f_theta
            sampling_seed: integer random seed for rejection sampler
            number_samples: number of samples
            p: non-negative integer (number of perturbations)
            s: positive number (smoothness parameter of Sobolev ball (Eq. (1))
            perturbation_multiplier: positive number (c_d in Eq. (17)) 
            non-negative integer (dimension of input of density)
    output: number_samples samples from f_theta
    """
    density_max = 1 + perturbation_multiplier * p ** (-s) * np.exp(-d)  # maximum of f_theta
    return rejection_sampler(
        sampling_seed, 
        lambda x: f_theta(x, p, s, perturbation_multiplier, f_theta_seed), 
        d, 
        density_max, 
        number_samples, 
        0, 
        1,
    )


def sampler_perturbations(
    m,
    n,
    d,
    number_perturbations,
    seed,
    scale=1,
):
    """
    Sample from a uniform distribution and a perturbed uniform distribution.
    inputs: m: integer (sample size perturbed uniform)
            n: integer (sample size uniform)
            d: integer (dimension)
            number_perturbations: integer (number of perturbations)
            seed: integer (randomness seed)
            scale: scalar (amplitude of permutation between 0 and 1)
    output: samples from perturbed uniform distribution (m, d)
            samples from uniform distribution (n, d)
    """
    assert scale >= 0 and scale <= 1
    s = 1
    p = number_perturbations
    # this makes the pertubation stretch all the way to zero
    perturbation_multiplier = np.exp(d) * p ** s * scale 
    rs = np.random.RandomState(seed)
    if number_perturbations == 0 or scale == 0:
        X = rs.uniform(0, 1, (m, d))
        Y = rs.uniform(0, 1, (n, d))     
    else:
        s = 1
        X = f_theta_sampler(seed + 1, seed + 2, m, p, s, perturbation_multiplier, d)
        Y = rs.uniform(0, 1, (n, d))
    return jnp.array(X), jnp.array(Y)
