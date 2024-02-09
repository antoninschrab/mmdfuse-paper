# Reproducibility code for MMD-FUSE

This GitHub repository contains the code for the reproducible experiments presented in our paper MMD-FUSE: Learning and Combining Kernels for Two-Sample Testing Without Data Splitting.

We provide the code to run the experiments to generate the figures and tables from our paper,
these can be found in [figures](figures).

To use MMD-FUSE in practice, we recommend using our `mmdfuse` package, more details available on the [mmdfuse](https://github.com/antoninschrab/mmdfuse) repository.

## Requirements
- `python 3.9`

Only the `jax` and `jaxlib` packages are required to run MMD-FUSE (see [mmdfuse](https://github.com/antoninschrab/mmdfuse)), several other packages are required to run other tests we compare against (see [env_mmdfuse.yml](env_mmdfuse.yml) and [env_autogluon.yml](env_autogluon.yml))

## Installation

In a chosen directory, clone the repository and change to its directory by executing 
```
git clone git@github.com:antoninschrab/mmdfuse-paper.git
cd mmdfuse-paper
```
We then recommend creating and activating a virtual environment using `conda` by running
```bash
conda env create -f env_mmdfuse.yml
conda env create -f env_autogluon.yml
conda activate mmdfuse-env
# conda activate autogluon-env
# can be deactivated by running:
# conda deactivate
```

## Reproducing the experiments of the paper

The results of the six experiments can be reproduced by running the code in the notebooks: [experiments_mixture.ipynb](experiments_mixture.ipynb), [experiments_perturbations.ipynb](experiments_perturbations.ipynb), [experiment_perturbations_vary_kernel.ipynb](experiment_perturbations_vary_kernel.ipynb), [experiments_galaxy.ipynb](experiments_galaxy.ipynb), [experiments_cifar.ipynb](experiments_cifar.ipynb), and [experiments_runtimes.ipynb](experiments_runtimes.ipynb).

The results are saved as `.npy` files in the directory `results`.
The figures of the paper can be obtained from these by running the code in the [figures.ipynb](figures.ipynb) notebook.

All the experiments are comprised of 'embarrassingly parallel for loops', significant speed up can be obtained by using parallel computing libraries such as `joblib` or `dask`.

## Datasets

- [GalaxyMNIST](https://github.com/mwalmsley/galaxy_mnist): [Galaxy Zoo DECaLS, Walmsley et al., 2021](https://arxiv.org/pdf/2102.08414.pdf)
- [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html): [Learning Multiple Layers of Features from Tiny Images, Krizhevsky, 2009](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)
- [CIFAR-10.1](cifar_data/cifar10.1_v4_data.npy): [Do CIFAR-10 Classifiers Generalize to CIFAR-10?, Recht et al., 2018](https://arxiv.org/pdf/1806.00451.pdf)

## Samplers

- [Sampler Mixture of Gaussians](sampler_mixture.py)
- [Sampler Perturbed Uniform](sampler_perturbations.py)
- [Sampler GalaxyMNIST](sampler_galaxy.py)

## How to use MMD-FUSE in practice?

The MMD-FUSE test is implemented as the function `mmdfuse` in [mmdfuse.py](mmdfuse.py) in Jax. It requires only the `jax` and `jaxlib` packages.

To use our tests in practice, we recommend using our `mmdfuse` package which is available on the [mmdfuse](https://github.com/antoninschrab/mmdfuse) repository. It can be installed by running
```bash
pip install git+https://github.com/antoninschrab/mmdfuse.git
```
Installation instructions and example code are available on the [mmdfuse](https://github.com/antoninschrab/mmdfuse) repository. 

We also provide some code showing how to use MMD-FUSE in the [demo_speed.ipynb](demo_speed.ipynb) notebook which also contains speed comparisons between running the code on CPU or on GPU:

| Speed in s | Jax (GPU) | Jax (CPU) | 
| -- | -- | -- |
| MMD-FUSE | 0.0054 | 2.95 | 
 
## References

*Interpretable Distribution Features with Maximum Testing Power.*
Wittawat Jitkrittum, Zoltán Szabó, Kacper Chwialkowski, Arthur Gretn.
([paper](https://proceedings.neurips.cc/paper/2016/file/0a09c8844ba8f0936c20bd791130d6b6-Paper.pdf), [code](https://github.com/wittawatj/interpretable-test))

*Learning Deep Kernels for Non-Parametric Two-Sample Tests.*
Feng Liu, Wenkai Xu, Jie Lu, Guangquan Zhang, Arthur Gretton, Danica J. Sutherland.
([paper](https://arxiv.org/abs/2002.09116), [code](https://github.com/fengliu90/DK-for-TST))

*MMD Aggregated Two-Sample Test.*
Antonin Schrab, Ilmun Kim, Mélisande Albert, Béatrice Laurent, Benjamin Guedj, Arthur Grett.
([paper](https://arxiv.org/abs/2110.15073), [code](https://github.com/antoninschrab/mmdagg))

*AutoML Two-Sample Test.*
Jonas M. Kübler, Vincent Stimper, Simon Buchholz, Krikamol Muandet, Bernhard Schölkopf.
([paper](https://arxiv.org/abs/2206.08843), [code](https://github.com/jmkuebler/auto-tst))

*Compress Then Test: Powerful Kernel Testing in Near-linear Time.*
Carles Domingo-Enrich, Raaz Dwivedi, Lester Mackey.
([paper](https://arxiv.org/abs/2301.05974), [code](https://arxiv.org/pdf/2301.05974.pdf))

## Contact

If you have any issues running the code, please do not hesitate to contact [Antonin Schrab](https://antoninschrab.github.io).

## Affiliations

Centre for Artificial Intelligence, Department of Computer Science, University College London

Gatsby Computational Neuroscience Unit, University College London

Inria London

## Bibtex

```
@article{biggs2023mmdfuse,
  author        = {Biggs, Felix and Schrab, Antonin and Gretton, Arthur},
  title         = {{MMD-FUSE}: {L}earning and Combining Kernels for Two-Sample Testing Without Data Splitting},
  year          = {2023},
  journal       = {Advances in Neural Information Processing Systems},
  volume        = {36}
}
```

## License

MIT License (see [LICENSE.md](LICENSE.md)).
