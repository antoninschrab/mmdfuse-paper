import jax
import jax.numpy as jnp
from jax import random, jit
from functools import partial
from kernel import jax_distances, kernel_matrix


@partial(jit, static_argnums=(3, 4, 5, 6))
def mmd_median(
    X,
    Y,
    key,
    alpha=0.05,
    kernel="gaussian",
    B1=2000,
    return_p_val=False,
):
    # Assertions
    m = X.shape[0]
    n = Y.shape[0]
    mn = m + n
    assert n >= 2 and m >= 2
    assert 0 < alpha and alpha < 1
    assert B1 > 0 and type(B1) == int
    if kernel in (
        "imq",
        "rq",
        "gaussian",
        "matern_0.5_l2",
        "matern_1.5_l2",
        "matern_2.5_l2",
        "matern_3.5_l2",
        "matern_4.5_l2",
    ):
        l = "l2"
    elif kernel in (
        "laplace",
        "matern_0.5_l1",
        "matern_1.5_l1",
        "matern_2.5_l1",
        "matern_3.5_l1",
        "matern_4.5_l1",
    ):
        l = "l1"
    else:
        raise ValueError("Kernel not implemented")

    # Setup for permutations
    key, subkey = random.split(key)
    B = B1
    # (B, m+n): rows of permuted indices
    idx = random.permutation(
        subkey,
        jnp.array([[i for i in range(m + n)]] * (B + 1)),
        axis=1,
        independent=True,
    )
    # 11
    v11 = jnp.concatenate((jnp.ones(m), -jnp.ones(n)))  # (m+n, )
    V11i = jnp.tile(v11, (B + 1, 1))  # (B, m+n)
    V11 = jnp.take_along_axis(
        V11i, idx, axis=1
    )  # (B, m+n): permute the entries of the rows
    V11 = V11.at[B].set(v11)  # (B+1)th entry is the original MMD (no permutation)
    V11 = V11.transpose()  # (m+n, B+1)
    # 10
    v10 = jnp.concatenate((jnp.ones(m), jnp.zeros(n)))
    V10i = jnp.tile(v10, (B + 1, 1))
    V10 = jnp.take_along_axis(V10i, idx, axis=1)
    V10 = V10.at[B].set(v10)
    V10 = V10.transpose()
    # 01
    v01 = jnp.concatenate((jnp.zeros(m), -jnp.ones(n)))
    V01i = jnp.tile(v01, (B + 1, 1))
    V01 = jnp.take_along_axis(V01i, idx, axis=1)
    V01 = V01.at[B].set(v01)
    V01 = V01.transpose()

    # Compute kernel matrix
    Z = jnp.concatenate((X, Y))
    pairwise_matrix = jax_distances(Z, Z, l, matrix=True)
    distances = pairwise_matrix[jnp.triu_indices(pairwise_matrix.shape[0])]
    bandwidth = jnp.median(distances)
    K = kernel_matrix(pairwise_matrix, l, kernel, bandwidth, 0.5)
    K = K.at[jnp.diag_indices(K.shape[0])].set(0)  # set diagonal elements to zero
    # Compute MMD permuted values
    M = (
        jnp.sum(V10 * (K @ V10), 0)  * (1 / (m * (m - 1)) - 1 / (m * n))
        + jnp.sum(V01 * (K @ V01), 0) * (1 / (n * (n - 1)) - 1 / (m * n))
        + jnp.sum(V11 * (K @ V11), 0) / (m * n)
    )
    MMD_original = M[B]
    p_val = jnp.mean(M >= MMD_original)
    output = p_val <= alpha

    # Return output
    if return_p_val:
        return p_val
    else:
        return output.astype(int)

