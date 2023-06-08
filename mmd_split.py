import jax
import jax.numpy as jnp
from jax import random, jit
from functools import partial
from kernel import jax_distances, kernel_matrix


@partial(jit, static_argnums=(3, 4, 5, 6, 7, 8))
def mmd_split(
    X,
    Y,
    key,
    alpha=0.05,
    split_ratio=0.5,
    kernel="gaussian",
    number_bandwidths=100,
    number_permutations=2000,
    return_p_val=False,
):
    # Assertions
    m = X.shape[0]
    n = Y.shape[0]
    assert m == n
    assert n >= 2 and m >= 2
    assert 0 < alpha and alpha < 1
    assert 0 < split_ratio and split_ratio < 1
    assert number_bandwidths > 1 and type(number_bandwidths) == int
    assert number_permutations > 0 and type(number_permutations) == int
    assert kernel in (
        "imq",
        "rq",
        "gaussian",
        "matern_0.5_l2",
        "matern_1.5_l2",
        "matern_2.5_l2",
        "matern_3.5_l2",
        "matern_4.5_l2",
        "laplace",
        "matern_0.5_l1",
        "matern_1.5_l1",
        "matern_2.5_l1",
        "matern_3.5_l1",
        "matern_4.5_l1",
    )
    if kernel in (
        "laplace",
        "matern_0.5_l1",
        "matern_1.5_l1",
        "matern_2.5_l1",
        "matern_3.5_l1",
        "matern_4.5_l1",
    ):
        l = "l1"
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
    
    # Split the keys
    key, subkey = random.split(key)
    subkeys = random.split(subkey, num=6)
    
    # Shuffle the data
    X_shuffle = jax.random.permutation(subkeys[0], X, axis=0)
    Y_shuffle = jax.random.permutation(subkeys[1], Y, axis=0)
    
    # Split the data
    split = int(n * split_ratio)
    X_selection = X[:split]
    Y_selection = Y[:split]
    X_test = X[split:]
    Y_test = Y[split:]
    
    ################## 
    # Kernel selection
    ##################
    N = number_bandwidths
    R = jnp.zeros((N, ))
    # Pairwise distance matrix
    Z = jnp.concatenate((X_selection, Y_selection))
    pairwise_matrix = jax_distances(Z, Z, l, matrix=True)

    # Collection of bandwidths
    def compute_bandwidths(distances, number_bandwidths):
        median = jnp.median(distances)
        distances = distances + (distances == 0) * median
        dd = jnp.sort(distances)
        lambda_min = dd[(jnp.floor(len(dd) * 0.05).astype(int))] / 2
        lambda_max = dd[(jnp.floor(len(dd) * 0.95).astype(int))] * 2
        bandwidths = jnp.linspace(lambda_min, lambda_max, number_bandwidths)
        return bandwidths

    distances = pairwise_matrix[jnp.triu_indices(pairwise_matrix.shape[0])]
    bandwidths = compute_bandwidths(distances, number_bandwidths)

    # Compute all permuted MMD estimates for either l1 or l2
    for i in range(number_bandwidths):
        # compute kernel matrix and set diagonal to zero
        bandwidth = bandwidths[i]
        K = kernel_matrix(pairwise_matrix, l, kernel, bandwidth)
        R = R.at[i].set(ratio_mmd_std(K))
    index_selected = jnp.argmax(R)
    bandwidth_selected = bandwidths[index_selected]
    
    # Setup for permutations
    n = X_test.shape[0]
    m = Y_test.shape[0]
    key, subkey = random.split(key)
    B = number_permutations
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
    
    ##########
    # Run test
    ##########
    Z = jnp.concatenate((X_test, Y_test))
    pairwise_matrix = jax_distances(Z, Z, l, matrix=True)
    K = kernel_matrix(pairwise_matrix, l, kernel, bandwidth_selected)
    M = (
        jnp.sum(V10 * (K @ V10), 0)  * (1 / (m * (m - 1)) - 1 / (m * n))
        + jnp.sum(V01 * (K @ V01), 0) * (1 / (n * (n - 1)) - 1 / (m * n))
        + jnp.sum(V11 * (K @ V11), 0) / (m * n)
    )
    
    # Compute test output
    all_MMD = M  # (B+1,)
    original_MMD = M[-1]  # (1,)
    p_val = jnp.mean(all_MMD >= original_MMD)
    output = p_val <= alpha

    # Return output
    if return_p_val:
        return p_val
    else:
        return output.astype(int)

    
# this function cannot be jitted
def mmd_split_different_kernels(
    X,
    Y,
    key,
    alpha=0.05,
    split_ratio=0.5,
    kernels=("imq", "gaussian", "laplace", "matern_0.5_l2", "matern_1.5_l2", "matern_1.5_l1"),
    number_bandwidths=20,
    number_permutations=2000,
    return_p_val=False,
):
    # Assertions
    m = X.shape[0]
    n = Y.shape[0]
    assert m == n
    assert n >= 2 and m >= 2
    assert 0 < alpha and alpha < 1
    assert 0 < split_ratio and split_ratio < 1
    assert number_bandwidths > 1 and type(number_bandwidths) == int
    assert number_permutations > 0 and type(number_permutations) == int
    if type(kernels) is str:
        # convert to list
        kernels = (kernels,)
    for kernel in kernels:
        assert kernel in (
            "imq",
            "rq",
            "gaussian",
            "matern_0.5_l2",
            "matern_1.5_l2",
            "matern_2.5_l2",
            "matern_3.5_l2",
            "matern_4.5_l2",
            "laplace",
            "matern_0.5_l1",
            "matern_1.5_l1",
            "matern_2.5_l1",
            "matern_3.5_l1",
            "matern_4.5_l1",
        )

    # Lists of kernels for l1 and l2
    all_kernels_l1 = (
        "laplace",
        "matern_0.5_l1",
        "matern_1.5_l1",
        "matern_2.5_l1",
        "matern_3.5_l1",
        "matern_4.5_l1",
    )
    all_kernels_l2 = (
        "imq",
        "rq",
        "gaussian",
        "matern_0.5_l2",
        "matern_1.5_l2",
        "matern_2.5_l2",
        "matern_3.5_l2",
        "matern_4.5_l2",
    )
    number_kernels = len(kernels)
    kernels_l1 = [k for k in kernels if k in all_kernels_l1]
    kernels_l2 = [k for k in kernels if k in all_kernels_l2]
    
    # Split the keys
    key, subkey = random.split(key)
    subkeys = random.split(subkey, num=6)
    
    # Shuffle the data
    X_shuffle = jax.random.permutation(subkeys[0], X, axis=0)
    Y_shuffle = jax.random.permutation(subkeys[1], Y, axis=0)
    
    # Split the data
    split = int(n * split_ratio)
    X_selection = X[:split]
    Y_selection = Y[:split]
    X_test = X[split:]
    Y_test = Y[split:]
    
    # Kernel selection
    N = number_bandwidths * number_kernels
    R = jnp.zeros((N, ))
    kernel_l_bandwidth = jnp.zeros((N, 3))  # kernel_index, l_index, bandwidth
    kernel_count = -1  # first kernel will have kernel_count = 0
    for r in range(2):
        kernels_l = (kernels_l1, kernels_l2)[r]
        l = ("l1", "l2")[r]
        if len(kernels_l) > 0:
            # Pairwise distance matrix
            Z = jnp.concatenate((X_selection, Y_selection))
            pairwise_matrix = jax_distances(Z, Z, l, matrix=True)

            # Collection of bandwidths
            def compute_bandwidths(distances, number_bandwidths):
                median = jnp.median(distances)
                distances = distances + (distances == 0) * median
                dd = jnp.sort(distances)
                lambda_min = dd[(jnp.floor(len(dd) * 0.05).astype(int))] / 2
                lambda_max = dd[(jnp.floor(len(dd) * 0.95).astype(int))] * 2
                bandwidths = jnp.linspace(lambda_min, lambda_max, number_bandwidths)
                return bandwidths

            distances = pairwise_matrix[jnp.triu_indices(pairwise_matrix.shape[0])]
            bandwidths = compute_bandwidths(distances, number_bandwidths)

            # Compute all permuted MMD estimates for either l1 or l2
            for j in range(len(kernels_l)):
                kernel = kernels_l[j]
                kernel_count += 1
                for i in range(number_bandwidths):
                    # compute kernel matrix and set diagonal to zero
                    bandwidth = bandwidths[i]
                    K = kernel_matrix(pairwise_matrix, l, kernel, bandwidth)
                    kernel_l_bandwidth = kernel_l_bandwidth.at[
                        kernel_count * number_bandwidths + i
                    ].set(
                        jnp.array([j, r, bandwidth])
                    )
                    R = R.at[kernel_count * number_bandwidths + i].set(
                         ratio_mmd_std(K)
                    )
    index_selected = jnp.argmax(R)
    kernel_index, l_index, bandwidth_selected = kernel_l_bandwidth[index_selected]
    kernel_selected = (kernels_l1, kernels_l2)[int(l_index)][int(kernel_index)]
    l_selected = ("l1", "l2")[int(l_index)]
    
    # Setup for permutations
    n = X_test.shape[0]
    m = Y_test.shape[0]
    key, subkey = random.split(key)
    B = number_permutations
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
    
    # Run test
    Z = jnp.concatenate((X_test, Y_test))
    pairwise_matrix = jax_distances(Z, Z, l_selected, matrix=True)
    K = kernel_matrix(pairwise_matrix, l_selected, kernel_selected, bandwidth_selected)
    M = (
        jnp.sum(V10 * (K @ V10), 0)
        * (n - m + 1)
        / (m * n * (m - 1))
        + jnp.sum(V01 * (K @ V01), 0)
        * (m - n + 1)
        / (m * n * (n - 1))
        + jnp.sum(V11 * (K @ V11), 0) / (m * n)
    )
    
    # Compute test output
    all_MMD = M  # (B+1,)
    original_MMD = M[-1]  # (1,)
    p_val = jnp.mean(all_MMD >= original_MMD)
    output = p_val <= alpha

    # Return output
    if return_p_val:
        return p_val
    else:
        return output.astype(int)

def ratio_mmd_std(K):
    n = int(K.shape[0]/2)
    regulariser = 10 ** (-8)

    # compute variance
    Kxx = K[:n, :n]
    Kxy = K[:n, n:]
    Kyx = K[n:, :n]
    Kyy = K[n:, n:]
    H_column_sum = (
        jnp.sum(Kxx, axis=1)
        + jnp.sum(Kyy, axis=1)
        - jnp.sum(Kxy, axis=1)
        - jnp.sum(Kyx, axis=1)
    )
    var = (
        4 / n ** 3 * jnp.sum(H_column_sum ** 2)
        - 4 / n ** 4 * jnp.sum(H_column_sum) ** 2
    )
    var = jnp.maximum(var, 0)
    var = var + regulariser

    # compute MMD_a estimate
    Kxx = K[:n, :n]
    Kxy = K[:n, n:]
    Kyy = K[n:, n:]
    Kxx = Kxx.at[jnp.diag_indices(Kxx.shape[0])].set(0)
    Kyy = Kyy.at[jnp.diag_indices(Kyy.shape[0])].set(0)
    s = jnp.ones(n)
    mmd = (
        s @ Kxx @ s / (n * (n - 1))
        + s @ Kyy @ s / (n * (n - 1))
        - 2 * s @ Kxy @ s / (n ** 2)
    )
    
    return mmd / jnp.sqrt(var)
