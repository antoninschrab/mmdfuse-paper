from freqopttest.tst import MeanEmbeddingTest as fot_MeanEmbeddingTest
from freqopttest.data import TSTData as fot_TSTData


# based on job_met_opt() function
# https://github.com/wittawatj/interpretable-test/blob/master/freqopttest/ex/ex1_power_vs_n.py
def met(X, Y, r, J=10, alpha=0.05):
    """MeanEmbeddingTest with test locations optimzied.
    Return results from calling perform_test()"""
    # MeanEmbeddingTest. optimize the test locations
    
    assert X.shape[0] == Y.shape[0]
    data = fot_TSTData(X, Y)
    tr, te = data.subsample(X.shape[0], seed=r+4).split_tr_te(tr_proportion=0.5, seed=r+5)

    met_opt_options = {'n_test_locs': J, 'max_iter': 200, 
            'locs_step_size': 0.1, 'gwidth_step_size': 0.1, 'seed': r+92856,
            'tol_fun': 1e-3}
    test_locs, gwidth, info = fot_MeanEmbeddingTest.optimize_locs_width(tr, alpha, **met_opt_options)
    met_opt = fot_MeanEmbeddingTest(test_locs, gwidth, alpha)
    met_opt_test  = met_opt.perform_test(te)

    return int(met_opt_test['h0_rejected'])


from freqopttest.tst import SmoothCFTest as fot_SmoothCFTest


# based on job_scf_opt() function 
# https://github.com/wittawatj/interpretable-test/blob/master/freqopttest/ex/ex1_power_vs_n.py
def scf(X, Y, r, J=10, alpha=0.05):
    """SmoothCFTest with frequencies optimized."""
    
    assert X.shape[0] == Y.shape[0]
    data = fot_TSTData(X, Y)
    tr, te = data.subsample(X.shape[0], seed=r+4).split_tr_te(tr_proportion=0.5, seed=r+5)
    
    op = {'n_test_freqs': J, 'max_iter': 200, 'freqs_step_size': 0.1, 
            'gwidth_step_size': 0.1, 'seed': r+92856, 'tol_fun': 1e-3}
    test_freqs, gwidth, info = fot_SmoothCFTest.optimize_freqs_width(tr, alpha, **op)
    scf_opt = fot_SmoothCFTest(test_freqs, gwidth, alpha)
    scf_opt_test = scf_opt.perform_test(te)

    return int(scf_opt_test['h0_rejected'])