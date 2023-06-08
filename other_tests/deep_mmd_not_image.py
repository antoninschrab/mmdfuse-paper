"""
The methods here are taken from Liu et al
https://github.com/fengliu90/DK-for-TST/blob/master/Baselines_Blob.py
"""
from sklearn.utils import check_random_state
from argparse import Namespace
import argparse
import os
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch
import pickle
from tqdm.auto import tqdm

# Setup seeds
np.random.seed(1102)
torch.manual_seed(1102)
torch.cuda.manual_seed(1102)
torch.backends.cudnn.deterministic = True
is_cuda = True

dtype = torch.float
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
cuda = True if torch.cuda.is_available() else False


def deep_mmd_not_image(sample_p, sample_q, n_epochs=1000):
    assert sample_p.shape[1] == sample_q.shape[1]
    sample_p = np.array(sample_p, dtype='float32')
    sample_q = np.array(sample_q, dtype='float32')
    
    # Setup for all experiments
    alpha = 0.05 # test threshold
    x_in = sample_p.shape[1] # number of neurons in the input layer, i.e., dimension of data
    H = 50 # number of neurons in the hidden layer
    x_out = 50 # number of neurons in the output layer
    learning_rate = 0.0005 # learning rate for MMD-D on Blob
    N_epoch = n_epochs # number of training epochs

    # prepare datasets
    sample_p = torch.from_numpy(sample_p)
    sample_q = torch.from_numpy(sample_q)
    
    # split data 50/50
    x_train, x_test = sample_p[:len(sample_p)//2], sample_p[len(sample_p)//2:]
    y_train, y_test = sample_q[:len(sample_q) // 2], sample_q[len(sample_q) // 2:]

    # Initialize parameters
    model_u = ModelLatentF(x_in, H, x_out)
    if cuda:
        model_u.cuda()
    epsilonOPT = MatConvert(np.random.rand(1) * (10 ** (-10)), device, dtype)
    epsilonOPT.requires_grad = True
    sigmaOPT = MatConvert(np.sqrt(np.random.rand(1) * 0.3), device, dtype)
    sigmaOPT.requires_grad = True
    sigma0OPT = MatConvert(np.sqrt(np.random.rand(1) * 0.002), device, dtype)
    sigma0OPT.requires_grad = True

    # Setup optimizer for training deep kernel
    optimizer_u = torch.optim.Adam(list(model_u.parameters())+[epsilonOPT]+[sigmaOPT]+[sigma0OPT], lr=learning_rate) #

    # Train deep kernel to maximize test power
    S = torch.cat([x_train.cpu(), y_train.cpu()], 0).to(device)
    # S = MatConvert(S, device, dtype)
    N1 = len(x_train)
    np.random.seed(seed=1102)
    torch.manual_seed(1102)
    torch.cuda.manual_seed(1102)
    for t in tqdm(range(N_epoch)):
        # Compute epsilon, sigma and sigma_0
        ep = torch.exp(epsilonOPT)/(1+torch.exp(epsilonOPT))
        sigma = sigmaOPT ** 2
        sigma0_u = sigma0OPT ** 2
        # Compute output of the deep network
        modelu_output = model_u(S)
        # Compute J (STAT_u)
        TEMP = MMDu(modelu_output, N1, S, sigma, sigma0_u, ep)
        mmd_value_temp = -1 * TEMP[0]
        mmd_std_temp = torch.sqrt(TEMP[1]+10**(-8))
        STAT_u = torch.div(mmd_value_temp, mmd_std_temp)
        # Initialize optimizer and Compute gradient
        optimizer_u.zero_grad()
        STAT_u.backward(retain_graph=True)
        # Update weights using gradient descent
        optimizer_u.step()

    # Compute test power of deep kernel based MMD
    S = torch.cat([x_test.cpu(), y_test.cpu()], 0).to(device)
    # S = MatConvert(S, device, dtype)
    N1 = len(x_test)
    N_per = 500
    alpha = 0.05
    # MMD-D
    dec, pvalue, _ = TST_MMD_u(model_u(S), N_per, N1, S, sigma, sigma0_u, alpha, device, dtype, ep)
    return dec


class ModelLatentF(torch.nn.Module):
    """Latent space for both domains."""
    def __init__(self, x_in, H, x_out):
        """Init latent features."""
        super(ModelLatentF, self).__init__()
        self.restored = False
        self.latent = torch.nn.Sequential(
            torch.nn.Linear(x_in, H, bias=True),
            torch.nn.Softplus(),
            torch.nn.Linear(H, H, bias=True),
            torch.nn.Softplus(),
            torch.nn.Linear(H, H, bias=True),
            torch.nn.Softplus(),
            torch.nn.Linear(H, x_out, bias=True),
        )
    def forward(self, input):
        """Forward the LeNet."""
        fealant = self.latent(input)
        return fealant


def MatConvert(x, device, dtype):
    """convert the numpy to a torch tensor."""
    x = torch.from_numpy(x).to(device, dtype)
    return x


def MMDu(Fea, len_s, Fea_org, sigma, sigma0=0.1, epsilon=10 ** (-10), is_smooth=True, is_var_computed=True, use_1sample_U=True):
    """compute value of deep-kernel MMD and std of deep-kernel MMD using merged data."""
    X = Fea[0:len_s, :] # fetch the sample 1 (features of deep networks)
    Y = Fea[len_s:, :] # fetch the sample 2 (features of deep networks)
    X_org = Fea_org[0:len_s, :] # fetch the original sample 1
    Y_org = Fea_org[len_s:, :] # fetch the original sample 2
    L = 1 # generalized Gaussian (if L>1)
    Dxx = Pdist2(X, X)
    Dyy = Pdist2(Y, Y)
    Dxy = Pdist2(X, Y)
    Dxx_org = Pdist2(X_org, X_org)
    Dyy_org = Pdist2(Y_org, Y_org)
    Dxy_org = Pdist2(X_org, Y_org)
    if is_smooth:
        Kx = (1-epsilon) * torch.exp(-(Dxx / sigma0) - (Dxx_org / sigma))**L + epsilon * torch.exp(-Dxx_org / sigma)
        Ky = (1-epsilon) * torch.exp(-(Dyy / sigma0) - (Dyy_org / sigma))**L + epsilon * torch.exp(-Dyy_org / sigma)
        Kxy = (1-epsilon) * torch.exp(-(Dxy / sigma0) - (Dxy_org / sigma))**L + epsilon * torch.exp(-Dxy_org / sigma)
    else:
        Kx = torch.exp(-Dxx / sigma0)
        Ky = torch.exp(-Dyy / sigma0)
        Kxy = torch.exp(-Dxy / sigma0)
    return h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed, use_1sample_U)


def TST_MMD_u(Fea, N_per, N1, Fea_org, sigma, sigma0, alpha, device, dtype, epsilon=10 ** (-10),is_smooth=True):
    """run two-sample test (TST) using deep kernel kernel."""
    TEMP = MMDu(Fea, N1, Fea_org, sigma, sigma0, epsilon,is_smooth)
    Kxyxy = TEMP[2]
    count = 0
    nxy = Fea.shape[0]
    nx = N1
    mmd_value_nn, p_val, rest = mmd2_permutations(Kxyxy, nx, permutations=200)
    if p_val > alpha:
        h = 0
    else:
        h = 1
    threshold = "NaN"
    return h,threshold,mmd_value_nn


def Pdist2(x, y):
    """compute the paired distance between x and y."""
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)
    Pdist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    Pdist[Pdist<0]=0
    return Pdist


def h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed, use_1sample_U=True):
    """compute value of MMD and std of MMD using kernel matrix."""
    Kxxy = torch.cat((Kx,Kxy),1)
    Kyxy = torch.cat((Kxy.transpose(0,1),Ky),1)
    Kxyxy = torch.cat((Kxxy,Kyxy),0)
    nx = Kx.shape[0]
    ny = Ky.shape[0]
    is_unbiased = True
    if is_unbiased:
        xx = torch.div((torch.sum(Kx) - torch.sum(torch.diag(Kx))), (nx * (nx - 1)))
        yy = torch.div((torch.sum(Ky) - torch.sum(torch.diag(Ky))), (ny * (ny - 1)))
        # one-sample U-statistic.
        if use_1sample_U:
            xy = torch.div((torch.sum(Kxy) - torch.sum(torch.diag(Kxy))), (nx * (ny - 1)))
        else:
            xy = torch.div(torch.sum(Kxy), (nx * ny))
        mmd2 = xx - 2 * xy + yy
    else:
        xx = torch.div((torch.sum(Kx)), (nx * nx))
        yy = torch.div((torch.sum(Ky)), (ny * ny))
        # one-sample U-statistic.
        if use_1sample_U:
            xy = torch.div((torch.sum(Kxy)), (nx * ny))
        else:
            xy = torch.div(torch.sum(Kxy), (nx * ny))
        mmd2 = xx - 2 * xy + yy
    if not is_var_computed:
        return mmd2, None

    hh = Kx+Ky-Kxy-Kxy.transpose(0,1)
    V1 = torch.dot(hh.sum(1)/ny,hh.sum(1)/ny) / ny
    V2 = (hh).sum() / (nx) / nx
    varEst = 4*(V1 - V2**2)
    if  varEst == 0.0:
        print('error!!'+str(V1))
    return mmd2, varEst, Kxyxy


def mmd2_permutations(K, n_X, permutations=500):
    """
        Fast implementation of permutations using kernel matrix.
    """
    K = torch.as_tensor(K)
    n = K.shape[0]
    assert K.shape[0] == K.shape[1]
    n_Y = n_X
    assert n == n_X + n_Y
    w_X = 1
    w_Y = -1
    ws = torch.full((permutations + 1, n), w_Y, dtype=K.dtype, device=K.device)
    ws[-1, :n_X] = w_X
    for i in range(permutations):
        ws[i, torch.randperm(n)[:n_X].numpy()] = w_X
    biased_ests = torch.einsum("pi,ij,pj->p", ws, K, ws)
    if True:  # u-stat estimator
        # need to subtract \sum_i k(X_i, X_i) + k(Y_i, Y_i) + 2 k(X_i, Y_i)
        # first two are just trace, but last is harder:
        is_X = ws > 0
        X_inds = is_X.nonzero()[:, 1].view(permutations + 1, n_X)
        Y_inds = (~is_X).nonzero()[:, 1].view(permutations + 1, n_Y)
        del is_X, ws
        cross_terms = K.take(Y_inds * n + X_inds).sum(1)
        del X_inds, Y_inds
        ests = (biased_ests - K.trace() + 2 * cross_terms) / (n_X * (n_X - 1))
    est = ests[-1]
    rest = ests[:-1]
    p_val = (rest > est).float().mean()
    return est.item(), p_val.item(), rest
