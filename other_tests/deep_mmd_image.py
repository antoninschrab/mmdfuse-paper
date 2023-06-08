"""
The methods here are taken from Liu et al:
https://github.com/fengliu90/DK-for-TST/blob/master/Deep_Baselines_CIFAR10.py
"""
from argparse import Namespace
import argparse
import os
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch
import torchvision 
import scipy.stats as stats
from tqdm.auto import tqdm

torch.backends.cudnn.deterministic = True
is_cuda = True

dtype = torch.float
device = torch.device("cuda:0")
cuda = True if torch.cuda.is_available() else False

def deep_mmd_image(sample_p, sample_q, n_epochs=1000):
    assert sample_p.shape[1] == sample_q.shape[1]
    
    # Setup seeds
    np.random.seed(819)
    torch.manual_seed(819)
    torch.cuda.manual_seed(819)
    
    # prepare datasets
    sample_p = np.array(sample_p, dtype='float32')
    sample_q = np.array(sample_q, dtype='float32')
    sample_p = torch.from_numpy(sample_p)
    sample_q = torch.from_numpy(sample_q)
    #sample_p = sample_p / 255
    #sample_q = sample_q / 255
    #sample_p = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(sample_p)
    #sample_q = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(sample_q)
    
    # split data 50/50
    x_train, x_test = sample_p[:opt.n // 2], sample_p[opt.n // 2:]
    y_train, y_test = sample_q[:opt.n // 2], sample_q[opt.n // 2:]
        
    # Parameters
    opt = Namespace()
    opt.n_epochs = n_epochs
    opt.batch_size = 100
    opt.img_size = sample_p.shape[-1]
    opt.orig_img_size = sample_p.shape[-1]
    opt.channels = sample_p.shape[1]
    opt.lr = 0.0002
    opt.n = sample_p.shape[0]
    N_per = 100 # permutation times
    alpha = 0.05 # test threshold

    # Loss function
    adversarial_loss = torch.nn.CrossEntropyLoss()

    # Define the deep network for MMD-D
    class Featurizer(nn.Module):
        def __init__(self):
            super(Featurizer, self).__init__()

            def discriminator_block(in_filters, out_filters, bn=True):
                block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0)] #0.25
                if bn:
                    block.append(nn.BatchNorm2d(out_filters, 0.8))
                return block

            self.model = nn.Sequential(
                *discriminator_block(opt.channels, 16, bn=False),
                *discriminator_block(16, 32),
                *discriminator_block(32, 64),
                *discriminator_block(64, 128),
            )

            # The height and width of downsampled image
            ds_size = opt.img_size // 2 ** 4
            self.adv_layer = nn.Sequential(
                nn.Linear(128 * ds_size ** 2, 300))

        def forward(self, img):
            out = self.model(img)
            out = out.view(out.shape[0], -1)
            feature = self.adv_layer(out)

            return feature

    featurizer = Featurizer()
    # Initialize parameters
    epsilonOPT = torch.log(MatConvert(np.random.rand(1) * 10 ** (-10), device, dtype))
    epsilonOPT.requires_grad = True
    sigmaOPT = MatConvert(np.ones(1) * np.sqrt(2 * 32 * 32), device, dtype)
    sigmaOPT.requires_grad = True
    sigma0OPT = MatConvert(np.ones(1) * np.sqrt(0.005), device, dtype)
    sigma0OPT.requires_grad = True
    if cuda:
        featurizer.cuda()
        adversarial_loss.cuda()

    # Initialize optimizers
    optimizer_F = torch.optim.Adam(list(featurizer.parameters()) + [epsilonOPT] + [sigmaOPT] + [sigma0OPT], lr=opt.lr)
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    
    # Dataloader
    dataloader_x_train = torch.utils.data.DataLoader(
        x_train,
        batch_size=opt.batch_size,
        shuffle=True,
    )

    # -----------------------------------------------------
    #  Training deep networks for MMD-D (called featurizer)
    # -----------------------------------------------------
    np.random.seed(seed=1102)
    torch.manual_seed(1102)
    torch.cuda.manual_seed(1102)
    for epoch in tqdm(range(opt.n_epochs)):
        for _, x_train_batch in enumerate(dataloader_x_train):
            ind = np.random.choice(y_train.shape[0], x_train_batch.shape[0], replace=False)
            y_train_batch = y_train[ind]

            x_train_batch = Variable(x_train_batch.type(Tensor))
            y_train_batch = Variable(y_train_batch.type(Tensor))
            X = torch.cat([x_train_batch, y_train_batch], 0)

            # ------------------------------
            #  Train deep network for MMD-D
            # ------------------------------
            # Initialize optimizer
            optimizer_F.zero_grad()
            # Compute output of deep network
            modelu_output = featurizer(X)
            # Compute epsilon, sigma and sigma_0
            ep = torch.exp(epsilonOPT) / (1 + torch.exp(epsilonOPT))
            sigma = sigmaOPT ** 2
            sigma0_u = sigma0OPT ** 2
            # Compute Compute J (STAT_u)
            TEMP = MMDu(modelu_output, x_train_batch.shape[0], X.reshape(X.shape[0], -1), sigma, sigma0_u, ep)
            mmd_value_temp = -1 * (TEMP[0])
            mmd_std_temp = torch.sqrt(TEMP[1] + 10 ** (-8))
            STAT_u = torch.div(mmd_value_temp, mmd_std_temp)
            # Compute gradient
            STAT_u.backward()
            # Update weights using gradient descent
            optimizer_F.step()

    # Run two-sample test on the test set
    S = torch.cat([x_test.cpu(), y_test.cpu()], 0).to(device)  
    Sv = S.view(x_test.shape[0] + y_test.shape[0], -1)
    h, threshold, mmd_value = TST_MMD_u(
        featurizer(S), 
        N_per, 
        x_test.shape[0], 
        Sv, 
        sigma, 
        sigma0_u, 
        ep, 
        alpha, 
        device, 
        dtype,
    )
    return h

# functions from utils_HD.py below

def get_item(x, is_cuda):
    """get the numpy value from a torch tensor."""
    if is_cuda:
        x = x.cpu().detach().numpy()
    else:
        x = x.detach().numpy()
    return x

def MatConvert(x, device, dtype):
    """convert the numpy to a torch tensor."""
    x = torch.from_numpy(x).to(device, dtype)
    return x

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
        return mmd2, None, Kxyxy
    hh = Kx+Ky-Kxy-Kxy.transpose(0,1)
    V1 = torch.dot(hh.sum(1)/ny,hh.sum(1)/ny) / ny
    V2 = (hh).sum() / (nx) / nx
    varEst = 4*(V1 - V2**2)
    if  varEst == 0.0:
        raise ValueError("error var")
    return mmd2, varEst, Kxyxy


def MMDu(Fea, len_s, Fea_org, sigma, sigma0=0.1, epsilon = 10**(-10), is_smooth=True, is_var_computed=True, use_1sample_U=True):
    """compute value of deep-kernel MMD and std of deep-kernel MMD using merged data."""
    X = Fea[0:len_s, :] # fetch the sample 1 (features of deep networks)
    Y = Fea[len_s:, :] # fetch the sample 2 (features of deep networks)
    X_org = Fea_org[0:len_s, :] # fetch the original sample 1
    Y_org = Fea_org[len_s:, :] # fetch the original sample 2
    L = 1 # generalized Gaussian (if L>1)

    nx = X.shape[0]
    ny = Y.shape[0]
    Dxx = Pdist2(X, X)
    Dyy = Pdist2(Y, Y)
    Dxy = Pdist2(X, Y)
    Dxx_org = Pdist2(X_org, X_org)
    Dyy_org = Pdist2(Y_org, Y_org)
    Dxy_org = Pdist2(X_org, Y_org)
    K_Ix = torch.eye(nx).cuda()
    K_Iy = torch.eye(ny).cuda()
    if is_smooth:
        Kx = (1-epsilon) * torch.exp(-(Dxx / sigma0)**L -Dxx_org / sigma) + epsilon * torch.exp(-Dxx_org / sigma)
        Ky = (1-epsilon) * torch.exp(-(Dyy / sigma0)**L -Dyy_org / sigma) + epsilon * torch.exp(-Dyy_org / sigma)
        Kxy = (1-epsilon) * torch.exp(-(Dxy / sigma0)**L -Dxy_org / sigma) + epsilon * torch.exp(-Dxy_org / sigma)
    else:
        Kx = torch.exp(-Dxx / sigma0)
        Ky = torch.exp(-Dyy / sigma0)
        Kxy = torch.exp(-Dxy / sigma0)
    return h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed, use_1sample_U)


def TST_MMD_u(Fea, N_per, N1, Fea_org, sigma, sigma0, ep, alpha, device, dtype, is_smooth=True):
    """run two-sample test (TST) using deep kernel kernel."""
    mmd_vector = np.zeros(N_per)
    TEMP = MMDu(Fea, N1, Fea_org, sigma, sigma0, ep, is_smooth)
    mmd_value = get_item(TEMP[0], is_cuda)
    Kxyxy = TEMP[2]
    count = 0
    nxy = Fea.shape[0]
    nx = N1
    for r in range(N_per):
        # print r
        ind = np.random.choice(nxy, nxy, replace=False)
        # divide into new X, Y
        indx = ind[:nx]
        indy = ind[nx:]
        Kx = Kxyxy[np.ix_(indx, indx)]
        Ky = Kxyxy[np.ix_(indy, indy)]
        Kxy = Kxyxy[np.ix_(indx, indy)]

        TEMP = h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed=False)
        mmd_vector[r] = TEMP[0]
        if mmd_vector[r] > mmd_value:
            count = count + 1
        if count > np.ceil(N_per * alpha):
            h = 0
            threshold = "NaN"
            break
        else:
            h = 1
    if h == 1:
        S_mmd_vector = np.sort(mmd_vector)
        threshold = S_mmd_vector[np.int(np.ceil(N_per * (1 - alpha)))]
    return h, threshold, mmd_value.item()

