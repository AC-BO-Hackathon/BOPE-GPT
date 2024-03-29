import os
import warnings
from itertools import combinations

import numpy as np
import torch
import time

# Suppress potential optimization warnings for cleaner notebook
warnings.filterwarnings("ignore")

SMOKE_TEST = os.environ.get("SMOKE_TEST")

import DTLZ2_model
import fischer_model
from botorch.test_functions.multi_objective import DTLZ2
from DTLZ2_model import neg_l1_dist
from DTLZ2_model import predict_DTLZ2_model
from fischer_model import predict_fischer_model
import tensorflow as tf
import random


from botorch.fit import fit_gpytorch_mll 
from botorch.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood 
from botorch.models.transforms.input import Normalize
from botorch.acquisition.preference import AnalyticExpectedUtilityOfBestOption
from botorch.optim import optimize_acqf

import pickle

#utils
# data generating helper functions
#function that defines the comparisons
def utility(X):
    y=predict_fischer_model(X)
    #weighted_y = y * torch.sqrt(torch.arange(y.size(-1), dtype=torch.float) + 1)
    #y = torch.sum(weighted_y, dim=-1)
    return y

def utility1(X): #The four outputs are equally important, and we want to maximize all of them.
    y=predict_fischer_model(X)
    y = torch.sum(y, dim=-1)
    return y

    
def ini(n,dim):
    X = torch.rand(n, dim, dtype=torch.float64)
    return X

def generate_data(X, dim=4):
    """Generate data X and y"""
    # X is randomly sampled from dim-dimentional unit cube
    # we recommend using double as opposed to float tensor here for
    # better numerical stability
    #X=ini(n,dim)
    x_2=tf.convert_to_tensor(X, dtype=tf.float32)
    y = utility(x_2)
    return y

def generate_data_u1(X, dim=4):
    """Generate data X and y"""
    # X is randomly sampled from dim-dimentional unit cube
    # we recommend using double as opposed to float tensor here for
    # better numerical stability
    #X=ini(n,dim)
    x_2=tf.convert_to_tensor(X, dtype=tf.float32)
    y = utility1(x_2)
    return y
def generate_comparisons(y, n_comp, noise=0.1, replace=False):
    """Create pairwise comparisons with noise"""
    # generate all possible pairs of elements in y
    all_pairs = np.array(list(combinations(range(y.shape[0]), 2)))
    # randomly select n_comp pairs from all_pairs
    comp_pairs = all_pairs[
        np.random.choice(range(len(all_pairs)), n_comp, replace=replace)
    ]
    # add gaussian noise to the latent y values
    c0 = y[comp_pairs[:, 0]] + np.random.standard_normal(len(comp_pairs)) * noise
    c1 = y[comp_pairs[:, 1]] + np.random.standard_normal(len(comp_pairs)) * noise
    reverse_comp = (c0 < c1).numpy()
    comp_pairs[reverse_comp, :] = np.flip(comp_pairs[reverse_comp, :], 1)
    comp_pairs = torch.tensor(comp_pairs).long()

    return comp_pairs


#wrapper for the model
def init_and_fit_model(X, comp):
    """Model fitting helper function"""
    model = PairwiseGP(
        X,
        comp,
        input_transform=Normalize(d=X.shape[-1]),
    )
    mll = PairwiseLaplaceMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return mll, model



def make_new_data(X, next_X, comps, q_comp):
    """Given X and next_X,
    generate q_comp new comparisons between next_X
    and return the concatenated X and comparisons
    """
    # next_X is float by default; cast it to the dtype of X (i.e., double)
    next_X = next_X.to(X)
    x_2=tf.convert_to_tensor(next_X, dtype=tf.float32)
    next_y = utility(x_2)
    next_comps = generate_comparisons_llm(next_y, n_comp=q_comp)
    comps = torch.cat([comps, next_comps + X.shape[-2]])
    X = torch.cat([X, next_X])
    return X, comps

def make_new_data_u1(X, next_X, comps, q_comp):
    """Given X and next_X,
    generate q_comp new comparisons between next_X
    and return the concatenated X and comparisons
    """
    # next_X is float by default; cast it to the dtype of X (i.e., double)
    next_X = next_X.to(X)
    x_2=tf.convert_to_tensor(next_X, dtype=tf.float32)
    next_y = utility1(x_2)
    next_comps = generate_comparisons(next_y, n_comp=q_comp)
    comps = torch.cat([comps, next_comps + X.shape[-2]])
    X = torch.cat([X, next_X])
    return X, comps