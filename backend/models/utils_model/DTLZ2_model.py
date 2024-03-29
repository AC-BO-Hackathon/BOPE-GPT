import os
from typing import Optional, Tuple

import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import torch
from botorch.acquisition import (
    GenericMCObjective,
    LearnedObjective,
    MCAcquisitionObjective,
)
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.monte_carlo import qSimpleRegret
from botorch.acquisition.preference import AnalyticExpectedUtilityOfBestOption
from botorch.fit import fit_gpytorch_mll
from botorch.models.deterministic import FixedSingleSampleModel
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim.optimize import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from botorch.test_functions.multi_objective import DTLZ2
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood


# Set plotting colors
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=colors)


#define DTLZ2 problem and utility function
def neg_l1_dist(Y: torch.Tensor, X: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Negative L1 distance from a Pareto optimal points"""
    if len(Y.shape) == 1:
        Y = Y.unsqueeze(0)
    dist = torch.cdist(
        Y, torch.full(Y.shape[-1:], fill_value=0.5, dtype=Y.dtype).unsqueeze(0), p=1
    ).squeeze(-1)
    return -dist



def predict_DTLZ2_model(x):
    X_dim = 5
    Y_dim = 4
    problem = DTLZ2(dim=X_dim, num_objectives=Y_dim)
    y=problem(x)
    return y


def DTLZ2_util(y):
    util_func = neg_l1_dist
    return util_func(y)