import os
import warnings
from itertools import combinations

import numpy as np
import torch
import time

# Suppress potential optimization warnings for cleaner notebook
warnings.filterwarnings("ignore")

SMOKE_TEST = os.environ.get("SMOKE_TEST")

#import DTLZ2_model
#import fischer_model
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


<<<<<<< HEAD
from utils_model.utils_1 utils_1 import generate_data,generate_data_u1,init_and_fit_model
=======
from utils_model.utils_1 import ini, generate_comparisons, make_new_data_u1, make_new_data, utility1, generate_data,generate_data_u1,init_and_fit_model
>>>>>>> 864031a20db346808232a745d5ba056e9ee6fae8
from utils_model.utils_llm import generate_comparisons_llm

# setting SEED
i = 1

#algos = ["EUBO","EUBO-LLM", "rand"]
def run_one_iteration_initial(algos,dim,q_inidata, prompt=None):

    #sampler options
    NUM_RESTARTS = 3
    RAW_SAMPLES = 512 if not SMOKE_TEST else 8

    #number of inputs
    #dim = 4
    #q_inidata=5 #number_of_initial_samples
    

    #interface
    #let's keep it fixed
    q_eubo = 2  # number of points per query
    q_comp_ini = 10  # number of comparisons per query
    q_comp_cycle=1 #tow options, one comp

    # initial evals keep the best values after each iteration
    best_vals = {}  # best observed values
    for algo in algos:
        best_vals[algo] = []

    # average over multiple trials

    #seed and dictionaries
    torch.manual_seed(i)
    np.random.seed(i)
    data = {}
    models = {}

     # X are within the unit cube
    bounds = torch.stack([torch.zeros(dim), torch.ones(dim)])

     # Create initial data
    init_X=ini(q_inidata,dim)

    #evaluate utility function and generate comparision, initial part

    if algo == "EUBO-LLM":
        init_y = generate_data(init_X, dim=dim)
        #add argument for LLM
        comparisons = generate_comparisons_llm(init_y, q_comp_ini)
    if algo == "EUBO":
        init_y = generate_data_u1(init_X, dim=dim)
        comparisons = generate_comparisons(init_y, q_comp_ini)
    if algo == "rand":
        init_y = generate_data_u1(init_X, dim=dim)
        comparisons = generate_comparisons(init_y, q_comp_ini)


    #saving the best value
    best_vals[algo].append([])
    #saving the data
    data[algo] = (init_X, comparisons)
    #surrogate model
    _, models[algo] = init_and_fit_model(init_X, comparisons)

    #evaluation of the initial_data and best value append
    best_next_y = utility1(init_X).max().item()
    best_vals[algo][-1].append(best_next_y)

    model = models[algo]

    #
    if algo == "EUBO-LLM":
        # create the acquisition function objec
        acq_func = AnalyticExpectedUtilityOfBestOption(pref_model=model)
        # optimize and get new observation
        next_X, acq_val = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=q_eubo,
                num_restarts=NUM_RESTARTS,
                raw_samples=RAW_SAMPLES,
        )
             
        # update data
        X, comps = data[algo]
        X, comps = make_new_data(X, next_X, comps, q_comp_cycle)
        data[algo] = (X, comps)
        
        # refit models
        _, models[algo] = init_and_fit_model(X, comps)
        
        # record the best observed values so far
        max_val = utility1(X).max().item()
        best_vals[algo][-1].append(max_val)
    elif algo == "EUBO":
        #create the acquisition function object
        acq_func = AnalyticExpectedUtilityOfBestOption(pref_model=model)
        # optimize and get new observation
        next_X, acq_val = optimize_acqf(
                        acq_function=acq_func,
                        bounds=bounds,
                        q=q_eubo,
                        num_restarts=NUM_RESTARTS,
                        raw_samples=RAW_SAMPLES,
                    )
        print(next_X)
            # update data
        X, comps = data[algo]
        X, comps = make_new_data_u1(X, next_X, comps, q_comp_cycle)
        data[algo] = (X, comps)

                # refit models
        _, models[algo] = init_and_fit_model(X, comps)

                # record the best observed values so far
        max_val = utility1(X).max().item()
        best_vals[algo][-1].append(max_val)
    else:
        # randomly sample data
        next_X= ini(q_eubo, dim=dim)
        print(next_X)
        # update data
        X, comps = data[algo]
        X, comps = make_new_data_u1(X, next_X, comps, q_comp_cycle)
        data[algo] = (X, comps)

        # refit models
        _, models[algo] = init_and_fit_model(X, comps)

        # record the best observed values so far
        max_val = utility1(X).max().item()
        best_vals[algo][-1].append(max_val)

    return data,best_vals

def run_one_iteration_normal(algo,dim,q_inidata,best_vals,data, prompt=None):

    #sampler options
    NUM_RESTARTS = 3
    RAW_SAMPLES = 512 if not SMOKE_TEST else 8

    q_eubo = 2  # number of points per query
    q_comp_ini = 10  # number of comparisons per query
    q_comp_cycle=1 #tow options, one comp

    # initial evals keep the best values after each iteration
    best_vals = {}  # best observed values
    for algo in algos:
        best_vals[algo] = []

    # average over multiple trials

    #seed and dictionaries
    torch.manual_seed(i)
    np.random.seed(i)
    data = {}
    models = {}

     # X are within the unit cube
    bounds = torch.stack([torch.zeros(dim), torch.ones(dim)])

   
    #surrogate model
    _, models[algo] = init_and_fit_model(init_X, comparisons)

    model = models[algo]

    #
    if algo == "EUBO-LLM":
        # create the acquisition function objec
        acq_func = AnalyticExpectedUtilityOfBestOption(pref_model=model)
        # optimize and get new observation
        next_X, acq_val = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=q_eubo,
                num_restarts=NUM_RESTARTS,
                raw_samples=RAW_SAMPLES,
        )
             
        # update data
        X, comps = data[algo]
        X, comps = make_new_data(X, next_X, comps, q_comp_cycle)
        data[algo] = (X, comps)
        
        # refit models
        _, models[algo] = init_and_fit_model(X, comps)
        
        # record the best observed values so far
        max_val = utility1(X).max().item()
        best_vals[algo][-1].append(max_val)
    elif algo == "EUBO":
        #create the acquisition function object
        acq_func = AnalyticExpectedUtilityOfBestOption(pref_model=model)
        # optimize and get new observation
        next_X, acq_val = optimize_acqf(
                        acq_function=acq_func,
                        bounds=bounds,
                        q=q_eubo,
                        num_restarts=NUM_RESTARTS,
                        raw_samples=RAW_SAMPLES,
                    )
        print(next_X)
            # update data
        X, comps = data[algo]
        X, comps = make_new_data_u1(X, next_X, comps, q_comp_cycle)
        data[algo] = (X, comps)

                # refit models
        _, models[algo] = init_and_fit_model(X, comps)

                # record the best observed values so far
        max_val = utility1(X).max().item()
        best_vals[algo][-1].append(max_val)
    else:
        # randomly sample data
        next_X= ini(q_eubo, dim=dim)
        print(next_X)
        # update data
        X, comps = data[algo]
        X, comps = make_new_data_u1(X, next_X, comps, q_comp_cycle)
        data[algo] = (X, comps)

        # refit models
        _, models[algo] = init_and_fit_model(X, comps)

        # record the best observed values so far
        max_val = utility1(X).max().item()
        best_vals[algo][-1].append(max_val)

    return data, best_vals