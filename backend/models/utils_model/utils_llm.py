import cohere

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

co = cohere.Client('9ylnov4iFULBLovujZIJLq6x8pkq4NkyNCw0oePR')

#remember to reduce the number of comparisons n_comp,
# Add prompt input
def generate_comparisons_llm(y, n_comp, replace=False):
    """Create pairwise comparisons with noise"""
    # generate all possible pairs of elements in y
    all_pairs = np.array(list(combinations(range(y.shape[0]), 2)))
    # randomly select n_comp pairs from all_pairs
    comp_pairs = all_pairs[
        np.random.choice(range(len(all_pairs)), n_comp, replace=replace)
    ]
    #parsing the tensor to get the strings for the LLM
    new_pairs=[]
    for opto in comp_pairs:
        firstoption=opto[0]
        secondoption=opto[1]
        numfirst=(y[firstoption,:])
        firstoutput1=f"{numfirst[0].cpu().numpy():.1f}"
        firstoutput2=f"{numfirst[1].cpu().numpy():.1f}"
        firstoutput3=f"{numfirst[2].cpu().numpy():.1f}"
        firstoutput4=f"{numfirst[3].cpu().numpy():.1f}"
        numsecond=(y[secondoption,:])
        secondoutput1=f"{numsecond[0].cpu().numpy():.1f}"
        secondoutput2=f"{numsecond[1].cpu().numpy():.1f}"
        secondoutput3=f"{numsecond[2].cpu().numpy():.1f}"
        secondoutput4=f"{numsecond[3].cpu().numpy():.1f}"
        mess="Suppose you're managing a Fischer-Tropsch synthesis process. The four outputs are equally important, and we want to maximize all of them. Option A: regime of "+firstoutput1+" CO conversion, "+firstoutput2+" methane production, "+firstoutput3+" paraffins, "+firstoutput4+" light olefins. Option B: regime of "+secondoutput1+" CO conversion, "+secondoutput2+" methane production, "+secondoutput3+" paraffins, "+secondoutput4+" light olefins. Choose only one option, only answer with 'Option A' or 'Option B'"
        print(mess)
        response = co.chat(message=mess,
        #perform web search before answering the question. You can also use your own custom connector.
                          #connectors=[{"id": "web-search"}]
        )
        print(response.text)
        opllm=response.text
        
        if "Option A" in opllm:
            new_pairs.append(opto.tolist())
        else:
            new_pairs.append(list(reversed(opto)))
        #api restrictions 20 API calls/minutes
        time.sleep(6)
    
    return torch.tensor(new_pairs)