import os
import warnings
from itertools import combinations
import numpy as np
import torch
import time
import tensorflow as tf
import random
import cohere
from botorch.test_functions.multi_objective import DTLZ2
from botorch.fit import fit_gpytorch_mll
from botorch.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from botorch.models.transforms.input import Normalize
from botorch.acquisition.preference import AnalyticExpectedUtilityOfBestOption
from botorch.optim import optimize_acqf
from keras.models import Sequential
from keras.layers import Dense
import json

warnings.filterwarnings("ignore")
SMOKE_TEST = os.environ.get("SMOKE_TEST")

# Cohere client initialization
co = cohere.Client("")


# DTLZ2 model functions
def neg_l1_dist(Y: torch.Tensor, X: torch.Tensor = None) -> torch.Tensor:
    if len(Y.shape) == 1:
        Y = Y.unsqueeze(0)
    dist = torch.cdist(
        Y, torch.full(Y.shape[-1:], fill_value=0.5, dtype=Y.dtype).unsqueeze(0), p=1
    ).squeeze(-1)
    return -dist


def predict_DTLZ2_model(x):
    X_dim, Y_dim = 5, 4
    problem = DTLZ2(dim=X_dim, num_objectives=Y_dim)
    return problem(x)


def DTLZ2_util(y):
    return neg_l1_dist(y)


# Fischer model functions
def load_fischer_model():
    N_neurons, N_layers = 20, 1
    model = Sequential()
    model.add(Dense(N_neurons, activation="sigmoid", input_dim=4))
    model.add(Dense(units=4))
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])

    with open("fischer_ann_weights.json", "r") as f:
        weights_list = json.load(f)
    weights = [np.array(w) for w in weights_list]
    model.set_weights(weights)
    return model


fischer_model = load_fischer_model()


def predict_fischer_model(X):
    y_pred = fischer_model.predict(X)
    return torch.Tensor(y_pred)


# Utility functions
def utility(X):
    x_2 = tf.convert_to_tensor(X, dtype=tf.float32)
    y = predict_fischer_model(x_2)
    return torch.sum(y, dim=-1)


def ini(n, dim):
    return torch.rand(n, dim, dtype=torch.float64)


def generate_data(X, dim=4):
    x_2 = tf.convert_to_tensor(X, dtype=tf.float32)
    return utility(x_2)


def generate_comparisons_llm(y, n_comp, replace=False):
    all_pairs = np.array(list(combinations(range(y.shape[0]), 2)))
    comp_pairs = all_pairs[
        np.random.choice(range(len(all_pairs)), n_comp, replace=replace)
    ]
    new_pairs = []
    for opto in comp_pairs:
        firstoption, secondoption = opto[0], opto[1]
        numfirst, numsecond = y[firstoption, :], y[secondoption, :]
        mess = f"Suppose you're managing a Fischer-Tropsch synthesis process. The four outputs are equally important, and we want to maximize all of them. Option A: regime of {numfirst[0]:.1f} CO conversion, {numfirst[1]:.1f} methane production, {numfirst[2]:.1f} paraffins, {numfirst[3]:.1f} light olefins. Option B: regime of {numsecond[0]:.1f} CO conversion, {numsecond[1]:.1f} methane production, {numsecond[2]:.1f} paraffins, {numsecond[3]:.1f} light olefins. Choose only one option, only answer with 'Option A' or 'Option B'"
        response = co.chat(message=mess)
        new_pairs.append(
            opto.tolist() if "Option A" in response.text else list(reversed(opto))
        )
        time.sleep(6)  # API restrictions: 20 API calls/minute
    return torch.tensor(new_pairs)


def init_and_fit_model(X, comp):
    model = PairwiseGP(X, comp, input_transform=Normalize(d=X.shape[-1]))
    mll = PairwiseLaplaceMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return mll, model


def make_new_data(X, next_X, comps, q_comp):
    next_X = next_X.to(X)
    x_2 = tf.convert_to_tensor(next_X, dtype=tf.float32)
    next_y = utility(x_2)
    next_comps = generate_comparisons_llm(next_y, n_comp=q_comp)
    comps = torch.cat([comps, next_comps + X.shape[-2]])
    X = torch.cat([X, next_X])
    return X, comps


# Main functions
def initialize_bope(dim, q_inidata, q_comp_ini):
    torch.manual_seed(0)
    np.random.seed(0)

    bounds = torch.stack([torch.zeros(dim), torch.ones(dim)])
    init_X = ini(q_inidata, dim)
    init_y = generate_data(init_X, dim=dim)
    comparisons = generate_comparisons_llm(init_y, q_comp_ini)

    _, model = init_and_fit_model(init_X, comparisons)
    best_val = utility(init_X).max().item()

    return {
        "X": init_X,
        "comparisons": comparisons,
        "model": model,
        "best_val": best_val,
        "bounds": bounds,
    }


def run_next_iteration(state, q_eubo=2, q_comp_cycle=1):
    NUM_RESTARTS, RAW_SAMPLES = 3, 512 if not SMOKE_TEST else 8

    acq_func = AnalyticExpectedUtilityOfBestOption(pref_model=state["model"])
    next_X, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=state["bounds"],
        q=q_eubo,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
    )

    X, comps = make_new_data(state["X"], next_X, state["comparisons"], q_comp_cycle)
    _, model = init_and_fit_model(X, comps)
    max_val = utility(X).max().item()

    state.update(
        {
            "X": X,
            "comparisons": comps,
            "model": model,
            "best_val": max(state["best_val"], max_val),
        }
    )

    return state


# Example usage
if __name__ == "__main__":
    dim, q_inidata, q_comp_ini = 4, 5, 10
    state = initialize_bope(dim, q_inidata, q_comp_ini)
    print(f"Initial best value: {state['best_val']}")

    for i in range(5):  # Run 5 iterations
        state = run_next_iteration(state)
        print(f"Iteration {i+1} best value: {state['best_val']}")
