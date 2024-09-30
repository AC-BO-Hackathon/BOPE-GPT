import os
import warnings
from typing import Any, Tuple
import numpy as np
import torch
import time
import tensorflow as tf
import cohere
from botorch.fit import fit_gpytorch_mll
from botorch.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from botorch.models.transforms.input import Normalize
from botorch.acquisition.preference import AnalyticExpectedUtilityOfBestOption
from botorch.optim import optimize_acqf
from keras.models import Sequential
from keras.layers import Dense
import json
from itertools import combinations
import asyncio
import aiofiles

from models import BopeState, State

warnings.filterwarnings("ignore")
SMOKE_TEST = os.environ.get("SMOKE_TEST")

# Cohere client initialization
cohere_api_key = os.getenv("COHERE_API_KEY", "-")
co = cohere.Client(cohere_api_key)


# Load Fischer model
async def load_fischer_model() -> Sequential:
    N_neurons, N_layers = 20, 1
    model = Sequential()
    model.add(Dense(N_neurons, activation="sigmoid", input_dim=4))
    for _ in range(N_layers):
        model.add(Dense(units=4))
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])

    async with aiofiles.open("fischer_ann_weights.json", "r") as f:
        weights_list = json.loads(await f.read())
    weights = [np.array(w) for w in weights_list]
    model.set_weights(weights)
    return model


# fischer_model = await load_fischer_model()

# note: slight diff b/w torch.tensor and torch.Tensor, keep in mind.
# also note: two kinds of tensors are used here- PyTorch and Tensorflow's


def predict_fischer_model(X: torch.Tensor, fischer_model: Sequential) -> torch.Tensor:
    y_pred = fischer_model.predict(X)
    return torch.tensor(y_pred)


def utility(X: tf.Tensor, fischer_model) -> torch.Tensor:
    x_2 = tf.convert_to_tensor(X, dtype=tf.float32)
    y = predict_fischer_model(x_2, fischer_model)
    return y


def ini(n: int, dim: int) -> torch.Tensor:
    return torch.rand(n, dim, dtype=torch.float64)


def generate_data(
    X: tf.Tensor, fischer_model: Sequential, dim: int = 4
) -> torch.Tensor:
    x_2 = tf.convert_to_tensor(X, dtype=tf.float32)
    return utility(x_2, fischer_model)


async def generate_comparisons_llm(
    y: torch.Tensor, n_comp: int, replace: bool = False
) -> torch.Tensor:
    all_pairs = np.array(
        list(combinations(range(y.shape[0]), 2))
    )  # len(y)=2 (hardcoded) when this is called from next iteration, and len(y)=q_inidata if called during initialization
    comp_pairs = all_pairs[
        np.random.choice(range(len(all_pairs)), n_comp, replace=replace)
    ]
    new_pairs = []
    print(f"\n Comp_pairs = {comp_pairs}")
    for opto in comp_pairs:
        firstoption, secondoption = opto[0], opto[1]
        numfirst, numsecond = y[firstoption, :], y[secondoption, :]
        mess = f"Suppose you're managing a Fischer-Tropsch synthesis process. The four outputs are equally important, and we want to maximize all of them. Option A: regime of {numfirst[0]:.1f} CO conversion, {numfirst[1]:.1f} methane production, {numfirst[2]:.1f} paraffins, {numfirst[3]:.1f} light olefins. Option B: regime of {numsecond[0]:.1f} CO conversion, {numsecond[1]:.1f} methane production, {numsecond[2]:.1f} paraffins, {numsecond[3]:.1f} light olefins. Choose only one option, only answer with 'Option A' or 'Option B'"
        response = co.chat(message=mess)
        print(f"\n LLM Response = {response}")
        new_pairs.append(
            opto.tolist() if "Option A" in response.text else list(reversed(opto))
        )
        await asyncio.sleep(6)  # API restrictions: 20 API calls/minute
    return torch.tensor(new_pairs)


def init_and_fit_model(
    X: torch.tensor, comp: torch.tensor
) -> Tuple[PairwiseLaplaceMarginalLogLikelihood, PairwiseGP]:
    model = PairwiseGP(X, comp, input_transform=Normalize(d=X.shape[-1]))
    mll = PairwiseLaplaceMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return mll, model


async def make_new_data(
    X: torch.Tensor,
    next_X: torch.Tensor,
    comps: torch.Tensor,
    q_comp: int,
    fischer_model: Sequential,
) -> Tuple[torch.Tensor, torch.Tensor]:
    next_X = next_X.to(X)
    x_2 = tf.convert_to_tensor(next_X, dtype=tf.float32)
    next_y = utility(
        x_2, fischer_model
    )  # outputs for the inputs chosen by acquistion function are generated via the ground truth model
    next_comps = await generate_comparisons_llm(
        next_y, n_comp=q_comp
    )  # llm compares the two outputs and rearranges them in the order of preference
    comps = torch.cat([comps, next_comps + X.shape[-2]])
    X = torch.cat([X, next_X])
    return X, comps


async def initialize_bope(
    dim: int,
    q_inidata: int,
    q_comp_ini: int,
    bounds: list,
    column_names: list[str],
) -> BopeState:
    print(f"\n Initializing BOPE...")
    torch.manual_seed(0)
    np.random.seed(0)

    # input_bounds = torch.stack(
    #    [torch.zeros(dim), torch.ones(dim)]
    # )  # bounds hard coded between 0 and 1 for all columns

    print(bounds)

    # Convert the first 'dim' number of lists in `bounds` to a tensor
    input_bounds = torch.tensor(bounds[:dim]).T

    print(f"\n Input_bounds = {input_bounds}")

    fischer_model = await load_fischer_model()

    init_X = ini(q_inidata, dim)
    init_y = generate_data(init_X, fischer_model, dim=dim)

    print(f"\n init_x = {init_X}\n init_y = {init_y}\n")
    comparisons = await generate_comparisons_llm(init_y, q_comp_ini)

    print(f"\nGenerated LLM comparisons = {comparisons}")

    _, model = init_and_fit_model(init_X, comparisons)

    print(f"\n Initialized model.")
    best_val = utility(init_X, fischer_model).max(dim=0).values

    print(f"\n best_val = {best_val}")

    input_columns = column_names[:dim]

    return BopeState(
        iteration=1,  # initialization iteration
        X=init_X,
        comparisons=comparisons,
        model=model,
        best_val=best_val,
        input_bounds=input_bounds,
        input_columns=input_columns,
        last_iteration_duration=None,
        updated_at=None,
    )


async def run_next_iteration(
    bope_state: BopeState, model: Any, q_eubo: int = 2, q_comp_cycle: int = 1
) -> BopeState:
    NUM_RESTARTS, RAW_SAMPLES = 3, 512 if not SMOKE_TEST else 8

    fischer_model = await load_fischer_model()

    acq_func = AnalyticExpectedUtilityOfBestOption(pref_model=model)
    next_X, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=bope_state.input_bounds,
        q=q_eubo,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
    )

    print(f"\n next_X = {next_X}")

    X, comps = await make_new_data(
        bope_state.X, next_X, bope_state.comparisons, q_comp_cycle, fischer_model
    )

    print(f"\n X, comps = {X}, {comps}")

    _, model = init_and_fit_model(X, comps)

    current_best = utility(X, fischer_model).max(dim=0).values
    bope_state.best_val = torch.max(bope_state.best_val, current_best)

    bope_state: BopeState = bope_state.copy(
        update={
            "iteration": bope_state.iteration + 1,
            "X": X,
            "comparisons": comps,
        }
    )

    return bope_state