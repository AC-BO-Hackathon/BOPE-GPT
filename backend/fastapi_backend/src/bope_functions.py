import os
import warnings
from typing import Any, Tuple, Dict
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

# Getting Pydantic models
from models import BopeState, State, ContourDataModel, VisualizationDataModel

# Getting helper functions
from helpers import matplotlib_visualization

import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")
SMOKE_TEST = os.environ.get("SMOKE_TEST")

# Cohere client initialization
cohere_api_key = os.getenv("COHERE_API_KEY", "-")
co = cohere.Client(cohere_api_key)


# Load Fischer model
async def load_fischer_model() -> Sequential:
    """
    Creates an ANN with predetermined weights representing the Fischer Tropsch dataset
    """
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
    """
    Predicts the output of the Fischer model for given input tensor X
    """
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
    model = PairwiseGP(
        X, comp, input_transform=Normalize(d=X.shape[-1]), jitter=1e-3
    )  # note: the pairwiseGP learns the latent utility function, NOT the actual output values of the dataset
    # ques: set a separate GP model (with customizable kernel) in parallel to learn to actual output values of the dataset given certain inputs?
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

    # Generate visualization data
    default_input_pairs = [0, 1]
    print("Generating visualization data...")
    vis_data: VisualizationDataModel = generate_contour_visualization_data(
        model, input_bounds, default_input_pairs
    )
    print(f"\n Vis_data.contour_data.keys(): {vis_data.contour_data.keys()}")

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
        visualization_data=vis_data,
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

    # Generate visualization data
    default_input_pairs = [0, 1]
    print("Generating visualization data...")
    vis_data: VisualizationDataModel = generate_contour_visualization_data(
        model, bope_state.input_bounds, default_input_pairs
    )
    print(f"\n Vis_data.contour_data.keys(): {vis_data.contour_data.keys()}")

    bope_state: BopeState = bope_state.model_copy(
        update={
            "iteration": bope_state.iteration + 1,
            "X": X,
            "comparisons": comps,
            "visualization_data": vis_data,
        }
    )

    return bope_state


def generate_contour_visualization_data(
    model: PairwiseGP,
    input_bounds: torch.Tensor,
    input_pairs: list[int],
    resolution: int = 50,
) -> VisualizationDataModel:
    """
    Generate data for visualizing the PairwiseGP model.

    Args:
        model (PairwiseGP): The trained PairwiseGP model.
        input_bounds (torch.Tensor): A tensor of shape (2, num_inputs) specifying the lower and upper bounds for each input.
        input_pairs (list[int]): The pair of input features to plot as plane axes of the contour plots.
        resolution (int): The number of points to sample in each dimension for the contour plots.

    Returns:
        Dict[str, Any]: A dictionary containing the visualization data.
    """

    num_inputs = input_bounds.shape[1]
    num_outputs = (
        model.num_outputs
    )  #  = 1 for pairwiseGP (represents preference degree of all inputs provided as a whole)

    # Generate grid only for provided pair of inputs
    grid_data = {}
    i, j = input_pairs
    x = torch.linspace(input_bounds[0, i], input_bounds[1, i], resolution)
    y = torch.linspace(input_bounds[0, j], input_bounds[1, j], resolution)
    grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
    grid_data[f"input_{i}_{j}"] = (grid_x, grid_y)

    # print(f"\n Grid_data generated: {grid_data}")

    # Generate data for contour plots
    contour_data = {}
    for pair, (grid_x, grid_y) in grid_data.items():
        i, j = map(int, pair.split("_")[1:])

        # Create input tensor for the model
        X = torch.zeros(resolution, resolution, num_inputs)
        X[:, :, i] = grid_x
        X[:, :, j] = grid_y

        # Set other inputs to their middle values (edit: now set to multiple values)
        non_ij_ranges = {}
        for k in range(num_inputs):
            if k != i and k != j:
                # X[:, :, k] = (input_bounds[0, k] + input_bounds[1, k]) / 2
                non_ij_ranges[k] = torch.linspace(
                    input_bounds[0, k], input_bounds[1, k], 10
                )

        """
        # Iterate over all combinations of non i and j inputs
        for values in torch.cartesian_prod(*non_ij_ranges.values()):
            for idx, k in enumerate(non_ij_ranges.keys()):
                X[:, :, k] = values[idx]

            print(
                f"\n Getting model predictions for pair- {pair} with non i and j inputs: {values}"
            )

            # print(f"\n Getting model predictions for pair- {pair}:")

            # Get model predictions
            with torch.no_grad():
                posterior = model.posterior(X.reshape(-1, num_inputs))
                mean = posterior.mean.reshape(resolution, resolution, num_outputs)
                std = posterior.stddev.reshape(resolution, resolution, num_outputs)

            # print(
            #    f"\nModel predictions:\nposterior: {posterior}\nmean: {mean}\nstd: {std}"
            # )

            # Store contour data for each output dimension
            contour_data[f"{pair}_{values}"] = ContourDataModel(
                x=grid_x.numpy(),  # converting from Torch tensor to numpy array for Matplotlib compatibility
                y=grid_y.numpy(),
                mean=[mean[:, :, k].numpy() for k in range(num_outputs)],
                std=[std[:, :, k].numpy() for k in range(num_outputs)],
            )
            """

        for values in torch.cartesian_prod(*non_ij_ranges.values()):
            for idx, k in enumerate(non_ij_ranges.keys()):
                # Ensure that X is correctly filled for non-i,j inputs
                X[:, :, k] = values[idx].expand_as(
                    grid_x
                )  # expand values to match the 50x50 grid

            print(
                f"Getting model predictions for pair {pair} with non-i,j inputs: {values}"
            )

            # Get model predictions
            with torch.no_grad():
                posterior = model.posterior(X.reshape(-1, num_inputs))
                mean = posterior.mean.reshape(resolution, resolution, num_outputs)
                std = posterior.stddev.reshape(resolution, resolution, num_outputs)

            # Store the contour data properly
            contour_data[f"{pair}_{values}"] = ContourDataModel(
                x=grid_x.numpy(),
                y=grid_y.numpy(),
                mean=[
                    mean[:, :, 0].numpy()
                ],  # Assuming num_outputs == 1 for pairwiseGP
                std=[std[:, :, 0].numpy()],
            )

    # matplotlib_visualization(contour_data, num_outputs) # uncomment this for local runs/testing

    # Generate slider data (WIP: modify for only for non primary input pair inputs)
    slider_data = {}
    for i in range(num_inputs):
        slider_data[f"input_{i}"] = {
            "min": input_bounds[0, i].item(),
            "max": input_bounds[1, i].item(),
            # "default": (input_bounds[0, i] + input_bounds[1, i]).item() / 2,
            "default_range": torch.linspace(
                input_bounds[0, k], input_bounds[1, k], 10
            ).tolist(),
        }

    visualization_data = VisualizationDataModel(
        contour_data=contour_data,
        slider_data=slider_data,
        num_inputs=num_inputs,
        num_outputs=num_outputs,
    )

    return visualization_data
