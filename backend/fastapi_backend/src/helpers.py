# helper functions
import os
from models import (
    BopeState,
    State,
    SerializedBopeState,
    SerializedState,
    ContourDataModel,
    SerializedContourDataModel,
    VisualizationDataModel,
    SerializedVisualizationDataModel,
)
import torch
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# for serializing/deserializing
# (MongoDB doc storing and API responses to the frontend are better without specialized data types like torch tensors)


def serialize_bope_state(bope_state: BopeState) -> SerializedBopeState:
    def round_floats(data, digits=5):
        if isinstance(data, list):
            return [round_floats(item, digits) for item in data]
        elif isinstance(data, float):
            return round(data, digits)
        return data

    serialized_visualization_data = None
    if bope_state.visualization_data:
        serialized_visualization_data = SerializedVisualizationDataModel(
            contour_data={
                key: SerializedContourDataModel(
                    x=round_floats(value.x.tolist()),
                    y=round_floats(value.y.tolist()),
                    mean=[round_floats(m.tolist()) for m in value.mean],
                    std=[round_floats(s.tolist()) for s in value.std],
                )
                for key, value in bope_state.visualization_data.contour_data.items()
            },
            # slider_data=bope_state.visualization_data.slider_data,
            slider_data={
                k: {sk: round_floats(sv) for sk, sv in v.items()}
                for k, v in bope_state.visualization_data.slider_data.items()
            },
            num_inputs=bope_state.visualization_data.num_inputs,
            num_outputs=bope_state.visualization_data.num_outputs,
        )

    return SerializedBopeState(
        iteration=bope_state.iteration,
        X=bope_state.X.tolist(),
        comparisons=bope_state.comparisons.tolist(),
        best_val=bope_state.best_val.tolist(),
        input_bounds=[
            b.tolist() for b in bope_state.input_bounds
        ],  # holds bounds for input columns only
        input_columns=bope_state.input_columns,
        output_columns=bope_state.output_columns,
        last_iteration_duration=bope_state.last_iteration_duration,
        updated_at=bope_state.updated_at,
        visualization_data=serialized_visualization_data,
        comparison_data=bope_state.comparison_data,
        pareto_plot_data=bope_state.pareto_plot_data,
    )


def deserialize_bope_state(serialized_bope_state: SerializedBopeState) -> BopeState:
    deserialized_visualization_data = None
    if serialized_bope_state.visualization_data:
        deserialized_visualization_data = VisualizationDataModel(
            contour_data={
                key: ContourDataModel(
                    x=np.array(value.x),  # Convert list back to numpy.ndarray
                    y=np.array(value.y),  # Convert list back to numpy.ndarray
                    mean=[
                        np.array(m) for m in value.mean
                    ],  # Convert each list back to numpy.ndarray
                    std=[
                        np.array(s) for s in value.std
                    ],  # Convert each list back to numpy.ndarray
                )
                for key, value in serialized_bope_state.visualization_data.contour_data.items()
            },
            slider_data=serialized_bope_state.visualization_data.slider_data,
            num_inputs=serialized_bope_state.visualization_data.num_inputs,
            num_outputs=serialized_bope_state.visualization_data.num_outputs,
        )

    return BopeState(
        iteration=serialized_bope_state.iteration,
        X=torch.tensor(serialized_bope_state.X),
        comparisons=torch.tensor(serialized_bope_state.comparisons),
        best_val=torch.tensor(serialized_bope_state.best_val),
        input_bounds=torch.stack(
            [torch.tensor(b) for b in serialized_bope_state.input_bounds]
        ),  # holds bounds for input columns only
        input_columns=serialized_bope_state.input_columns,
        output_columns=serialized_bope_state.output_columns,
        last_iteration_duration=serialized_bope_state.last_iteration_duration,
        updated_at=serialized_bope_state.updated_at,
        visualization_data=deserialized_visualization_data,
        comparison_data=serialized_bope_state.comparison_data,
        pareto_plot_data=serialized_bope_state.pareto_plot_data,
    )


def serialize_state(state: State) -> SerializedState:
    return SerializedState(
        created_at=state.created_at,
        dataset_id=state.dataset_id,
        column_names=state.column_names,
        bounds=state.bounds,
        bope_state=serialize_bope_state(state.bope_state) if state.bope_state else None,
    )


# matplotlib plots generation from visualization data for local runs/testing
# NOTE: if you want to run this process locally and generate Matplotlib plots, uncomment out this function in `def generate_contour_visualization_data` in `bope_functions.py`

output_dir = "heatmaps"
os.makedirs(output_dir, exist_ok=True)


def matplotlib_visualization(contour_data: ContourDataModel, num_outputs: int):
    print(f"\n Showing matplotlib renders of generated contour_data:\n")
    print(f"\nnum_outputs: {num_outputs}")
    for pair, data in contour_data.items():
        x = data["x"]
        y = data["y"]
        mean = data["mean"]
        std = data["std"]

        # num_outputs = mean.shape[-1]  # Number of output dimensions
        # Extract the non i and j input values from the pair string
        non_ij_values = pair.split("_")[2:]

        for output_idx in range(num_outputs):

            # Create a figure with subplots
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))

            # Extract the mean and std for the current output
            mean_output = mean[output_idx]  # (resolution, resolution)
            std_output = std[output_idx]  # (resolution, resolution)

            # Ensure the shapes match the expected dimensions
            assert (
                mean_output.shape == std_output.shape
            ), f"Shape mismatch: mean {mean_output.shape}, std {std_output.shape}"

            # Plot the mean heatmap
            c1 = ax[0].imshow(
                mean_output,
                extent=(x.min(), x.max(), y.min(), y.max()),
                origin="lower",
                aspect="auto",
            )
            ax[0].set_title(
                f"Mean Heatmap for {pair}, Output {output_idx}\nNon i and j inputs: {non_ij_values}"
            )
            ax[0].set_xlabel("X")
            ax[0].set_ylabel("Y")
            fig.colorbar(c1, ax=ax[0], orientation="vertical")

            # Plot the standard deviation heatmap
            c2 = ax[1].imshow(
                std_output,
                extent=(x.min(), x.max(), y.min(), y.max()),
                origin="lower",
                aspect="auto",
            )
            ax[1].set_title(
                f"Standard Deviation Heatmap for {pair}, Output {output_idx}\nNon i and j inputs: {non_ij_values}"
            )
            ax[1].set_xlabel("X")
            ax[1].set_ylabel("Y")
            fig.colorbar(c2, ax=ax[1], orientation="vertical")

            # Show the plots
            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, f"heatmap_{pair}_output_{output_idx}.png")
            )
            plt.close(fig)


# recursive function to add brevity to large Pydantic objects (after converting them to 'dict' with `model_dump()`- helps
# for logging purposes):


def brief_summary(data, max_length=100, max_depth=2, current_depth=0):
    if current_depth > max_depth:
        if isinstance(data, dict):
            return f"<{type(data).__name__} (length: {len(data)}, depth: {get_depth(data)})>"
        elif isinstance(data, list):
            return f"<{type(data).__name__} (length: {len(data)}, depth: {get_depth(data)})>"
        else:
            return f"<{type(data).__name__}>"

    if isinstance(data, dict):
        return {
            k: brief_summary(v, max_length, max_depth, current_depth + 1)
            for k, v in data.items()
        }
    elif isinstance(data, list):
        return [
            brief_summary(item, max_length, max_depth, current_depth + 1)
            for item in data[:3]
        ] + (["..."] if len(data) > 3 else [])
    elif isinstance(data, str):
        return data[:max_length] + "..." if len(data) > max_length else data
    elif isinstance(data, float):
        return round(data, 5)  # Round floats to 5 significant digits
    elif isinstance(data, datetime):
        return data.isoformat()  # Convert datetime to ISO format string
    return data


def get_depth(data, current_depth=0):
    if isinstance(data, dict):
        return (
            max(get_depth(v, current_depth + 1) for v in data.values())
            if data
            else current_depth
        )
    elif isinstance(data, list):
        return (
            max(get_depth(item, current_depth + 1) for item in data)
            if data
            else current_depth
        )
    return current_depth
