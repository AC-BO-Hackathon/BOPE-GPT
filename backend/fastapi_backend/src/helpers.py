# helper functions
from models import BopeState, State, SerializedBopeState, SerializedState
import torch

# for serializing/deserializing
# (MongoDB doc storing and API responses to the frontend are better without specialized data types like torch tensors)


def serialize_bope_state(bope_state: BopeState) -> SerializedBopeState:
    return SerializedBopeState(
        iteration=bope_state.iteration,
        X=bope_state.X.tolist(),
        comparisons=bope_state.comparisons.tolist(),
        best_val=bope_state.best_val.tolist(),
        input_bounds=[
            b.tolist() for b in bope_state.input_bounds
        ],  # holds bounds for input columns only
        input_columns=bope_state.input_columns,
        last_iteration_duration=bope_state.last_iteration_duration,
        updated_at=bope_state.updated_at,
    )


def deserialize_bope_state(serialized_bope_state: SerializedBopeState) -> BopeState:
    return BopeState(
        iteration=serialized_bope_state.iteration,
        X=torch.tensor(serialized_bope_state.X),
        comparisons=torch.tensor(serialized_bope_state.comparisons),
        best_val=torch.tensor(serialized_bope_state.best_val),
        input_bounds=torch.stack(
            [torch.tensor(b) for b in serialized_bope_state.input_bounds]
        ),  # holds bounds for input columns only
        input_columns=serialized_bope_state.input_columns,
        last_iteration_duration=serialized_bope_state.last_iteration_duration,
        updated_at=serialized_bope_state.updated_at,
    )


def serialize_state(state: State) -> SerializedState:
    return SerializedState(
        created_at=state.created_at,
        dataset_id=state.dataset_id,
        column_names=state.column_names,
        bounds=state.bounds,
        bope_state=serialize_bope_state(state.bope_state) if state.bope_state else None,
    )
