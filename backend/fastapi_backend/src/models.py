from pydantic import BaseModel
from typing import List, Optional, Any, Dict
from datetime import datetime, timezone
import torch
import numpy as np


# Pydantic models for api request validation
class InitializeBOPERequest(BaseModel):
    llm_prompt: str = "Enter a prompt here"  # ""
    num_inputs: int = 4  # 4
    num_initial_samples: int = 5  # 5
    num_initial_comparisons: int = 10  # 10
    enable_flexible_prompt: Optional[bool] = False
    enable_llm_explanations: Optional[bool] = False
    state_id: str = (
        "Insert whatever state ID received after hitting the `upload_dataset` endpoint"
    )


class RunNextIterationRequest(BaseModel):
    llm_prompt: str
    comparison_explanations: bool = False
    enable_flexible_prompt: bool = False
    state_id: str


# Pydantic models for visualization data
class ContourDataModel(BaseModel):
    x: np.ndarray
    y: np.ndarray
    mean: List[np.ndarray]
    std: List[np.ndarray]

    class Config:
        arbitrary_types_allowed = True


class SerializedContourDataModel(BaseModel):
    x: List[List[float]]
    y: List[List[float]]
    mean: List[List[List[float]]]
    std: List[List[List[float]]]


class VisualizationDataModel(BaseModel):
    contour_data: Dict[str, ContourDataModel]
    slider_data: Dict[str, Dict[str, Any]]
    num_inputs: int
    num_outputs: int


class SerializedVisualizationDataModel(BaseModel):
    contour_data: Dict[str, SerializedContourDataModel]
    slider_data: Dict[str, Dict[str, Any]]
    num_inputs: int
    num_outputs: int


# Pydantic models for comparison data


class ComparisonDataModel(BaseModel):
    pair_indices: List[List[int]]
    pair_input_values: List[List[List[float]]]
    pair_output_values: List[List[List[float]]]


# Pydantic models for pareto plot data

class DataPoint(BaseModel):
    id: int
    input_values: Dict[str, float]
    output_values: Dict[str, float]


class ParetoPlotData(BaseModel):
    x_label: str
    y_label: str
    point_indices: List[int]
    is_pareto: List[bool]


class ParetoVisualizationData(BaseModel):
    pareto_plots: List[ParetoPlotData]
    data_points: List[DataPoint]
    input_columns: List[str]
    output_columns: List[str]


# Pydantic models for bope-state, state, dataset ('state' includes 'bope-state' plus auxiliary info) and serialied versions
class BopeState(BaseModel):
    iteration: int
    X: torch.Tensor
    comparisons: torch.Tensor
    best_val: torch.Tensor
    input_bounds: torch.Tensor
    input_columns: List[str]
    output_columns: List[str]
    last_iteration_duration: Optional[float]
    updated_at: Optional[datetime]
    visualization_data: Optional[VisualizationDataModel] = (
        None  # not all bope state instances need this (or will have at each stage)
    )
    comparison_data: Optional[ComparisonDataModel] = None
    pareto_plot_data: Optional[ParetoVisualizationData] = None

    class Config:
        arbitrary_types_allowed = True


class SerializedBopeState(BaseModel):
    iteration: int
    X: List[List[float]]
    comparisons: List[List[float]]
    best_val: List[float]
    input_bounds: List[List[float]]
    input_columns: List[str]
    output_columns: List[str]
    last_iteration_duration: float
    updated_at: datetime
    visualization_data: Optional[SerializedVisualizationDataModel] = None
    comparison_data: Optional[ComparisonDataModel] = None
    pareto_plot_data: Optional[ParetoVisualizationData] = None


class State(BaseModel):
    created_at: datetime
    dataset_id: Optional[str]
    column_names: List[str]
    bounds: List[List[float]]
    bope_state: Optional[BopeState]

    class Config:
        arbitrary_types_allowed = True


class SerializedState(BaseModel):
    created_at: datetime
    dataset_id: str
    column_names: List[str]
    bounds: List[List[float]]
    bope_state: Optional[SerializedBopeState]


class UploadedDataset(BaseModel):
    data: List[dict]
    column_names: List[str]
    uploaded_at: datetime
