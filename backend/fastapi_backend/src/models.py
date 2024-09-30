from pydantic import BaseModel
from typing import List, Optional, Any
from datetime import datetime, timezone
import torch


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


# Pydantic models for bope-state, state, dataset ('state' includes 'bope-state' plus auxiliary info) and serialied versions
class BopeState(BaseModel):
    iteration: int
    X: torch.Tensor
    comparisons: torch.Tensor
    best_val: torch.Tensor
    input_bounds: torch.Tensor
    input_columns: List[str]
    last_iteration_duration: Optional[float]
    updated_at: Optional[datetime]

    class Config:
        arbitrary_types_allowed = True


class State(BaseModel):
    created_at: datetime
    dataset_id: Optional[str]
    column_names: List[str]
    bounds: List[List[float]]
    bope_state: Optional[BopeState]

    class Config:
        arbitrary_types_allowed = True


class SerializedBopeState(BaseModel):
    iteration: int
    X: List[List[float]]
    comparisons: List[List[float]]
    best_val: List[float]
    input_bounds: List[List[float]]
    input_columns: List[str]
    last_iteration_duration: float
    updated_at: datetime


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
