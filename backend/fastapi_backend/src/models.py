from pydantic import BaseModel
from typing import List, Optional

# WIP FILE (See README)

# pydantic defines the structure of data to validate them in API requests
# TODO: Modify these models based on the data structures I want received from the frontend


class UploadResponse(BaseModel):
    message: str


class OptimizationInitResponse(BaseModel):
    status: str
    details: dict


class NextIterationResponse(BaseModel):
    status: str
    results: dict


class DataPoint(BaseModel):
    id: int
    features: List[float]


class ComparisonResult(BaseModel):
    chosen_id: int
    other_id: int
    score: float


class OptimizationResult(BaseModel):
    iteration: int
    comparisons: List[ComparisonResult]
    best_candidate: Optional[DataPoint]
