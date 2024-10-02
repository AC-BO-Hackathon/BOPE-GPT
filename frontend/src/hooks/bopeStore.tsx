import { create } from 'zustand'

interface AppState {
  stateId: string | null
  latestBopeData: InitializeBopeResponse | IterationBopeResponse | null // response could be either after initialization or iteration type
  uploadedDatasetData: UploadDatasetSuccessResponse | null
  loading: boolean;
  setStateId: (id: string) => void
  setLatestBopeData: (data: InitializeBopeResponse | IterationBopeResponse ) => void
  setUploadedDatasetData: (data: UploadDatasetSuccessResponse) => void
  setLoading: (loading: boolean) => void; 
}

// Interface for the successful response from /upload_dataset/
export interface UploadDatasetSuccessResponse {
  message: string;
  dataset_id: string;
  state_id: string;
  column_names: string[];
}

interface ContourDataModel {
  x: number[][]; // List[List[float]] in Python
  y: number[][]; // List[List[float]] in Python
  mean: number[][][]; // List[List[List[float]]] in Python
  std: number[][][]; // List[List[List[float]]] in Python
}

interface VisualizationDataModel {
  contour_data: { [key: string]: ContourDataModel }; // Dict[str, SerializedContourDataModel] in Python
  slider_data: { [key: string]: { [key: string]: any } }; // Dict[str, Dict[str, Any]] in Python
  num_inputs: number; // int in Python
  num_outputs: number; // int in Python
}

export interface BopeState {
  iteration: number;
  X: number[][];
  comparisons: number[][];
  best_val: number[];
  input_bounds: number[][];
  input_columns: string[];
  last_iteration_duration: number;
  visualization_data: VisualizationDataModel
}

export interface InitializeBopeResponse { // interface equivalent to NextIterationBopeResponse 
  message: string;
  bope_state: BopeState;
  state_id: string;
}

export interface IterationBopeResponse {
  message: string;
  bope_state: BopeState;
  state_id: string;
}


export const useBopeStore = create<AppState>((set) => ({
  stateId: null,
  latestBopeData: null,
  uploadedDatasetData: null,
  loading: false,
  setStateId: (id) => set({ stateId: id }),
  setLatestBopeData: (data) => set({ latestBopeData: data }),
  setUploadedDatasetData: (data) => set({uploadedDatasetData: data}),
  setLoading: (loading) => set({ loading }),
}))