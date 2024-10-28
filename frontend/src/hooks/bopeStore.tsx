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

export interface ContourDataModel {
  x: number[][]; // List[List[float]] in Python
  y: number[][]; // List[List[float]] in Python
  mean: number[][][]; // List[List[List[float]]] in Python
  std: number[][][]; // List[List[List[float]]] in Python
}

export interface SliderData {
  min: number;
  max: number;
  default_range: number[];
}

export interface VisualizationDataModel {
  contour_data: { [key: string]: ContourDataModel }; // Dict[str, SerializedContourDataModel] in Python
  slider_data: { [key: string]: SliderData }; // Dict[str, SliderData] in Python
  num_inputs: number; // int in Python
  num_outputs: number; // int in Python
}

export interface ComparisonDataModel {
  pair_indices: number[][];
  pair_input_values: number[][][];
  pair_output_values: number[][][];
}

/*
export interface ParetoPointData {
  x: number;
  y: number;
  is_pareto: boolean;
  input_values: { [key: string]: number };
  output_values: { [key: string]: number };
}

export interface ParetoPlotData {
  x_label: string;
  y_label: string;
  points: ParetoPointData[];
}

export interface ParetoVisualizationData {
  pareto_plots: ParetoPlotData[];
  input_columns: string[];
  output_columns: string[];
}
*/

export interface DataPoint {
  id: number;
  input_values: { [key: string]: number };
  output_values: { [key: string]: number };
}

export interface ParetoPlotData {
  x_label: string;
  y_label: string;
  point_indices: number[];
  is_pareto: boolean[];
}

export interface ParetoVisualizationData {
  pareto_plots: ParetoPlotData[];
  data_points: DataPoint[];
  input_columns: string[];
  output_columns: string[];
}

export interface BopeState {
  iteration: number;
  X: number[][];
  comparisons: number[][];
  best_val: number[];
  input_bounds: number[][];
  input_columns: string[];
  output_columns: string[];
  last_iteration_duration: number;
  visualization_data: VisualizationDataModel
  comparison_data: ComparisonDataModel
  pareto_plot_data?: ParetoVisualizationData; // Optional
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