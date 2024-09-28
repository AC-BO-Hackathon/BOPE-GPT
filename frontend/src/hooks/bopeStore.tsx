import { create } from 'zustand'

interface AppState {
  stateId: string | null
  visualizationData: InitializeBopeResponse | null
  uploadedDatasetData: UploadDatasetSuccessResponse | null
  loading: boolean;
  setStateId: (id: string) => void
  setVisualizationData: (data: InitializeBopeResponse) => void
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

export interface BopeState {
  iteration: number;
  X: number[][];
  comparisons: number[][];
  best_val: number[];
  input_bounds: number[][];
  input_columns: string[];
  iteration_duration: number;
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
  visualizationData: null,
  uploadedDatasetData: null,
  loading: false,
  setStateId: (id) => set({ stateId: id }),
  setVisualizationData: (data) => set({ visualizationData: data }),
  setUploadedDatasetData: (data) => set({uploadedDatasetData: data}),
  setLoading: (loading) => set({ loading }),
}))