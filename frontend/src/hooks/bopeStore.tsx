import type { UploadDatasetSuccessResponse } from '@/pages' // actual interface defined in pages/index.tsx
import { create } from 'zustand'

interface AppState {
  stateId: string | null
  visualizationData: InitializeBopeResponse | null
  uploadedDatasetData: UploadDatasetSuccessResponse | null
  setStateId: (id: string) => void
  setVisualizationData: (data: InitializeBopeResponse) => void
  setUploadedDatasetData: (data: UploadDatasetSuccessResponse) => void
}

interface BopeState {
  iteration: number;
  X: number[][];
  comparisons: number[][];
  best_val: number[];
  input_bounds: number[][];
  input_columns: string[];
}

interface InitializeBopeResponse {
  message: string;
  bope_state: BopeState;
  state_id: string;
}

export const useBopeStore = create<AppState>((set) => ({
  stateId: null,
  visualizationData: null,
  uploadedDatasetData: null,
  setStateId: (id) => set({ stateId: id }),
  setVisualizationData: (data) => set({ visualizationData: data }),
  setUploadedDatasetData: (data) => set({uploadedDatasetData: data}),
}))