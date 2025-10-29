import { defineStore } from 'pinia';
import datasetApi from '@/services/api/datasetApi';
import { useRobotStore } from './robotStore';

export const useDatasetStore = defineStore('dataset', {
  state: () => ({
    count: 0,
    datasets: [],
    recording: {
      active: false,
      currentDataset: null,
      error: null
    },
    replay: {
      active: false,
      currentDataset: null,
      error: null
    }
  }),

  getters: {
    hasDatasets: (state) => state.count > 0,
    canStartTraining: (state) => state.count >= 5,
    isRecording: (state) => state.recording.active,
    isReplaying: (state) => state.replay.active
  },

  actions: {
    async fetchCount() {
      try {
        const response = await datasetApi.getCount();
        this.count = response.data.count;
      } catch (error) {
        console.error('Failed to fetch dataset count:', error);
        this.count = 0;
      }
    },

    async fetchDatasets() {
      try {
        const response = await datasetApi.list();
        this.datasets = response.data.datasets;
        this.count = this.datasets.length;
      } catch (error) {
        console.error('Failed to fetch datasets:', error);
        this.datasets = [];
      }
    },

    async startRecording(config = {}) {
      try {
        this.recording.error = null;
        const robotStore = useRobotStore();
        const teleopConfig = robotStore.teleoperationConfig || {};
        const operationMode = teleopConfig.operationMode
          || robotStore.status?.teleoperation?.configuration?.operation_mode
          || robotStore.status?.teleoperation?.configuration?.operationMode
          || 'bimanual';
        const payload = {
          ...config,
          operation_mode: operationMode,
          interactive: typeof config.interactive === 'undefined' ? false : !!config.interactive
        };
        const response = await datasetApi.startRecording(payload);
        this.recording.active = true;
        this.recording.currentDataset = response.data.dataset_id;
        return response;
      } catch (error) {
        this.recording.error = error.message;
        throw error;
      }
    },

    async stopRecording() {
      try {
        const response = await datasetApi.stopRecording();
        this.recording.active = false;
        this.recording.currentDataset = null;
        await this.fetchCount(); // Refresh count
        return response;
      } catch (error) {
        this.recording.error = error.message;
        throw error;
      }
    }
  }
});