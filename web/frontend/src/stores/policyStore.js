import { defineStore } from 'pinia';
import policyApi from '@/services/api/policyApi';

export const usePolicyStore = defineStore('policy', {
  state: () => ({
    training: {
      active: false,
      progress: 0,
      currentModel: null,
      error: null
    },
    models: [],
    evaluation: {
      active: false,
      results: null,
      error: null
    }
  }),

  getters: {
    isTraining: (state) => state.training.active,
    hasModels: (state) => state.models.length > 0,
    trainingProgress: (state) => state.training.progress,
    isEvaluating: (state) => state.evaluation.active
  },

  actions: {
    async startTraining(config = {}) {
      try {
        this.training.error = null;
        const response = await policyApi.startTraining(config);
        this.training.active = true;
        this.training.currentModel = response.data.model_id;
        return response;
      } catch (error) {
        this.training.error = error.message;
        throw error;
      }
    },

    async stopTraining() {
      try {
        const response = await policyApi.stopTraining();
        this.training.active = false;
        this.training.currentModel = null;
        this.training.progress = 0;
        return response;
      } catch (error) {
        this.training.error = error.message;
        throw error;
      }
    },

    async fetchTrainingStatus() {
      if (!this.training.active) return;
      
      try {
        const response = await policyApi.getStatus();
        this.training.progress = response.data.progress;
        if (response.data.status === 'completed') {
          this.training.active = false;
          await this.fetchModels();
        }
      } catch (error) {
        console.error('Failed to fetch training status:', error);
      }
    },

    async fetchModels() {
      try {
        const response = await policyApi.listModels();
        this.models = response.data.models;
      } catch (error) {
        console.error('Failed to fetch models:', error);
        this.models = [];
      }
    }
  }
});