const API_BASE = '/api/training';

// Helper function for making API calls with fetch
async function apiCall(endpoint, options = {}) {
  const url = `${API_BASE}${endpoint}`;

  const config = {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers
    },
    ...options
  };

  if (config.body && typeof config.body === 'object') {
    config.body = JSON.stringify(config.body);
  }

  try {
    const response = await fetch(url, config);
    const data = await response.json();

    if (!response.ok) {
      const errorMessage = data.message || `HTTP ${response.status}: ${response.statusText}`;
      throw new Error(errorMessage);
    }

    return { data };
  } catch (error) {
    const apiError = new Error(error.message);
    apiError.response = {
      data: { message: error.message },
      status: error.status || 500
    };
    throw apiError;
  }
}

export default {
  // ============================================
  // 🧠 TRAINING CARD APIs
  // ============================================
  
  // Start training a policy
  startTraining(config = {}) {
    console.log('Calling startTraining with config:', config);
    return apiCall('/start', {
      method: 'POST',
      body: config
    });
  },

  // Stop training
  stopTraining() {
    console.log('Calling stopTraining...');
    return apiCall('/stop', {
      method: 'POST'
    });
  },

  // Get training status
  getStatus() {
    return apiCall('/status');
  },

  // Get training progress/metrics
  getProgress() {
    return apiCall('/progress');
  },

  // Get training logs
  getLogs(lines = 100) {
    return apiCall(`/logs?lines=${lines}`);
  },

  // Pause training
  pauseTraining() {
    return apiCall('/pause', {
      method: 'POST'
    });
  },

  // Resume training
  resumeTraining() {
    return apiCall('/resume', {
      method: 'POST'
    });
  },

  // List trained models
  listModels() {
    return apiCall('/models');
  },

  // Get model details
  getModel(modelId) {
    return apiCall(`/models/${modelId}`);
  },

  // Evaluate model
  evaluateModel(modelId, config = {}) {
    console.log('Calling evaluateModel for:', modelId);
    return apiCall(`/models/${modelId}/evaluate`, {
      method: 'POST',
      body: config
    });
  },

  // Download trained model
  downloadModel(modelId) {
    return apiCall(`/models/${modelId}/download`);
  },

  // Delete model
  deleteModel(modelId) {
    return apiCall(`/models/${modelId}`, {
      method: 'DELETE'
    });
  },

  // Get available training configurations
  getTrainingConfigs() {
    return apiCall('/configs');
  },

  // Validate dataset for training
  validateDataset(datasetId) {
    return apiCall('/validate-dataset', {
      method: 'POST',
      body: { dataset_id: datasetId }
    });
  }
};