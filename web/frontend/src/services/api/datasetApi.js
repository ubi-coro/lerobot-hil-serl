const API_BASE = '/api/dataset';

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
  // 📹 RECORD DATASET CARD APIs
  // ============================================
  
  // Start recording a new dataset
  startRecording(config = {}) {
    console.log('Calling startRecording with config:', config);
    return apiCall('/record/start', {
      method: 'POST',
      body: config
    });
  },

  // Stop recording
  stopRecording() {
    console.log('Calling stopRecording...');
    return apiCall('/record/stop', {
      method: 'POST'
    });
  },

  // Get recording status
  getRecordingStatus() {
    return apiCall('/record/status');
  },

  // Pause/resume recording
  pauseRecording() {
    return apiCall('/record/pause', {
      method: 'POST'
    });
  },

  resumeRecording() {
    return apiCall('/record/resume', {
      method: 'POST'
    });
  },

  // ============================================
  // 📊 REPLAY DATASET CARD APIs
  // ============================================
  
  // List all datasets
  list() {
    console.log('Calling dataset list...');
    return apiCall('/list');
  },

  // Get dataset details
  getDataset(datasetId) {
    return apiCall(`/details/${datasetId}`);
  },

  // Replay dataset
  replay(datasetId, config = {}) {
    console.log('Calling replay for dataset:', datasetId);
    return apiCall(`/replay/${datasetId}`, {
      method: 'POST',
      body: config
    });
  },

  // Stop replay
  stopReplay() {
    return apiCall('/replay/stop', {
      method: 'POST'
    });
  },

  // Get replay status
  getReplayStatus() {
    return apiCall('/replay/status');
  },

  // ============================================
  // 📈 DATA VISUALIZATION CARD APIs
  // ============================================
  
  // Get dataset count (used by dashboard)
  getCount() {
    return apiCall('/count');
  },

  // List available datasets/repos for visualization
  listAvailableRepos() {
    console.log('Fetching available repos for visualization...');
    return apiCall('/visualization/repos');
  },

  // Get local datasets
  listLocalDatasets() {
    console.log('Fetching local datasets...');
    return apiCall('/list');
  },

  // Launch LeRobot's HTML visualizer for specific repo
  launchHtmlVisualizer(repoId, options = {}) {
    console.log('Launching HTML visualizer for repo:', repoId);
    return apiCall('/visualization/launch', {
      method: 'POST',
      body: { 
        repo_id: repoId,
        visualizer_type: 'html',
        port: options.port || 9090,
        episodes: options.episodes || 'all',
        local_files_only: options.localFilesOnly || false,
        episode_index: options.episodeIndex,
        ...options
      }
    });
  },

  // Launch visualizer for local dataset specifically
  launchLocalDatasetVisualizer(datasetPath, visualizerType = 'html', options = {}) {
    console.log('Launching local dataset visualizer:', datasetPath);
    return apiCall('/visualization/launch-local', {
      method: 'POST',
      body: { 
        dataset_path: datasetPath,
        visualizer_type: visualizerType,
        port: options.port || 9090,
        episode_index: options.episodeIndex || 0,
        episodes: options.episodes || 'all',
        ...options
      }
    });
  },

  // Stop running visualizer
  stopVisualizer(visualizerType = 'html') {
    console.log('Stopping visualizer:', visualizerType);
    return apiCall('/visualization/stop', {
      method: 'POST',
      body: { visualizer_type: visualizerType }
    });
  },

  // Get visualizer status (running/stopped)
  getVisualizerStatus() {
    return apiCall('/visualization/status');
  },

  // Open visualizer URL in new window
  openVisualizerWindow(port = 9090) {
    const url = `http://localhost:${port}`;
    
    console.log('Opening HTML visualizer window:', url);
    const windowFeatures = 'width=1400,height=900,scrollbars=yes,resizable=yes';
    window.open(url, 'lerobot_html_visualizer', windowFeatures);
  },

  // Check if repo exists and is accessible
  validateRepo(repoId) {
    return apiCall('/validation/repo', {
      method: 'POST',
      body: { repo_id: repoId }
    });
  },

  // Get repo info (episodes count, size, etc.)
  getRepoInfo(repoId) {
    return apiCall(`/info/${encodeURIComponent(repoId)}`);
  },

  // Get dataset statistics
  getStatistics(datasetId) {
    return apiCall(`/statistics/${datasetId}`);
  },

  // Export dataset
  exportDataset(datasetId, format = 'hdf5') {
    return apiCall(`/export/${datasetId}?format=${format}`, {
      method: 'POST'
    });
  },

  // Delete dataset
  deleteDataset(datasetId) {
    return apiCall(`/delete/${datasetId}`, {
      method: 'DELETE'
    });
  },

  // ============================================
  // 📊 HTML VISUALIZER APIs  
  // ============================================

  // Launch HTML visualizer with manual parameters
  launchHtmlVisualizerCustom(params) {
    console.log('Launching HTML visualizer with params:', params);
    return fetch('/api/dataset/visualize', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        repo_id: params.repoId,
        root_path: params.rootPath,
        episode_index: params.episodeIndex
      })
    }).then(response => response.json());
  },

  // Browse directory contents for folder selection
  browseDirectory(path) {
    return apiCall('/browse-directory', {
      method: 'POST',
      body: {
        path: path
      }
    });
  }
};