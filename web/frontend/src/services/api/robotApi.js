const API_BASE = '/api/robot';

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

  // Convert body object to JSON string if needed
  if (config.body && typeof config.body === 'object') {
    config.body = JSON.stringify(config.body);
  }

  console.log(`Making ${config.method || 'GET'} request to ${url}`);

  try {
    const response = await fetch(url, config);

    // Parse JSON response
    let data;
    try {
      data = await response.json();
    } catch (parseError) {
      console.error('Failed to parse JSON response:', parseError);
      throw new Error('Invalid JSON response from server');
    }

    console.log(`Response from ${url}:`, data);

    // Check if response is successful
    if (!response.ok) {
      const errorMessage = data.detail || data.message || `HTTP ${response.status}: ${response.statusText}`;
      throw new Error(errorMessage);
    }

    // Return in axios-like format for compatibility
    return { data };

  } catch (error) {
    console.error(`Error calling ${url}:`, error);

    // Create axios-like error object for compatibility
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
  // 🎮 TELEOPERATION CARD APIs (FastAPI Backend)
  // ============================================
  
  // Start teleoperation with ALOHA-compatible configuration
  startTeleoperation(config = {}) {
    console.log('Starting ALOHA teleoperation with config:', config);

    // Prepare body matching simplified backend (single config dict)
    const teleoperationBody = {
      config: {
        fps: config.fps || 30,
        operation_mode: config.operation_mode || 'bimanual',
        show_cameras: config.show_cameras !== false,
        display_data: config.display_data || false,
        safety_limits: config.safety_limits !== false,
        performance_monitoring: true
      }
    };

    return fetch('/api/aloha-teleoperation/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(teleoperationBody)
    }).then(async response => {
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || data.message || 'Failed to start teleoperation');
      }
      return { data };
    });
  },

  // Stop teleoperation
  stopTeleoperation() {
    console.log('Stopping ALOHA teleoperation...');
    return fetch('/api/aloha-teleoperation/stop', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      }
    }).then(async response => {
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || data.message || 'Failed to stop teleoperation');
      }
      return { data };
    });
  },

  // Get teleoperation status
  getTeleoperationStatus() {
    return fetch('/api/aloha-teleoperation/status', {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json'
      }
    }).then(async response => {
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || data.message || 'Failed to get teleoperation status');
      }
      return { data };
    });
  },

  // Emergency stop (placeholder - would need to be implemented in backend)
  emergencyStop() {
    console.log('Emergency stop called...');
    return this.stopTeleoperation();
  },

  // ============================================
  // ⚙️ CALIBRATION CARD APIs
  // ============================================
  
  // Get available robot configurations
  getConfigs() {
    console.log('Calling getConfigs...');
    return apiCall('/configs');
  },

  // Connect to robot
  connect(operationMode = 'bimanual', configSettings = {}) {
    console.log('Calling connect with operation mode:', operationMode);
    console.log('Calling connect with config settings:', configSettings);

    const connectRequest = {
      robot_type: configSettings.robot_type || 'aloha',
      operation_mode: operationMode,
      profile_name: configSettings.profile_name || null,
      show_cameras: configSettings.show_cameras !== false,
      display_data: !!configSettings.display_data,
      fps: configSettings.fps || 30,
      calibrate: !!configSettings.calibrate,
      force_reconnect: !!configSettings.force_reconnect,
      overrides: configSettings.overrides || []
    };

    return apiCall('/connect', {
      method: 'POST',
      body: connectRequest
    });
  },

  // Disconnect from robot
  disconnect() {
    console.log('Calling disconnect...');
    return apiCall('/disconnect', {
      method: 'POST'
    });
  },

  // Get robot status
  getStatus() {
    console.log('Calling getStatus...');
    return apiCall('/status');
  },

  // Move robot to safe position
  moveToSafePosition(config = {}) {
    console.log('Calling moveToSafePosition with config:', config);
    return apiCall('/robot/safe-position', {
      method: 'POST',
      body: config
    });
  },

  // Run system diagnostics
  runDiagnostics() {
    console.log('Calling runDiagnostics...');
    return apiCall('/diagnostics', {
      method: 'POST'
    });
  },

  // Calibrate cameras
  calibrateCameras(config = {}) {
    console.log('Calling calibrateCameras with config:', config);
    return apiCall('/calibrate/cameras', {
      method: 'POST',
      body: config
    });
  },

  // Calibrate arms
  calibrateArms(config = {}) {
    console.log('Calling calibrateArms with config:', config);
    return apiCall('/calibrate/arms', {
      method: 'POST',
      body: config
    });
  }
};
