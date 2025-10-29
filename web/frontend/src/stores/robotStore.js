import { defineStore } from 'pinia';
import { io } from 'socket.io-client';
import robotApi from '@/services/api/robotApi';

export const useRobotStore = defineStore('robot', {
  state: () => ({
    configs: [],
  selectedRobotType: (typeof localStorage !== 'undefined' && localStorage.getItem('lerobot.selectedRobotType')) || 'aloha_bimanual',
    status: {
      connected: false,
      available_arms: [],
      cameras: [],
      mode: null, // added default mode so components relying on it don't error
      error: null // include error slot so getters referencing it are safe
    },
    internalErrorMessage: '',
    internalHasError: false,
    socket: null,
    cameraStreams: {},
    statusPollingTimer: null,
    // Add teleoperation configuration state
    teleoperationConfig: {
      fps: 30,
      showCameras: true,
      maxRelativeTarget: 25,
      operationMode: 'bimanual',
      enableSafeShutdown: true,
      movingTime: 0.1,
      teleopTimeLimit: null,
      performanceMonitoring: false,
      debugLevel: 'INFO'
    },
    // Performance monitoring state
    performanceMetrics: {
      actualFps: 0,
      latency: 0,
      cpuUsage: 0,
      memoryUsage: 0,
      timestamp: null
    }
  }),

  getters: {
  isConnected: (state) => !!state.status.connected,
  isTeleoperating: (state) => state.status.mode === 'teleoperating',
  hasError: (state) => state.internalHasError || !!state.status.error,
  // unified public error message accessor
  errorMessage: (state) => state.internalErrorMessage || state.status.error || '',
  availableCameras: (state) => state.status.cameras || [],
  robotType: (state) => state.selectedRobotType,
  },

  actions: {
    // Initialize socket connection
    initSocket() {
      if (this.socket) return;
      const envUrl = import.meta?.env?.VITE_BACKEND_URL;
      let backendUrl = envUrl || window.location.origin;
      if (!envUrl && window.location.port === '5173') backendUrl = 'http://localhost:8000';
      console.log(`[robotStore] Initializing Socket.IO -> ${backendUrl}`);

      const connectWithOptions = (opts, label) => {
        try {
          console.log(`[robotStore] Attempting Socket.IO connect (${label})`);
          this.socket = io(backendUrl, opts);
        } catch (e) {
          console.error('[robotStore] Socket creation failed:', e);
        }
      };

      // First attempt: default (allows polling then upgrade)
      connectWithOptions({ path: '/socket.io', transports: ['polling','websocket'], withCredentials: false, timeout: 8000 }, 'polling+websocket');
      if (!this.socket) return;

      let retried = false;
      this.socket.on('connect', () => {
        console.log('[robotStore] Socket connected to backend (id=' + this.socket.id + ')');
      });
      this.socket.on('disconnect', (reason) => {
        console.log('[robotStore] Socket disconnected:', reason);
      });
      this.socket.on('connect_error', (err) => {
        console.error('[robotStore] Socket connection error:', err.message);
        if (!retried) {
          retried = true;
          // Retry forcing new and allowing all transports
            console.log('[robotStore] Retrying socket connection with forceNew');
            this.socket = null;
            connectWithOptions({ path: '/socket.io', transports: ['polling','websocket'], forceNew: true, reconnectionAttempts: 2, timeout: 10000 }, 'retry');
        }
      });
      this.socket.on('error', (err) => {
        console.error('[robotStore] Socket error event:', err);
      });
      this.socket.on('camera_frame', (data) => {
        if (!data || !data.camera_id) return;
        if (!this.cameraStreams[data.camera_id]) console.log(`[robotStore] First frame for ${data.camera_id}`);
        this.cameraStreams[data.camera_id] = data.frame;
      });
      this.socket.on('camera_list', (data) => {
        if (data && Array.isArray(data.cameras)) {
          this.status.cameras = data.cameras;
          console.log('[robotStore] Updated camera list from event:', data.cameras);
        }
      });
      this.socket.on('teleoperation_status', (payload) => {
        try {
          const active = !!payload?.active;
          this.status.mode = active ? 'teleoperating' : null;
          // Optionally store snapshot fields used by UI
          if (!this.status.teleoperation) this.status.teleoperation = {};
          this.status.teleoperation = {
            active,
            stage: payload?.stage,
            session_duration: payload?.session_duration ?? 0,
            display_data_active: payload?.display_data_active ?? false,
            configuration: payload?.configuration || null,
            // Pass-through performance metrics and convenient fields for UI parity with recorder
            performance_metrics: payload?.performance_metrics || null,
            fps_target: (payload?.configuration && typeof payload.configuration.fps === 'number') ? payload.configuration.fps : null,
            fps_current: (payload?.performance_metrics && typeof payload.performance_metrics.average_fps === 'number') ? payload.performance_metrics.average_fps : null,
          };
        } catch (e) {
          console.debug('[robotStore] teleoperation_status handling error:', e);
        }
      });
    },

    // Set and persist chosen robot type (UI only for now)
    setRobotType(type) {
  this.selectedRobotType = type || 'aloha_bimanual';
      try { localStorage.setItem('lerobot.selectedRobotType', this.selectedRobotType); } catch (_) {}
    },

    // Fetch robot configurations
    async fetchRobotConfigs() {
      try {
        console.log('Fetching robot configurations from API...');
        const response = await robotApi.getConfigs();
        console.log('API response:', response);

        if (response.data && response.data.status === 'success' && response.data.data) {
          this.configs = response.data.data;
          console.log('Configs stored:', this.configs);
        } else {
          console.error('Invalid response format:', response);
          this.internalHasError = true;
      this.internalErrorMessage = 'Invalid API response format';
        }
      } catch (error) {
        console.error('Error fetching robot configurations:', error);
  this.internalHasError = true;
  this.internalErrorMessage = error.message || 'Failed to load robot configurations';
      }
    },

    // Connect to robot
    async connect(mode, profile = null, displayData = false, showCameras = true) {
      if (this.isConnecting || this.isConnected) {
        console.warn("Connect ignored: already connected or connecting.");
        return;
      }
      this.isConnecting = true;
      this.connectionError = null;
      this.connectionStatus = "Connecting to robot hardware...";

      // Force camera connection for the main robot instance
      const connectPayload = {
        robot_type: `aloha_${mode}`,
        operation_mode: mode,
        profile_name: profile,
        show_cameras: true, // Always connect cameras on initial connection
        display_data: displayData,
        force_reconnect: true,
      };

      try {
        const response = await robotApi.connect(connectPayload);

        if (response.data.status === 'success') {
          this.status = {
            ...this.status,
            ...response.data.data,
            connected: true
          };
          console.log('Robot connected successfully');
          this.isConnecting = false;
          this.connectionStatus = null;
        } else {
          this.internalHasError = true;
          this.internalErrorMessage = response.data.message || 'Connection failed';
          this.isConnecting = false;
        }
      } catch (error) {
        console.error('Error connecting robot:', error);
        this.internalHasError = true;
        this.internalErrorMessage = error.response?.data?.message || 'Failed to connect to robot';
        this.isConnecting = false;
        this.connectionStatus = null;
      }
    },

    // Disconnect robot    // Disconnect robot
    async disconnectRobot() {
      try {
        const response = await robotApi.disconnect();

        if (response.data.status === 'success') {
          this.status = {
            connected: false,
            available_arms: [],
            cameras: [],
            mode: null
          };
          console.log('Robot disconnected successfully');
        }
      } catch (error) {
        console.error('Error disconnecting robot:', error);
  this.internalHasError = true;
  this.internalErrorMessage = error.response?.data?.message || 'Disconnect failed';
      }
    },

    // Start teleoperation
    async startTeleoperation(fps = 30) {
      try {
        const response = await robotApi.startTeleoperation(fps, false);

        if (response.data.status === 'success') {
          this.status.mode = 'teleoperating';
          console.log('Teleoperation started');
        }
      } catch (error) {
        console.error('Error starting teleoperation:', error);
  this.internalHasError = true;
  this.internalErrorMessage = error.response?.data?.message || 'Failed to start teleoperation';
      }
    },

    // Stop teleoperation
    async stopTeleoperation() {
      try {
        const response = await robotApi.stopTeleoperation();

        if (response.data.status === 'success') {
          this.status.mode = null;
          console.log('Teleoperation stopped');
        }
      } catch (error) {
        console.error('Error stopping teleoperation:', error);
  this.internalHasError = true;
  this.internalErrorMessage = error.response?.data?.message || 'Failed to stop teleoperation';
      }
    },

    // Get robot status
    async fetchRobotStatus() {
      try {
        const response = await robotApi.getStatus();

        if (response.data.status === 'success') {
          this.status = { ...this.status, ...response.data.data };
        }
      } catch (error) {
        console.error('Error fetching robot status:', error);
        // Don't set error state for status polling failures
      }
    },

    // Backwards-compatible alias used by several views (e.g. Dashboard, Teleoperation)
    async updateStatus() {
      return this.fetchRobotStatus();
    },

    // Start status polling
    startStatusPolling(interval = 1000) {
      if (this.statusPollingTimer) {
        clearInterval(this.statusPollingTimer);
      }

      this.statusPollingTimer = setInterval(() => {
        if (this.status.connected) {
          this.fetchRobotStatus();
        }
      }, interval);
    },

    // Stop status polling
    stopStatusPolling() {
      if (this.statusPollingTimer) {
        clearInterval(this.statusPollingTimer);
        this.statusPollingTimer = null;
      }
    },

    // Set teleoperation configuration
    setTeleoperationConfig(config) {
      this.teleoperationConfig = { ...this.teleoperationConfig, ...config };
      console.log('Teleoperation configuration updated:', this.teleoperationConfig);
    },

    // Enhanced teleoperation start with configuration
    async startTeleoperationWithConfig(config = null) {
      try {
        const finalConfig = config || this.teleoperationConfig;
        
        // Prepare configuration for backend
        const teleoperationParams = {
          fps: finalConfig.fps,
          show_cameras: finalConfig.showCameras,
          max_relative_target: finalConfig.maxRelativeTarget,
          operation_mode: finalConfig.operationMode,
          enable_safe_shutdown: finalConfig.enableSafeShutdown,
          moving_time: finalConfig.movingTime,
          teleop_time_limit: finalConfig.teleopTimeLimit,
          performance_monitoring: finalConfig.performanceMonitoring,
          debug_level: finalConfig.debugLevel
        };

        const response = await robotApi.startTeleoperationAdvanced(teleoperationParams);

        if (response.data.status === 'success') {
          this.status.mode = 'teleoperating';
          
          // Start performance monitoring if enabled
          if (finalConfig.performanceMonitoring) {
            this.startPerformanceMonitoring();
          }
          
          console.log('Advanced teleoperation started with config:', finalConfig);
        } else {
          this.internalHasError = true;
            this.internalErrorMessage = response.data.message || 'Failed to start teleoperation';
        }
      } catch (error) {
        console.error('Error starting advanced teleoperation:', error);
  this.internalHasError = true;
  this.internalErrorMessage = error.response?.data?.message || 'Failed to start teleoperation';
      }
    },

    // Stop enhanced teleoperation
    async stopTeleoperationAdvanced() {
      try {
        const response = await robotApi.stopTeleoperation();

        if (response.data.status === 'success') {
          this.status.mode = null;
          this.stopPerformanceMonitoring();
          console.log('Advanced teleoperation stopped');
        }
      } catch (error) {
        console.error('Error stopping teleoperation:', error);
  this.internalHasError = true;
  this.internalErrorMessage = error.response?.data?.message || 'Failed to stop teleoperation';
      }
    },

    // Performance monitoring
    startPerformanceMonitoring() {
      if (this.performanceTimer) {
        clearInterval(this.performanceTimer);
      }

      this.performanceTimer = setInterval(async () => {
        try {
          const response = await robotApi.getPerformanceMetrics();
          if (response.data.status === 'success') {
            this.performanceMetrics = {
              ...this.performanceMetrics,
              ...response.data.data,
              timestamp: new Date()
            };
          }
        } catch (error) {
          console.error('Error fetching performance metrics:', error);
        }
      }, 1000);
    },

    stopPerformanceMonitoring() {
      if (this.performanceTimer) {
        clearInterval(this.performanceTimer);
        this.performanceTimer = null;
      }
    },

    // Emergency stop functionality
    emergencyStop() {
      if (this.status.mode === 'teleoperating') {
        this.stopTeleoperationAdvanced();
        console.log('Emergency stop activated');
      }
    }
  }
});
