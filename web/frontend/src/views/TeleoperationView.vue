<template>
  <div class="teleoperation-view">
    <h1>Teleoperation</h1>
    <h6>
      The connection panel decides whether you operate bimanual or single arm. Pick your environment and any display options, then press <em>Start Teleoperation</em>; hit the <kbd>Space</kbd> bar anytime for an emergency stop.
    </h6>
    

  <!-- Connection status card removed: dashboard handles connection -->

    <!-- Teleoperation Configuration (shown when connected) -->
  <div v-if="robotStore.isConnected" class="config-section">
      <div class="config-card">
        <h3><i class="bi bi-sliders me-2"></i>Teleoperation Settings</h3>
        
        <div class="config-grid">
          <!-- Environment Type -->
          <div class="config-group">
            <label>Environment</label>
            <div class="env-selector">
              <button 
                :class="['env-btn', { active: teleoperationConfig.environment === 'real' }]"
                @click="teleoperationConfig.environment = 'real'"
              >
                <i class="bi bi-robot"></i>
                <span>Real Robot</span>
              </button>
              <button 
                :class="['env-btn', { active: teleoperationConfig.environment === 'sim' }]"
                @click="teleoperationConfig.environment = 'sim'"
                disabled
              >
                <i class="bi bi-display"></i>
                <span>Simulation</span>
              </button>
            </div>
          </div>

          <!-- Camera Display Options -->
          <div class="config-group">
            <label>Display Options</label>
            <div class="display-options">
              <label class="checkbox-label">
                <input 
                  type="checkbox" 
                  v-model="teleoperationConfig.showCameras"
                  class="config-checkbox"
                />
                Enable Camera Streaming
                <small>Stream camera feeds to the browser during teleoperation</small>
              </label>
              
              <label class="checkbox-label disabled">
                <input 
                  type="checkbox" 
                  v-model="teleoperationConfig.displayData"
                  class="config-checkbox"
                  disabled
                />
                Show External Display
                <small>Open LeRobot's display window with cameras and telemetry</small>
              </label>
            </div>
          </div>
        </div>

        <!-- Start/Stop Controls -->
        <div class="operation-controls">
          <button 
            v-if="!isOperating" 
            @click="startTeleoperation" 
            class="btn btn-success btn-lg"
            :disabled="isStarting"
          >
            <i class="bi bi-play-circle me-2"></i>
            {{ isStarting ? 'Starting...' : 'Start Teleoperation' }}
          </button>
          
          <div v-if="isOperating" class="active-controls">
            <button @click="stopTeleoperation" class="btn btn-warning btn-lg">
              <i class="bi bi-stop-circle me-2"></i>Stop Teleoperation
            </button>
            <button 
              v-if="teleoperationConfig.showCameras" 
              @click="showCameraModal = true" 
              class="btn btn-info"
            >
              <i class="bi bi-camera-video me-2"></i>View Cameras
            </button>
            <button @click="emergencyStop" class="btn btn-danger">
              <i class="bi bi-exclamation-triangle me-2"></i>Emergency Stop
            </button>
          </div>
        </div>
      </div>
    </div>

    <!-- Active Teleoperation Status -->
    <div v-if="isOperating" class="operation-status">
      <div class="status-grid">
        <div class="status-item">
          <span class="status-label">Mode</span>
          <span class="status-value">{{ getCurrentModeDisplay() }}</span>
        </div>
        <div class="status-item">
          <span class="status-label">Environment</span>
          <span class="status-value">{{ teleoperationConfig.environment === 'real' ? 'Real Robot' : 'Simulation' }}</span>
        </div>
        <div class="status-item">
          <span class="status-label">FPS Current</span>
          <span class="status-value">{{ teleopStatus.fps_current != null ? teleopStatus.fps_current.toFixed(1) : '-' }}</span>
        </div>
        <div class="status-item">
          <span class="status-label">Duration</span>
          <span class="status-value">{{ formatDuration(operationDuration) }}</span>
        </div>
      </div>
    </div>

    <!-- Display Data Info (if enabled) -->
    <div v-if="isOperating && teleoperationConfig.displayData" class="display-data-info">
      <div class="alert alert-info">
        <i class="bi bi-window me-2"></i>
        <strong>External Display Active:</strong> LeRobot's display window should be open showing real-time camera feeds and telemetry data.
        If you don't see it, check your system for a new rerun window.
      </div>
    </div>

    <!-- Camera Modal (appears when cameras enabled during teleoperation) -->
    <CameraModal
      :open="showCameraModal"
      @update:open="showCameraModal = $event"
      @close="showCameraModal = false"
    />
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import { useRobotStore } from '@/stores/robotStore'
import robotApi from '@/services/api/robotApi'
import CameraModal from '@/components/CameraModal.vue'

const robotStore = useRobotStore()
// Removed storeToRefs usage (not imported) – we access reactive store state directly.

// State
const isStarting = ref(false)
const isOperating = ref(false)
const operationStartTime = ref(null)
const operationDuration = ref(0)
const showCameraModal = ref(false)
const teleopStatus = computed(() => robotStore.status?.teleoperation || {})

// Teleoperation configuration
const teleoperationConfig = ref({
  environment: 'real',
  showCameras: false, // disabled by default
  displayData: false  // External LeRobot display window
})

// Watch for changes in display options to make them mutually exclusive
watch(() => teleoperationConfig.value.showCameras, (newValue) => {
  if (newValue) {
    teleoperationConfig.value.displayData = false
  }
})

watch(() => teleoperationConfig.value.displayData, (newValue) => {
  if (newValue) {
    teleoperationConfig.value.showCameras = false
  }
})

// Close camera modal when teleoperation stops
watch(() => isOperating.value, (operating) => {
  if (!operating) {
    showCameraModal.value = false
  }
})

const deriveOperationModeForTeleop = () => {
  const status = robotStore.status || {}
  const rawMode = status.mode
  if (rawMode) {
    const lowered = String(rawMode).toLowerCase()
    if (lowered.includes('left')) return 'left_only'
    if (lowered.includes('right')) return 'right_only'
    if (lowered.includes('bi') || lowered.includes('dual')) return 'bimanual'
  }

  const arms = Array.isArray(status.available_arms) ? status.available_arms : []
  if (arms.length === 1) {
    const arm = String(arms[0]).toLowerCase()
    if (arm.includes('left')) return 'left_only'
    if (arm.includes('right')) return 'right_only'
  }

  return 'bimanual'
}

const readableOperationMode = computed(() => {
  const effective = deriveOperationModeForTeleop()
  if (effective === 'left_only') return 'Left Arm'
  if (effective === 'right_only') return 'Right Arm'
  return 'Bimanual'
})

// Methods
// Connection / disconnection handled elsewhere

const startTeleoperation = async () => {
  isStarting.value = true
  
  try {
    const config = {
      operation_mode: deriveOperationModeForTeleop(),
      show_cameras: teleoperationConfig.value.showCameras,
      display_data: teleoperationConfig.value.displayData,  // Add display_data parameter
      fps: 30,
      safety_limits: true,
      performance_monitoring: true
    }
  // fps target comes from backend configuration in teleopStatus; no direct set here
    
    // Use the dedicated teleoperation API with 'normal' preset as default
    const response = await robotApi.startTeleoperation({ ...config, preset: 'normal' })
    
    isOperating.value = true
    operationStartTime.value = Date.now()
    
    // Auto-open camera modal if camera streaming was enabled
    if (teleoperationConfig.value.showCameras) {
      showCameraModal.value = true
    }
    
    console.log('✅ Teleoperation started successfully:', response.data)
    
    // Add status polling to detect issues early
  const statusCheckInterval = setInterval(async () => {
      try {
        const status = await robotApi.getTeleoperationStatus()
        console.log('📊 Teleoperation status check:', status.data)
        
        // Check for error conditions
        if (status.data.status === 'error' || status.data.active === false) {
          console.error('⚠️ Teleoperation stopped unexpectedly:', status.data)
          clearInterval(statusCheckInterval)
          isOperating.value = false
          // Log unexpected stop; surface via alert
          console.error('Teleoperation stopped unexpectedly:', status.data)
        }
      } catch (error) {
        console.error('❌ Status check failed:', error)
        // Don't stop teleoperation just because status check failed
      }
  }, 2000) // Check every 2 seconds
    
    // Store interval for cleanup
    window.teleoperationStatusInterval = statusCheckInterval
    
  } catch (error) {
    console.error('Failed to start teleoperation:', error)
    alert(`Failed to start teleoperation: ${error.message}`)
  } finally {
    isStarting.value = false
  }
}

const stopTeleoperation = async () => {
  try {
    // Clean up status checking
    if (window.teleoperationStatusInterval) {
      clearInterval(window.teleoperationStatusInterval)
      window.teleoperationStatusInterval = null
    }
    
    await robotApi.stopTeleoperation()
    isOperating.value = false
    operationStartTime.value = null
    operationDuration.value = 0
  // clear derived fps (teleopStatus will reflect null on next status)
  // Clear any cached frames
  robotStore.cameraStreams = {}
    console.log('✅ Teleoperation stopped successfully')
  } catch (error) {
    console.error('❌ Failed to stop teleoperation:', error)
  }
}

const emergencyStop = async () => {
  try {
    await robotApi.emergencyStop()
    isOperating.value = false
    operationStartTime.value = null
    operationDuration.value = 0
  // clear derived fps (teleopStatus will reflect null on next status)
  } catch (error) {
    console.error('Emergency stop failed:', error)
  }
}

const getCurrentModeDisplay = () => readableOperationMode.value

const formatDuration = (seconds) => {
  const mins = Math.floor(seconds / 60)
  const secs = seconds % 60
  return `${mins}:${secs.toString().padStart(2, '0')}`
}

// Update operation duration
let durationInterval = null

const startDurationTracking = () => {
  if (durationInterval) clearInterval(durationInterval)
  
  durationInterval = setInterval(() => {
    if (operationStartTime.value) {
      operationDuration.value = Math.floor((Date.now() - operationStartTime.value) / 1000)
    }
  }, 1000)
}

const stopDurationTracking = () => {
  if (durationInterval) {
    clearInterval(durationInterval)
    durationInterval = null
  }
}

// Lifecycle
const startStatusPolling = () => {
  if (window.teleoperationStatusInterval) return
  window.teleoperationStatusInterval = setInterval(async () => {
    try {
      const { data } = await robotApi.getTeleoperationStatus()
      const s = data.data || {}
      if (!s.active) {
        clearInterval(window.teleoperationStatusInterval)
        window.teleoperationStatusInterval = null
        isOperating.value = false
        return
      }
      // keep duration fresh using server session_duration if provided
      if (typeof s.session_duration === 'number' && s.session_duration >= 0) {
        operationStartTime.value = Date.now() - Math.floor(s.session_duration) * 1000
      }
  // FPS metrics come via teleopStatus from socket and config; no direct assignment needed here
    } catch {
      // ignore transient errors
    }
  }, 2000)
}

const stopStatusPolling = () => {
  if (window.teleoperationStatusInterval) {
    clearInterval(window.teleoperationStatusInterval)
    window.teleoperationStatusInterval = null
  }
}

const syncTeleopStatus = async () => {
  try {
    const { data } = await robotApi.getTeleoperationStatus()
    const s = data.data || {}
    if (s.active) {
      isOperating.value = true
      // Restore config snapshot if available
      if (s.configuration) {
        const cfg = s.configuration
        teleoperationConfig.value.showCameras = !!cfg.show_cameras
        teleoperationConfig.value.displayData = !!cfg.display_data
      }
      // Restore duration based on server session time
      if (typeof s.session_duration === 'number' && s.session_duration >= 0) {
        operationStartTime.value = Date.now() - Math.floor(s.session_duration) * 1000
      } else if (!operationStartTime.value) {
        operationStartTime.value = Date.now()
      }
  // FPS values are derived from store/socket; no direct set here
      startDurationTracking()
      startStatusPolling()
    } else {
      isOperating.value = false
      stopStatusPolling()
    }
  } catch {
    // If status endpoint fails, leave current UI state unchanged
  }
}

onMounted(() => {
  // Ensure socket connected for receiving camera_frame events
  robotStore.initSocket()
  robotStore.updateStatus()
  // Sync teleoperation status when (re)entering the view
  syncTeleopStatus()
  startDurationTracking()
  const handleKeyPress = (event) => {
    if (event.code === 'Space' && isOperating.value) {
      event.preventDefault()
      emergencyStop()
    }
  }
  document.addEventListener('keydown', handleKeyPress)
  const handleVisibility = () => {
    if (!document.hidden) {
      // Refresh state when the tab/view regains focus
      syncTeleopStatus()
    }
  }
  document.addEventListener('visibilitychange', handleVisibility)
  onUnmounted(() => {
    document.removeEventListener('keydown', handleKeyPress)
    document.removeEventListener('visibilitychange', handleVisibility)
    stopDurationTracking()
    stopStatusPolling()
  })
})

onUnmounted(() => {
  stopDurationTracking()
})
</script>

<style scoped>
.teleoperation-view {
  max-width: 1400px;
  margin: 0 auto;
  padding: 1.5rem;
}

h1 { 
  margin: 0 0 .75rem; 
  font-size: 1.8rem; 
  font-weight:600; 
  letter-spacing:-0.5px; 
}

h6 { 
  margin: 0 0 1.25rem; 
  font-size: .75rem; 
  line-height: 1.6; 
  opacity: .7; 
}

h6 kbd {
  background: #1f2937;
  color: #f9fafb;
  padding: 0.15rem 0.4rem;
  border-radius: 4px;
  font-size: 0.7rem;
  font-weight: 600;
}

/* Status card removed */

/* Configuration Section */
.config-section {
  margin-bottom: 2rem;
}

.config-card {
  background: linear-gradient(135deg,#ffffff 0%,#f8fafc 100%);
  border-radius: 14px;
  padding: 1.1rem 1.35rem 1.25rem;
  border: 1px solid #e5e7eb;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
}

.config-card h3 {
  margin: 0 0 1.5rem 0;
  color: #1f2937;
  font-size: 1.05rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: .06em;
  opacity: .75;
}

.config-grid {
  display: grid;
  gap: 1.5rem; /* reduced to feel like tighter layout */
  margin-bottom: 1.4rem;
}

.config-group {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.config-group label {
  font-weight: 600;
  color: #374151;
}

/* Environment Selector */
.env-selector {
  display: flex;
  gap: 0.85rem;
}

.env-btn {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.45rem;
  padding: 0.85rem 0.9rem;
  background: white;
  border: 2px solid #e5e7eb;
  border-radius: 10px;
  cursor: pointer;
  transition: all 0.2s ease;
  color: #111827;
}

.env-btn:hover:not(:disabled) {
  border-color: #3b82f6;
  background: #eff6ff;
  color: #111827;
}

.env-btn.active {
  border-color: #3b82f6;
  background: #dbeafe;
  color: #111827;
}

.env-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.env-btn i {
  font-size: 1.5rem;
  color: currentColor;
}

/* Display Options */
.display-options {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

/* Unified checkbox label style with centered checkbox */
.checkbox-label {
  position: relative;
  display: block;
  cursor: pointer;
  padding: 0.75rem 0.85rem 1.05rem 2.5rem; /* proportionally reduced */
  background: #f9fafb;
  border: 1px solid #e5e7eb;
  border-radius: 10px;
  transition: all 0.2s ease;
  line-height: 1.1;
}
.checkbox-label:hover {
  background: #f3f4f6;
  border-color: #d1d5db;
}
.checkbox-label.disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
.checkbox-label.disabled:hover {
  background: #f9fafb;
  border-color: #e5e7eb;
}
.checkbox-label input.config-checkbox {
  position: absolute;
  left: 1rem;
  top: 50%;
  transform: translateY(-50%);
  margin: 0;
  width: 1.1rem;
  height: 1.1rem;
}
.checkbox-label small {
  display: block;
  color: #6b7280;
  font-size: 0.7rem;
  margin-top: 0.35rem;
}

/* Operation Controls */
.operation-controls {
  display: flex;
  justify-content: center;
  padding-top: 1.5rem;
  border-top: 1px solid #f3f4f6;
}

.active-controls {
  display: flex;
  gap: 1rem;
  align-items: center;
}

/* Operation Status */
.operation-status {
  background: linear-gradient(135deg,#ffffff 0%,#f8fafc 100%);
  border-radius: 14px;
  padding: 1.1rem 1.35rem 1.25rem;
  border: 1px solid #e5e7eb;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
  margin-bottom: 1.75rem;
}

.status-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 1rem;
}

.status-item {
  text-align: center;
  padding: 1rem;
  background: #f9fafb;
  border-radius: 0.5rem;
}

.status-label {
  display: block;
  font-size: 0.85rem;
  color: #6b7280;
  margin-bottom: 0.25rem;
}

.status-value {
  display: block;
  font-weight: 600;
  color: #1f2937;
}

/* Camera Section */
.camera-section {
  background: linear-gradient(135deg,#ffffff 0%,#f8fafc 100%);
  border-radius: 14px;
  padding: 1.1rem 1.35rem 1.25rem;
  border: 1px solid #e5e7eb;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
}

.camera-section h3 {
  margin: 0 0 1.5rem 0;
  color: #1f2937;
  font-size: 1.05rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: .06em;
  opacity: .75;
}

/* Display Data Info */
.display-data-info {
  margin-bottom: 2rem;
}

.alert {
  padding: 1rem;
  border-radius: 0.5rem;
  border: 1px solid transparent;
}

.alert-info {
  background-color: #e0f2fe;
  border-color: #0288d1;
  color: #01579b;
}

.camera-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 1rem;
}

.camera-feed {
  border: 1px solid #e5e7eb;
  border-radius: 0.5rem;
  overflow: hidden;
}

.camera-header {
  background: #f9fafb;
  padding: 0.75rem;
  font-weight: 600;
  color: #374151;
  border-bottom: 1px solid #e5e7eb;
}

.camera-stream {
  aspect-ratio: 16/9;
  background: #000;
  position: relative;
}

.stream-placeholder {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: #6b7280;
  background: #f3f4f6;
}

.stream-placeholder i {
  font-size: 2rem;
  margin-bottom: 0.5rem;
}

/* Buttons */
.btn {
  padding: 0.75rem 1.5rem;
  border: none;
  border-radius: 0.5rem;
  cursor: pointer;
  font-weight: 500;
  transition: all 0.2s ease;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
}

.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn-primary {
  background: #3b82f6;
  color: white;
}

.btn-primary:hover:not(:disabled) {
  background: #2563eb;
}

.btn-secondary {
  background: #6b7280;
  color: white;
}

.btn-secondary:hover {
  background: #4b5563;
}

.btn-success {
  background: #10b981;
  color: white;
}

.btn-success:hover:not(:disabled) {
  background: #059669;
}

.btn-warning {
  background: #f59e0b;
  color: white;
}

.btn-warning:hover {
  background: #d97706;
}

.btn-danger {
  background: #ef4444;
  color: white;
}

.btn-danger:hover {
  background: #dc2626;
}

.btn-outline {
  background: white;
  color: #3b82f6;
  border: 1px solid #3b82f6;
}

.btn-outline:hover {
  background: #3b82f6;
  color: white;
}

.btn-lg {
  padding: 1rem 2rem;
  font-size: 1.1rem;
}

/* Dark mode */
body.dark-mode .teleoperation-view .config-card,
body.dark-mode .teleoperation-view .operation-status,
body.dark-mode .teleoperation-view .camera-section {
  background: linear-gradient(135deg,#1f2937 0%,#111827 100%);
  border-color: #374151;
  box-shadow: 0 4px 18px rgba(0,0,0,0.45);
}

body.dark-mode .teleoperation-view h1 {
  color: #f1f5f9;
}

body.dark-mode .teleoperation-view h6 {
  color: #94a3b8;
}

body.dark-mode .teleoperation-view h3 {
  color: #cbd5e1;
}

body.dark-mode .teleoperation-view h6 kbd {
  background: #374151;
  color: #e5e7eb;
}

/* Responsive Design */
@media (max-width: 768px) {
  .teleoperation-view {
    padding: 1rem;
  }
  
  h1 {
    font-size: 1.5rem;
  }
  
  .status-header {
    flex-direction: column;
    text-align: center;
    gap: 1rem;
  }
  
  .mode-selector {
    grid-template-columns: 1fr;
  }
  
  .env-selector {
    flex-direction: column;
  }
  
  .active-controls {
    flex-direction: column;
    width: 100%;
  }
  
  .camera-grid {
    grid-template-columns: 1fr;
  }
}
</style>