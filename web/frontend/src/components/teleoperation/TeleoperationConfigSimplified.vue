<template>
  <div class="teleoperation-config">
    <div class="card">
      <div class="card-header">
        <h5 class="mb-0">🚀 Quick Start Configuration</h5>
      </div>
      <div class="card-body">
        <!-- Simple Presets Only -->
        <div class="row g-3">
          <div class="col-12">
            <label class="form-label fw-bold">Choose Configuration Preset</label>
            <select v-model="selectedPreset" class="form-select form-select-lg" @change="applyPreset">
              <option value="safe">🛡️ Safe Mode (Beginner)</option>
              <option value="normal">⚡ Normal Mode (Recommended)</option>
              <option value="performance">🚄 Performance Mode (Advanced)</option>
            </select>
            <div class="form-text mt-2">
              <strong>{{ PRESETS[selectedPreset].description }}</strong>
            </div>
          </div>
          
          <div class="col-12">
            <div class="card bg-light">
              <div class="card-body">
                <h6 class="card-title">Current Settings:</h6>
                <div class="row">
                  <div class="col-md-4">
                    <small class="text-muted">Control Frequency:</small>
                    <div class="fw-bold">{{ PRESETS[selectedPreset].fps }}Hz</div>
                  </div>
                  <div class="col-md-4">
                    <small class="text-muted">Safety Limit:</small>
                    <div class="fw-bold">
                      {{ PRESETS[selectedPreset].maxRelativeTarget ? PRESETS[selectedPreset].maxRelativeTarget + '°' : 'Unlimited' }}
                    </div>
                  </div>
                  <div class="col-md-4">
                    <small class="text-muted">Operation Mode:</small>
                    <div class="fw-bold">{{ PRESETS[selectedPreset].operationMode }}</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
          
          <div class="col-12">
            <div class="form-check form-switch">
              <input v-model="showCameras" class="form-check-input" type="checkbox" id="showCameras">
              <label class="form-check-label" for="showCameras">
                📷 Enable Camera Display
              </label>
              <div class="form-text">Toggle camera feeds (disable to improve performance)</div>
            </div>
          </div>
          
          <div class="col-12">
            <div class="form-check form-switch">
              <input v-model="enableEmergencyStop" class="form-check-input" type="checkbox" id="enableEmergencyStop">
              <label class="form-check-label" for="enableEmergencyStop">
                🚨 Enable Emergency Stop (Space key)
              </label>
              <div class="form-text">Press spacebar to instantly stop teleoperation</div>
            </div>
          </div>
          
          <div class="col-12 mt-4">
            <div class="d-grid">
              <button 
                @click="applyConfiguration" 
                class="btn btn-success btn-lg"
                :disabled="robotStore.status.mode === 'teleoperating'"
              >
                <i class="bi bi-play-circle me-2"></i>
                {{ robotStore.status.mode === 'teleoperating' ? 'Teleoperation Active' : 'Start Teleoperation' }}
              </button>
            </div>
          </div>
          
          <!-- Emergency Stop Button -->
          <div v-if="robotStore.status.mode === 'teleoperating'" class="col-12">
            <div class="d-grid">
              <button 
                @click="handleEmergencyStop" 
                class="btn btn-danger btn-lg"
              >
                <i class="bi bi-stop-circle me-2"></i>
                🚨 EMERGENCY STOP
              </button>
            </div>
          </div>
        </div>

        <!-- Status Display -->
        <div v-if="robotStore.status.mode" class="mt-4">
          <div class="alert" :class="getStatusAlertClass()">
            <div class="d-flex align-items-center">
              <i :class="getStatusIcon()" class="me-2"></i>
              <div>
                <strong>Status:</strong> {{ getStatusText() }}
                <div v-if="robotStore.status.mode === 'teleoperating'" class="small mt-1">
                  Press <kbd>Space</kbd> for emergency stop
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Advanced Settings Toggle -->
        <div class="mt-4">
          <div class="text-center">
            <button 
              @click="showAdvanced = !showAdvanced" 
              class="btn btn-outline-secondary btn-sm"
            >
              {{ showAdvanced ? 'Hide' : 'Show' }} Advanced Settings
            </button>
          </div>
          
          <!-- Advanced Settings (Collapsed by default) -->
          <div v-if="showAdvanced" class="mt-3">
            <div class="card border-secondary">
              <div class="card-header bg-secondary text-white">
                <h6 class="mb-0">⚙️ Advanced Configuration</h6>
              </div>
              <div class="card-body">
                <div class="alert alert-warning">
                  <i class="bi bi-exclamation-triangle me-2"></i>
                  <strong>Advanced Mode:</strong> Only modify these if you know what you're doing.
                </div>
                
                <div class="row g-3">
                  <div class="col-md-6">
                    <label class="form-label">Custom FPS Override</label>
                    <input 
                      type="number" 
                      class="form-control" 
                      v-model.number="customFps"
                      placeholder="Leave empty to use preset"
                      min="1" 
                      max="300"
                    >
                  </div>
                  
                  <div class="col-md-6">
                    <label class="form-label">Custom Safety Limit (degrees)</label>
                    <input 
                      type="number" 
                      class="form-control" 
                      v-model.number="customMaxTarget"
                      placeholder="Leave empty to use preset"
                      min="1" 
                      max="180"
                    >
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue';
import { useRobotStore } from '@/stores/robotStore';

const robotStore = useRobotStore();

// Configuration state
const selectedPreset = ref('normal');
const showCameras = ref(true);
const enableEmergencyStop = ref(true);
const showAdvanced = ref(false);
const customFps = ref(null);
const customMaxTarget = ref(null);

// Preset configurations
const PRESETS = {
  safe: { 
    fps: 30, 
    maxRelativeTarget: 5, 
    operationMode: 'bimanual',
    description: 'Safe settings for learning and training. Limited movement range.'
  },
  normal: { 
    fps: 30, 
    maxRelativeTarget: 25, 
    operationMode: 'bimanual',
    description: 'Balanced performance and safety. Recommended for most users.'
  },
  performance: { 
    fps: 60, 
    maxRelativeTarget: null, 
    operationMode: 'bimanual',
    description: 'Maximum performance with unlimited movement range. For experts only.'
  }
};

// Apply preset automatically when selection changes
const applyPreset = () => {
  console.log(`Applied preset: ${selectedPreset.value}`);
};

// Build final configuration
const buildConfiguration = () => {
  const preset = PRESETS[selectedPreset.value];
  
  return {
    // Use custom values if provided, otherwise use preset
    fps: customFps.value || preset.fps,
    maxRelativeTarget: customMaxTarget.value !== null ? customMaxTarget.value : preset.maxRelativeTarget,
    operationMode: preset.operationMode,
    showCameras: showCameras.value,
    enableSafeShutdown: enableEmergencyStop.value
  };
};

// Apply configuration and start teleoperation
const applyConfiguration = async () => {
  const config = buildConfiguration();
  
  try {
    console.log('Starting teleoperation with config:', config);
    
    // Start teleoperation with the configured settings
    await robotStore.startTeleoperation(config.fps);
    
    console.log('✅ Teleoperation started successfully');
    
  } catch (error) {
    console.error('❌ Failed to start teleoperation:', error);
    // Could add toast notification here
  }
};

// Enhanced emergency stop
const handleEmergencyStop = async () => {
  try {
    console.log('🚨 Emergency stop initiated');
    await robotStore.emergencyStop();
    console.log('✅ Emergency stop completed');
  } catch (error) {
    console.error('❌ Emergency stop failed:', error);
    // Force stop even if API fails
    robotStore.status.mode = null;
  }
};

// Keyboard emergency stop
const handleKeydown = (event) => {
  if (enableEmergencyStop.value && event.code === 'Space' && robotStore.status.mode === 'teleoperating') {
    event.preventDefault();
    handleEmergencyStop();
  }
};

// Status helpers
const getStatusAlertClass = () => {
  switch (robotStore.status.mode) {
    case 'teleoperating': return 'alert-success';
    case 'connected': return 'alert-info';
    case 'error': return 'alert-danger';
    default: return 'alert-secondary';
  }
};

const getStatusIcon = () => {
  switch (robotStore.status.mode) {
    case 'teleoperating': return 'bi bi-play-circle-fill text-success';
    case 'connected': return 'bi bi-wifi text-info';
    case 'error': return 'bi bi-exclamation-triangle-fill text-danger';
    default: return 'bi bi-circle text-secondary';
  }
};

const getStatusText = () => {
  switch (robotStore.status.mode) {
    case 'teleoperating': return 'Teleoperation Active';
    case 'connected': return 'Robot Connected';
    case 'error': return 'Connection Error';
    default: return 'Disconnected';
  }
};

// Lifecycle
onMounted(() => {
  // Add keyboard listener for emergency stop
  if (enableEmergencyStop.value) {
    document.addEventListener('keydown', handleKeydown);
  }
});

onUnmounted(() => {
  document.removeEventListener('keydown', handleKeydown);
});
</script>

<style scoped>
.form-select-lg {
  font-size: 1.1rem;
  padding: 0.75rem;
}

.card {
  box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
}

.bg-light {
  background-color: #f8f9fa !important;
}

kbd {
  background-color: #212529;
  color: #fff;
  padding: 0.2rem 0.4rem;
  border-radius: 0.25rem;
  font-size: 0.875em;
}
</style>
