<template>
  <div class="teleoperation-config">
    <div class="card">
      <div class="card-header d-flex justify-content-between align-items-center">
        <h5 class="mb-0">Teleoperation Configuration</h5>
        <div class="btn-group btn-group-sm">
          <button 
            class="btn" 
            :class="configMode === 'standard' ? 'btn-primary' : 'btn-outline-primary'"
            @click="configMode = 'standard'"
          >
            Standard
          </button>
          <button 
            class="btn" 
            :class="configMode === 'expert' ? 'btn-primary' : 'btn-outline-primary'"
            @click="configMode = 'expert'"
          >
            Expert
          </button>
        </div>
      </div>
      
      <div class="card-body">
        <!-- Standard Configuration -->
        <div v-if="configMode === 'standard'">
          <div class="row g-3">
            <!-- Control Frequency -->
            <div class="col-md-6">
              <label class="form-label">Control Frequency</label>
              <select v-model="config.fps" class="form-select">
                <option :value="null">Maximum (No Limit)</option>
                <option :value="30">30 Hz (Standard)</option>
                <option :value="60">60 Hz (Smooth)</option>
                <option :value="100">100 Hz (High Performance)</option>
                <option :value="200">200 Hz (Ultra High)</option>
              </select>
              <div class="form-text">Higher frequencies provide smoother control but require more processing power</div>
            </div>

            <!-- Camera Settings -->
            <div class="col-md-6">
              <label class="form-label">Camera Display</label>
              <div class="form-check">
                <input 
                  class="form-check-input" 
                  type="checkbox" 
                  id="showCameras"
                  v-model="config.showCameras"
                >
                <label class="form-check-label" for="showCameras">
                  Show camera feeds during teleoperation
                </label>
              </div>
              <div class="form-text">Disable to reduce system load if cameras aren't needed</div>
            </div>

            <!-- Safety Settings -->
            <div class="col-md-6">
              <label class="form-label">Movement Safety Limit (degrees)</label>
              <select v-model="config.maxRelativeTarget" class="form-select">
                <option :value="5">5° (Very Safe - Training)</option>
                <option :value="15">15° (Safe - Normal Operation)</option>
                <option :value="25">25° (Standard - Default)</option>
                <option :value="45">45° (Extended Range)</option>
                <option :value="null">Unlimited (Expert Only)</option>
              </select>
              <div class="form-text">Limits maximum joint movement per command</div>
            </div>

            <!-- Operation Mode -->
            <div class="col-md-6">
              <label class="form-label">Operation Mode</label>
              <select v-model="config.operationMode" class="form-select">
                <option value="bimanual">Bimanual (Both Arms)</option>
                <option value="right_only">Right Arm Only</option>
                <option value="left_only">Left Arm Only</option>
              </select>
              <div class="form-text">Configure which arms to control</div>
            </div>

            <!-- Emergency Stop -->
            <div class="col-12">
              <div class="form-check">
                <input 
                  class="form-check-input" 
                  type="checkbox" 
                  id="enableSafeShutdown"
                  v-model="config.enableSafeShutdown"
                >
                <label class="form-check-label" for="enableSafeShutdown">
                  Enable emergency stop hotkey (Space bar)
                </label>
              </div>
              <div class="form-text">Allows instant teleoperation stop via keyboard</div>
            </div>
          </div>
        </div>

        <!-- Expert Configuration -->
        <div v-else-if="configMode === 'expert'">
          <div class="alert alert-warning">
            <i class="bi bi-exclamation-triangle me-2"></i>
            <strong>Expert Mode:</strong> Advanced settings that can affect robot safety and performance.
          </div>

          <div class="row g-3">
            <!-- Custom FPS -->
            <div class="col-md-6">
              <label class="form-label">Custom Control Frequency (Hz)</label>
              <input 
                type="number" 
                class="form-control" 
                v-model.number="config.customFps"
                :placeholder="config.fps || 'Unlimited'"
                min="1" 
                max="300"
              >
              <div class="form-text">Override standard frequency options (1-300 Hz)</div>
            </div>

            <!-- Movement Smoothness -->
            <div class="col-md-6">
              <label class="form-label">Movement Time (seconds)</label>
              <input 
                type="number" 
                class="form-control" 
                v-model.number="config.movingTime"
                step="0.01"
                min="0.01"
                max="1.0"
              >
              <div class="form-text">Time for robot to execute each movement command</div>
            </div>

            <!-- Advanced Safety -->
            <div class="col-md-6">
              <label class="form-label">Custom Safety Limit (degrees)</label>
              <input 
                type="number" 
                class="form-control" 
                v-model.number="config.customMaxTarget"
                :placeholder="config.maxRelativeTarget || 'Unlimited'"
                min="1" 
                max="180"
              >
              <div class="form-text">Custom joint movement limit (overrides presets)</div>
            </div>

            <!-- Teleoperation Time Limit -->
            <div class="col-md-6">
              <label class="form-label">Session Time Limit (minutes)</label>
              <input 
                type="number" 
                class="form-control" 
                v-model.number="config.teleopTimeLimit"
                placeholder="No limit"
                min="1" 
                max="120"
              >
              <div class="form-text">Automatic stop after specified time</div>
            </div>

            <!-- Performance Monitoring -->
            <div class="col-md-6">
              <div class="form-check">
                <input 
                  class="form-check-input" 
                  type="checkbox" 
                  id="performanceMonitoring"
                  v-model="config.performanceMonitoring"
                >
                <label class="form-check-label" for="performanceMonitoring">
                  Enable performance monitoring
                </label>
              </div>
              <div class="form-text">Track FPS, latency, and system metrics</div>
            </div>

            <!-- Debug Logging -->
            <div class="col-md-6">
              <label class="form-label">Debug Log Level</label>
              <select v-model="config.debugLevel" class="form-select">
                <option value="INFO">INFO (Standard)</option>
                <option value="DEBUG">DEBUG (Verbose)</option>
                <option value="WARNING">WARNING (Minimal)</option>
                <option value="ERROR">ERROR (Critical Only)</option>
              </select>
              <div class="form-text">Control verbosity of teleoperation logs</div>
            </div>
          </div>
        </div>

        <!-- Configuration Actions -->
        <div class="mt-4">
          <div class="row g-2">
            <div class="col-md-6">
              <div class="d-grid">
                <button 
                  class="btn btn-success" 
                  @click="applyConfiguration"
                  :disabled="!isConnected || isTeleoperating"
                >
                  <i class="bi bi-check-circle me-2"></i>
                  Apply Configuration
                </button>
              </div>
            </div>
            <div class="col-md-3">
              <div class="d-grid">
                <button 
                  class="btn btn-outline-secondary" 
                  @click="saveConfiguration"
                >
                  <i class="bi bi-save me-2"></i>
                  Save
                </button>
              </div>
            </div>
            <div class="col-md-3">
              <div class="d-grid">
                <button 
                  class="btn btn-outline-secondary" 
                  @click="loadConfiguration"
                >
                  <i class="bi bi-folder-open me-2"></i>
                  Load
                </button>
              </div>
            </div>
          </div>
        </div>

        <!-- Current Configuration Summary -->
        <div class="mt-4">
          <div class="card bg-light">
            <div class="card-body">
              <h6 class="card-title">Current Configuration</h6>
              <div class="row g-2 small">
                <div class="col-6">
                  <strong>Frequency:</strong> {{ getEffectiveFps() }} Hz
                </div>
                <div class="col-6">
                  <strong>Safety Limit:</strong> {{ getEffectiveSafetyLimit() }}°
                </div>
                <div class="col-6">
                  <strong>Cameras:</strong> {{ config.showCameras ? 'Enabled' : 'Disabled' }}
                </div>
                <div class="col-6">
                  <strong>Mode:</strong> {{ config.operationMode }}
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
import { ref, computed, onMounted } from 'vue';
import { useRobotStore } from '@/stores/robotStore';

const robotStore = useRobotStore();

// Props
const emit = defineEmits(['configurationApplied']);

// Data
const configMode = ref('standard');

const config = ref({
  // Standard options
  fps: 30,
  showCameras: true,
  maxRelativeTarget: 25,
  operationMode: 'bimanual',
  enableSafeShutdown: true,
  
  // Expert options
  customFps: null,
  movingTime: 0.1,
  customMaxTarget: null,
  teleopTimeLimit: null,
  performanceMonitoring: false,
  debugLevel: 'INFO'
});

// Computed properties
const isConnected = computed(() => robotStore.isConnected);
const isTeleoperating = computed(() => robotStore.isTeleoperating);

// Methods
const getEffectiveFps = () => {
  if (configMode.value === 'expert' && config.value.customFps) {
    return config.value.customFps;
  }
  return config.value.fps || 'Unlimited';
};

const getEffectiveSafetyLimit = () => {
  if (configMode.value === 'expert' && config.value.customMaxTarget) {
    return config.value.customMaxTarget;
  }
  return config.value.maxRelativeTarget || 'Unlimited';
};

const applyConfiguration = () => {
  const finalConfig = {
    fps: configMode.value === 'expert' && config.value.customFps ? config.value.customFps : config.value.fps,
    showCameras: config.value.showCameras,
    maxRelativeTarget: configMode.value === 'expert' && config.value.customMaxTarget ? config.value.customMaxTarget : config.value.maxRelativeTarget,
    operationMode: config.value.operationMode,
    enableSafeShutdown: config.value.enableSafeShutdown,
    movingTime: config.value.movingTime,
    teleopTimeLimit: config.value.teleopTimeLimit,
    performanceMonitoring: config.value.performanceMonitoring,
    debugLevel: config.value.debugLevel
  };

  emit('configurationApplied', finalConfig);
  
  // Store configuration in robot store
  robotStore.setTeleoperationConfig(finalConfig);
  
  // Show success message
  console.log('Teleoperation configuration applied:', finalConfig);
};

const saveConfiguration = () => {
  try {
    localStorage.setItem('lerobot_teleoperation_config', JSON.stringify(config.value));
    console.log('Configuration saved to localStorage');
  } catch (error) {
    console.error('Failed to save configuration:', error);
  }
};

const loadConfiguration = () => {
  try {
    const saved = localStorage.getItem('lerobot_teleoperation_config');
    if (saved) {
      const savedConfig = JSON.parse(saved);
      config.value = { ...config.value, ...savedConfig };
      console.log('Configuration loaded from localStorage');
    }
  } catch (error) {
    console.error('Failed to load configuration:', error);
  }
};

// Initialize
onMounted(() => {
  loadConfiguration();
});
</script>

<style scoped>
.teleoperation-config {
  width: 100%;
}

.card-body {
  max-height: 600px;
  overflow-y: auto;
}

.form-text {
  font-size: 0.8rem;
}

.alert {
  font-size: 0.9rem;
}

.bg-light {
  background-color: #f8f9fa !important;
}

body.dark-mode .bg-light {
  background-color: #2d2d2d !important;
  color: #e4e6eb;
}

body.dark-mode .card {
  background-color: #2a2a2a;
  border-color: #333;
}
</style>
