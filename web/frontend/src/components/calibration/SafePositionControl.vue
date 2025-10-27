<template>
  <div class="safe-position-control">
    <div class="card">
      <div class="card-header">
        <h5 class="mb-0">
          <i class="bi bi-shield-check me-2"></i>
          Safe Position Control
        </h5>
      </div>
      <div class="card-body">
        <div class="row g-3">
          <!-- Arm Selection -->
          <div class="col-md-6">
            <label class="form-label">Arms to Move</label>
            <div class="form-check">
              <input 
                class="form-check-input" 
                type="radio" 
                id="allArms" 
                v-model="config.armSelection"
                value="all"
              >
              <label class="form-check-label" for="allArms">
                All Arms ({{ availableArms.length }} arms)
              </label>
            </div>
            <div class="form-check">
              <input 
                class="form-check-input" 
                type="radio" 
                id="selectiveArms" 
                v-model="config.armSelection"
                value="selective"
              >
              <label class="form-check-label" for="selectiveArms">
                Select Specific Arms
              </label>
            </div>
            
            <!-- Specific Arm Selection (when selective is chosen) -->
            <div v-if="config.armSelection === 'selective'" class="mt-2 ms-3">
              <div class="form-check" v-for="arm in availableArms" :key="arm">
                <input 
                  class="form-check-input" 
                  type="checkbox" 
                  :id="`arm_${arm}`"
                  v-model="config.selectedArms"
                  :value="arm"
                >
                <label class="form-check-label" :for="`arm_${arm}`">
                  {{ arm }}
                </label>
              </div>
            </div>
          </div>

          <!-- Movement Parameters -->
          <div class="col-md-6">
            <div class="mb-3">
              <label class="form-label">Speed Factor</label>
              <input 
                type="range" 
                class="form-range" 
                v-model.number="config.speedFactor"
                min="0.1" 
                max="1.0" 
                step="0.1"
              >
              <div class="d-flex justify-content-between">
                <small class="text-muted">Slow (0.1)</small>
                <small class="text-primary">{{ config.speedFactor }}</small>
                <small class="text-muted">Fast (1.0)</small>
              </div>
            </div>

            <div class="mb-3">
              <label class="form-label">Timeout (seconds)</label>
              <input 
                type="number" 
                class="form-control" 
                v-model.number="config.timeout"
                min="5" 
                max="60"
                step="1"
              >
              <div class="form-text">Maximum time to wait for movement completion</div>
            </div>
          </div>

          <!-- Display Options -->
          <div class="col-12">
            <div class="form-check">
              <input 
                class="form-check-input" 
                type="checkbox" 
                id="showMovementData"
                v-model="config.displayData"
              >
              <label class="form-check-label" for="showMovementData">
                Display movement data during operation
              </label>
            </div>
          </div>

          <!-- Control Buttons -->
          <div class="col-12">
            <div class="d-grid gap-2 d-md-flex justify-content-md-end">
              <button 
                class="btn btn-outline-secondary me-md-2" 
                type="button"
                @click="resetToDefaults"
              >
                <i class="bi bi-arrow-clockwise me-2"></i>
                Reset Defaults
              </button>
              <button 
                class="btn btn-success" 
                type="button"
                @click="moveToSafePosition"
                :disabled="!isConnected || isMoving || isTeleoperating"
              >
                <span v-if="isMoving" class="spinner-border spinner-border-sm me-2" role="status"></span>
                <i v-else class="bi bi-shield-check me-2"></i>
                {{ isMoving ? 'Moving...' : 'Move to Safe Position' }}
              </button>
            </div>
          </div>
        </div>

        <!-- Status Display -->
        <div v-if="movementStatus" class="mt-3">
          <div class="alert" :class="getStatusAlertClass()">
            <div class="d-flex justify-content-between align-items-center">
              <div>
                <i :class="getStatusIcon()" class="me-2"></i>
                {{ movementStatus.message }}
              </div>
              <div v-if="movementStatus.progress !== null" class="text-end">
                <small>{{ Math.round(movementStatus.progress) }}%</small>
              </div>
            </div>
            <div v-if="movementStatus.progress !== null" class="progress mt-2" style="height: 6px;">
              <div 
                class="progress-bar" 
                :class="getProgressBarClass()"
                :style="`width: ${movementStatus.progress}%`"
              ></div>
            </div>
          </div>
        </div>

        <!-- Warning for active teleoperation -->
        <div v-if="isTeleoperating" class="alert alert-warning">
          <i class="bi bi-exclamation-triangle me-2"></i>
          Safe position movement is not available during active teleoperation. Please stop teleoperation first.
        </div>

        <!-- Connection warning -->
        <div v-if="!isConnected" class="alert alert-info">
          <i class="bi bi-info-circle me-2"></i>
          Connect to a robot to use safe position control.
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, watch } from 'vue';
import { useRobotStore } from '@/stores/robotStore';
import robotApi from '@/services/api/robotApi';

const robotStore = useRobotStore();

// Data
const config = ref({
  armSelection: 'all',
  selectedArms: [],
  speedFactor: 0.3,
  timeout: 10,
  displayData: false
});

const isMoving = ref(false);
const movementStatus = ref(null);

// Computed properties
const isConnected = computed(() => robotStore.isConnected);
const isTeleoperating = computed(() => robotStore.isTeleoperating);
const availableArms = computed(() => robotStore.status.available_arms || []);

// Methods
const resetToDefaults = () => {
  config.value = {
    armSelection: 'all',
    selectedArms: [],
    speedFactor: 0.3,
    timeout: 10,
    displayData: false
  };
  movementStatus.value = null;
};

const moveToSafePosition = async () => {
  if (!isConnected.value || isTeleoperating.value) {
    return;
  }

  isMoving.value = true;
  movementStatus.value = {
    type: 'info',
    message: 'Initiating safe position movement...',
    progress: 0
  };

  try {
    // Prepare configuration
    const safePositionConfig = {
      arms: config.value.armSelection === 'all' ? null : config.value.selectedArms,
      timeout_s: config.value.timeout,
      speed_factor: config.value.speedFactor,
      display_data: config.value.displayData
    };

    // Simulate progress updates (in real implementation, this would come from the backend)
    const progressInterval = setInterval(() => {
      if (movementStatus.value && movementStatus.value.progress < 90) {
        movementStatus.value.progress += Math.random() * 20;
        movementStatus.value.message = `Moving to safe position... ${Math.round(movementStatus.value.progress)}%`;
      }
    }, 500);

    // Call the API
    const response = await robotApi.moveToSafePosition(safePositionConfig);

    clearInterval(progressInterval);

    if (response.data.status === 'success') {
      movementStatus.value = {
        type: 'success',
        message: 'Robot successfully moved to safe position',
        progress: 100
      };
    } else {
      movementStatus.value = {
        type: 'error',
        message: response.data.message || 'Failed to move to safe position',
        progress: null
      };
    }
  } catch (error) {
    console.error('Error moving to safe position:', error);
    movementStatus.value = {
      type: 'error',
      message: error.response?.data?.message || 'Failed to move to safe position',
      progress: null
    };
  } finally {
    isMoving.value = false;
    
    // Clear status after a delay
    setTimeout(() => {
      movementStatus.value = null;
    }, 5000);
  }
};

const getStatusAlertClass = () => {
  if (!movementStatus.value) return '';
  switch (movementStatus.value.type) {
    case 'success': return 'alert-success';
    case 'error': return 'alert-danger';
    case 'info': return 'alert-info';
    default: return 'alert-secondary';
  }
};

const getStatusIcon = () => {
  if (!movementStatus.value) return '';
  switch (movementStatus.value.type) {
    case 'success': return 'bi bi-check-circle';
    case 'error': return 'bi bi-exclamation-triangle';
    case 'info': return 'bi bi-info-circle';
    default: return 'bi bi-gear';
  }
};

const getProgressBarClass = () => {
  if (!movementStatus.value) return '';
  switch (movementStatus.value.type) {
    case 'success': return 'bg-success';
    case 'error': return 'bg-danger';
    case 'info': return 'bg-info';
    default: return 'bg-primary';
  }
};

// Initialize
onMounted(() => {
  // Load saved configuration if available
  try {
    const saved = localStorage.getItem('lerobot_safe_position_config');
    if (saved) {
      const savedConfig = JSON.parse(saved);
      config.value = { ...config.value, ...savedConfig };
    }
  } catch (error) {
    console.error('Failed to load safe position configuration:', error);
  }
});

// Watch for configuration changes and save them
watch(config, (newConfig) => {
  try {
    localStorage.setItem('lerobot_safe_position_config', JSON.stringify(newConfig));
  } catch (error) {
    console.error('Failed to save safe position configuration:', error);
  }
}, { deep: true });
</script>

<style scoped>
.safe-position-control {
  width: 100%;
}

.form-range::-webkit-slider-thumb {
  background: #0d6efd;
}

.form-range::-moz-range-thumb {
  background: #0d6efd;
}

.progress {
  border-radius: 3px;
}

.alert {
  border-radius: 4px;
}

body.dark-mode .form-range::-webkit-slider-thumb {
  background: #4dabf7;
}

body.dark-mode .form-range::-moz-range-thumb {
  background: #4dabf7;
}
</style>
