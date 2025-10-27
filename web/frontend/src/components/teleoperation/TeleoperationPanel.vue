<template>
  <div class="teleoperation-panel">
    <div class="card">
      <div class="card-header d-flex justify-content-between align-items-center">
        <h5 class="mb-0">Robot Teleoperation</h5>
        <span 
          class="badge" 
          :class="isTeleoperating ? 'bg-success' : 'bg-secondary'"
        >
          {{ isTeleoperating ? 'Active' : 'Inactive' }}
        </span>
      </div>
      
      <div class="card-body">
        <div v-if="!isConnected" class="alert alert-warning">
          Please connect to a robot first.
        </div>
        
        <div v-else>
          <!-- Teleoperation Settings -->
          <div v-if="!isTeleoperating" class="mb-3">
            <div class="mb-3">
              <label for="fpsInput" class="form-label">FPS (leave empty for max speed)</label>
              <input 
                type="number" 
                id="fpsInput" 
                v-model.number="fps" 
                class="form-control" 
                placeholder="e.g. 30"
                min="1"
                max="100"
              />
            </div>
            
            <div class="d-grid">
              <button 
                type="button" 
                class="btn btn-success" 
                @click="startTeleoperation" 
                :disabled="isLoading"
              >
                <span v-if="isLoading" class="spinner-border spinner-border-sm me-2" role="status"></span>
                Start Teleoperation
              </button>
            </div>
          </div>
          
          <!-- Active Teleoperation Controls -->
          <div v-else>
            <div class="alert alert-info mb-3">
              <i class="fas fa-info-circle me-2"></i>
              Teleoperation is active. The robot is now controllable.
            </div>
            
            <div class="d-grid">
              <button 
                type="button" 
                class="btn btn-danger" 
                @click="stopTeleoperation" 
                :disabled="isLoading"
              >
                <span v-if="isLoading" class="spinner-border spinner-border-sm me-2" role="status"></span>
                Stop Teleoperation
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
    
    <div v-if="hasError" class="alert alert-danger mt-3">
      {{ errorMessage }}
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue';
import { useRobotStore } from '@/stores/robotStore';

const robotStore = useRobotStore();
const fps = ref(null);

// Computed properties
const isConnected = computed(() => robotStore.isConnected);
const isTeleoperating = computed(() => robotStore.isTeleoperating);
const isLoading = computed(() => robotStore.isLoading);
const hasError = computed(() => robotStore.hasError);
const errorMessage = computed(() => robotStore.errorMessage);

// Methods
const startTeleoperation = async () => {
  try {
    await robotStore.startTeleoperation(fps.value);
  } catch (error) {
    console.error('Error starting teleoperation:', error);
  }
};

const stopTeleoperation = async () => {
  try {
    await robotStore.stopTeleoperation();
  } catch (error) {
    console.error('Error stopping teleoperation:', error);
  }
};
</script>

<style scoped>
.teleoperation-panel {
  width: 100%;
}
</style>