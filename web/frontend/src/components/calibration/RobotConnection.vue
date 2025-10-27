<template>
  <div class="aloha-robot-control">
    <div v-if="!isConnected" class="connection-form">
      <h3>ALOHA Robot Connection</h3>

      <div class="mb-3">
        <label for="operationMode" class="form-label">Operation Mode</label>
        <select id="operationMode" v-model="operationMode" class="form-select" :disabled="isLoading">
          <option value="bimanual">Bimanual (Both Arms)</option>
          <option value="right_only">Single-Handed (Right Arm Only)</option>
          <option value="left_only">Single-Handed (Left Arm Only)</option>
        </select>
      </div>

      <div class="mb-3">
        <label for="configuration" class="form-label">Configuration</label>
        <select id="configuration" v-model="configuration" class="form-select" :disabled="isLoading">
          <option value="demo_default_no_cameras">Demo Default without Cameras</option>
          <option value="demo_default_with_cameras">Demo Default with Cameras</option>
        </select>
      </div>

      <button
        @click="connectAloha"
        class="btn btn-primary"
        :disabled="isLoading"
      >
        {{ isLoading ? 'Connecting...' : 'Connect ALOHA Robot' }}
      </button>

      <!-- Show local error message if any -->
      <div v-if="errorMessage" class="alert alert-danger mt-3">
        {{ errorMessage }}
      </div>
    </div>

    <div v-else class="robot-controls">
      <h3>ALOHA Robot Connected</h3>
      
      <div class="alert alert-success mb-3">
        <i class="bi bi-check-circle me-2"></i>
        Robot successfully connected and ready for teleoperation.
      </div>

      <div class="d-grid">
        <button
          @click="disconnectRobot"
          class="btn btn-danger"
          :disabled="isLoading"
        >
          <span v-if="isLoading" class="spinner-border spinner-border-sm me-2" role="status"></span>
          <i v-else class="bi bi-plug me-2"></i>
          Disconnect Robot
        </button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue';
import { useRobotStore } from '@/stores/robotStore';
import robotApi from '@/services/api/robotApi';

// Initialize the robot store
const robotStore = useRobotStore();

const operationMode = ref('bimanual');
const configuration = ref('demo_default_no_cameras');
const isLoading = ref(false);
const errorMessage = ref('');

// Use computed properties to get reactive state from the store
const isConnected = computed(() => robotStore.isConnected);

// Helper function to get configuration settings
const getConfigurationSettings = (configName) => {
  const configs = {
    'demo_default_no_cameras': { fps: 30, enableCameras: false },
    'demo_default_with_cameras': { fps: 30, enableCameras: true },
  };
  return configs[configName] || configs['demo_default_no_cameras'];
};

const connectAloha = async () => {
  console.log('Connect button clicked');
  
  try {
    isLoading.value = true;
    errorMessage.value = '';
    
  // Clear previous error state
  robotStore.internalErrorMessage = '';
    
    const configSettings = getConfigurationSettings(configuration.value);
    console.log('Making connection request:', { operationMode: operationMode.value, configSettings });
    
    const response = await robotApi.connect(operationMode.value, configSettings);
    
    if (response?.data?.status === 'success') {
      // Update store
      robotStore.status.connected = true;
      robotStore.status = { ...robotStore.status, ...response.data.data };
      robotStore.teleoperationConfig.showCameras = configSettings.enableCameras;
      
      console.log('Connection successful');
    } else {
      const error = response?.data?.message || 'Connection failed';
      errorMessage.value = error;
  robotStore.internalErrorMessage = error;
    }
  } catch (error) {
    console.error('Connection error:', error);
    const errorMsg = error?.response?.data?.message || error?.message || 'Connection failed';
    errorMessage.value = errorMsg;
  robotStore.internalErrorMessage = errorMsg;
  } finally {
    isLoading.value = false;
  }
};

const disconnectRobot = async () => {
  try {
    isLoading.value = true;
    await robotApi.disconnect();
    
    // Update store - clear error and reset connection status
    robotStore.status.connected = false;
    robotStore.status.mode = null;
    robotStore.status.available_arms = [];
    robotStore.status.cameras = [];
  robotStore.internalErrorMessage = ''; // Clear any error state
    
    console.log('Disconnected successfully');
  } catch (error) {
    console.error('Disconnect error:', error);
  } finally {
    isLoading.value = false;
  }
};
</script>

<style scoped>
.aloha-robot-control {
  padding: 15px;
}
</style>

<style scoped>
.robot-connection {
  padding: 15px;
}
</style>
