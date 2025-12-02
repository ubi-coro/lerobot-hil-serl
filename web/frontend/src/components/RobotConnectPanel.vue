<template>
  <div class="status-card" :class="connectionStatusClass" ref="panelRoot">
    <div class="status-header">
      <div class="status-icon">
        <i :class="statusIcon"></i>
      </div>
      <div class="status-info">
        <h3>{{ statusTitle }}</h3>
        <p>{{ statusMessage }}</p>
      </div>
      <div class="status-actions" v-if="!busy">
        <div class="status-controls">
          <div class="robot-type-picker" v-if="!isConnected">
            <label class="rt-label">Robot Type</label>
            <select v-model="selectedType" @change="onTypeChange">
              <option value="aloha_bimanual_demo">ALOHA Bimanual Demonstration</option>
              <option value="aloha_bimanual">ALOHA Bimanual</option>
              <option value="aloha_right">ALOHA Single Right</option>
              <option value="aloha_left">ALOHA Single Left</option>
              <option value="koch" disabled>Koch</option>
              <option value="koch_bimanual" disabled>Koch Bimanual</option>
              <option value="so101" disabled>So101</option>
              <option value="so100" disabled>So100</option>
              <option value="lekiwi" disabled>LeKiwi</option>
              <option value="stretch" disabled>Stretch</option>
            </select>
          </div>
        </div>
        <button 
          v-if="!isConnected" 
          @click="connect" 
          class="btn btn-primary" 
          :disabled="busy"
        >
          <i class="bi bi-power me-2"></i>Connect Robot
        </button>
        <button 
          v-if="isConnected" 
          @click="disconnect" 
          class="btn btn-secondary"
        >
          <i class="bi bi-power me-2"></i>Disconnect
        </button>
        <button 
          v-if="isConnected" 
          @click="goHome" 
          class="btn btn-warning"
          :disabled="busy"
          title="Move to safe home position"
        >
          <i class="bi bi-house-door me-2"></i>Home
        </button>
      </div>
    </div>
    <div v-if="error && !isConnected" class="error-details">
      <h4><i class="bi bi-exclamation-triangle me-2"></i>Connection Failed</h4>
      <p>{{ error }}</p>
      <ul v-if="errorTips.length">
        <li v-for="tip in errorTips" :key="tip">{{ tip }}</li>
      </ul>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch } from 'vue';
const emit = defineEmits(['connect-error','connected']);
import { useRobotStore } from '@/stores/robotStore';
import robotApi from '@/services/api/robotApi';

const robotStore = useRobotStore();
const busy = ref(false);
const error = ref('');
const errorTips = ref([]);

const ROBOT_LABELS = {
  aloha_bimanual: 'ALOHA Bimanual',
  aloha_left: 'ALOHA Single Left',
  aloha_right: 'ALOHA Single Right'
};

const selectedType = ref(robotStore.robotType || 'aloha_bimanual');

const isConnected = computed(()=> robotStore.status.connected);
// Mirror TeleoperationView naming for exact design parity
const connectionStatusClass = computed(() => {
  if (busy.value) return 'connecting';
  if (error.value) return 'error';
  if (isConnected.value) return 'connected';
  return 'disconnected';
});
const statusIcon = computed(() => {
  if (busy.value) return 'bi bi-hourglass-split';
  if (error.value) return 'bi bi-exclamation-triangle';
  if (isConnected.value) return 'bi bi-check-circle';
  return 'bi bi-x-circle';
});
const statusTitle = computed(() => {
  if (busy.value) return 'Connecting...';
  if (error.value) return 'Connection Failed';
  if (isConnected.value) return 'Robot Connected';
  return 'Robot Disconnected';
});
const statusMessage = computed(() => {
  if (busy.value) return 'Establishing connection to robot hardware';
  if (error.value) return 'Unable to connect to robot';
  if (isConnected.value) return 'Ready for operations';
  const label = ROBOT_LABELS[selectedType.value] || selectedType.value.toUpperCase();
  return `Click Connect Robot to begin (type: ${label})`;
});

function deriveOperationMode(type){
  if (type === 'aloha_left') return 'left_only';
  if (type === 'aloha_right') return 'right_only';
  return 'bimanual';
}

async function connect(){
  try {
    busy.value = true; error.value=''; errorTips.value = [];
    // Minimal connect: fetch configs first (if not loaded), then connect using first config
    if (!robotStore.configs || robotStore.configs.length === 0){
      await robotStore.fetchRobotConfigs();
    }
    const operationMode = deriveOperationMode(selectedType.value);
    robotStore.setRobotType(selectedType.value);
    const response = await robotApi.connect(operationMode, {
      robot_type: selectedType.value,
      show_cameras: true,
      fps: 30,
      force_reconnect: true
    });
    if (response.data.status !== 'success') {
      throw new Error(response.data.message || 'Connect failed');
    }
    await robotStore.updateStatus();
    emit('connected');
  } catch(e){
    const message = e.message || 'Connection error';
    error.value = message;
    const lower = message.toLowerCase();
    if (lower.includes('realsense')) {
      errorTips.value = [
        'Unplug and reconnect the affected Intel RealSense camera, then wait a few seconds.',
        'If the camera still fails to start, power-cycle the USB hub or workstation port.',
        'After reconnecting the hardware, press “Connect Robot” again.'
      ];
    }
    if ((message || '').toLowerCase().includes('calibr')) {
      emit('connect-error', message);
    }
  } finally { busy.value=false; }
}

async function disconnect(){
  try { await robotStore.disconnectRobot(); } catch { /* ignore */ }
}

async function goHome(){
  if(!confirm("Move robot to Home position? Ensure the workspace is clear.")) return;
  try {
    busy.value = true;
    await robotApi.goHome(true);
  } catch(e){
    error.value = e.message || 'Failed to go home';
  } finally {
    busy.value = false;
  }
}

function onTypeChange(){
  robotStore.setRobotType(selectedType.value);
}

watch(() => robotStore.robotType, (val) => { if (val) selectedType.value = val; });
</script>

<style scoped>
/* Exact teleoperation card styling duplicated for parity */
.status-card { background: white; border-radius: 1rem; padding: 2rem; margin-bottom: 2rem; border: 2px solid #e5e7eb; transition: all 0.3s ease; }
.status-card.disconnected { border-color: #ef4444; background: linear-gradient(135deg, #fef2f2 0%, #ffffff 100%); }
.status-card.connecting { border-color: #f59e0b; background: linear-gradient(135deg, #fffbeb 0%, #ffffff 100%); }
.status-card.connected { border-color: #10b981; background: linear-gradient(135deg, #ecfdf5 0%, #ffffff 100%); }
.status-card.error { border-color: #ef4444; background: linear-gradient(135deg, #fef2f2 0%, #ffffff 100%); }
.status-header { display: flex; align-items: center; gap: 1.5rem; }
.status-icon { font-size: 3rem; display: flex; align-items: center; justify-content: center; }
.status-card.disconnected .status-icon { color: #ef4444; }
.status-card.connecting .status-icon { color: #f59e0b; }
.status-card.connected .status-icon { color: #10b981; }
.status-card.error .status-icon { color: #ef4444; }
.status-info { flex: 1; }
.status-info h3 { margin: 0 0 0.5rem 0; font-size: 1.5rem; font-weight: 600; color: #1f2937; }
.status-info p { margin: 0; color: #6b7280; font-size: 1rem; line-height: 1.4; }
.status-actions { display: flex; gap: 1rem; }
.robot-type-picker { display:flex; align-items:center; gap:.5rem; margin-right: .75rem; }
.robot-type-picker .rt-label { font-size:.8rem; color:#374151; font-weight:600; }
.robot-type-picker select { padding:.5rem .6rem; border:1px solid #d1d5db; border-radius:.5rem; background:#fff; font-size:.9rem; color:#111827; }
button.btn { padding: 0.75rem 1.5rem; border: none; border-radius: 0.5rem; cursor: pointer; font-weight: 500; transition: all 0.2s ease; display: inline-flex; align-items: center; justify-content: center; gap: 0.5rem; }
button.btn-primary { background: #3b82f6; color: white; }
button.btn-primary:hover:not(:disabled) { background: #2563eb; }
button.btn-secondary { background: #6b7280; color: white; }
button.btn-secondary:hover { background: #4b5563; }
button.btn-warning { background: #f59e0b; color: white; }
button.btn-warning:hover:not(:disabled) { background: #d97706; }
button[disabled] { opacity: 0.5; cursor: not-allowed; }
.error-details { margin-top: 1.5rem; padding-top: 1.5rem; border-top: 1px solid #f3f4f6; }
.error-details h4 { color: #dc2626; margin: 0 0 0.5rem 0; }
.error-details p { color: #6b7280; margin: 0; }
</style>
