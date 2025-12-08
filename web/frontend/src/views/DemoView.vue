<template>
  <div class="demo-view">
    <div class="demo-card">
      <div class="demo-header">
        <h1>🎤 Presentation Mode</h1>
        <p class="subtitle">Run the pre-trained policy on the robot</p>
      </div>
      
      <div class="demo-info">
        <div class="info-item">
          <span class="label">Task</span>
          <span class="value">{{ demoTask }}</span>
        </div>
        <div class="info-item">
          <span class="label">Policy</span>
          <span class="value path">{{ policyPathShort }}</span>
        </div>
        <div class="info-item">
          <span class="label">Episode Duration</span>
          <span class="value">{{ episodeTime }}s</span>
        </div>
        <div class="info-item">
          <span class="label">Reset Duration</span>
          <span class="value">{{ resetTime }}s</span>
        </div>
      </div>
      
      
      <div class="demo-options">
        <label class="option">
          <input type="checkbox" v-model="showCamerasOnStart" />
          <span>Show Camera Feeds when running</span>
        </label>
      </div>
      
      <div class="demo-status" v-if="status.active">
        <div class="status-phase">{{ phaseLabel }}</div>
        <div class="status-bar">
          <div class="status-fill" :style="{ width: progressPct + '%' }"></div>
        </div>
        <div class="status-info">
          <span>Episode {{ status.episode_index + 1 }} / {{ numEpisodes }}</span>
          <span v-if="remainingTimeText" class="remaining">⏱ {{ remainingTimeText }} left</span>
        </div>
      </div>
      
      <div class="demo-actions">
        <button 
          v-if="!status.active" 
          class="btn-start" 
          @click="startDemo" 
          :disabled="!canStart || starting"
        >
          <span v-if="starting">Starting...</span>
          <span v-else>▶ Start Presentation</span>
        </button>
        
        <!-- Move robots to user-defined start position -->
        <button
          v-if="!status.active"
          class="btn-start"
          style="margin-left: 0.75rem; background: linear-gradient(135deg, #3b82f6, #2563eb)"
          @click="moveToStartPosition"
          :disabled="!canStart || movingToStart"
        >
          <span v-if="movingToStart">Moving...</span>
          <span v-else>⟲ Robot to Start-Pos</span>
        </button>

        <button 
          v-if="status.active" 
          class="btn-stop" 
          @click="stopDemo"
        >
          ⬛ Stop Presentation
        </button>
        
        <!-- View Cameras button when running (only if camera streaming is enabled) -->
        <button 
          v-if="status.active && showCamerasOnStart" 
          class="btn-cameras"
          @click="showCameraModal = true"
        >
          📹 View Cameras
        </button>
      </div>
      
      <div class="demo-error" v-if="error">
        {{ error }}
      </div>
      
      <div class="demo-note" v-if="!status.active">
        <p>💡 No data will be saved in presentation mode. The robot will execute the trained policy autonomously.</p>
      </div>
    </div>
    
    <!-- Demo Camera Modal with Progress -->
    <DemoCameraModal
      :open="showCameraModal"
      :phase="status.phase"
      :current-episode="status.episode_index + 1"
      :total-episodes="numEpisodes"
      :phase-elapsed="status.phase_elapsed_s"
      :phase-total="status.phase_total_s"
      @update:open="showCameraModal = $event"
      @close="showCameraModal = false"
      @stop="stopDemo"
    />
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, toRaw, watch } from 'vue';
import { useRobotStore } from '@/stores/robotStore';
import DemoCameraModal from '@/components/DemoCameraModal.vue';

const robotStore = useRobotStore();

// Demo configuration - loaded from backend or fallback to defaults
const demoConfig = ref({
  task_description: 'Loading...',
  policy_path: '',
  repo_id: '',
  root: '',
  episode_time_s: 60,
  reset_time_s: 10,
  num_episodes: 50,
  fps: 30,
  interactive: true
});

const demoTask = computed(() => demoConfig.value.task_description);
const policyPath = computed(() => demoConfig.value.policy_path);
const repoId = computed(() => demoConfig.value.repo_id);
const root = computed(() => demoConfig.value.root);
const episodeTime = computed(() => demoConfig.value.episode_time_s);
const resetTime = computed(() => demoConfig.value.reset_time_s);
const numEpisodes = computed(() => demoConfig.value.num_episodes);
const fps = computed(() => demoConfig.value.fps);

const policyPathShort = computed(() => {
  const parts = policyPath.value.split('/');
  return '.../' + parts.slice(-3).join('/');
});

// State
const showCamerasOnStart = ref(true);
const showCameraModal = ref(false);
const starting = ref(false);
const movingToStart = ref(false);
const error = ref('');
const status = ref({
  active: false,
  episode_index: 0,
  phase: 'idle',
  phase_elapsed_s: 0,
  phase_total_s: 0
});

// Auto-open camera modal when presentation starts (if option enabled)
watch(() => status.value.active, (active) => {
  if (active && showCamerasOnStart.value) {
    showCameraModal.value = true;
  } else if (!active) {
    showCameraModal.value = false;
  }
});

// Computed
const canStart = computed(() => robotStore.isConnected);

const progressPct = computed(() => {
  if (!status.value.phase_total_s) return 0;
  return Math.min(100, (status.value.phase_elapsed_s / status.value.phase_total_s) * 100);
});

// Remaining time for current phase (episode/warmup/reset)
const remainingTimeText = computed(() => {
  if (!status.value.phase_total_s) return '';
  const remaining = Math.max(0, status.value.phase_total_s - status.value.phase_elapsed_s);
  const m = Math.floor(remaining / 60);
  const s = Math.floor(remaining % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
});

const phaseLabel = computed(() => {
  const ph = status.value.phase;
  if (ph === 'warmup') return '⏳ Warming up...';
  if (ph === 'recording') return '🤖 Policy running...';
  if (ph === 'resetting') return '🔄 Resetting...';
  return '⏸ Idle';
});

// Socket for status updates
let socket = null;

async function loadDemoConfig() {
  try {
    const operationMode = robotStore.teleoperationConfig?.operationMode || 'bimanual';
    const response = await fetch(`/api/configuration/demo-config/${operationMode}`);
    if (response.ok) {
      const data = await response.json();
      if (data.status === 'success' && data.data) {
        demoConfig.value = { ...demoConfig.value, ...data.data };
      }
    }
  } catch (e) {
    console.warn('Could not load demo config, using defaults:', e);
  }
}

onMounted(async () => {
  robotStore.initSocket();
  
  // Wait a moment for socket to initialize
  await new Promise(resolve => setTimeout(resolve, 500));
  
  socket = robotStore.socket;
  console.log('[DemoView] Socket reference:', socket);
  console.log('[DemoView] Socket connected:', socket?.connected);
  
  // Load demo config from backend
  await loadDemoConfig();
  
  if (socket) {
    socket.on('recording_status', (payload) => {
      console.log('[DemoView] Received recording_status:', payload);
      if (payload) {
        status.value = { ...status.value, ...payload };
      }
    });
    
    socket.on('recording_error', (payload) => {
      console.log('[DemoView] Received recording_error:', payload);
      error.value = payload?.error || 'Demo error';
      starting.value = false;
      status.value.active = false;
    });
    
    socket.on('recording_started', () => {
      console.log('[DemoView] Received recording_started');
      starting.value = false;
      status.value.active = true;
    });
  } else {
    console.error('[DemoView] Socket is null after init!');
  }
});

onUnmounted(() => {
  if (socket) {
    socket.off('recording_status');
    socket.off('recording_error');
    socket.off('recording_started');
  }
});

function startDemo() {
  if (!canStart.value || starting.value) return;
  
  // Re-get the socket in case it was initialized after onMounted
  // Use toRaw to get the actual socket object, not the Vue Proxy
  const rawSocket = toRaw(robotStore.socket);
  
  if (!rawSocket) {
    error.value = 'Socket unavailable - please refresh the page';
    console.error('Socket is null when trying to start demo');
    return;
  }
  
  if (!rawSocket.connected) {
    error.value = 'Socket not connected - please wait and try again';
    console.error('Socket exists but not connected:', rawSocket);
    return;
  }
  
  starting.value = true;
  error.value = '';
  
  // Get operation mode from robotStore
  const teleopConfig = robotStore.teleoperationConfig || {};
  const operationMode = teleopConfig.operationMode || 'bimanual';
  
  const payload = {
    mode: 'replay',  // This triggers demo_mode in backend
    repo_id: repoId.value,
    root: root.value,
    single_task: demoTask.value,
    fps: fps.value,
    warmup_time_s: 5,
    episode_time_s: episodeTime.value,
    reset_time_s: resetTime.value,
    num_episodes: numEpisodes.value,
    video: false,
    push_to_hub: false,
    policy_path: policyPath.value,
    operation_mode: operationMode,
    resume: true,
    show_cameras: showCamerasOnStart.value,  // Enable camera streaming for frontend display
  };
  
  console.log('Emitting start_recording with payload:', payload);
  console.log('Using raw socket:', rawSocket);
  console.log('Socket ID:', rawSocket.id);
  rawSocket.emit('start_recording', payload);
}

function stopDemo() {
  const rawSocket = toRaw(robotStore.socket);
  if (rawSocket) {
    rawSocket.emit('stop_recording', {});
  }
  status.value.active = false;
}

// Move robots to measured start positions (defined in backend robot.py)
async function moveToStartPosition() {
  if (!canStart.value || movingToStart.value) return;

  movingToStart.value = true;
  error.value = '';

  try {
    const response = await fetch('/api/robot/start-position', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        duration_seconds: 5.0,
        move_leaders: true
      })
    });

    const data = await response.json();
    if (!response.ok) {
      error.value = data.detail || 'Failed to move to start position';
    }
  } catch (e) {
    error.value = `Error moving to start position: ${e.message}`;
  } finally {
    movingToStart.value = false;
  }
}
</script>

<style scoped>
.demo-view {
  display: flex;
  justify-content: center;
  align-items: flex-start;
  padding: 2rem;
  min-height: 100%;
}

.demo-card {
  background: linear-gradient(135deg, #ffffff 0%, #f0fdf4 100%);
  border: 2px solid #10b981;
  border-radius: 1.5rem;
  padding: 2.5rem;
  max-width: 500px;
  width: 100%;
  box-shadow: 0 8px 32px rgba(16, 185, 129, 0.15);
}

.demo-header {
  text-align: center;
  margin-bottom: 2rem;
}

.demo-header h1 {
  font-size: 2rem;
  font-weight: 700;
  color: #065f46;
  margin: 0 0 0.5rem;
}

.demo-header .subtitle {
  font-size: 1rem;
  color: #6b7280;
  margin: 0;
}

.demo-info {
  background: white;
  border-radius: 0.75rem;
  padding: 1rem;
  margin-bottom: 1.5rem;
  border: 1px solid #d1fae5;
}

.info-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.5rem 0;
  border-bottom: 1px solid #ecfdf5;
}

.info-item:last-child {
  border-bottom: none;
}

.info-item .label {
  font-size: 0.8rem;
  font-weight: 600;
  text-transform: uppercase;
  color: #6b7280;
}

.info-item .value {
  font-size: 0.9rem;
  color: #1f2937;
  font-weight: 500;
}

.info-item .value.path {
  font-family: monospace;
  font-size: 0.75rem;
  color: #059669;
}

.demo-options {
  margin-bottom: 1.5rem;
}

.demo-options .option {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  font-size: 0.9rem;
  color: #374151;
  cursor: pointer;
}

.demo-options input[type="checkbox"] {
  width: 1.25rem;
  height: 1.25rem;
  accent-color: #10b981;
}

.demo-status {
  background: #ecfdf5;
  border-radius: 0.75rem;
  padding: 1rem;
  margin-bottom: 1.5rem;
  text-align: center;
}

.status-phase {
  font-size: 1.1rem;
  font-weight: 600;
  color: #065f46;
  margin-bottom: 0.75rem;
}

.status-bar {
  background: #d1fae5;
  border-radius: 0.5rem;
  height: 8px;
  overflow: hidden;
  margin-bottom: 0.5rem;
}

.status-fill {
  background: linear-gradient(90deg, #10b981, #059669);
  height: 100%;
  transition: width 0.3s ease;
}

.status-info {
  font-size: 0.85rem;
  color: #6b7280;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.status-info .remaining {
  font-weight: 600;
  color: #065f46;
}

.demo-actions {
  display: flex;
  justify-content: center;
  margin-bottom: 1rem;
}

.btn-start, .btn-stop {
  padding: 1rem 3rem;
  font-size: 1.2rem;
  font-weight: 700;
  border: none;
  border-radius: 0.75rem;
  cursor: pointer;
  transition: all 0.2s ease;
}

.btn-start {
  background: linear-gradient(135deg, #10b981, #059669);
  color: white;
  box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);
}

.btn-start:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 6px 16px rgba(16, 185, 129, 0.5);
}

.btn-start:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn-stop {
  background: #ef4444;
  color: white;
  box-shadow: 0 4px 12px rgba(239, 68, 68, 0.4);
}

.btn-stop:hover {
  background: #dc2626;
}

.btn-cameras {
  padding: 0.75rem 1.5rem;
  font-size: 1rem;
  font-weight: 600;
  border: none;
  border-radius: 0.75rem;
  cursor: pointer;
  transition: all 0.2s ease;
  background: linear-gradient(135deg, #3b82f6, #2563eb);
  color: white;
  box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
  margin-left: 0.75rem;
}

.btn-cameras:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 16px rgba(59, 130, 246, 0.5);
}

.demo-error {
  background: #fef2f2;
  border: 1px solid #fecaca;
  border-radius: 0.5rem;
  padding: 0.75rem;
  color: #dc2626;
  font-size: 0.85rem;
  text-align: center;
  margin-bottom: 1rem;
}

.demo-note {
  text-align: center;
}

.demo-note p {
  font-size: 0.8rem;
  color: #6b7280;
  margin: 0;
  line-height: 1.5;
}

/* Dark mode */
body.dark-mode .demo-card {
  background: linear-gradient(135deg, #1f2937 0%, #064e3b 100%);
  border-color: #059669;
}

body.dark-mode .demo-header h1 {
  color: #a7f3d0;
}

body.dark-mode .demo-info {
  background: #1f2937;
  border-color: #065f46;
}

body.dark-mode .info-item {
  border-color: #064e3b;
}

body.dark-mode .info-item .value {
  color: #f3f4f6;
}

body.dark-mode .demo-options .option {
  color: #d1d5db;
}
</style>
