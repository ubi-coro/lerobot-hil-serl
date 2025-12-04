<template>
  <div class="main-operation-selector">
    <!-- Hero Section -->
    <div class="hero-section">
      <h1 class="hero-title">Robot Control Center</h1>
      <p class="hero-subtitle">Professional robot control and data collection</p>
    </div>
    
  <!-- Global Robot Connect Panel -->
  <RobotConnectPanel class="connect-wrapper robot-connect-panel" @connect-error="handleCalibrationNeeded" @connected="clearCalibrationWarning" />

  <!-- Operation Grid -->
  <div class="operation-grid">
      <!-- Primary Operations (Most Important) -->
  <div class="operation-card primary" @click="cardClick(startTeleoperation, canOperate)" :class="{ disabled: !canOperate }">
        <div class="card-icon">🎮</div>
        <h3>Teleoperation</h3>
        <p>Real-time bimanual robot control</p>
        <div class="card-features">
          <span>• 4-arm ALOHA control</span>
          <span>• Safety presets</span>
          <span>• Real-time feedback</span>
        </div>
        <div class="card-status" v-if="!canStartTeleoperation">
          <i class="bi bi-exclamation-triangle"></i>
          Robot connection required
        </div>
      </div>
      
    <!-- Presentation moved to center column -->
  <div class="operation-card secondary demo-mode" v-if="isDemoRobotConnected" @click="startDemo" :class="{ disabled: !canOperate }">
        <div class="card-icon">🚀</div>
      <h3>Presentation</h3>
        <p>One-click policy demonstration</p>
        <div class="card-features">
          <span>• Pre-configured settings</span>
          <span>• Instant start</span>
          <span>• No setup required</span>
        </div>
        <div class="card-status demo-ready" v-if="canOperate">
          <i class="bi bi-check-circle"></i>
          Ready to run
        </div>
      </div>

  <div class="operation-card primary" v-if="!isDemoRobotConnected" @click="cardClick(startRecording, canOperate)" :class="{ disabled: !canOperate }">
        <div class="card-icon">📹</div>
        <h3>Record Dataset</h3>
        <p>Capture training demonstrations</p>
        <div class="card-features">
          <span>• Multi-camera recording</span>
          <span>• Sensor data capture</span>
          <span>• Episode management</span>
        </div>
        <div class="card-status" v-if="!canStartRecording">
          <i class="bi bi-exclamation-triangle"></i>
          System setup required
        </div>
      </div>
      
      <!-- Secondary Operations (Important) -->
  <div class="operation-card secondary" v-if="!isDemoRobotConnected" @click="cardClick(replayDataset, canOperate)" :class="{ disabled: !canOperate }">
        <div class="card-icon">📊</div>
        <h3>Replay Dataset</h3>
        <p>Analyze recorded episodes</p>
        <div class="card-features">
          <span>• Episode visualization</span>
          <span>• Data analysis</span>
          <span>• Quality review</span>
        </div>
        <div class="card-status" v-if="!hasDatasets">
          <i class="bi bi-info-circle"></i>
          {{ datasetCount }} datasets available
        </div>
      </div>
      
  <div class="operation-card secondary" v-if="!isDemoRobotConnected" @click="canStartTraining ? startTraining() : null" :class="{ inactive: !canStartTraining }">
        <div class="card-icon">🧠</div>
        <h3>Training</h3>
        <p>Train AI models on collected data</p>
        <div class="card-features">
          <span>• Policy training</span>
          <span>• Model evaluation</span>
          <span>• Progress monitoring</span>
        </div>
        <div class="card-status" v-if="!canStartTraining">
          <i class="bi bi-info-circle"></i>
          Need 5+ episodes for training
        </div>
      </div>
      
      <!-- Utility Operations (When Needed) -->
  <div class="operation-card utility" v-if="!isDemoRobotConnected" @click="openCalibration" :class="{ highlight: calibrationNeeded, inactive: !canAccessCalibration }">
        <div class="card-icon">⚙️</div>
        <h3>Calibration</h3>
        <p>System setup & remote support</p>
        <div class="card-features">
          <span>• Remote calibration</span>
          <span>• Expert support</span>
          <span>• System validation</span>
        </div>
      </div>
      
      <div class="operation-card utility" v-if="!isDemoRobotConnected" @click="openVisualization" :class="{ inactive: !canAccessVisualization }">
        <div class="card-icon">📈</div>
        <h3>Data Visualization</h3>
        <p>LeRobot dataset explorer</p>
        <div class="card-features">
          <span>• Interactive plots</span>
          <span>• Episode browser</span>
          <span>• Data insights</span>
        </div>
      </div>
    </div>
    
    <!-- System Recommendations (Smart suggestions) -->
    <div class="recommendations" v-if="recommendations.length > 0">
      <h4><i class="bi bi-lightbulb"></i> Recommended Actions</h4>
      <div class="recommendation-list">
        <div 
          v-for="rec in recommendations" 
          :key="rec.operation"
          class="recommendation-item"
          :class="rec.priority"
          @click="executeRecommendation(rec)"
        >
          <span class="rec-icon">{{ rec.icon }}</span>
          <div class="rec-content">
            <span class="rec-title">{{ rec.title }}</span>
            <span class="rec-reason">{{ rec.reason }}</span>
          </div>
          <i class="bi bi-chevron-right"></i>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'
import { useRobotStore } from '@/stores/robotStore'
import RobotConnectPanel from '@/components/RobotConnectPanel.vue'

const router = useRouter()
const robotStore = useRobotStore()

// Reactive state
const datasetCount = ref(0)
const recommendations = ref([])

// Computed properties for operation availability
const canOperate = computed(() => robotStore.status.connected);

const calibrationNeeded = ref(false);

const hasDatasets = computed(() => {
  return datasetCount.value > 0
})

const canStartTraining = computed(() => datasetCount.value >= 5);

// Permanently disabled features (under construction)
const canAccessCalibration = computed(() => false);

// These are referenced in template but were missing
const canStartTeleoperation = computed(() => robotStore.status.connected);
const canStartRecording = computed(() => robotStore.status.connected);
const canAccessVisualization = computed(() => false); // Permanently disabled

// Check if a demo robot type is connected
const isDemoRobotConnected = computed(() => {
  const robotType = robotStore.selectedRobotType || '';
  return robotType.toLowerCase().includes('demo') && robotStore.status.connected;
});

// Operation handlers
const startTeleoperation = () => { if (canOperate.value) router.push('/teleoperation') }

const startRecording = () => { if (canOperate.value) router.push('/record-dataset') }

const replayDataset = () => { if (canOperate.value) router.push('/replay-dataset') }

const startDemo = () => { if (canOperate.value) router.push('/demo') }

const startTraining = () => { if (canStartTraining.value) router.push('/training') }

const openCalibration = () => { /* disabled */ }

const openVisualization = () => { /* disabled */ }

// Smart recommendations system
const generateRecommendations = () => {
  const recs = []
  
  // Check robot connection
  // Calibration recommendation removed (feature disabled)
  
  // Check for datasets
  if (datasetCount.value === 0) {
    recs.push({
      operation: 'recording',
      priority: 'high',
      title: 'Create First Dataset',
      reason: 'No datasets found - start with data collection',
      icon: '📹'
    })
  }
  
  // Ready for teleoperation
  if (canOperate.value && robotStore.status.mode === 'idle') {
    recs.push({
      operation: 'teleoperation',
      priority: 'medium',
      title: 'Start Robot Control',
      reason: 'System ready for teleoperation',
      icon: '🎮'
    })
  }
  
  // Ready for training
  if (canStartTraining.value) {
    recs.push({
      operation: 'training',
      priority: 'medium',
      title: 'Train AI Model',
      reason: `${datasetCount.value} episodes available for training`,
      icon: '🧠'
    })
  }
  
  // Sort by priority (high first)
  recs.sort((a, b) => {
    const weights = { high: 1, medium: 2, low: 3 }
    return weights[a.priority] - weights[b.priority]
  })
  
  recommendations.value = recs.slice(0, 3) // Show max 3 recommendations
}

const executeRecommendation = (rec) => {
  switch (rec.operation) {
    case 'calibration':
      // Disabled - ignore
      break
    case 'recording':
      startRecording()
      break
    case 'teleoperation':
      startTeleoperation()
      break
    case 'training':
      startTraining()
      break
  }
}

// Data fetching
const loadSystemData = async () => {
  try {
    // Fetch dataset count
    const datasetResponse = await fetch('/api/dataset/count')
    if (datasetResponse.ok) {
      const data = await datasetResponse.json()
      datasetCount.value = data.count
    }
    
    // Update robot status
    await robotStore.updateStatus()
    
    // Generate smart recommendations
    generateRecommendations()
  } catch (error) {
    console.error('Failed to load system data:', error)
  }
}

// Lifecycle
onMounted(() => {
  // Ensure socket (camera/status events) is initialized early
  try { robotStore.initSocket(); } catch (e) { /* ignore */ }
  loadSystemData()
  const interval = setInterval(loadSystemData, 5000)
  onUnmounted(() => clearInterval(interval))
})

const cardClick = (fn, enabled) => {
  // enabled might already be a plain boolean due to template ref unwrapping
  const isEnabled = (enabled && typeof enabled === 'object' && 'value' in enabled) ? enabled.value : enabled;
  if (isEnabled) return fn();
  const el = document.querySelector('.robot-connect-panel');
  if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' });
};

function handleCalibrationNeeded(msg){
  calibrationNeeded.value = true;
}
function clearCalibrationWarning(){
  calibrationNeeded.value = false;
}
</script>

<style scoped>
.main-operation-selector {
  max-width: 1400px;
  margin: 0 auto;
  padding: 2rem;
  /* Removed local font-family override to ensure global Inter stack consistency across views */
}

/* Hero Section */
.hero-section {
  text-align: center;
  margin-bottom: 3rem;
}

.hero-title {
  font-size: 2.5rem;
  font-weight: 600;
  margin-bottom: 0.5rem;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.hero-subtitle {
  font-size: 1.2rem;
  color: #6b7280;
  margin-bottom: 1.5rem;
}

/* Operation Grid */
.operation-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1.5rem;
  margin-bottom: 2rem;
}

.operation-card {
  background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
  border: 2px solid #e5e7eb;
  border-radius: 1rem;
  padding: 1.5rem;
  cursor: pointer;
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  min-height: 270px; /* enforce consistent height for alignment */
}

.operation-card:hover:not([disabled]) {
  transform: translateY(-4px);
  box-shadow: 0 12px 32px rgba(0, 0, 0, 0.15);
}

.operation-card.primary {
  border-color: #10b981;
  background: linear-gradient(135deg, #ecfdf5 0%, #f0fdf4 100%);
}

.operation-card.primary:hover:not([disabled]) {
  border-color: #059669;
  box-shadow: 0 12px 32px rgba(16, 185, 129, 0.25);
}

.operation-card.secondary {
  border-color: #3b82f6;
  background: linear-gradient(135deg, #eff6ff 0%, #f0f9ff 100%);
}

.operation-card.secondary.demo-mode {
  border-color: #8b5cf6;
  background: linear-gradient(135deg, #f5f3ff 0%, #ede9fe 100%);
  animation: demo-pulse 2s ease-in-out infinite;
}

.operation-card.secondary.demo-mode:hover:not(.disabled) {
  border-color: #7c3aed;
  box-shadow: 0 12px 32px rgba(139, 92, 246, 0.35);
}

@keyframes demo-pulse {
  0%, 100% { box-shadow: 0 0 0 0 rgba(139, 92, 246, 0.2); }
  50% { box-shadow: 0 0 0 8px rgba(139, 92, 246, 0); }
}

.card-status.demo-ready {
  color: #059669;
  font-weight: 600;
}

.operation-card.secondary:hover:not([disabled]) {
  border-color: #2563eb;
  box-shadow: 0 12px 32px rgba(59, 130, 246, 0.25);
}

.operation-card.utility {
  border-color: #6b7280;
  background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%);
}

.operation-card.utility:hover:not([disabled]) {
  border-color: #4b5563;
  box-shadow: 0 12px 32px rgba(107, 114, 128, 0.25);
}

.operation-card[disabled] {
  opacity: 0.6;
  cursor: not-allowed;
  transform: none !important;
}

.operation-card.disabled { position:relative; cursor:not-allowed; opacity:.55; }
.operation-card.disabled::after { content:"Robot connection required"; position:absolute; inset:0; display:flex; align-items:center; justify-content:center; font-size:.75rem; font-weight:600; color:#374151; backdrop-filter:blur(2px); background:rgba(255,255,255,0.35); text-align:center; padding:.5rem; border-radius:inherit; }
/* Training card uses .inactive (no overlay text) */
.operation-card.inactive { opacity:.55; cursor:not-allowed; }
body.dark-mode .operation-card.disabled::after { background:rgba(17,24,39,0.55); color:#cbd5e1; }

/* Calibration highlight when needed */
.operation-card.utility.highlight {
  animation: pulse 1.5s ease-in-out infinite;
  border-color: #f59e0b;
  box-shadow: 0 0 0 4px rgba(245,158,11,0.25);
}
@keyframes pulse {
  0% { box-shadow: 0 0 0 0 rgba(245,158,11,0.4); }
  70% { box-shadow: 0 0 0 12px rgba(245,158,11,0); }
  100% { box-shadow: 0 0 0 0 rgba(245,158,11,0); }
}

.card-icon {
  font-size: 3rem;
  margin-bottom: 1rem;
  display: block;
}

.operation-card h3 {
  font-size: 1.5rem;
  font-weight: 600;
  margin-bottom: 0.5rem;
  color: #1f2937;
}

.operation-card p {
  color: #6b7280;
  margin-bottom: 1rem;
  line-height: 1.5;
}

.card-features {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
  font-size: 0.9rem;
  color: #9ca3af;
  flex-grow: 1; /* push status to bottom for consistent layout */
}

.card-status {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-top: 1rem;
  padding: 0.5rem;
  background: rgba(245, 158, 11, 0.1);
  border-radius: 0.5rem;
  font-size: 0.85rem;
  color: #d97706;
}

/* Recommendations */
.recommendations {
  background: linear-gradient(135deg, #fefbff 0%, #f8fafc 100%);
  border: 1px solid #e5e7eb;
  border-radius: 1rem;
  padding: 1.5rem;
}

.recommendations h4 {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 1rem;
  color: #1f2937;
  font-size: 1.1rem;
}

.recommendation-list {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.recommendation-item {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 1rem;
  background: white;
  border: 1px solid #e5e7eb;
  border-radius: 0.75rem;
  cursor: pointer;
  transition: all 0.2s ease;
}

.recommendation-item:hover {
  border-color: #d1d5db;
  transform: translateX(4px);
}

.recommendation-item.high {
  border-left: 4px solid #ef4444;
}

.recommendation-item.medium {
  border-left: 4px solid #f59e0b;
}

.rec-icon {
  font-size: 1.5rem;
}

.rec-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.rec-title {
  font-weight: 600;
  color: #1f2937;
}

.rec-reason {
  font-size: 0.9rem;
  color: #6b7280;
}

/* Dark mode support */
@media (prefers-color-scheme: dark) {
  .operation-card {
    background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
    border-color: #374151;
    color: #f9fafb;
  }
  
  .operation-card h3 {
    color: #f9fafb;
  }
  
  .hero-title {
    background: linear-gradient(135deg, #818cf8 0%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  
  .recommendations {
    background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
    border-color: #374151;
  }
}

/* Responsive design */
@media (max-width: 1200px) {
  .operation-grid {
    gap: 1rem;
  }
  
  .operation-card { padding: 1.25rem; min-height: 250px; }
  
  .card-icon {
    font-size: 2.5rem;
  }
  
  .operation-card h3 {
    font-size: 1.3rem;
  }
  
  .operation-card p {
    font-size: 0.9rem;
  }
}

@media (max-width: 900px) {
  .operation-grid {
    gap: 0.75rem;
  }
  
  .operation-card { padding: 1rem; min-height: 230px; }
  
  .card-icon {
    font-size: 2rem;
    margin-bottom: 0.75rem;
  }
  
  .operation-card h3 {
    font-size: 1.2rem;
    margin-bottom: 0.4rem;
  }
  
  .operation-card p {
    font-size: 0.85rem;
    margin-bottom: 0.75rem;
  }
  
  .card-features {
    font-size: 0.8rem;
  }
}

@media (max-width: 768px) {
  .main-operation-selector {
    padding: 1rem;
  }
  
  .hero-title {
    font-size: 2rem;
  }
  
  .operation-grid {
    grid-template-columns: repeat(2, 1fr);
    gap: 0.5rem;
  }
  
  .operation-card { padding: 0.75rem; min-height: 210px; }
  
  .card-icon {
    font-size: 1.75rem;
    margin-bottom: 0.5rem;
  }
  
  .operation-card h3 {
    font-size: 1.1rem;
    margin-bottom: 0.3rem;
  }
  
  .operation-card p {
    font-size: 0.8rem;
    margin-bottom: 0.5rem;
  }
  
  .card-features {
    font-size: 0.75rem;
    gap: 0.2rem;
  }
  
  .card-status {
    font-size: 0.75rem;
    padding: 0.4rem;
    margin-top: 0.75rem;
  }
}

@media (max-width: 480px) {
  .hero-title {
    font-size: 1.75rem;
  }
  
  .hero-subtitle {
    font-size: 1rem;
  }
  
  .operation-grid {
    gap: 0.4rem;
  }
  
  .operation-card { padding: 0.6rem; min-height: 200px; }
  
  .card-icon {
    font-size: 1.5rem;
    margin-bottom: 0.4rem;
  }
  
  .operation-card h3 {
    font-size: 1rem;
    margin-bottom: 0.25rem;
  }
  
  .operation-card p {
    font-size: 0.75rem;
    margin-bottom: 0.4rem;
  }
  
  .card-features {
    font-size: 0.7rem;
  }
  
  .card-status {
    font-size: 0.7rem;
    padding: 0.3rem;
    margin-top: 0.5rem;
  }
}

@media (max-width: 360px) {
  .operation-card { padding: 0.5rem; min-height: 190px; }
  
  .card-icon {
    font-size: 1.25rem;
    margin-bottom: 0.3rem;
  }
  
  .operation-card h3 {
    font-size: 0.9rem;
  }
  
  .operation-card p {
    font-size: 0.7rem;
  }
  
  .card-features {
    font-size: 0.65rem;
  }
}
</style>
