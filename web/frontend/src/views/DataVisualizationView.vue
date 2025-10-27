<template>
  <div class="data-visualization-view">
    <div class="page-header">
      <h1><i class="bi bi-bar-chart me-3"></i>Data Visualization</h1>
      <p class="subtitle">Explore and analyze your robot datasets with LeRobot's powerful visualization tools</p>
    </div>

    <!-- Quick Launch Section -->
    <div class="quick-launch">
      <button @click="showSelector = true" class="launch-btn primary">
        <div class="btn-icon">📊</div>
        <div class="btn-content">
          <h3>Launch Visualizer</h3>
          <p>Select dataset and start visualization</p>
        </div>
        <i class="bi bi-chevron-right"></i>
      </button>
    </div>

    <!-- Running Visualizers Status -->
    <div v-if="runningVisualizers.length > 0" class="running-visualizers">
      <h3><i class="bi bi-play-circle me-2"></i>Running Visualizers</h3>
      <div class="visualizer-grid">
        <div 
          v-for="viz in runningVisualizers" 
          :key="`${viz.type}-${viz.target}`"
          class="visualizer-card"
        >
          <div class="viz-info">
            <div class="viz-type">
              <i class="bi bi-window"></i>
              HTML Visualizer
            </div>
            <div class="viz-target">{{ viz.target }}</div>
            <div class="viz-status">
              <span class="status-dot active"></span>
              Running on port {{ viz.port }}
            </div>
          </div>
          <div class="viz-actions">
            <button @click="openVisualizer(viz)" class="btn-sm primary">
              <i class="bi bi-box-arrow-up-right me-1"></i>Open
            </button>
            <button @click="stopVisualizer(viz)" class="btn-sm secondary">
              <i class="bi bi-stop me-1"></i>Stop
            </button>
          </div>
        </div>
      </div>
    </div>

    <!-- Info Cards -->
    <div class="info-cards">
      <div class="info-card">
        <div class="card-icon">🌐</div>
        <h4>HTML Visualizer</h4>
        <p>Interactive web-based exploration with video playback, episode browser, and data plots. Perfect for detailed analysis and sharing.</p>
        <ul>
          <li>Episode-by-episode navigation</li>
          <li>Multi-camera video streams</li>
          <li>Interactive data plots</li>
          <li>Web-based sharing</li>
        </ul>
      </div>
    </div>

    <!-- Dataset Selector Modal -->
    <div v-if="showSelector" class="modal-overlay" @click.self="showSelector = false">
      <DatasetSelector 
        @close="showSelector = false"
        @launch="handleVisualizerLaunch"
      />
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import DatasetSelector from '@/components/dataVisualization/DatasetSelector.vue'
import datasetApi from '@/services/api/datasetApi'

// State
const showSelector = ref(false)
const runningVisualizers = ref([])
let statusCheckInterval = null

// Methods
const handleVisualizerLaunch = (launchInfo) => {
  console.log('Visualizer launched:', launchInfo)
  
  // Add to running visualizers list
  const visualizer = {
    type: 'html',
    target: launchInfo.target,
    port: 8080,
    startedAt: new Date()
  }
  
  runningVisualizers.value.push(visualizer)
}

const openVisualizer = (visualizer) => {
  datasetApi.openVisualizerWindow('html', visualizer.port)
}

const stopVisualizer = async (visualizer) => {
  try {
    await datasetApi.stopVisualizer('html')
    
    // Remove from running list
    const index = runningVisualizers.value.findIndex(
      v => v.target === visualizer.target
    )
    if (index > -1) {
      runningVisualizers.value.splice(index, 1)
    }
  } catch (error) {
    console.error('Failed to stop visualizer:', error)
    alert(`Failed to stop visualizer: ${error.message}`)
  }
}

const checkVisualizerStatus = async () => {
  try {
    const response = await datasetApi.getVisualizerStatus()
    
    // Update running visualizers based on actual status
    if (response.data.running) {
      runningVisualizers.value = response.data.visualizers || []
    } else {
      runningVisualizers.value = []
    }
  } catch (error) {
    console.error('Failed to check visualizer status:', error)
  }
}

// Lifecycle
onMounted(() => {
  // Check status immediately and then every 5 seconds
  checkVisualizerStatus()
  statusCheckInterval = setInterval(checkVisualizerStatus, 5000)
})

onUnmounted(() => {
  if (statusCheckInterval) {
    clearInterval(statusCheckInterval)
  }
})
</script>

<style scoped>
.data-visualization-view {
  max-width: 1200px;
  margin: 0 auto;
  padding: 2rem;
}

.page-header {
  text-align: center;
  margin-bottom: 3rem;
}

.page-header h1 {
  font-size: 2.5rem;
  font-weight: 600;
  margin-bottom: 0.5rem;
  color: #1f2937;
}

.subtitle {
  font-size: 1.2rem;
  color: #6b7280;
  margin: 0;
}

.quick-launch {
  margin-bottom: 3rem;
}

.launch-btn {
  display: flex;
  align-items: center;
  gap: 1.5rem;
  width: 100%;
  padding: 2rem;
  background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
  color: white;
  border: none;
  border-radius: 1rem;
  cursor: pointer;
  transition: all 0.3s ease;
  text-align: left;
}

.launch-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 12px 32px rgba(59, 130, 246, 0.3);
}

.btn-icon {
  font-size: 3rem;
}

.btn-content {
  flex: 1;
}

.btn-content h3 {
  margin: 0 0 0.5rem 0;
  font-size: 1.5rem;
}

.btn-content p {
  margin: 0;
  opacity: 0.9;
}

.running-visualizers {
  margin-bottom: 3rem;
}

.running-visualizers h3 {
  color: #1f2937;
  margin-bottom: 1rem;
}

.visualizer-grid {
  display: grid;
  gap: 1rem;
}

.visualizer-card {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1.5rem;
  background: white;
  border: 1px solid #e5e7eb;
  border-radius: 0.75rem;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}

.viz-info {
  flex: 1;
}

.viz-type {
  font-weight: 600;
  color: #1f2937;
  margin-bottom: 0.25rem;
}

.viz-target {
  color: #3b82f6;
  font-family: monospace;
  font-size: 0.9rem;
  margin-bottom: 0.25rem;
}

.viz-status {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.85rem;
  color: #10b981;
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #10b981;
  animation: pulse 2s infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

.viz-actions {
  display: flex;
  gap: 0.5rem;
}

.btn-sm {
  padding: 0.5rem 1rem;
  border: none;
  border-radius: 0.5rem;
  cursor: pointer;
  font-size: 0.85rem;
  transition: all 0.2s ease;
}

.btn-sm.primary {
  background: #3b82f6;
  color: white;
}

.btn-sm.primary:hover {
  background: #2563eb;
}

.btn-sm.secondary {
  background: #6b7280;
  color: white;
}

.btn-sm.secondary:hover {
  background: #4b5563;
}

.info-cards {
  display: flex;
  justify-content: center;
}

.info-card {
  padding: 2rem;
  background: white;
  border: 1px solid #e5e7eb;
  border-radius: 1rem;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05);
  max-width: 500px;
}

.card-icon {
  font-size: 3rem;
  margin-bottom: 1rem;
}

.info-card h4 {
  color: #1f2937;
  margin-bottom: 1rem;
}

.info-card p {
  color: #6b7280;
  margin-bottom: 1rem;
  line-height: 1.6;
}

.info-card ul {
  list-style: none;
  padding: 0;
  margin: 0;
}

.info-card li {
  color: #059669;
  margin-bottom: 0.5rem;
  position: relative;
  padding-left: 1.5rem;
}

.info-card li::before {
  content: '✓';
  position: absolute;
  left: 0;
  color: #10b981;
  font-weight: bold;
}

.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
  padding: 2rem;
}

  @media (max-width: 768px) {
  .data-visualization-view {
    padding: 1rem;
  }
  
  .page-header h1 {
    font-size: 2rem;
  }
  
  .info-card {
    max-width: none;
  }
  
  .launch-btn {
    padding: 1.5rem;
  }
  
  .btn-icon {
    font-size: 2rem;
  }
  
  .btn-content h3 {
    font-size: 1.2rem;
  }
}
</style>