<template>
  <!-- Demo Camera Modal with Progress Footer -->
  <Teleport to="body">
    <transition name="modal-fade">
      <div v-if="isOpen" class="modal-overlay" @click.self="closeModal">
        <div class="modal-content">
          <!-- Header -->
          <div class="modal-header">
            <div class="header-title">
              <i class="bi bi-camera-video"></i>
              <span>Camera Feeds</span>
              <span class="live-badge">● LIVE</span>
            </div>
            <button class="close-btn" @click="closeModal" title="Close (ESC)">
              <i class="bi bi-x-lg"></i>
            </button>
          </div>

          <!-- Camera Viewer -->
          <div class="modal-body">
            <CameraViewer :modal-mode="true" />
          </div>

          <!-- Progress Footer (Always Visible) -->
          <div class="progress-footer">
            <div class="progress-info">
              <div class="phase-indicator">
                <span class="phase-icon">{{ phaseIcon }}</span>
                <span class="phase-text">{{ phaseLabel }}</span>
              </div>
              <div class="episode-counter">
                <span class="episode-label">Episode</span>
                <span class="episode-value">{{ currentEpisode }} / {{ totalEpisodes }}</span>
              </div>
              <div class="time-remaining" v-if="remainingTime">
                <span class="time-icon">⏱</span>
                <span class="time-value">{{ remainingTime }}</span>
                <span class="time-label">left</span>
              </div>
            </div>
            
            <div class="progress-bar-container">
              <div class="progress-bar">
                <div class="progress-fill" :style="{ width: progressPct + '%' }"></div>
              </div>
              <span class="progress-pct">{{ progressPct.toFixed(0) }}%</span>
            </div>

            <div class="footer-actions">
              <button class="stop-btn" @click="$emit('stop')" title="Stop Presentation">
                <i class="bi bi-stop-fill"></i>
                <span>Stop</span>
              </button>
            </div>
          </div>
        </div>
      </div>
    </transition>
  </Teleport>
</template>

<script setup>
import { ref, watch, computed, onMounted, onUnmounted } from 'vue'
import CameraViewer from '@/components/dataVisualization/CameraViewer.vue'

const props = defineProps({
  open: { type: Boolean, default: false },
  phase: { type: String, default: 'idle' },
  currentEpisode: { type: Number, default: 0 },
  totalEpisodes: { type: Number, default: 1 },
  phaseElapsed: { type: Number, default: 0 },
  phaseTotal: { type: Number, default: 0 }
})

const emit = defineEmits(['close', 'update:open', 'stop'])

const isOpen = ref(props.open)

// Watch for prop changes
watch(() => props.open, (newVal) => {
  isOpen.value = newVal
})

watch(() => isOpen.value, (newVal) => {
  emit('update:open', newVal)
})

const closeModal = () => {
  isOpen.value = false
  emit('close')
}

// Progress calculations
const progressPct = computed(() => {
  if (!props.phaseTotal) return 0
  return Math.min(100, (props.phaseElapsed / props.phaseTotal) * 100)
})

const remainingTime = computed(() => {
  if (!props.phaseTotal) return ''
  const remaining = Math.max(0, props.phaseTotal - props.phaseElapsed)
  const m = Math.floor(remaining / 60)
  const s = Math.floor(remaining % 60)
  return `${m}:${s.toString().padStart(2, '0')}`
})

const phaseLabel = computed(() => {
  switch (props.phase) {
    case 'warmup': return 'Warming up...'
    case 'recording': return 'Policy running'
    case 'resetting': return 'Resetting...'
    case 'processing': return 'Processing...'
    case 'pushing': return 'Uploading...'
    default: return 'Idle'
  }
})

const phaseIcon = computed(() => {
  switch (props.phase) {
    case 'warmup': return '⏳'
    case 'recording': return '🤖'
    case 'resetting': return '🔄'
    case 'processing': return '⚙️'
    case 'pushing': return '☁️'
    default: return '⏸'
  }
})

// Handle ESC key
const handleKeydown = (e) => {
  if (e.key === 'Escape' && isOpen.value) {
    e.preventDefault()
    closeModal()
  }
}

onMounted(() => {
  document.addEventListener('keydown', handleKeydown)
})

onUnmounted(() => {
  document.removeEventListener('keydown', handleKeydown)
})
</script>

<style scoped>
/* Modal Overlay */
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(0, 0, 0, 0.9);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 2000;
  padding: 1rem;
}

/* Modal Content */
.modal-content {
  background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
  border-radius: 12px;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.6), 0 0 0 1px rgba(255, 255, 255, 0.1);
  display: flex;
  flex-direction: column;
  max-width: 95vw;
  max-height: 95vh;
  width: 100%;
  height: 100%;
  border: 1px solid rgba(16, 185, 129, 0.3);
  overflow: hidden;
}

/* Modal Header */
.modal-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.75rem 1.25rem;
  background: rgba(0, 0, 0, 0.4);
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  flex-shrink: 0;
}

.header-title {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  color: #f1f5f9;
  font-weight: 600;
  font-size: 1rem;
}

.header-title i {
  color: #10b981;
  font-size: 1.2rem;
}

.live-badge {
  background: #dc2626;
  color: white;
  font-size: 0.65rem;
  font-weight: 700;
  padding: 0.2rem 0.5rem;
  border-radius: 4px;
  animation: pulse-live 1.5s ease-in-out infinite;
}

@keyframes pulse-live {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.6; }
}

/* Close Button */
.close-btn {
  width: 36px;
  height: 36px;
  border: none;
  background: rgba(255, 255, 255, 0.1);
  color: #9ca3af;
  border-radius: 6px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.2rem;
  transition: all 0.15s ease;
}

.close-btn:hover {
  background: #ef4444;
  color: white;
}

/* Modal Body (Camera Content) */
.modal-body {
  flex: 1;
  overflow: hidden;
  padding: 0.5rem;
  min-height: 0;
  display: flex;
  flex-direction: column;
}

/* Progress Footer */
.progress-footer {
  padding: 0.75rem 1.25rem;
  background: linear-gradient(180deg, rgba(16, 185, 129, 0.15) 0%, rgba(16, 185, 129, 0.25) 100%);
  border-top: 1px solid rgba(16, 185, 129, 0.3);
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  flex-shrink: 0;
}

.progress-info {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1.5rem;
  flex-wrap: wrap;
}

.phase-indicator {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.phase-icon {
  font-size: 1.25rem;
}

.phase-text {
  font-weight: 600;
  color: #a7f3d0;
  font-size: 0.95rem;
}

.episode-counter {
  display: flex;
  align-items: baseline;
  gap: 0.4rem;
}

.episode-label {
  font-size: 0.75rem;
  color: #9ca3af;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.episode-value {
  font-size: 1.1rem;
  font-weight: 700;
  color: #f1f5f9;
}

.time-remaining {
  display: flex;
  align-items: center;
  gap: 0.35rem;
  background: rgba(0, 0, 0, 0.3);
  padding: 0.4rem 0.75rem;
  border-radius: 6px;
}

.time-icon {
  font-size: 1rem;
}

.time-value {
  font-size: 1.1rem;
  font-weight: 700;
  color: #fbbf24;
  font-family: monospace;
}

.time-label {
  font-size: 0.75rem;
  color: #9ca3af;
}

/* Progress Bar */
.progress-bar-container {
  display: flex;
  align-items: center;
  gap: 0.75rem;
}

.progress-bar {
  flex: 1;
  height: 10px;
  background: rgba(0, 0, 0, 0.4);
  border-radius: 5px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #10b981, #059669);
  transition: width 0.3s ease;
  border-radius: 5px;
}

.progress-pct {
  font-size: 0.85rem;
  font-weight: 600;
  color: #a7f3d0;
  min-width: 3rem;
  text-align: right;
}

/* Footer Actions */
.footer-actions {
  display: flex;
  justify-content: flex-end;
  margin-top: 0.25rem;
}

.stop-btn {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 1rem;
  background: #dc2626;
  color: white;
  border: none;
  border-radius: 6px;
  font-weight: 600;
  font-size: 0.85rem;
  cursor: pointer;
  transition: all 0.15s ease;
}

.stop-btn:hover {
  background: #b91c1c;
  transform: scale(1.02);
}

.stop-btn i {
  font-size: 1rem;
}

/* Fade Transition */
.modal-fade-enter-active,
.modal-fade-leave-active {
  transition: opacity 0.2s ease;
}

.modal-fade-enter-active .modal-content,
.modal-fade-leave-active .modal-content {
  transition: transform 0.2s ease;
}

.modal-fade-enter-from,
.modal-fade-leave-to {
  opacity: 0;
}

.modal-fade-enter-from .modal-content,
.modal-fade-leave-to .modal-content {
  transform: scale(0.95);
}

/* Responsive */
@media (max-width: 768px) {
  .progress-info {
    flex-direction: column;
    align-items: flex-start;
    gap: 0.75rem;
  }

  .footer-actions {
    justify-content: center;
  }

  .stop-btn {
    width: 100%;
    justify-content: center;
  }
}
</style>
