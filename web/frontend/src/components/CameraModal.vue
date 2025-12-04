<template>
  <!-- Modal Overlay -->
  <Teleport to="body">
    <transition name="modal-fade">
      <div v-if="isOpen" class="modal-overlay" @click.self="closeModal">
        <!-- Modal Content -->
        <div class="modal-content">
          <!-- Header -->
          <div class="modal-header">
            <div class="header-title">
              <i class="bi bi-camera-video"></i>
              <span>Camera Feeds</span>
            </div>
            <button class="close-btn" @click="closeModal" title="Close (ESC)">
              <i class="bi bi-x-lg"></i>
            </button>
          </div>

          <!-- Camera Viewer -->
          <div class="modal-body">
            <CameraViewer :modal-mode="true" />
          </div>

          <!-- Footer Info -->
          <div class="modal-footer">
            <p class="footer-hint">
              <i class="bi bi-info-circle me-2"></i>
              Press <kbd>ESC</kbd> or click the <strong>X</strong> to close this view. You can still control the robot while this modal is open.
            </p>
          </div>
        </div>
      </div>
    </transition>
  </Teleport>
</template>

<script setup>
import { ref, watch, onMounted, onUnmounted } from 'vue'
import CameraViewer from '@/components/dataVisualization/CameraViewer.vue'

const props = defineProps({
  open: {
    type: Boolean,
    default: false
  }
})

const emit = defineEmits(['close', 'update:open'])

const isOpen = ref(props.open)

// Watch for prop changes
watch(() => props.open, (newVal) => {
  isOpen.value = newVal
})

// Watch for modal state changes and emit updates
watch(() => isOpen.value, (newVal) => {
  emit('update:open', newVal)
})

const closeModal = () => {
  isOpen.value = false
  emit('close')
}

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
  background: rgba(0, 0, 0, 0.85);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 2000;
  backdrop-filter: blur(2px);
  -webkit-backdrop-filter: blur(2px);
  padding: 1rem;
}

/* Modal Content */
.modal-content {
  background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
  border-radius: 12px;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.6), 0 0 0 1px rgba(255, 255, 255, 0.1);
  display: flex;
  flex-direction: column;
  max-width: 90vw;
  max-height: 90vh;
  width: 100%;
  height: 100%;
  border: 1px solid rgba(255, 255, 255, 0.1);
  overflow: hidden;
}

/* Modal Header */
.modal-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.6rem 1rem;
  background: rgba(0, 0, 0, 0.3);
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  flex-shrink: 0;
}

.header-title {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  color: #f1f5f9;
  font-weight: 600;
  font-size: 0.95rem;
}

.header-title i {
  color: #60a5fa;
  font-size: 1.1rem;
}

/* Close Button */
.close-btn {
  width: 32px;
  height: 32px;
  border: none;
  background: rgba(255, 255, 255, 0.1);
  color: #9ca3af;
  border-radius: 6px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1rem;
  transition: all 0.15s ease;
  flex-shrink: 0;
}

.close-btn:hover {
  background: #ef4444;
  color: white;
  transform: scale(1.05);
}

.close-btn:active {
  transform: scale(0.95);
}

/* Modal Body (Camera Content) */
.modal-body {
  flex: 1 1 0;
  overflow: hidden; /* No scrolling - fit all cameras */
  padding: 0.5rem;
  min-height: 0;
  display: flex;
  flex-direction: column;
}

/* Ensure CameraViewer fills the body */
.modal-body > * {
  flex: 1;
  min-height: 0;
}

/* Modal Footer */
.modal-footer {
  padding: 0.4rem 1rem;
  background: rgba(0, 0, 0, 0.2);
  border-top: 1px solid rgba(255, 255, 255, 0.1);
  flex-shrink: 0;
}

.footer-hint {
  margin: 0;
  font-size: 0.7rem;
  color: #9ca3af;
  display: flex;
  align-items: center;
  line-height: 1.3;
}

.footer-hint kbd {
  background: rgba(255, 255, 255, 0.1);
  color: #e5e7eb;
  padding: 0.2rem 0.4rem;
  border-radius: 3px;
  font-size: 0.75rem;
  font-weight: 600;
  margin: 0 0.25rem;
  border: 1px solid rgba(255, 255, 255, 0.2);
}

.footer-hint strong {
  color: #e5e7eb;
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

/* Responsive Design */
@media (max-width: 768px) {
  .modal-overlay {
    padding: 0.5rem;
  }

  .modal-content {
    max-width: 100vw;
    max-height: 100vh;
    border-radius: 8px;
  }

  .modal-header {
    padding: 0.75rem 1rem;
  }

  .header-title {
    font-size: 1rem;
    gap: 0.5rem;
  }

  .header-title i {
    font-size: 1.1rem;
  }

  .close-btn {
    width: 32px;
    height: 32px;
    font-size: 1rem;
  }

  .modal-body {
    padding: 0.75rem;
  }

  .modal-footer {
    padding: 0.5rem 1rem;
  }

  .footer-hint {
    font-size: 0.75rem;
  }
}
</style>
