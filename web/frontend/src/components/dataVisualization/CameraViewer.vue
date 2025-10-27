<template>
  <div class="camera-viewer" :class="{ expanded: isExpanded }" ref="viewerRef">
    <!-- Actions bar -->
    <div class="viewer-actions d-flex justify-content-end align-items-center mb-2">
      <button class="btn btn-sm" :class="isExpanded ? 'btn-outline-light' : 'btn-outline-secondary'" @click="toggleExpanded">
        {{ isExpanded ? 'Collapse' : 'Expand' }}
      </button>
      <button class="btn btn-sm ms-2" :class="isFullscreen ? 'btn-outline-light' : 'btn-outline-secondary'" @click="toggleFullscreen">
        {{ isFullscreen ? 'Exit Fullscreen' : 'Fullscreen' }}
      </button>
    </div>
    <div v-if="!displayedCameras.length" class="no-cameras">
      <p class="text-muted">No cameras available</p>
    </div>
    
    <div v-else class="row g-3">
      <!-- Display available cameras (up to 4) -->
      <div 
        v-for="(camera, index) in displayedCameras" 
        :key="getCameraId(camera, index)" 
        class="col-md-6 mb-3"
      >
        <div class="card h-100">
          <div class="card-header d-flex justify-content-between align-items-center">
            <span>{{ getCameraName(camera, index) }}</span>
            <span v-if="cameraStreams[getCameraId(camera, index)]" class="badge bg-success">Live</span>
            <span v-else class="badge bg-secondary">Offline</span>
          </div>
          
          <div class="card-body p-0">
            <div class="camera-feed">
              <img 
                v-if="cameraStreams[getCameraId(camera, index)]" 
                :src="cameraStreams[getCameraId(camera, index)]" 
                class="img-fluid" 
                alt="Camera feed"
                @error="onImageError(getCameraId(camera, index))"
              />
              <div v-else class="camera-placeholder">
                <i class="bi bi-camera-video-off me-2"></i>
                <span>{{ getCameraStatus(camera, index) }}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      <!-- Add placeholder slots to always have 4 camera positions -->
      <div 
        v-for="index in (4 - displayedCameras.length)" 
        :key="`placeholder-${index}`" 
        class="col-md-6 mb-3"
        v-if="displayedCameras.length < 4"
      >
        <div class="card h-100">
          <div class="card-header d-flex justify-content-between align-items-center">
            <span>Camera {{ displayedCameras.length + index }}</span>
            <span class="badge bg-secondary">Not Connected</span>
          </div>
          
          <div class="card-body p-0">
            <div class="camera-placeholder">
              <i class="bi bi-camera-video me-2"></i>
              <span>Camera not connected</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted, onUnmounted, ref } from 'vue';
import { useRobotStore } from '@/stores/robotStore';

const robotStore = useRobotStore();
const isExpanded = ref(false);
const isFullscreen = ref(false);
const viewerRef = ref(null);

// Computed properties
const cameras = computed(() => robotStore.status.cameras || []);
const cameraStreams = computed(() => robotStore.cameraStreams || {});

// Always show up to 4 cameras (limit if more than 4)
const displayedCameras = computed(() => {
  return cameras.value.slice(0, 4);
});

// Helper methods to handle different camera data formats
const getCameraId = (camera, index) => {
  // Handle different camera data structures
  if (typeof camera === 'string') {
    return camera; // Camera is just a string ID
  } else if (camera && typeof camera === 'object') {
    return camera.id || camera.name || camera.key || `camera_${index}`;
  }
  return `camera_${index}`;
};

const getCameraName = (camera, index) => {
  if (typeof camera === 'string') {
    return camera;
  } else if (camera && typeof camera === 'object') {
    return camera.name || camera.id || camera.key || `Camera ${index + 1}`;
  }
  return `Camera ${index + 1}`;
};

const getCameraStatus = (camera, index) => {
  const cameraId = getCameraId(camera, index);
  if (cameraStreams.value[cameraId]) {
    return 'Loading...';
  }
  return 'Camera feed unavailable';
};

const onImageError = (cameraId) => {
  console.warn(`Failed to load camera image for ${cameraId}`);
};

// Initialize socket connection when component mounts
onMounted(() => {
  robotStore.initSocket();

  // Track fullscreen changes to keep state in sync
  const onFsChange = () => {
    const fsEl = document.fullscreenElement || document.webkitFullscreenElement;
    isFullscreen.value = !!fsEl;
  };
  document.addEventListener('fullscreenchange', onFsChange);
  document.addEventListener('webkitfullscreenchange', onFsChange);

  // Cleanup listeners on unmount
  onUnmounted(() => {
    document.removeEventListener('fullscreenchange', onFsChange);
    document.removeEventListener('webkitfullscreenchange', onFsChange);
    // Exit fullscreen if this element owns it
    if (document.fullscreenElement === viewerRef.value) {
      document.exitFullscreen?.();
    }
  });
});

// UI actions
const toggleExpanded = () => {
  isExpanded.value = !isExpanded.value;
};

const toggleFullscreen = async () => {
  try {
    if (!isFullscreen.value) {
      const el = viewerRef.value;
      if (el?.requestFullscreen) await el.requestFullscreen();
      else if (el?.webkitRequestFullscreen) await el.webkitRequestFullscreen();
      isFullscreen.value = true;
    } else {
      await document.exitFullscreen?.();
      isFullscreen.value = false;
    }
  } catch (e) {
    // Fallback to expanded mode if Fullscreen API fails
    isExpanded.value = true;
  }
};
</script>

<style scoped>
.camera-viewer {
  width: 100%;
}

.camera-viewer .viewer-actions {
  position: sticky;
  top: 0;
  z-index: 2;
}

.camera-feed {
  position: relative;
  min-height: 220px; /* Slightly reduced height for 2x2 grid */
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: #222;
  overflow: hidden;
  border-radius: 0 0 4px 4px;
}

.camera-feed img {
  width: 100%;
  height: auto;
  object-fit: cover; /* Ensures image fills space nicely */
}

.camera-placeholder {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  height: 100%;
  min-height: 220px;
  background-color: #222;
  color: #999;
  border-radius: 0 0 4px 4px;
}

.no-cameras {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 220px;
  background-color: #f8f9fa;
  border-radius: 5px;
}

/* Expanded overlay mode */
.camera-viewer.expanded {
  position: fixed;
  inset: 0;
  background: #111;
  z-index: 1050; /* above typical app chrome */
  padding: 1rem;
  display: flex;
  flex-direction: column;
  overflow: hidden; /* prevent page scroll in expanded mode */
}

.camera-viewer.expanded .viewer-actions {
  background: rgba(20, 20, 20, 0.6);
  padding: 0.5rem;
  border-radius: 6px;
}

.camera-viewer.expanded > .row.g-3 {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  grid-template-rows: repeat(2, 1fr);
  gap: 1rem;
  flex: 1 1 auto; /* fill remaining height under actions */
  height: auto;
  overflow: hidden;
}

.camera-viewer.expanded .row {
  margin: 0 !important; /* neutralize bootstrap row negative margins */
}

.camera-viewer.expanded .col-md-6.mb-3 { /* remove bootstrap spacing in grid */
  margin-bottom: 0 !important;
  padding-left: 0 !important;
  padding-right: 0 !important; /* remove gutters for precise alignment */
}

.camera-viewer.expanded .card {
  height: 100%;
  display: flex;
  flex-direction: column;
}

.camera-viewer.expanded .card-body {
  flex: 1 1 auto;
  display: flex;
  padding: 0; /* already p-0 in template, keep consistent */
}

.camera-viewer.expanded .camera-feed,
.camera-viewer.expanded .camera-placeholder {
  min-height: 0;
  height: 100%;
}

.camera-viewer.expanded .camera-feed img {
  width: 100%;
  height: 100%;
  object-fit: contain; /* ensure full video visible without cropping */
}

/* Make cameras taller on larger screens */
@media (min-width: 1200px) {
  .camera-feed, .camera-placeholder {
    min-height: 250px;
  }
}

/* Make cameras shorter on smaller screens */
@media (max-width: 991px) {
  .camera-feed, .camera-placeholder {
    min-height: 200px;
  }
}
</style>