<template>
  <div class="dataset-selector">
    <div class="modal-header">
      <h3><i class="bi bi-chart-bar me-2"></i>Select Dataset to Visualize</h3>
      <button @click="$emit('close')" class="btn-close">
        <i class="bi bi-x-lg"></i>
      </button>
    </div>

    <div class="selector-tabs">
      <button 
        :class="['tab-btn', { active: activeTab === 'repos' }]"
        @click="activeTab = 'repos'"
      >
        <i class="bi bi-cloud me-2"></i>HuggingFace Repos
      </button>
      <button 
        :class="['tab-btn', { active: activeTab === 'local' }]"
        @click="activeTab = 'local'"
      >
        <i class="bi bi-hdd me-2"></i>Local Datasets
      </button>
    </div>

    <!-- HuggingFace Repos Tab -->
    <div v-if="activeTab === 'repos'" class="tab-content">
      <div class="input-section">
        <label for="repoInput">Repository ID:</label>
        <div class="input-group">
          <input
            id="repoInput"
            v-model="repoInput"
            type="text"
            placeholder="e.g., lerobot/aloha_sim_insertion_human"
            class="form-control"
            @keyup.enter="validateRepo"
          />
          <button @click="validateRepo" class="btn btn-outline" :disabled="!repoInput.trim()">
            <i class="bi bi-search"></i>
          </button>
        </div>
        <small class="hint">Enter HuggingFace repository ID (username/dataset-name)</small>
      </div>

      <!-- Popular repos suggestions -->
      <div class="suggestions">
        <h5>Popular Datasets:</h5>
        <div class="repo-grid">
          <div 
            v-for="repo in popularRepos" 
            :key="repo.id"
            class="repo-card"
            @click="selectRepo(repo.id)"
          >
            <div class="repo-icon">{{ repo.icon }}</div>
            <div class="repo-info">
              <span class="repo-name">{{ repo.name }}</span>
              <span class="repo-id">{{ repo.id }}</span>
              <span class="repo-desc">{{ repo.description }}</span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Local Datasets Tab -->
    <div v-if="activeTab === 'local'" class="tab-content">
      <!-- Existing Local Datasets -->
      <div v-if="localDatasets.length > 0" class="dataset-section">
        <h5><i class="bi bi-hdd me-2"></i>Found Local Datasets</h5>
        <div class="dataset-list">
          <div 
            v-for="dataset in localDatasets" 
            :key="dataset.id"
            class="dataset-item"
            @click="selectLocalDataset(dataset)"
          >
            <div class="dataset-icon">📊</div>
            <div class="dataset-info">
              <span class="dataset-name">{{ dataset.name }}</span>
              <span class="dataset-meta">{{ dataset.episodes }} episodes • {{ dataset.size }}</span>
              <span class="dataset-date">{{ formatDate(dataset.created) }}</span>
              <span class="dataset-path">{{ dataset.path }}</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Manual Dataset Input -->
      <div class="dataset-section">
        <h5><i class="bi bi-gear me-2"></i>Manual Dataset Specification</h5>
        <p class="form-description">Specify dataset parameters manually to launch a web-based visualization</p>
        
        <div class="form-group">
          <label for="manualRepoId">Repository ID:</label>
          <input
            id="manualRepoId"
            v-model="htmlParams.repoId"
            type="text"
            placeholder="e.g., jannick-st/hoodie-unfolding-lemgo"
            class="form-control"
          />
          <small class="hint">Dataset repository identifier</small>
        </div>

        <div class="form-group">
          <label for="manualRootPath">Root Path:</label>
          <div class="input-group">
            <input
              id="manualRootPath"
              v-model="htmlParams.rootPath"
              type="text"
              placeholder="e.g., /home/jannick/data/jannick-st/eval_hoodie-unfolding-lemgo-lennart-dagger/"
              class="form-control"
            />
            <button @click="openFolderBrowser" class="btn btn-outline" type="button">
              <i class="bi bi-folder2-open"></i>
            </button>
          </div>
          <small class="hint">Full path to the dataset directory</small>
        </div>

        <div v-if="htmlParams.repoId || htmlParams.rootPath" class="example-section">
          <h6>Preview Command:</h6>
          <code class="example-command">
            visualize_dataset_html.py --root {{ htmlParams.rootPath || '[root-path]' }} --repo-id {{ htmlParams.repoId || '[repo-id]' }} --serve 1 --port {{ VISUALIZER_PORT }}
          </code>
        </div>
      </div>
    </div>

    <!-- Validation Result -->
    <div v-if="validationResult" class="validation-result" :class="validationResult.valid ? 'valid' : 'invalid'">
      <i :class="validationResult.valid ? 'bi bi-check-circle' : 'bi bi-exclamation-triangle'"></i>
      <span>{{ validationResult.message }}</span>
    </div>

    <!-- Action Buttons -->
    <div class="action-buttons">
      <button @click="$emit('close')" class="btn btn-secondary">
        Cancel
      </button>
      
      <!-- Launch Visualizer Button -->
      <button 
        @click="launchVisualizer" 
        :disabled="!canLaunchVisualizer"
        class="btn btn-primary"
      >
        <i class="bi bi-window me-2"></i>Launch Visualizer
      </button>
    </div>

    <!-- Folder Browser Modal -->
    <div v-if="showFolderBrowser" class="folder-browser-overlay" @click="closeFolderBrowser">
      <div class="folder-browser-modal" @click.stop>
        <div class="folder-browser-header">
          <h4><i class="bi bi-folder2-open me-2"></i>Select Folder</h4>
          <button @click="closeFolderBrowser" class="btn-close">
            <i class="bi bi-x-lg"></i>
          </button>
        </div>

        <!-- Breadcrumb Navigation -->
        <div class="breadcrumb-nav">
          <button 
            v-for="(crumb, index) in breadcrumbs" 
            :key="index"
            @click="navigateToPath(crumb.path)"
            class="breadcrumb-item"
            :class="{ active: index === breadcrumbs.length - 1 }"
          >
            {{ crumb.name }}
          </button>
        </div>

        <!-- Current Path Display -->
        <div class="current-path">
          <i class="bi bi-folder me-2"></i>
          <span>{{ currentPath }}</span>
          <button @click="selectCurrentPath" class="btn btn-sm btn-primary">
            <i class="bi bi-check2 me-1"></i>Select This Folder
          </button>
        </div>

        <!-- Folder Contents -->
        <div class="folder-contents">
          <div v-if="isLoadingFolders" class="loading-state">
            <i class="bi bi-hourglass-split"></i>
            <span>Loading folders...</span>
          </div>
          
          <div v-else-if="folderContents.length === 0" class="empty-folder">
            <i class="bi bi-folder-x"></i>
            <span>No subfolders found</span>
          </div>
          
          <div v-else class="folder-list">
            <!-- Parent directory navigation -->
            <div 
              v-if="currentPath !== '/'"
              @click="navigateUp"
              class="folder-item parent-folder"
            >
              <i class="bi bi-arrow-up-circle"></i>
              <span>.. (Parent Directory)</span>
            </div>
            
            <!-- Folder items -->
            <div 
              v-for="folder in folderContents" 
              :key="folder.name"
              @click="navigateToFolder(folder)"
              class="folder-item"
            >
              <i class="bi bi-folder"></i>
              <span>{{ folder.name }}</span>
              <small class="folder-meta">{{ folder.permissions || '' }}</small>
            </div>
          </div>
        </div>

        <!-- Browser Actions -->
        <div class="folder-browser-actions">
          <button @click="closeFolderBrowser" class="btn btn-secondary">
            Cancel
          </button>
          <button @click="selectCurrentPath" class="btn btn-primary">
            <i class="bi bi-check2 me-1"></i>Use This Path
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, computed } from 'vue'
import datasetApi from '@/services/api/datasetApi'

const emit = defineEmits(['close', 'launch'])

// Constants
const VISUALIZER_PORT = 9090
const LAUNCH_DELAY_MS = 1000
const WINDOW_OPEN_DELAY_MS = 1500

// State
const activeTab = ref('repos')
const repoInput = ref('')
const selectedRepo = ref('')
const selectedDataset = ref('')
const validationResult = ref(null)
const localDatasets = ref([])

// Folder browser state
const showFolderBrowser = ref(false)
const currentPath = ref('/home')
const folderContents = ref([])
const breadcrumbs = ref([])
const isLoadingFolders = ref(false)

// HTML visualizer parameters
const htmlParams = ref({
  repoId: '',
  rootPath: ''
})

// Computed
const isHtmlFormValid = computed(() => {
  return htmlParams.value.repoId.trim() && 
         htmlParams.value.rootPath.trim()
})

const canLaunchVisualizer = computed(() => {
  if (activeTab.value === 'repos') {
    return !!selectedRepo.value
  } else if (activeTab.value === 'local') {
    return !!selectedDataset.value || isHtmlFormValid.value
  }
  return false
})

// Popular repositories - could be made dynamic via API in the future
const popularRepos = ref([
  {
    id: 'lerobot/aloha_sim_insertion_human',
    name: 'ALOHA Insertion',
    icon: '🤖',
    description: 'Human demonstrations for insertion tasks'
  },
  {
    id: 'lerobot/pusht',
    name: 'PushT',
    icon: '📦',
    description: 'Push task demonstrations'
  },
  {
    id: 'lerobot/aloha_sim_transfer_cube_human',
    name: 'ALOHA Transfer',
    icon: '🎯',
    description: 'Cube transfer demonstrations'
  },
  {
    id: 'lerobot/droid_100',
    name: 'DROID-100',
    icon: '🦾',
    description: 'Diverse robot interaction dataset'
  }
])

// Methods
const selectRepo = (repoId) => {
  repoInput.value = repoId
  selectedRepo.value = repoId
  selectedDataset.value = ''
  validationResult.value = null
}

const selectLocalDataset = (dataset) => {
  selectedDataset.value = dataset
  selectedRepo.value = ''
  validationResult.value = {
    valid: true,
    message: `✓ Local dataset: ${dataset.episodes} episodes, ${dataset.size}`
  }
}

const validateRepo = async () => {
  if (!repoInput.value.trim()) return
  
  try {
    validationResult.value = { valid: false, message: 'Validating repository...' }
    
    const response = await datasetApi.validateRepo(repoInput.value.trim())
    
    if (response.data.valid) {
      selectedRepo.value = repoInput.value.trim()
      validationResult.value = {
        valid: true,
        message: `✓ Repository found: ${response.data.info.episodes} episodes`
      }
    } else {
      validationResult.value = {
        valid: false,
        message: `Repository not found or inaccessible`
      }
    }
  } catch (error) {
    validationResult.value = {
      valid: false,
      message: `Error: ${error.message}`
    }
  }
}

const launchHtmlVisualizer = async () => {
  try {
    // Call backend API to launch dataset visualizer with the provided parameters
    const response = await datasetApi.launchHtmlVisualizerCustom({
      repoId: htmlParams.value.repoId,
      rootPath: htmlParams.value.rootPath
    })

    // Show success message and open the visualizer
    const message = `✅ LeRobot HTML Visualizer Launched!\n\n` +
                   `Dataset: ${htmlParams.value.repoId}\n` +
                   `Path: ${htmlParams.value.rootPath}\n\n` +
                   `🌐 Opening in browser: http://localhost:${VISUALIZER_PORT}`
    
    alert(message)

    // Open the visualizer in a new window/tab
    setTimeout(() => {
      window.open(`http://localhost:${VISUALIZER_PORT}`, '_blank')
    }, LAUNCH_DELAY_MS)

    emit('launch', { 
      type: 'html', 
      target: `${htmlParams.value.repoId}`,
      isLocal: true,
      params: htmlParams.value
    })
    emit('close')
    
  } catch (error) {
    console.error('Failed to launch HTML visualizer:', error)
    alert(`Failed to launch HTML visualizer: ${error.message}`)
  }
}

const launchVisualizer = async () => {
  // Check if we should use manual input instead of selected dataset/repo
  if (activeTab.value === 'local' && !selectedDataset.value && isHtmlFormValid.value) {
    return launchHtmlVisualizer()
  }

  const target = selectedRepo.value || selectedDataset.value
  if (!target) return

  try {
    if (selectedDataset.value) {
      // Local dataset - use local visualizer API
      await datasetApi.launchLocalDatasetVisualizer(
        selectedDataset.value.path,
        'html',
        { port: VISUALIZER_PORT }
      )
    } else {
      // HuggingFace repo - launch HTML visualizer
      await datasetApi.launchHtmlVisualizer(selectedRepo.value, {
        port: VISUALIZER_PORT
      })
    }

    // Open visualizer window after launch
    setTimeout(() => {
      datasetApi.openVisualizerWindow(VISUALIZER_PORT)
    }, WINDOW_OPEN_DELAY_MS)

    emit('launch', { 
      type: 'html', 
      target: selectedDataset.value?.name || selectedRepo.value,
      isLocal: !!selectedDataset.value 
    })
    emit('close')
    
  } catch (error) {
    console.error('Failed to launch visualizer:', error)
    alert(`Failed to launch visualizer: ${error.message}`)
  }
}

const loadLocalDatasets = async () => {
  try {
    const response = await datasetApi.listLocalDatasets()
    localDatasets.value = response.data.datasets || []
  } catch (error) {
    console.error('Failed to load local datasets:', error)
    localDatasets.value = []
  }
}

const formatDate = (date) => {
  return new Date(date).toLocaleDateString()
}

// Folder browser methods
const openFolderBrowser = async () => {
  showFolderBrowser.value = true
  
  // Initialize with a sensible starting path
  let initialPath = htmlParams.value.rootPath
  
  if (!initialPath) {
    // Try common data directories first
    const commonPaths = ['/home/jannick/data', '/home/jannick', '/data', '/home']
    initialPath = commonPaths[0] // Start with most likely location
  }
  
  await navigateToPath(initialPath)
}

const closeFolderBrowser = () => {
  showFolderBrowser.value = false
}

const navigateToPath = async (path) => {
  try {
    isLoadingFolders.value = true
    currentPath.value = path
    
    // Call backend API to list directory contents
    const response = await datasetApi.browseDirectory(path)
    folderContents.value = response.data.folders || []
    
    // Update breadcrumbs
    updateBreadcrumbs(path)
    
  } catch (error) {
    console.error('Failed to browse directory:', error)
    
    // More detailed error message
    let errorMessage = 'Failed to browse directory'
    if (error.response && error.response.data && error.response.data.message) {
      errorMessage = error.response.data.message
    } else if (error.message) {
      errorMessage = error.message
    }
    
    alert(`${errorMessage}: ${path}`)
    
    // If it fails, try to go back to parent or home
    if (path !== '/home' && path !== '/') {
      const parentPath = path.split('/').slice(0, -1).join('/') || '/home'
      setTimeout(() => navigateToPath(parentPath), 100)
    }
    
  } finally {
    isLoadingFolders.value = false
  }
}

const navigateToFolder = async (folder) => {
  const newPath = currentPath.value === '/' 
    ? `/${folder.name}` 
    : `${currentPath.value}/${folder.name}`
  await navigateToPath(newPath)
}

const navigateUp = async () => {
  const parentPath = currentPath.value.split('/').slice(0, -1).join('/') || '/'
  await navigateToPath(parentPath)
}

const updateBreadcrumbs = (path) => {
  const parts = path.split('/').filter(part => part)
  breadcrumbs.value = [
    { name: 'Root', path: '/' },
    ...parts.map((part, index) => ({
      name: part,
      path: '/' + parts.slice(0, index + 1).join('/')
    }))
  ]
}

const selectCurrentPath = () => {
  htmlParams.value.rootPath = currentPath.value
  closeFolderBrowser()
}

// Lifecycle
onMounted(() => {
  loadLocalDatasets()
})
</script>

<style scoped>
.dataset-selector {
  background: white;
  border-radius: 1rem;
  padding: 1.5rem;
  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.15);
  max-width: 600px;
  max-height: 80vh;
  overflow-y: auto;
}

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1.5rem;
  padding-bottom: 1rem;
  border-bottom: 1px solid #e5e7eb;
}

.modal-header h3 {
  margin: 0;
  color: #1f2937;
}

.btn-close {
  background: none;
  border: none;
  font-size: 1.2rem;
  color: #6b7280;
  cursor: pointer;
  padding: 0.5rem;
  border-radius: 0.5rem;
}

.btn-close:hover {
  background: #f3f4f6;
  color: #374151;
}

.selector-tabs {
  display: flex;
  gap: 0.5rem;
  margin-bottom: 1.5rem;
  border-bottom: 1px solid #e5e7eb;
}

.tab-btn {
  background: none;
  border: none;
  padding: 0.75rem 1rem;
  cursor: pointer;
  border-radius: 0.5rem 0.5rem 0 0;
  color: #6b7280;
  border-bottom: 2px solid transparent;
}

.tab-btn.active {
  color: #3b82f6;
  border-bottom-color: #3b82f6;
  background: #eff6ff;
}

.tab-content {
  min-height: 300px;
}

.input-section {
  margin-bottom: 1.5rem;
}

.input-section label {
  display: block;
  margin-bottom: 0.5rem;
  font-weight: 500;
  color: #374151;
}

.input-group {
  display: flex;
  gap: 0.5rem;
}

.form-control {
  flex: 1;
  padding: 0.75rem;
  border: 1px solid #d1d5db;
  border-radius: 0.5rem;
  font-size: 0.95rem;
}

.form-control:focus {
  outline: none;
  border-color: #3b82f6;
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}

.hint {
  color: #6b7280;
  font-size: 0.85rem;
  margin-top: 0.25rem;
  display: block;
}

.suggestions h5 {
  margin-bottom: 1rem;
  color: #374151;
}

.repo-grid {
  display: grid;
  gap: 0.75rem;
}

.repo-card {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 1rem;
  border: 1px solid #e5e7eb;
  border-radius: 0.75rem;
  cursor: pointer;
  transition: all 0.2s ease;
}

.repo-card:hover {
  border-color: #3b82f6;
  background: #eff6ff;
}

.repo-icon {
  font-size: 1.5rem;
}

.repo-info {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.repo-name {
  font-weight: 600;
  color: #1f2937;
}

.repo-id {
  font-size: 0.85rem;
  color: #3b82f6;
  font-family: monospace;
}

.repo-desc {
  font-size: 0.85rem;
  color: #6b7280;
}

.dataset-list {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.dataset-item {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 1rem;
  border: 1px solid #e5e7eb;
  border-radius: 0.75rem;
  cursor: pointer;
  transition: all 0.2s ease;
}

.dataset-item:hover {
  border-color: #10b981;
  background: #ecfdf5;
}

.dataset-icon {
  font-size: 1.5rem;
}

.dataset-info {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.dataset-name {
  font-weight: 600;
  color: #1f2937;
}

.dataset-meta {
  font-size: 0.85rem;
  color: #059669;
}

.dataset-date {
  font-size: 0.8rem;
  color: #6b7280;
}

.dataset-path {
  font-size: 0.75rem;
  color: #9ca3af;
  font-family: monospace;
  margin-top: 0.25rem;
}

.manual-form {
  padding: 1rem;
}

.manual-form h5 {
  color: #1f2937;
  margin-bottom: 0.5rem;
}

.form-description {
  color: #6b7280;
  margin-bottom: 1.5rem;
  font-size: 0.9rem;
}

.form-group {
  margin-bottom: 1.25rem;
}

.form-group label {
  display: block;
  margin-bottom: 0.5rem;
  font-weight: 500;
  color: #374151;
}

.example-section {
  margin-top: 1.5rem;
  padding: 1rem;
  background: #f9fafb;
  border-radius: 0.5rem;
  border: 1px solid #e5e7eb;
}

.example-section h6 {
  margin: 0 0 0.75rem 0;
  color: #374151;
  font-size: 0.9rem;
}

.example-command {
  display: block;
  background: #1f2937;
  color: #f9fafb;
  padding: 0.75rem;
  border-radius: 0.375rem;
  font-family: 'Courier New', monospace;
  font-size: 0.8rem;
  line-height: 1.4;
  word-break: break-all;
  overflow-wrap: break-word;
}

.validation-result {
  padding: 0.75rem;
  border-radius: 0.5rem;
  margin: 1rem 0;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.validation-result.valid {
  background: #ecfdf5;
  color: #059669;
  border: 1px solid #a7f3d0;
}

.validation-result.invalid {
  background: #fef2f2;
  color: #dc2626;
  border: 1px solid #fecaca;
}

.action-buttons {
  display: flex;
  gap: 1rem;
  justify-content: flex-end;
  padding-top: 1.5rem;
  border-top: 1px solid #e5e7eb;
  margin-top: 1.5rem;
}

.btn {
  padding: 0.75rem 1.5rem;
  border-radius: 0.5rem;
  border: none;
  cursor: pointer;
  font-weight: 500;
  transition: all 0.2s ease;
}

.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn-primary {
  background: #3b82f6;
  color: white;
}

.btn-primary:hover:not(:disabled) {
  background: #2563eb;
}

.btn-secondary {
  background: #6b7280;
  color: white;
}

.btn-secondary:hover {
  background: #4b5563;
}

.btn-outline {
  background: white;
  color: #3b82f6;
  border: 1px solid #3b82f6;
}

.btn-outline:hover:not(:disabled) {
  background: #3b82f6;
  color: white;
}

.dataset-section {
  margin-bottom: 2rem;
}

.dataset-section:not(:last-child) {
  border-bottom: 1px solid #e5e7eb;
  padding-bottom: 2rem;
}

.dataset-section h5 {
  margin-bottom: 1rem;
  color: #374151;
}

/* Folder Browser Styles */
.folder-browser-overlay {
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
}

.folder-browser-modal {
  background: white;
  border-radius: 1rem;
  width: 90%;
  max-width: 600px;
  max-height: 80vh;
  overflow: hidden;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.2);
}

.folder-browser-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1.5rem;
  border-bottom: 1px solid #e5e7eb;
  background: #f9fafb;
}

.folder-browser-header h4 {
  margin: 0;
  color: #1f2937;
}

.breadcrumb-nav {
  display: flex;
  align-items: center;
  padding: 1rem 1.5rem;
  border-bottom: 1px solid #e5e7eb;
  background: #fafafa;
  overflow-x: auto;
  gap: 0.25rem;
}

.breadcrumb-item {
  background: none;
  border: none;
  color: #3b82f6;
  cursor: pointer;
  padding: 0.25rem 0.5rem;
  border-radius: 0.25rem;
  font-size: 0.9rem;
  white-space: nowrap;
}

.breadcrumb-item:hover {
  background: #eff6ff;
}

.breadcrumb-item.active {
  color: #6b7280;
  font-weight: 500;
}

.breadcrumb-item:not(:last-child)::after {
  content: '/';
  margin-left: 0.5rem;
  color: #9ca3af;
}

.current-path {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 1rem 1.5rem;
  border-bottom: 1px solid #e5e7eb;
  background: #f8fafc;
  font-family: monospace;
  font-size: 0.9rem;
}

.current-path span {
  flex: 1;
  color: #374151;
}

.folder-contents {
  height: 300px;
  overflow-y: auto;
}

.loading-state,
.empty-folder {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: #6b7280;
  gap: 0.5rem;
}

.loading-state i,
.empty-folder i {
  font-size: 2rem;
}

.folder-list {
  padding: 0.5rem;
}

.folder-item {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.75rem;
  border-radius: 0.5rem;
  cursor: pointer;
  transition: all 0.2s ease;
}

.folder-item:hover {
  background: #f3f4f6;
}

.folder-item.parent-folder {
  border-bottom: 1px solid #e5e7eb;
  margin-bottom: 0.5rem;
  color: #6b7280;
}

.folder-item i {
  color: #3b82f6;
  font-size: 1.1rem;
}

.folder-item.parent-folder i {
  color: #6b7280;
}

.folder-item span {
  flex: 1;
  font-weight: 500;
  color: #374151;
}

.folder-meta {
  color: #9ca3af;
  font-size: 0.8rem;
}

.folder-browser-actions {
  display: flex;
  gap: 1rem;
  justify-content: flex-end;
  padding: 1.5rem;
  border-top: 1px solid #e5e7eb;
  background: #f9fafb;
}

.btn-sm {
  padding: 0.5rem 1rem;
  font-size: 0.85rem;
}
</style>
