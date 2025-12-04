<template>
  <div class="app-container">
    <!-- Left-aligned sidebar navigation -->
    <div :class="['sidebar bg-dark text-light', sidebarCollapsed ? 'sidebar-collapsed' : '']">
      <div class="d-flex align-items-center p-3 border-bottom border-secondary">
        <img src="@/assets/Ubi_CoRo.png" alt="LeRobot" class="logo me-2" v-if="!sidebarCollapsed">
        <span class="fw-bold fs-5" v-if="!sidebarCollapsed">Robot Control</span>
        <button @click="toggleSidebar" class="btn btn-sm btn-outline-light ms-auto">
          <i :class="['bi', sidebarCollapsed ? 'bi-chevron-right' : 'bi-chevron-left']"></i>
        </button>
      </div>
      
      <ul class="nav nav-pills flex-column mb-auto p-2">
        <li class="nav-item mb-1">
          <router-link to="/" class="nav-link text-light d-flex align-items-center" active-class="active">
            <i class="bi bi-house me-2"></i>
            <span v-if="!sidebarCollapsed">Dashboard</span>
          </router-link>
        </li>
        
        <li class="nav-item mb-1">
          <router-link 
            to="/teleoperation" 
            class="nav-link text-light d-flex align-items-center" 
            active-class="active"
            :class="{ disabled: !robotConnected }"
            :tabindex="!robotConnected ? -1 : 0"
            @click.prevent="!robotConnected && scrollToConnect()"
          >
            <i class="bi bi-joystick me-2"></i>
            <span v-if="!sidebarCollapsed">Teleoperation</span>
          </router-link>
        </li>
        
        <li class="nav-item mb-1" v-if="!isDemoRobotConnected">
          <router-link 
            to="/record-dataset" 
            class="nav-link text-light d-flex align-items-center" 
            active-class="active"
            :class="{ disabled: !robotConnected }"
            :tabindex="!robotConnected ? -1 : 0"
            @click.prevent="!robotConnected && scrollToConnect()"
          >
            <i class="bi bi-record-circle me-2"></i>
            <span v-if="!sidebarCollapsed">Record Dataset</span>
          </router-link>
        </li>
        
        <li class="nav-item mb-1" v-if="!isDemoRobotConnected">
          <div 
            class="nav-link text-light d-flex align-items-center" 
            :class="{ disabled: !robotConnected }"
            :style="!robotConnected ? 'opacity:0.5;cursor:not-allowed;' : ''"
            @click="robotConnected && $router.push('/replay-dataset')"
          >
            <i class="bi bi-play-circle me-2"></i>
            <span v-if="!sidebarCollapsed">Replay Dataset</span>
          </div>
        </li>
        
        <li class="nav-item mb-1" v-if="!isDemoRobotConnected">
          <div class="nav-link text-light d-flex align-items-center disabled" style="opacity:0.5;cursor:not-allowed;">
            <i class="bi bi-cpu me-2"></i>
            <span v-if="!sidebarCollapsed">Training</span>
          </div>
        </li>
        
        <li class="nav-item mb-1" v-if="!isDemoRobotConnected">
          <div class="nav-link text-light d-flex align-items-center disabled" style="opacity:0.5;cursor:not-allowed;">
            <i class="bi bi-tools me-2"></i>
            <span v-if="!sidebarCollapsed">Calibration</span>
          </div>
        </li>
        
        <li class="nav-item mb-1" v-if="!isDemoRobotConnected">
          <div class="nav-link text-light d-flex align-items-center disabled" style="opacity:0.5;cursor:not-allowed;">
            <i class="bi bi-graph-up me-2"></i>
            <span v-if="!sidebarCollapsed">Data Visualization</span>
          </div>
        </li>
        
        <!-- Presentation link - only visible when demo robot connected -->
        <li class="nav-item mb-1" v-if="isDemoRobotConnected">
          <router-link 
            to="/demo" 
            class="nav-link text-light d-flex align-items-center" 
            active-class="active"
          >
            <i class="bi bi-rocket me-2"></i>
            <span v-if="!sidebarCollapsed">Presentation</span>
          </router-link>
        </li>
        
      </ul>
      
      <div class="mt-auto border-top border-secondary p-3">
        <div class="d-flex align-items-center">
          <i :class="['bi', robotConnected ? 'bi-circle-fill text-success' : 'bi-circle-fill text-danger']"></i>
          <span class="ms-2" v-if="!sidebarCollapsed">{{ currentModeLabel }}</span>
        </div>
      </div>
    </div>

    <!-- Main content area -->
    <div :class="['main-content', sidebarCollapsed ? 'main-expanded' : '']">
      <!-- Top header with controls -->
      <header class="header bg-white border-bottom shadow-sm">
        <div class="container-fluid d-flex justify-content-between align-items-center">
          <h1 class="h3 m-0">{{ currentPageTitle }}</h1>
          <div class="d-flex">
            <button class="btn btn-outline-secondary me-2" @click="toggleDarkMode">
              <i :class="['bi', darkMode ? 'bi-sun' : 'bi-moon']"></i>
            </button>
            <div class="dropdown">
              <button class="btn btn-outline-secondary dropdown-toggle" type="button" data-bs-toggle="dropdown">
                <i class="bi bi-person-circle"></i>
                <span class="ms-1">User</span>
              </button>
              <ul class="dropdown-menu dropdown-menu-end">
                <li><a class="dropdown-item" href="#">Profile</a></li>
                <li><a class="dropdown-item" href="#">Settings</a></li>
                <li><hr class="dropdown-divider"></li>
                <li><a class="dropdown-item" href="#">Logout</a></li>
              </ul>
            </div>
          </div>
        </div>
      </header>
      
      <!-- Main page content -->
      <main class="content">
        <div class="router-view-container">
          <router-view />
        </div>
      </main>
      
      <!-- Footer -->
      <footer class="footer bg-white border-top">
        <div class="container-fluid">
          <div class="d-flex justify-content-between align-items-center">
            <span>Robot Control Interface &copy; {{ new Date().getFullYear() }}</span>
            <span class="text-muted">Version 1.0</span>
          </div>
        </div>
      </footer>
    </div>
  </div>
</template>

<script>
import { useRobotStore } from '@/stores/robotStore';
import { useRecordingStore } from '@/stores/recordingStore';
import { useDatasetStore } from '@/stores/datasetStore';

export default {
  name: 'App',
  setup() {
    const robotStore = useRobotStore();
    const recordingStore = useRecordingStore();
    const datasetStore = useDatasetStore();
    
    return {
      robotStore,
      recordingStore,
      datasetStore
    };
  },
  data() {
    return {
      sidebarCollapsed: false,
      darkMode: false,
    };
  },
  computed: {
    robotConnected() {
      return this.robotStore.status.connected;
    },
    isDemoRobotConnected(){
      const robotType = this.robotStore.selectedRobotType || '';
      return robotType.toLowerCase().includes('demo') && this.robotStore.status.connected;
    },
    currentModeLabel() {
      if (!this.robotConnected) return 'Disconnected';
      // Priority: Recording > Teleoperating > Replaying > Connected
      if (this.recordingStore?.isActive) return 'Recording';
      if (this.robotStore?.isTeleoperating) return 'Teleoperating';
      if (this.datasetStore?.isReplaying) return 'Replaying';
      return 'Connected';
    },
    calibrationNeeded(){
      const err = (this.robotStore.errorMessage || '').toLowerCase();
      return err.includes('calibr');
    },
    currentPageTitle() {
      const routeName = this.$route.name;
      const titles = {
        'home': 'Dashboard',
        'teleoperation': 'Teleoperation',
        'record-dataset': 'Record Dataset',
        'replay-dataset': 'Replay Dataset',
        'training': 'Training',
        'calibration': 'Calibration',
        'data-visualization': 'Data Visualization',
        'datasets': 'Dataset Management',
        'policies': 'Policy Management',
        'demo': 'Presentation'
      };
      return titles[routeName] || 'LeRobot Interface';
    }
  },
  methods: {
    toggleSidebar() {
      this.sidebarCollapsed = !this.sidebarCollapsed;
    },
    toggleDarkMode() {
      this.darkMode = !this.darkMode;
      document.body.classList.toggle('dark-mode', this.darkMode);
    },
    scrollToConnect(){
      const el = document.querySelector('.robot-connect-panel');
      if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  },
  mounted() {
    // Initialize Bootstrap components that require JS
    import('bootstrap');
    
    // Check for dark mode preference
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
      this.darkMode = true;
      document.body.classList.add('dark-mode');
    }

  // Ensure sockets/listeners for global status
  try { this.robotStore.initSocket(); } catch {}
  try { this.recordingStore.ensureSocketListeners(); } catch {}
  // Initialize recording store persistence after Pinia is ready
  try { this.recordingStore._initPersistence?.(); } catch {}
  }
}
</script>

<style>
@import "bootstrap/dist/css/bootstrap.min.css";
@import "bootstrap-icons/font/bootstrap-icons.css";

:root {
  --sidebar-width: 250px;
  --sidebar-collapsed-width: 70px;
  --header-height: 60px;
  --footer-height: 50px;
}

/* Reset and base styles */
* {
  box-sizing: border-box;
}

body {
  margin: 0;
  padding: 0;
  font-family: 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
  overflow: hidden;
}

body.dark-mode {
  --bs-body-bg: #121212;
  --bs-body-color: #e4e6eb;
}

/* Main layout container */
.app-container {
  display: flex;
  width: 100%; 
  height: 100vh;
  position: relative;
  overflow: hidden;
}

/* Sidebar styles */
.sidebar {
  width: var(--sidebar-width);
  height: 100vh;
  position: fixed;
  left: 0;
  top: 0;
  z-index: 1030;
  overflow-y: auto;
  transition: width 0.3s ease;
}

.sidebar-collapsed {
  width: var(--sidebar-collapsed-width);
}

.logo {
  height: 30px;
}

.sidebar .nav-link {
  border-radius: 4px;
  margin-bottom: 2px;
  padding: 10px 15px;
}

.sidebar .nav-link.active {
  background-color: rgba(255, 255, 255, 0.2);
}

.sidebar .nav-link:hover:not(.active) {
  background-color: rgba(255, 255, 255, 0.1);
}

.sidebar .nav-link.disabled {
  pointer-events: none;
  opacity: 0.5;
}

.sidebar .nav-link.calibration-needed {
  animation: pulseCal 1.5s ease-in-out infinite;
  background-color: rgba(245,158,11,0.15);
  border-left: 4px solid #f59e0b;
}

@keyframes pulseCal {
  0% { box-shadow: 0 0 0 0 rgba(245,158,11,0.6); }
  70% { box-shadow: 0 0 0 8px rgba(245,158,11,0); }
  100% { box-shadow: 0 0 0 0 rgba(245,158,11,0); }
}

/* Main content area */
.main-content {
  position: relative;
  margin-left: var(--sidebar-width);
  width: calc(100% - var(--sidebar-width));
  transition: margin-left 0.3s ease, width 0.3s ease;
  display: flex;
  flex-direction: column;
  height: 100vh;
  overflow-x: hidden;
}

.main-expanded {
  margin-left: var(--sidebar-collapsed-width);
  width: calc(100% - var(--sidebar-collapsed-width));
}

/* Header */
/* .header {
  height: var(--header-height);
  padding: 0 20px;
  display: flex;
  align-items: center;
} */

/* Main content */
.content {
  flex: 1;
  padding: 15px; 
  overflow-y: auto;
  overflow-x: hidden;
  background-color: #f8f9fa;
  width: 100%;
  box-sizing: border-box;
  /* Add this to ensure it takes full width */
  max-width: 100%;
}

.container-fluid {
  width: 100%;
  max-width: 100%;
  padding-left: 10px;
  padding-right: 10px;
  /* Add this to ensure it takes full available width */
  margin-left: 0;
  margin-right: 0;
}

/* Footer */
/* .footer {
  height: var(--footer-height);
  padding: 0 20px;
  display: flex;
  align-items: center;
} */
.header, .footer {
  width: 100%;
}

/* Dark mode styles */
body.dark-mode .content {
  background-color: #1a1a1a;
}

body.dark-mode .bg-white {
  background-color: #1e1e1e !important;
}

body.dark-mode .border-bottom,
body.dark-mode .border-top {
  border-color: #333 !important;
}

body.dark-mode .text-secondary {
  color: #adb5bd !important;
}

body.dark-mode .card {
  background-color: #2a2a2a;
  border-color: #333;
}

body.dark-mode .card-header {
  background-color: #252525;
  border-color: #333;
}

body.dark-mode .table {
  color: #e4e6eb;
}

body.dark-mode .dropdown-menu {
  background-color: #2a2a2a;
  border-color: #333;
}

body.dark-mode .dropdown-item {
  color: #e4e6eb;
}

body.dark-mode .dropdown-item:hover {
  background-color: #333;
}

body.dark-mode .dropdown-divider {
  border-color: #444;
}

/* Fix content overflow in router view */
.router-view-container {
  width: 100%;
  max-width: none;
  overflow-x: hidden;
}

/* Responsive adjustments */
@media (max-width: 992px) {
  .sidebar {
    width: var(--sidebar-collapsed-width);
  }
  
  .sidebar:not(.sidebar-collapsed) {
    width: var(--sidebar-width);
  }
  
  .main-content {
    margin-left: var(--sidebar-collapsed-width);
    width: calc(100% - var(--sidebar-collapsed-width));
  }
}

/* Media query for fullscreen - ensure no overflow at any screen size */
@media (min-width: 1920px) {
  .content {
    /* Remove this line or change to 100% */
    max-width: 100%;
    /* Remove any margin that might be centering it */
    margin: 0;
  }
  
  .container-fluid {
    max-width: 100%;
    /* Make sure container has full width */
    width: 100%;
  }
}

@media (max-width: 576px) {
  .header {
    padding: 0 10px;
  }
  
  .content {
    padding: 10px;
  }
  
  .header h1 {
    font-size: 1.25rem;
  }
}
</style>