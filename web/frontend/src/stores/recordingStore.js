import { defineStore } from 'pinia';
import { useRobotStore } from './robotStore';

// Simple field validation helper
function validateConfig(cfg, mode = 'recording') {
  const errors = {};
  if (!cfg.repo_id || !cfg.repo_id.includes('/')) errors.repo_id = 'Format: user/dataset';
  if (!cfg.single_task || cfg.single_task.trim().length < 3) errors.single_task = 'Describe the task';
  if (!cfg.fps || cfg.fps <= 0) errors.fps = 'FPS > 0';
  if (!cfg.episode_time_s || cfg.episode_time_s < 1) errors.episode_time_s = '>=1s';
  if (!cfg.num_episodes || cfg.num_episodes < 1) errors.num_episodes = '>=1';
  if (!cfg.root || cfg.root.trim().length === 0) errors.root = 'Root path required';
  
  // Replay-specific validation
  if (mode === 'replay') {
    if (!cfg.policyPath || cfg.policyPath.trim().length === 0) {
      errors.policyPath = 'Policy path required for evaluation';
    } else if (!cfg.policyPath.endsWith('pretrained_model')) {
      errors.policyPath = 'Policy path should end with pretrained_model';
    }
  }
  
  // Frontend best-effort check: if root exists & not resume, warn user
  try {
    if (cfg.root && !cfg.resume) {
      const existingDatasets = JSON.parse(localStorage.getItem('lerobot.recording.created_roots') || '[]');
      if (existingDatasets.includes(cfg.root)) {
        errors.root = "Dataset already exists at this root. Enable 'Resume' or change root.";
      }
    }
  } catch (_) { /* ignore storage issues */ }
  return errors;
}

export const useRecordingStore = defineStore('recording', {
  state: () => ({
  config: {
      repo_id: '',
      single_task: '',
      fps: 30,
      warmup_time_s: 10,
      episode_time_s: 30,
      reset_time_s: 10,
      num_episodes: 1,
  // video always on for dataset recording; no toggle needed (backend assumes video)
      push_to_hub: false,
      private: false,
      resume: false,
  root: '',
  display_data: false,
  policyPath: '',
  interactive: false
    },
    mode: 'recording',
    demoConfig: null,  // Cached demo configuration from backend (used for pre-filling form)
    status: {
      active: false,
      episode_index: 0,
      total_episodes: 0,
      episode_frames: 0,
      total_frames: 0,
      episode_elapsed_s: null,
      episode_duration_s: null,
      fps_target: null,
      fps_current: null,
  state: 'idle',
  phase: 'idle',
  phase_elapsed_s: null,
  phase_total_s: null
    },
    starting: false,
    error: null,
    validationErrors: {},
    lastUpdate: null,
    initializedSocket: false
  }),
  getters: {
    isActive: (s) => s.status.active,
    isIdle: (s) => !s.status.active,
    canStart: (s) => Object.keys(s.validationErrors).length === 0 && !s.status.active && !s.starting,
    hasDemoConfig: (s) => !!s.demoConfig,
    progressPct: (s) => {
      if (!s.status.total_episodes || s.status.total_episodes === 0) return 0;
      return Math.min(100, Math.round((s.status.episode_index / s.status.total_episodes) * 100));
    },
    episodeProgressPct: (s) => {
      if (!s.status.episode_duration_s || !s.status.episode_elapsed_s) return 0;
      return Math.min(100, Math.round((s.status.episode_elapsed_s / s.status.episode_duration_s) * 100));
    },
    phaseCountdownPct: (s) => {
      const total = s.status.phase_total_s;
      const elapsed = s.status.phase_elapsed_s;
      if (!total || total <= 0 || elapsed == null) return 0;
      const remaining = Math.max(0, total - elapsed);
      return Math.max(0, Math.min(100, Math.round((remaining / total) * 100)));
    },
    // Mixed-mode bar percent: countdown for warmup/reset, count up for recording
    phaseBarPct: (s) => {
      const total = s.status.phase_total_s;
      const elapsed = s.status.phase_elapsed_s;
      if (!total || total <= 0 || elapsed == null) return 0;
      const ph = s.status.phase || s.status.state;
      if (ph === 'recording') {
        return Math.max(0, Math.min(100, Math.round((elapsed / total) * 100)));
      }
      const remaining = Math.max(0, total - elapsed);
      return Math.max(0, Math.min(100, Math.round((remaining / total) * 100)));
    },
    // Friendly time label per phase: recording shows elapsed/total, others show remaining
    phaseTimeText: (s) => {
      const ph = s.status.phase || s.status.state;
      const total = s.status.phase_total_s || s.status.episode_duration_s;
      const elapsed = s.status.phase_elapsed_s || 0;
      if (!total || total <= 0) {
        // Indeterminate phases like processing/pushing: no time shown
        return '';
      }
      if (ph === 'recording') {
        const e = Math.max(0, elapsed).toFixed(1);
        return `${e}s / ${total}`;
      }
      const remaining = Math.max(0, total - elapsed).toFixed(1);
      return `${remaining}s remaining`;
    },
    phaseLabel: (s) => {
      const ph = s.status.phase || s.status.state;
      if (ph === 'warmup') return 'Warmup';
      if (ph === 'recording') return 'Recording';
      if (ph === 'resetting') return 'Resetting';
      if (ph === 'processing') return 'Processing';
      if (ph === 'pushing') return 'Pushing';
      if (ph === 'transition') return 'Preparing';
      return 'Idle';
    }
  },
  actions: {
    _initPersistence() {
      const savedRoot = localStorage.getItem('lerobot.recording.root');
      const savedRepo = localStorage.getItem('lerobot.recording.repo_id');
      const savedTask = localStorage.getItem('lerobot.recording.single_task');
      const savedPolicyPath = localStorage.getItem('lerobot.recording.policy_path');
  const savedInteractive = localStorage.getItem('lerobot.recording.interactive');
      if (savedRoot && !this.config.root) {
        this.config.root = savedRoot;
        this.validationErrors = validateConfig(this.config, this.mode);
      }
      if (savedRepo && !this.config.repo_id) {
        this.config.repo_id = savedRepo;
        this.validationErrors = validateConfig(this.config, this.mode);
      }
      if (savedTask && !this.config.single_task) {
        this.config.single_task = savedTask;
        this.validationErrors = validateConfig(this.config, this.mode);
      }
      if (savedPolicyPath && !this.config.policyPath) {
        this.config.policyPath = savedPolicyPath;
        this.validationErrors = validateConfig(this.config, this.mode);
      }
      if (savedInteractive !== null) {
        this.config.interactive = savedInteractive === 'true';
      }
    },
    ensureSocketListeners() {
      if (this.initializedSocket) return;
      const robotStore = useRobotStore();
      robotStore.initSocket();
      const sock = robotStore.socket;
      if (!sock) return;
      sock.on('recording_status', (payload) => {
        if (!payload) return;
        this.status = { ...this.status, ...payload };
        this.status.active = !!(payload.active);
        this.lastUpdate = Date.now();
      });
      sock.on('recording_error', (payload) => {
        this.error = payload?.error || 'Unknown recording error';
        this.starting = false;
        // Reset status on error to allow retry
        this.status.active = false;
        this.status.phase = 'idle';
      });
      sock.on('recording_started', () => {
        this.starting = false;
        this.status.active = true;
      });
      this.initializedSocket = true;
    },
    updateConfig(partial) {
      this.config = { ...this.config, ...partial };
      // If push_to_hub was turned off, also clear private flag to avoid stale state
      if (!this.config.push_to_hub && this.config.private) {
        this.config.private = false;
      }
      this.validationErrors = validateConfig(this.config, this.mode);
      if (typeof partial.root !== 'undefined') {
        try { localStorage.setItem('lerobot.recording.root', this.config.root || ''); } catch (_) { /* ignore */ }
      }
      if (typeof partial.repo_id !== 'undefined') {
        try { localStorage.setItem('lerobot.recording.repo_id', this.config.repo_id || ''); } catch (_) { /* ignore */ }
      }
      if (typeof partial.single_task !== 'undefined') {
        try { localStorage.setItem('lerobot.recording.single_task', this.config.single_task || ''); } catch (_) { /* ignore */ }
      }
      if (typeof partial.policyPath !== 'undefined') {
        try { localStorage.setItem('lerobot.recording.policy_path', this.config.policyPath || ''); } catch (_) { /* ignore */ }
      }
      if (typeof partial.interactive !== 'undefined') {
        try { localStorage.setItem('lerobot.recording.interactive', this.config.interactive ? 'true' : 'false'); } catch (_) { /* ignore */ }
      }
    },
    setMode(newMode) {
      if (newMode !== 'recording' && newMode !== 'replay') {
        console.error('Invalid mode:', newMode);
        return;
      }
      this.mode = newMode;
      this.validationErrors = validateConfig(this.config, this.mode);
    },
    
    /**
     * Fetch and apply demo configuration for the given operation mode.
     * This pre-fills the form with all necessary settings for demo/evaluation.
     * @param {string} operationMode - 'bimanual', 'left', or 'right'
     */
    async fetchDemoConfig(operationMode = 'bimanual') {
      try {
        const response = await fetch(`/api/configuration/demo-config/${operationMode}`);
        const result = await response.json();
        
        if (result.status === 'success' && result.data) {
          this.demoConfig = result.data;
          
          // Pre-fill the form with demo config values
          this.config.policyPath = result.data.policy_path || '';
          this.config.single_task = result.data.task_description || '';
          this.config.fps = result.data.fps || 30;
          this.config.episode_time_s = result.data.episode_time_s || 60;
          this.config.reset_time_s = result.data.reset_time_s || 10;
          this.config.num_episodes = result.data.num_episodes || 50;
          this.config.interactive = result.data.interactive !== false;
          
          // Set sensible defaults for dataset paths
          this.config.repo_id = 'local/eval_demo';
          this.config.root = '/tmp/demo_session';
          this.config.push_to_hub = false;
          this.config.resume = false;
          
          // Set mode to replay
          this.mode = 'replay';
          
          console.log('[recordingStore] Demo config applied:', result.data);
          this.validationErrors = validateConfig(this.config, this.mode);
          return true;
        } else {
          console.warn('[recordingStore] No demo config available:', result.message);
          this.demoConfig = null;
          return false;
        }
      } catch (error) {
        console.error('[recordingStore] Failed to fetch demo config:', error);
        this.demoConfig = null;
        return false;
      }
    },
    
    validateAll() {
      this.validationErrors = validateConfig(this.config, this.mode);
      return Object.keys(this.validationErrors).length === 0;
    },
    start() {
      const robotStore = useRobotStore();
      this.ensureSocketListeners();
      // Require a connected robot; connection happens from the overview panel.
      if (!robotStore.isConnected) {
        this.error = 'Robot not connected';
        return;
      }
      if (!this.validateAll()) return;
      const sock = robotStore.socket;
      if (!sock) {
        this.error = 'Socket unavailable';
        return;
      }
      this.starting = true;
      this.error = null;
      const payload = { ...this.config, mode: this.mode };
      // enforce video true implicitly
      payload.video = true;
      const teleopConfig = robotStore.teleoperationConfig || {};
      const operationMode = teleopConfig.operationMode
        || robotStore.status?.teleoperation?.configuration?.operation_mode
        || robotStore.status?.teleoperation?.configuration?.operationMode
        || 'bimanual';
      payload.operation_mode = operationMode;
      payload.interactive = !!this.config.interactive;
      sock.emit('start_recording', payload);
      // Mark this root as used so subsequent attempts without resume will show a validation error early.
      try {
        if (this.config.root) {
          const arr = JSON.parse(localStorage.getItem('lerobot.recording.created_roots') || '[]');
          if (!arr.includes(this.config.root)) {
            arr.push(this.config.root);
            localStorage.setItem('lerobot.recording.created_roots', JSON.stringify(arr));
          }
        }
      } catch (_) { /* ignore */ }
    },
    stop() {
      const robotStore = useRobotStore();
      const sock = robotStore.socket;
      if (sock) sock.emit('stop_recording', {});
    },
    rerecordEpisode() {
      const sock = useRobotStore().socket; if (sock) sock.emit('recording_command', { action: 'rerecord_episode' });
    },
    skipEpisode() {
      const sock = useRobotStore().socket; if (sock) sock.emit('recording_command', { action: 'skip_episode' });
    },
    finishEarlyEpisode() {
      const sock = useRobotStore().socket; if (sock) sock.emit('recording_command', { action: 'exit_early' });
    },
    emergencyStop() {
      const sock = useRobotStore().socket; if (sock) sock.emit('recording_command', { action: 'stop' });
    },
    resetForm() {
      this.config = { ...this.config, repo_id: '', single_task: '' };
      this.validationErrors = {};
    },
    setError(errorMsg) {
      this.error = errorMsg;
    }
  }
});

// Note: Do not call useRecordingStore() at module load time; Pinia may not be installed yet.
// Initialization is triggered explicitly from the app after Pinia is ready.
