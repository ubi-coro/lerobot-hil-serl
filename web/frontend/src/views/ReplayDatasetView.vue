<template>
  <div class="replay-dataset-view">
    <h1>Replay Dataset (Policy Evaluation)</h1>
    <h6>
      Run a trained policy on the robot and record evaluation episodes. Fill robot config, policy path, and dataset settings.
      The dataset name will automatically get an <code>eval_</code> prefix by convention.
      Status shows: Warmup → Recording (Policy) → Resetting → Processing → Pushing (optional).
    </h6>
    <div class="layout">
      <!-- LEFT: Configuration -->
      <section class="config" :class="{ disabled: isActive }">
        <h2>Configuration</h2>
        <form @submit.prevent>
          <!-- Policy Path (Required for Replay) -->
          <div class="field">
            <label>Policy Path (Required) <span class="err" v-if="errors.policyPath">*</span></label>
            <div class="root-with-browse">
              <div class="root-row">
                <input 
                  class="root-input" 
                  v-model="cfg.policyPath" 
                  @input="onChange" 
                  placeholder="/path/to/model/checkpoints/last/pretrained_model"
                  :disabled="isActive"
                />
                <button 
                  type="button" 
                  class="browse-btn" 
                  @click="openBrowsePolicy" 
                  :disabled="isActive"
                  title="Browse for policy checkpoint"
                >
                  <svg width="18" height="18" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z"/>
                  </svg>
                </button>
              </div>
            </div>
            <div class="err" v-if="errors.policyPath">{{ errors.policyPath }}</div>
            <div class="hint">Path to pretrained_model directory containing policy checkpoint</div>
          </div>

          <!-- Dataset Repo ID -->
          <div class="field">
            <label>Dataset Repo ID <span class="err" v-if="errors.repoId">*</span></label>
            <input v-model="cfg.repoId" @input="onChange" placeholder="local/eval_my_task" :disabled="isActive"/>
            <div class="err" v-if="errors.repoId">{{ errors.repoId }}</div>
            <div class="hint">Convention: prefix with "eval_" for evaluation datasets</div>
          </div>

          <!-- Dataset Root Path -->
          <div class="field">
            <label>Dataset Root Path <span class="err" v-if="errors.root">*</span></label>
            <div class="root-with-browse">
              <div class="root-row">
                <input 
                  class="root-input" 
                  v-model="cfg.root" 
                  @input="onChange" 
                  placeholder="/media/user/DATA/eval_datasets/my_task"
                  :disabled="isActive"
                />
                <button 
                  type="button" 
                  class="browse-btn" 
                  @click="openBrowseRoot" 
                  :disabled="isActive"
                  title="Browse for root directory"
                >
                  <svg width="18" height="18" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z"/>
                  </svg>
                </button>
              </div>
            </div>
            <div class="err" v-if="errors.root">{{ errors.root }}</div>
          </div>

          <!-- Task Description -->
          <div class="field">
            <label>Task Description <span class="err" v-if="errors.singleTask">*</span></label>
            <textarea v-model="cfg.singleTask" @input="onChange" rows="2" placeholder="Describe the evaluation task" :disabled="isActive"></textarea>
            <div class="err" v-if="errors.singleTask">{{ errors.singleTask }}</div>
          </div>

          <!-- Timings Grid -->
          <div class="grid">
            <div class="field">
              <label>FPS</label>
              <input type="number" v-model.number="cfg.fps" @input="onChange" min="1" max="60" :disabled="isActive"/>
            </div>
            <div class="field">
              <label>Episode Time (s)</label>
              <input type="number" v-model.number="cfg.episodeTimeS" @input="onChange" min="1" :disabled="isActive"/>
            </div>
            <div class="field">
              <label>Reset Time (s)</label>
              <input type="number" v-model.number="cfg.resetTimeS" @input="onChange" min="1" :disabled="isActive"/>
            </div>
            <div class="field">
              <label>Num Episodes</label>
              <input type="number" v-model.number="cfg.numEpisodes" @input="onChange" min="1" :disabled="isActive"/>
            </div>
          </div>

          <!-- Toggles -->
          <div class="toggles">
            <label>
              <input type="checkbox" v-model="cfg.video" @change="onChange" :disabled="isActive"/>
              Encode Video
            </label>
            <label>
              <input type="checkbox" v-model="cfg.pushToHub" @change="onChange" :disabled="isActive"/>
              Push to Hub
            </label>
            <label>
              <input type="checkbox" v-model="cfg.displayData" @change="onChange" :disabled="isActive"/>
              Display Data
            </label>
            <label>
              <input type="checkbox" v-model="cfg.interactive" @change="onChange" :disabled="isActive"/>
              Allow Interventions
            </label>
          </div>

          <!-- Actions -->
          <div class="actions">
            <button type="button" @click="start" :disabled="!canStart || isActive">
              <svg width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
                <path d="M5 3.5v9l7-4.5z"/>
              </svg>
              Start Evaluation
            </button>
            <button type="button" @click="resetForm" :disabled="isActive">Reset Form</button>
          </div>
          <div class="error" v-if="error">{{ error }}</div>
        </form>
      </section>

      <!-- RIGHT: Runtime Status -->
      <section class="runtime">
        <h2>Evaluation Status</h2>

        <!-- Overall Progress -->
        <div class="overall">
          <label>
            Overall Progress
            <span class="existing-total" v-if="status.existingEpisodes > 0">
              ({{ status.existingEpisodes }} existing episodes found)
            </span>
          </label>
          <div class="bar">
            <div class="fill" :style="{ width: progressPct + '%' }"></div>
          </div>
          <div class="metrics-row">
            <span>Episode {{ status.currentEpisode }} / {{ cfg.numEpisodes }}</span>
            <span>{{ progressPct.toFixed(1) }}%</span>
          </div>
        </div>

        <!-- Current Episode Phase -->
        <div class="episode">
          <label>Current Phase: <strong>{{ phaseLabel }}</strong></label>
          <div 
            class="bar small" 
            :class="{ indeterminate: status.phase === 'processing' || status.phase === 'pushing' }"
          >
            <div class="fill" :style="{ width: phaseBarPct + '%' }"></div>
          </div>
          <div class="metrics-row">
            <span>{{ phaseTimeText }}</span>
            <span v-if="status.phase !== 'processing' && status.phase !== 'pushing'">
              {{ phaseBarPct.toFixed(1) }}%
            </span>
          </div>
        </div>

        <!-- Metrics -->
        <div class="metrics">
          <div class="metric">
            <div class="lbl">Recorded Frames</div>
            <div class="val">{{ status.numFrames }}</div>
          </div>
          <div class="metric">
            <div class="lbl">Completed Episodes</div>
            <div class="val">{{ status.completedEpisodes }}</div>
          </div>
          <div class="metric">
            <div class="lbl">FPS Target</div>
            <div class="val">{{ cfg.fps }}</div>
          </div>
        </div>

        <!-- Runtime Actions -->
        <div class="runtime-actions">
          <button @click="stop" :disabled="!isActive" class="danger">
            <svg width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
              <path d="M5 3.5h6A1.5 1.5 0 0 1 12.5 5v6a1.5 1.5 0 0 1-1.5 1.5H5A1.5 1.5 0 0 1 3.5 11V5A1.5 1.5 0 0 1 5 3.5"/>
            </svg>
            Stop Evaluation
          </button>
          <button @click="skipEpisode" :disabled="!isActive || status.phase === 'processing' || status.phase === 'pushing'">
            Skip Episode
          </button>
          <button @click="rerecordEpisode" :disabled="!isActive || status.phase === 'processing' || status.phase === 'pushing'">
            Re-record Episode
          </button>
          <button @click="emergencyStop" class="danger" :disabled="!isActive">
            Emergency Stop
          </button>
        </div>

        <div class="state-line" v-if="status.phase">
          Phase: {{ status.phase }} | Frame: {{ status.currentFrame }} | Elapsed: {{ status.elapsedTime }}s
        </div>
      </section>
    </div>
  </div>

  <!-- Directory Pickers -->
  <DirectoryPickerModal v-model="browseRootOpen" :start-path="initialBrowsePathRoot" @select="setRoot" />
  <DirectoryPickerModal v-model="browsePolicyOpen" :start-path="initialBrowsePathPolicy" @select="setPolicyPath" />
</template>

<script setup>
import { ref, computed } from 'vue';
import { storeToRefs } from 'pinia';
import { useRecordingStore } from '@/stores/recordingStore';
import { useRobotStore } from '@/stores/robotStore';
import DirectoryPickerModal from '@/components/DirectoryPickerModal.vue';

const recStore = useRecordingStore();
const robotStore = useRobotStore();
recStore.ensureSocketListeners();

const { config: cfg, status, validationErrors: errors, error } = storeToRefs(recStore);
const { isActive, canStart, progressPct, phaseBarPct, phaseLabel, phaseTimeText } = storeToRefs(recStore);

function onChange() { 
  recStore.updateConfig({ ...cfg.value }); 
}

function start() { 
  // For replay, we need to ensure policyPath is set
  if (!cfg.value.policyPath || cfg.value.policyPath.trim().length === 0) {
    recStore.setError('Policy path is required for evaluation');
    return;
  }
  recStore.start(); 
}

function stop() { recStore.stop(); }
function rerecordEpisode() { recStore.rerecordEpisode(); }
function skipEpisode() { recStore.skipEpisode(); }
function emergencyStop() { recStore.emergencyStop(); }
function resetForm() { 
  recStore.resetForm();
  // Set eval_ prefix by default
  if (!cfg.value.repoId.startsWith('local/eval_')) {
    recStore.updateConfig({ repoId: 'local/eval_' });
  }
}

// Directory picker for root
const browseRootOpen = ref(false);
const initialBrowsePathRoot = computed(() => 
  cfg.value.root && cfg.value.root.trim().length > 0 ? cfg.value.root : '/'
);
function openBrowseRoot() { browseRootOpen.value = true; }
function setRoot(path) { recStore.updateConfig({ root: path }); }

// Directory picker for policy
const browsePolicyOpen = ref(false);
const initialBrowsePathPolicy = computed(() => 
  cfg.value.policyPath && cfg.value.policyPath.trim().length > 0 ? cfg.value.policyPath : '/'
);
function openBrowsePolicy() { browsePolicyOpen.value = true; }
function setPolicyPath(path) { recStore.updateConfig({ policyPath: path }); }

// Ensure robot socket ready
if (!robotStore.socket) robotStore.initSocket();

// Initialize with eval_ prefix
if (!cfg.value.repoId || !cfg.value.repoId.startsWith('eval_')) {
  recStore.updateConfig({ repoId: 'local/eval_' });
}

// Set replay mode for policy evaluation
recStore.setMode('replay');
</script>

<style scoped>
.replay-dataset-view { padding: 1.5rem; display: flex; flex-direction: column; gap: 1.25rem; max-width:1400px; margin:0 auto; }
.layout { display: grid; grid-template-columns: minmax(340px,420px) 1fr; gap: 2rem; align-items: start; }
section { background: linear-gradient(135deg,#ffffff 0%,#f8fafc 100%); border: 1px solid #e5e7eb; border-radius: 14px; padding: 1.1rem 1.35rem 1.25rem; box-shadow: 0 4px 12px rgba(0,0,0,0.04); position:relative; overflow:hidden; }
section.runtime { display:flex; flex-direction:column; }
section.config { display:flex; flex-direction:column; }
h1 { margin: 0 0 .75rem; font-size: 1.8rem; font-weight:600; letter-spacing:-0.5px; }
h2 { margin: 0 0 .85rem; font-size: 1.05rem; font-weight: 600; text-transform:uppercase; letter-spacing:.06em; opacity:.75; }
h6 { margin: 0 0 1.25rem; font-size: .75rem; line-height: 1.6; opacity: .7; }
h6 code { background: #f1f5f9; padding: .15rem .35rem; border-radius: 3px; font-size: .7rem; }
form { display: flex; flex-direction: column; gap: .9rem; }
.field { display: flex; flex-direction: column; gap: .35rem; }
.field label { font-size:.7rem; font-weight:600; text-transform:uppercase; letter-spacing:.05em; color:#4b5563; }
.field input, .field textarea { background:#ffffff; border:1px solid #d1d5db; border-radius:6px; padding:.55rem .7rem; color:#111827; font-size:.85rem; line-height:1.25; box-shadow: inset 0 0 0 1px rgba(0,0,0,0.02); }
.field input:focus, .field textarea:focus { outline:none; border-color:#3b82f6; box-shadow:0 0 0 2px rgba(59,130,246,0.25); }
.field input:disabled, .field textarea:disabled { background:#f3f4f6; color:#6b7280; }
.grid { display:grid; grid-template-columns: repeat(auto-fill,minmax(110px,1fr)); gap:.85rem; }
.toggles { display:flex; flex-wrap:wrap; gap:.5rem; font-size:.7rem; }
.toggles label { display:flex; align-items:center; gap:.4rem; background:#f1f5f9; padding:.45rem .65rem; border-radius:20px; cursor:pointer; border:1px solid #e2e8f0; font-weight:500; color:#334155; }
.toggles label:has(input:disabled) { opacity:.45; cursor:not-allowed; }
.toggles input { accent-color:#3b82f6; }
.actions { display:flex; gap:.6rem; margin-top:.25rem; }
button { cursor:pointer; background:#3b82f6; border:none; color:#fff; padding:.6rem 1rem; border-radius:6px; font-size:.8rem; font-weight:600; letter-spacing:.03em; display:inline-flex; align-items:center; gap:.4rem; box-shadow:0 2px 4px rgba(0,0,0,0.08); }
button:hover:not([disabled]) { background:#2563eb; }
button[disabled] { opacity:.5; cursor:not-allowed; }
button.danger { background:#dc2626; }
button.danger:hover { background:#b91c1c; }
section.disabled { opacity:.65; pointer-events:none; }
.err { color:#dc2626; font-size:.65rem; font-weight:500; }
.hint { font-size:.65rem; opacity:.65; margin-top:.25rem; }
.error { color:#dc2626; font-size:.7rem; margin-top:.45rem; font-weight:500; }
.runtime { display:flex; flex-direction:column; gap:1.15rem; }
.bar { position:relative; background:#f1f5f9; border:1px solid #e2e8f0; border-radius:5px; height:12px; overflow:hidden; }
.bar.small { height:8px; }
.fill { position:absolute; top:0; left:0; bottom:0; background:linear-gradient(90deg,#10b981,#059669); transition:width .25s linear; }
.bar.indeterminate .fill { 
  width: 35% !important; 
  animation: shimmer 1.2s infinite linear; 
  background: linear-gradient(90deg, rgba(16,185,129,0.2), rgba(16,185,129,0.6), rgba(16,185,129,0.2));
}
@keyframes shimmer {
  0% { transform: translateX(-100%); }
  100% { transform: translateX(300%); }
}
.metrics { display:grid; grid-template-columns: repeat(auto-fill,minmax(140px,1fr)); gap:.65rem; }
.metric { background:#f1f5f9; padding:.6rem .7rem; border-radius:6px; display:flex; flex-direction:column; gap:.25rem; border:1px solid #e2e8f0; }
.metric .lbl { font-size:.6rem; font-weight:600; text-transform:uppercase; letter-spacing:.06em; opacity:.55; }
.metric .val { font-size:1.15rem; font-weight:700; }
.metrics-row { display:flex; justify-content:space-between; font-size:.7rem; opacity:.8; margin-top:.3rem; font-weight:500; }
.runtime-actions { display:flex; flex-wrap:wrap; gap:.55rem; }
.overall, .episode { display:flex; flex-direction:column; gap:.5rem; }
.overall label, .episode label { font-size:.7rem; font-weight:600; letter-spacing:.05em; text-transform:uppercase; color:#475569; }
.overall label .existing-total { font-size:.6rem; font-weight:500; text-transform:none; margin-left:.35rem; color:#64748b; }
.state-line { font-size:.6rem; opacity:.55; margin-top:.4rem; letter-spacing:.05em; }
.root-with-browse .root-row { display:flex; gap:.5rem; }
.root-with-browse .root-input { flex:1; padding:.65rem .75rem; font-size:.8rem; }
.root-with-browse .browse-btn { background:#6366f1; display:flex; align-items:center; justify-content:center; padding:.45rem; width:36px; min-width:36px; border-radius:6px; }
.root-with-browse .browse-btn svg { pointer-events:none; }
.root-with-browse .browse-btn:hover { background:#4f46e5; }
.root-with-browse .browse-btn:disabled { background:#6b7280; }

/* Dark mode */
body.dark-mode .replay-dataset-view section { background:linear-gradient(135deg,#1f2937 0%,#111827 100%); border-color:#374151; box-shadow:0 4px 18px rgba(0,0,0,0.45); }
body.dark-mode .replay-dataset-view h1 { color:#f1f5f9; }
body.dark-mode .replay-dataset-view h2 { color:#cbd5e1; opacity:.9; }
body.dark-mode .replay-dataset-view h6 { color:#94a3b8; }
body.dark-mode .replay-dataset-view h6 code { background:#273549; color:#cbd5e1; }
body.dark-mode .replay-dataset-view .field label { color:#94a3b8; }
body.dark-mode .replay-dataset-view .field input, 
body.dark-mode .replay-dataset-view .field textarea { background:#1e2532; border:1px solid #334155; color:#f1f5f9; }
body.dark-mode .replay-dataset-view .field input:disabled, 
body.dark-mode .replay-dataset-view .field textarea:disabled { background:#273042; color:#94a3b8; }
body.dark-mode .replay-dataset-view .toggles label { background:#273549; border-color:#334155; color:#e2e8f0; }
body.dark-mode .replay-dataset-view .bar { background:#1e2532; border-color:#334155; }
body.dark-mode .replay-dataset-view .metric { background:#273549; border-color:#334155; }
body.dark-mode .replay-dataset-view .overall label, 
body.dark-mode .replay-dataset-view .episode label { color:#cbd5e1; }
body.dark-mode .replay-dataset-view .overall label .existing-total { color:#94a3b8; }
body.dark-mode .replay-dataset-view .state-line { color:#94a3b8; opacity:.6; }

/* Responsive */
@media (max-width: 1220px) { .layout { gap:1.5rem; } }
@media (max-width: 1100px) { .layout { grid-template-columns:1fr; } }
@media (max-width: 620px) { .grid { grid-template-columns: repeat(auto-fill,minmax(100px,1fr)); } }
</style>
