<template>
  <div class="record-dataset-view">
    <h1>Record Dataset</h1>
    <h6>
      Fill the required fields (Repo ID, Root Path, Task) and timings, then press <em>Start Recording</em>.
      The status panel shows overall progress and the current phase: Warmup → Recording → Resetting → Processing → Pushing (optional).
      During recording you can <strong>Re-record</strong> to redo the current episode, <strong>Skip</strong> to move on, <strong>Stop</strong> to end, or use <strong>Emergency Stop</strong> if needed.
      Tip: Use the folder button to pick a root. Enable <strong>Live Display</strong> to open the external viewer; enable <strong>Push to Hub</strong> to upload after completion.
    </h6>
    <div class="layout">
      <section class="config" :class="{ disabled: isActive }">
        <h2>Configuration</h2>
        <form @submit.prevent="start">
          <div class="field">
            <label>Repo ID *</label>
            <input v-model="cfg.repo_id" :disabled="isActive" placeholder="user/dataset" @input="onChange" />
            <span class="err" v-if="errors.repo_id">{{ errors.repo_id }}</span>
          </div>
          <div class="field root-with-browse">
            <label>Root Path *</label>
            <div class="root-row">
              <input v-model="cfg.root" :disabled="isActive" placeholder="/data/my-datasets" @input="onChange" class="root-input" :title="cfg.root" />
              <button type="button" class="browse-btn icon-only" @click="openBrowse" :disabled="isActive" aria-label="Browse for folder" title="Browse for folder">
                <svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor" aria-hidden="true">
                  <path d="M10 4l2 2h8a2 2 0 0 1 2 2v1H2V6a2 2 0 0 1 2-2h6zm12 6v8a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2v-8h20z" />
                </svg>
              </button>
            </div>
            <span class="err" v-if="errors.root">{{ errors.root }}</span>
            <small class="hint" v-if="!errors.root">Pick a directory that will contain (or already contains) the dataset folder.</small>
          </div>
          <div class="field">
            <label>Task Description *</label>
            <textarea v-model="cfg.single_task" :disabled="isActive" rows="2" @input="onChange" />
            <span class="err" v-if="errors.single_task">{{ errors.single_task }}</span>
          </div>
          <div class="grid">
            <div class="field">
              <label>FPS *</label>
              <input type="number" v-model.number="cfg.fps" :disabled="isActive" @input="onChange" />
              <span class="err" v-if="errors.fps">{{ errors.fps }}</span>
            </div>
            <div class="field">
              <label>Episodes *</label>
              <input type="number" v-model.number="cfg.num_episodes" :disabled="isActive" @input="onChange" />
              <span class="err" v-if="errors.num_episodes">{{ errors.num_episodes }}</span>
            </div>
            <div class="field">
              <label>Episode Time (s)</label>
              <input type="number" v-model.number="cfg.episode_time_s" :disabled="isActive" @input="onChange" />
              <span class="err" v-if="errors.episode_time_s">{{ errors.episode_time_s }}</span>
            </div>
            <div class="field">
              <label>Warmup (s)</label>
              <input type="number" v-model.number="cfg.warmup_time_s" :disabled="isActive" @input="onChange" />
            </div>
            <div class="field">
              <label>Reset (s)</label>
              <input type="number" v-model.number="cfg.reset_time_s" :disabled="isActive" @input="onChange" />
            </div>
          </div>
          <div class="toggles">
            <label><input type="checkbox" v-model="cfg.display_data" :disabled="isActive" @change="onChange" /> Live Display</label>
            <label><input type="checkbox" v-model="cfg.push_to_hub" :disabled="isActive" @change="onChange" /> Push to Hub</label>
            <label><input type="checkbox" v-model="cfg.private" :disabled="isActive || !cfg.push_to_hub" @change="onChange" /> Private</label>
            <label><input type="checkbox" v-model="cfg.resume" :disabled="isActive" @change="onChange" /> Resume</label>
          </div>
          <div class="actions" v-if="!isActive">
            <button type="submit" :disabled="!canStart">Start Recording</button>
            <button type="button" @click="resetForm" :disabled="isActive">Reset</button>
          </div>
        </form>
        <p class="hint">* required fields. Robot must be connected.</p>
        <p class="error" v-if="error">{{ error }}</p>
      </section>

      <section class="runtime">
        <h2>Status</h2>
        <div v-if="!isActive && progressPct===0">Idle</div>
        <div class="overall" v-if="progressPct>0 || isActive">
          <label>
            Episodes: {{ status.episode_index }} / {{ status.total_episodes || cfg.num_episodes }}
            <span v-if="status.existing_episodes != null && status.existing_episodes > 0" class="existing-total">
              (existing total: {{ status.existing_episodes +  status.episode_index}}, new total: {{ status.existing_episodes + (status.total_episodes || cfg.num_episodes) }})
            </span>
          </label>
          <div class="bar"><div class="fill" :style="{width: progressPct+'%'}"></div></div>
        </div>
        <div class="episode" v-if="isActive">
          <label>{{ phaseLabel }} Progress</label>
          <div class="bar small" :class="{ indeterminate: !status.phase_total_s }">
            <div class="fill" :style="{width: (status.phase_total_s ? phaseBarPct : 35)+'%'}"></div>
          </div>
          <div class="metrics-row">
            <span v-if="status.phase==='recording'">{{ status.episode_frames }} frames</span>
            <span v-else>&nbsp;</span>
            <span v-if="phaseTimeText && status.phase !== 'processing' && status.phase !== 'pushing'">{{ phaseTimeText }}</span>
          </div>
        </div>
        <div class="metrics">
          <div class="metric"><span class="lbl">FPS Target</span><span>{{ status.fps_target || cfg.fps }}</span></div>
          <div class="metric"><span class="lbl">FPS Current</span><span>{{ status.fps_current?.toFixed(1) || '-' }}</span></div>
          <div class="metric"><span class="lbl">Total Frames</span><span>{{ status.total_frames }}</span></div>
        </div>
        <div class="runtime-actions" v-if="isActive">
          <button @click="stop">Stop</button>
          <button @click="rerecordEpisode" :disabled="status.rerecord_pending">Re-record</button>
          <button @click="skipEpisode" :disabled="status.rerecord_pending">Skip</button>
          <button class="danger" @click="emergencyStop">Emergency Stop</button>
        </div>
        <div v-if="status.rerecord_pending" class="hint" style="margin-top:.25rem;">
          Re-record requested — finishing this phase, then resetting and redoing the current episode.
        </div>
        <div class="state-line">State: {{ status.state }}</div>
      </section>
    </div>
  </div>
  <DirectoryPickerModal v-model="browseOpen" :start-path="initialBrowsePath" @select="setRoot" />
</template>

<script setup>
import { storeToRefs } from 'pinia';
import { useRecordingStore } from '@/stores/recordingStore';
import { useRobotStore } from '@/stores/robotStore';
import DirectoryPickerModal from '@/components/DirectoryPickerModal.vue';

const recStore = useRecordingStore();
const robotStore = useRobotStore();
recStore.ensureSocketListeners();

const { config: cfg, status, validationErrors: errors, error } = storeToRefs(recStore);
const { isActive, canStart, progressPct, phaseBarPct, phaseLabel, phaseTimeText } = storeToRefs(recStore);

function onChange() { recStore.updateConfig({ ...cfg.value }); }
function start() { recStore.start(); }
function stop() { recStore.stop(); }
function rerecordEpisode() { recStore.rerecordEpisode(); }
function skipEpisode() { recStore.skipEpisode(); }
function emergencyStop() { recStore.emergencyStop(); }
function resetForm() { recStore.resetForm(); }

// Directory picker integration
import { ref, computed } from 'vue';
const browseOpen = ref(false);
const initialBrowsePath = computed(()=> cfg.value.root && cfg.value.root.trim().length>0 ? cfg.value.root : '/');
function openBrowse(){ browseOpen.value = true; }
function setRoot(path){ recStore.updateConfig({ root: path }); }

// Ensure robot socket ready
if (!robotStore.socket) robotStore.initSocket();
</script>

<style scoped>
.record-dataset-view { padding: 1.5rem; display: flex; flex-direction: column; gap: 1.25rem; max-width:1400px; margin:0 auto; }
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
/* Indeterminate shimmer for processing/pushing phases */
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
.metrics-row { display:flex; justify-content:space-between; font-size:.7rem; opacity:.8; margin-top:.3rem; font-weight:500; }
.runtime-actions { display:flex; flex-wrap:wrap; gap:.55rem; }
.overall, .episode { display:flex; flex-direction:column; gap:.5rem; }
.overall label, .episode label { font-size:.7rem; font-weight:600; letter-spacing:.05em; text-transform:uppercase; color:#475569; }
.overall label .existing-total { font-size:.6rem; font-weight:500; text-transform:none; margin-left:.35rem; color:#64748b; }
body.dark-mode .overall label .existing-total { color:#94a3b8; }
.state-line { font-size:.6rem; opacity:.55; margin-top:.4rem; letter-spacing:.05em; }
.root-with-browse .root-row { display:flex; gap:.5rem; }
.root-with-browse .root-input { flex:1; padding:.65rem .75rem; font-size:.8rem; }
.root-with-browse .browse-btn { background:#6366f1; display:flex; align-items:center; justify-content:center; padding:.45rem; width:36px; min-width:36px; border-radius:6px; }
.root-with-browse .browse-btn svg { pointer-events:none; }
.root-with-browse .browse-btn:hover { background:#4f46e5; }
.root-with-browse .browse-btn:disabled { background:#6b7280; }

/* Dark mode overrides */
body.dark-mode .record-dataset-view section { background:linear-gradient(135deg,#1f2937 0%,#111827 100%); border-color:#374151; box-shadow:0 4px 18px rgba(0,0,0,0.45); }
body.dark-mode .record-dataset-view h1 { color:#f1f5f9; }
body.dark-mode .record-dataset-view h2 { color:#cbd5e1; opacity:.9; }
body.dark-mode .record-dataset-view h6 { color:#94a3b8; }
body.dark-mode .record-dataset-view h6 code { background:#273549; color:#cbd5e1; }
body.dark-mode .record-dataset-view .field label { color:#94a3b8; }
body.dark-mode .record-dataset-view .field input, 
body.dark-mode .record-dataset-view .field textarea { background:#1e2532; border:1px solid #334155; color:#f1f5f9; }
body.dark-mode .record-dataset-view .field input:disabled, 
body.dark-mode .record-dataset-view .field textarea:disabled { background:#273042; color:#94a3b8; }
body.dark-mode .record-dataset-view .toggles label { background:#273549; border-color:#334155; color:#e2e8f0; }
body.dark-mode .record-dataset-view .bar { background:#1e2532; border-color:#334155; }
body.dark-mode .record-dataset-view .metric { background:#273549; border-color:#334155; }
body.dark-mode .record-dataset-view .overall label, 
body.dark-mode .record-dataset-view .episode label { color:#cbd5e1; }
body.dark-mode .record-dataset-view .state-line { color:#94a3b8; opacity:.6; }

/* Responsive */
@media (max-width: 1220px) { .layout { gap:1.5rem; } }
@media (max-width: 1100px) { .layout { grid-template-columns:1fr; } }
@media (max-width: 620px) { .grid { grid-template-columns: repeat(auto-fill,minmax(100px,1fr)); } }
</style>
