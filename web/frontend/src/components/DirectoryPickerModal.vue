<template>
  <div v-if="modelValue" class="dir-modal-overlay" @keydown.esc="close">
    <div class="dir-modal" role="dialog" aria-modal="true">
      <header class="dm-header">
        <h3>Select Directory</h3>
        <button class="close" @click="close" aria-label="Close">×</button>
      </header>
      <div class="dm-pathbar">
        <button class="up" @click="goUp" :disabled="!canGoUp">↑</button>
        <input class="path-input" v-model="currentPath" @keyup.enter="jumpToPath" />
        <button class="go" @click="jumpToPath">Go</button>
      </div>
      <div class="dm-quick">
        <button v-for="q in quickLinks" :key="q.path" @click="quick(q.path)">{{ q.label }}</button>
      </div>
      <div v-if="mountInfo && mountInfo.mounts && mountInfo.mounts.length > 0" class="dm-mounts">
        <h4>Available Drives:</h4>
        <div class="mount-list">
          <div v-for="mount in mountInfo.mounts.slice(0, 5)" :key="mount.mount_point" class="mount-item">
            <span class="mount-path">{{ mount.mount_point }}</span>
            <span class="mount-size">{{ mount.total_gb }}GB</span>
            <span class="mount-usage" :class="{ 'high-usage': mount.usage_percent > 90 }">
              {{ mount.usage_percent }}% used
            </span>
          </div>
        </div>
      </div>
      <div class="dm-body" v-if="!loading">
        <ul class="folder-list" @dblclick.prevent>
          <li v-for="f in folders" :key="f.path" @click="select(f)" :class="{ selected: selectedPath===f.path }" @dblclick="enter(f)" :title="f.path">
            <span class="icon">📁</span>
            <span class="name">{{ f.name }}</span>
          </li>
          <li v-if="folders.length===0" class="empty">No subfolders</li>
        </ul>
      </div>
      <div class="dm-body loading" v-else>Loading...</div>
      <div v-if="error" class="dm-error">{{ error }}</div>
      <footer class="dm-footer">
        <div class="left">
          <span class="selected-label">Selected:</span>
          <span class="selected-path" :title="selectedPath || 'None'">{{ selectedPath || 'None' }}</span>
        </div>
        <div class="right">
          <button @click="confirm" :disabled="!selectedPath">Use This Folder</button>
          <button class="cancel" @click="close">Cancel</button>
        </div>
      </footer>
    </div>
  </div>
</template>

<script setup>
import { ref, watch, onMounted } from 'vue';
import datasetApi from '@/services/api/datasetApi';

const props = defineProps({
  modelValue: { type: Boolean, default: false },
  startPath: { type: String, default: '/' }
});
const emit = defineEmits(['update:modelValue','select']);

const currentPath = ref(props.startPath || '/');
const folders = ref([]);
const loading = ref(false);
const error = ref(null);
const selectedPath = ref(null);
const mountInfo = ref(null);

const quickLinks = [
  { label: 'Home', path: '~' },
  { label: 'Root', path: '/' },
  { label: 'Media', path: '/media' },
  { label: 'Mount', path: '/mnt' },
  { label: 'Data', path: '/data' },
  { label: 'Datasets', path: '~/datasets' }
];

function fetchDir(path){
  loading.value = true;
  error.value = null;
  datasetApi.browseDirectory(path).then(r => {
    folders.value = r.data.folders || [];
    currentPath.value = r.data.path;
    mountInfo.value = r.data.mounts || null;
  }).catch(e => {
    error.value = e?.response?.data?.message || e.message;
  }).finally(()=>{ loading.value = false; });
}

function goUp(){
  if(!canGoUp.value) return;
  const parent = currentPath.value === '/' ? '/' : currentPath.value.replace(/\/$/,'').split('/').slice(0,-1).join('/') || '/';
  fetchDir(parent);
  selectedPath.value = null;
}
function enter(folder){ fetchDir(folder.path); selectedPath.value = folder.path; }
function select(folder){ selectedPath.value = folder.path; }
function confirm(){ if(selectedPath.value){ emit('select', selectedPath.value); close(); } }
function close(){ emit('update:modelValue', false); }
function quick(path){ fetchDir(path); selectedPath.value = null; }
function jumpToPath(){ if(currentPath.value) fetchDir(currentPath.value); }

const canGoUp = ref(false);
watch(currentPath, (p)=>{ canGoUp.value = p !== '/'; });

watch(()=>props.modelValue, (open)=>{ if(open){ fetchDir(currentPath.value || props.startPath || '/'); } });

onMounted(()=>{ if(props.modelValue){ fetchDir(currentPath.value); } });
</script>

<style scoped>
.dir-modal-overlay { position:fixed; inset:0; background:rgba(0,0,0,0.45); display:flex; align-items:center; justify-content:center; z-index:5000; }
.dir-modal { width:720px; max-width:95vw; background:#fff; border-radius:12px; box-shadow:0 10px 30px rgba(0,0,0,0.35); display:flex; flex-direction:column; max-height:90vh; overflow:hidden; }
.dm-header { display:flex; align-items:center; justify-content:space-between; padding:.85rem 1rem .65rem; border-bottom:1px solid #e5e7eb; }
.dm-header h3 { margin:0; font-size:1rem; font-weight:600; }
.dm-header .close { background:none; border:none; font-size:1.2rem; cursor:pointer; line-height:1; }
.dm-pathbar { display:flex; gap:.5rem; padding:.6rem 1rem; align-items:center; border-bottom:1px solid #f1f5f9; }
.dm-pathbar .path-input { flex:1; padding:.45rem .55rem; border:1px solid #d1d5db; border-radius:6px; font-size:.8rem; }
.dm-pathbar button { background:#3b82f6; color:#fff; border:none; padding:.45rem .7rem; border-radius:6px; font-size:.7rem; cursor:pointer; }
.dm-pathbar button:disabled { opacity:.4; cursor:not-allowed; }
.dm-quick { display:flex; flex-wrap:wrap; gap:.4rem; padding:0 1rem .5rem; border-bottom:1px solid #f1f5f9; }
.dm-quick button { background:#f1f5f9; border:1px solid #e2e8f0; border-radius:16px; padding:.35rem .65rem; font-size:.65rem; cursor:pointer; color:#334155; font-weight:500; transition:background .15s, color .15s, border-color .15s; }
.dm-quick button:hover { background:#e2e8f0; }
/* Make browsing area fixed height to avoid layout jumping between folders */
.dm-body { flex:1 1 auto; overflow:auto; padding:.5rem 1rem .75rem; min-height:200px; max-height:400px; }
ul.folder-list { list-style:none; margin:0; padding:0; display:grid; grid-template-columns:repeat(auto-fill,180px); grid-auto-rows:52px; justify-content:start; gap:.5rem; min-height:260px; }
ul.folder-list li { background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:.55rem .55rem; display:flex; gap:.5rem; align-items:center; cursor:pointer; font-size:.7rem; font-weight:500; width:180px; height:52px; box-sizing:border-box; }
ul.folder-list li .name { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; flex:1; }
ul.folder-list li:hover { background:#eef6ff; border-color:#bfdbfe; }
ul.folder-list li.selected { background:#2563eb; border-color:#1d4ed8; color:#fff; }
ul.folder-list li.empty { grid-column:1 / -1; justify-content:center; }
.icon { font-size:.9rem; }
.dm-footer { display:flex; align-items:center; justify-content:space-between; padding:.65rem 1rem .75rem; border-top:1px solid #e5e7eb; gap:1rem; flex-wrap:wrap; min-height:3rem; }
.dm-footer .left { flex:1; min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-size:.75rem; display:flex; align-items:center; gap:.5rem; }
.selected-label { font-weight:500; color:#6b7280; flex-shrink:0; }
.selected-path { font-weight:600; color:#1f2937; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.dm-footer .right { display:flex; gap:.55rem; flex-shrink:0; }
.dm-footer button { background:#3b82f6; color:#fff; border:none; padding:.5rem .8rem; border-radius:6px; font-size:.7rem; font-weight:600; cursor:pointer; white-space:nowrap; min-width:fit-content; }
.dm-footer button.cancel { background:#6b7280; }
.dm-error { padding:.4rem 1rem; color:#b91c1c; font-size:.65rem; font-weight:500; }
.loading { padding:2rem 1rem; text-align:center; font-size:.75rem; opacity:.7; }
@media (max-width:600px){ .dir-modal { width:100%; height:100%; border-radius:0; max-height:none; } ul.folder-list { grid-template-columns:repeat(auto-fill,minmax(140px,1fr)); } .dm-footer { flex-direction:column; align-items:stretch; gap:.5rem; } .dm-footer .left { text-align:center; margin-bottom:.25rem; } .dm-footer .right { justify-content:center; } }
body.dark-mode .dir-modal { background:#1f2937; color:#f1f5f9; }
body.dark-mode .dm-header { border-color:#334155; }
body.dark-mode .dm-pathbar { border-color:#334155; }
body.dark-mode .dm-pathbar .path-input { background:#1e293b; border-color:#334155; color:#f1f5f9; }
body.dark-mode ul.folder-list li { background:#1e293b; border-color:#334155; color:#e2e8f0; }
body.dark-mode ul.folder-list li:hover { background:#273549; border-color:#3b82f6; }
body.dark-mode ul.folder-list li.selected { background:#2563eb; border-color:#1d4ed8; }
body.dark-mode .dm-quick button { background:#273549; border-color:#334155; color:#e2e8f0; }
body.dark-mode .dm-quick button:hover { background:#334155; border-color:#3b82f6; }
body.dark-mode .dm-footer { border-color:#334155; }
.dm-mounts { padding:0 1rem .75rem; border-bottom:1px solid #e5e7eb; }
.dm-mounts h4 { margin:0 0 .5rem; font-size:.75rem; font-weight:600; text-transform:uppercase; letter-spacing:.05em; color:#4b5563; }
.mount-list { display:flex; flex-direction:column; gap:.35rem; }
.mount-item { display:flex; align-items:center; justify-content:space-between; padding:.4rem .6rem; background:#f8fafc; border:1px solid #e2e8f0; border-radius:6px; font-size:.7rem; }
.mount-path { font-weight:500; color:#1f2937; flex:1; }
.mount-size { color:#6b7280; margin-right:.5rem; }
.mount-usage { font-weight:500; color:#059669; }
.mount-usage.high-usage { color:#dc2626; }
body.dark-mode .dm-mounts { border-color:#334155; }
body.dark-mode .dm-mounts h4 { color:#cbd5e1; }
body.dark-mode .mount-item { background:#1e293b; border-color:#334155; }
body.dark-mode .mount-path { color:#f1f5f9; }
body.dark-mode .mount-size { color:#94a3b8; }
body.dark-mode .mount-usage { color:#10b981; }
body.dark-mode .selected-label { color:#94a3b8; }
body.dark-mode .selected-path { color:#f1f5f9; }
</style>
