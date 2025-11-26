import { createRouter, createWebHistory } from 'vue-router';
import DashboardView from '../views/DashboardView.vue';

const routes = [
  { path: '/', name: 'home', component: DashboardView },
  {
    path: '/teleoperation',
    name: 'teleoperation',
    component: () => import('../views/TeleoperationView.vue')
  },
  {
    path: '/record-dataset',
    name: 'record-dataset',
    component: () => import('../views/RecordDatasetView.vue')
  },
  {
    path: '/replay-dataset',
    name: 'replay-dataset',
    component: () => import('../views/ReplayDatasetView.vue')
  },
  {
    path: '/training',
    name: 'training',
    component: () => import('../views/TrainingView.vue')
  },
  {
    path: '/calibration',
    name: 'calibration',
    component: () => import('../views/CalibrationView.vue')
  },
  {
    path: '/data-visualization',
    name: 'data-visualization',
    component: () => import('../views/DataVisualizationView.vue')
  },
  {
    path: '/demo',
    name: 'demo',
    component: () => import('../views/DemoView.vue')
  }
];

const router = createRouter({
  history: createWebHistory('/'),
  routes
});

export default router;