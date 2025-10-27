<template>
  <div class="teleoperation-docs">
    <div class="card">
      <div class="card-header">
        <h5 class="mb-0">
          <i class="bi bi-book me-2"></i>
          Teleoperation Documentation
        </h5>
      </div>
      <div class="card-body">
        <div class="row">
          <div class="col-md-3">
            <div class="nav nav-pills flex-column" role="tablist">
              <button 
                class="nav-link active" 
                data-bs-toggle="pill" 
                data-bs-target="#quick-start"
                type="button"
                role="tab"
              >
                Quick Start
              </button>
              <button 
                class="nav-link" 
                data-bs-toggle="pill" 
                data-bs-target="#standard-config"
                type="button"
                role="tab"
              >
                Standard Config
              </button>
              <button 
                class="nav-link" 
                data-bs-toggle="pill" 
                data-bs-target="#expert-config"
                type="button"
                role="tab"
              >
                Expert Config
              </button>
              <button 
                class="nav-link" 
                data-bs-toggle="pill" 
                data-bs-target="#safety"
                type="button"
                role="tab"
              >
                Safety
              </button>
              <button 
                class="nav-link" 
                data-bs-toggle="pill" 
                data-bs-target="#troubleshooting"
                type="button"
                role="tab"
              >
                Troubleshooting
              </button>
            </div>
          </div>
          <div class="col-md-9">
            <div class="tab-content">
              <!-- Quick Start -->
              <div class="tab-pane fade show active" id="quick-start" role="tabpanel">
                <h6>Quick Start Guide</h6>
                <ol>
                  <li><strong>Connect Robot:</strong> Select operation mode and click "Connect ALOHA Robot"</li>
                  <li><strong>Configure:</strong> Use Standard mode for common settings</li>
                  <li><strong>Start:</strong> Click "Apply Configuration" then "Start Teleoperation"</li>
                  <li><strong>Control:</strong> Use your input device to control the robot</li>
                  <li><strong>Stop:</strong> Click "Stop Teleoperation" or press Space for emergency stop</li>
                </ol>
                
                <div class="alert alert-info">
                  <strong>Default Settings:</strong>
                  <ul class="mb-0">
                    <li>Frequency: 30 Hz</li>
                    <li>Safety Limit: 25°</li>
                    <li>Cameras: Enabled</li>
                    <li>Mode: Bimanual</li>
                  </ul>
                </div>
              </div>

              <!-- Standard Config -->
              <div class="tab-pane fade" id="standard-config" role="tabpanel">
                <h6>Standard Configuration Options</h6>
                
                <div class="table-responsive">
                  <table class="table table-sm">
                    <thead>
                      <tr>
                        <th>Setting</th>
                        <th>Options</th>
                        <th>Description</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td>Control Frequency</td>
                        <td>30, 60, 100, 200 Hz or Unlimited</td>
                        <td>Higher = smoother control, more CPU usage</td>
                      </tr>
                      <tr>
                        <td>Operation Mode</td>
                        <td>Bimanual, Right Only, Left Only</td>
                        <td>Which arms to control</td>
                      </tr>
                      <tr>
                        <td>Safety Limit</td>
                        <td>5°, 15°, 25°, 45°, Unlimited</td>
                        <td>Maximum joint movement per command</td>
                      </tr>
                      <tr>
                        <td>Camera Display</td>
                        <td>Enabled/Disabled</td>
                        <td>Show camera feeds during operation</td>
                      </tr>
                      <tr>
                        <td>Emergency Stop</td>
                        <td>Enabled/Disabled</td>
                        <td>Space bar to instantly stop</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>

              <!-- Expert Config -->
              <div class="tab-pane fade" id="expert-config" role="tabpanel">
                <h6>Expert Configuration Options</h6>
                
                <div class="alert alert-warning">
                  <i class="bi bi-exclamation-triangle me-2"></i>
                  <strong>Warning:</strong> These settings can affect robot safety and performance.
                </div>

                <div class="table-responsive">
                  <table class="table table-sm">
                    <thead>
                      <tr>
                        <th>Setting</th>
                        <th>Range</th>
                        <th>Default</th>
                        <th>Description</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td>Custom Frequency</td>
                        <td>1-300 Hz</td>
                        <td>-</td>
                        <td>Override standard frequency options</td>
                      </tr>
                      <tr>
                        <td>Movement Time</td>
                        <td>0.01-1.0 s</td>
                        <td>0.1 s</td>
                        <td>Time for robot to execute each command</td>
                      </tr>
                      <tr>
                        <td>Custom Safety Limit</td>
                        <td>1-180°</td>
                        <td>-</td>
                        <td>Custom joint movement limit</td>
                      </tr>
                      <tr>
                        <td>Session Time Limit</td>
                        <td>1-120 min</td>
                        <td>-</td>
                        <td>Automatic stop after specified time</td>
                      </tr>
                      <tr>
                        <td>Performance Monitoring</td>
                        <td>On/Off</td>
                        <td>Off</td>
                        <td>Track FPS, latency, system metrics</td>
                      </tr>
                      <tr>
                        <td>Debug Level</td>
                        <td>ERROR, WARNING, INFO, DEBUG</td>
                        <td>INFO</td>
                        <td>Control verbosity of logs</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>

              <!-- Safety -->
              <div class="tab-pane fade" id="safety" role="tabpanel">
                <h6>Safety Guidelines</h6>
                
                <div class="alert alert-danger">
                  <strong>⚠️ Important Safety Information</strong>
                </div>
                
                <h6 class="mt-3">Before Operation:</h6>
                <ul>
                  <li>Ensure workspace is clear of obstacles and people</li>
                  <li>Verify robot is properly calibrated</li>
                  <li>Test emergency stop functionality</li>
                  <li>Start with low safety limits (5-15°) for training</li>
                </ul>

                <h6 class="mt-3">During Operation:</h6>
                <ul>
                  <li>Keep emergency stop key (Space) easily accessible</li>
                  <li>Monitor robot behavior continuously</li>
                  <li>Use "Move to Safe Position" before stopping</li>
                  <li>Watch for unusual sounds or movements</li>
                </ul>

                <h6 class="mt-3">Emergency Procedures:</h6>
                <div class="row g-3">
                  <div class="col-md-6">
                    <div class="card bg-danger text-white">
                      <div class="card-body">
                        <h6 class="card-title">Immediate Stop</h6>
                        <p class="card-text">Press <kbd>Space</kbd> key for instant emergency stop</p>
                      </div>
                    </div>
                  </div>
                  <div class="col-md-6">
                    <div class="card bg-warning">
                      <div class="card-body">
                        <h6 class="card-title">Controlled Stop</h6>
                        <p class="card-text">Click "Stop Teleoperation" for controlled shutdown</p>
                      </div>
                    </div>
                  </div>
                </div>

                <h6 class="mt-3">Recommended Settings by Experience:</h6>
                <div class="table-responsive">
                  <table class="table table-sm">
                    <thead>
                      <tr>
                        <th>Experience Level</th>
                        <th>Safety Limit</th>
                        <th>Frequency</th>
                        <th>Cameras</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td>Beginner</td>
                        <td>5°</td>
                        <td>30 Hz</td>
                        <td>Enabled</td>
                      </tr>
                      <tr>
                        <td>Intermediate</td>
                        <td>15-25°</td>
                        <td>60 Hz</td>
                        <td>Enabled</td>
                      </tr>
                      <tr>
                        <td>Expert</td>
                        <td>25-45°</td>
                        <td>100+ Hz</td>
                        <td>Optional</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>

              <!-- Troubleshooting -->
              <div class="tab-pane fade" id="troubleshooting" role="tabpanel">
                <h6>Common Issues & Solutions</h6>
                
                <div class="accordion" id="troubleshootingAccordion">
                  <div class="accordion-item">
                    <h2 class="accordion-header">
                      <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#connection-issues">
                        Connection Issues
                      </button>
                    </h2>
                    <div id="connection-issues" class="accordion-collapse collapse" data-bs-parent="#troubleshootingAccordion">
                      <div class="accordion-body">
                        <strong>Robot won't connect:</strong>
                        <ul>
                          <li>Check USB connections</li>
                          <li>Verify calibration directory exists</li>
                          <li>Restart the application</li>
                          <li>Check robot power and status LEDs</li>
                        </ul>
                      </div>
                    </div>
                  </div>

                  <div class="accordion-item">
                    <h2 class="accordion-header">
                      <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#performance-issues">
                        Performance Issues
                      </button>
                    </h2>
                    <div id="performance-issues" class="accordion-collapse collapse" data-bs-parent="#troubleshootingAccordion">
                      <div class="accordion-body">
                        <strong>Low FPS or laggy response:</strong>
                        <ul>
                          <li>Reduce control frequency</li>
                          <li>Disable camera display</li>
                          <li>Close other applications</li>
                          <li>Check system resources</li>
                        </ul>
                      </div>
                    </div>
                  </div>

                  <div class="accordion-item">
                    <h2 class="accordion-header">
                      <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#camera-issues">
                        Camera Issues
                      </button>
                    </h2>
                    <div id="camera-issues" class="accordion-collapse collapse" data-bs-parent="#troubleshootingAccordion">
                      <div class="accordion-body">
                        <strong>Cameras not displaying:</strong>
                        <ul>
                          <li>Check camera connections</li>
                          <li>Verify camera permissions</li>
                          <li>Try disabling/enabling camera display</li>
                          <li>Restart teleoperation</li>
                        </ul>
                      </div>
                    </div>
                  </div>

                  <div class="accordion-item">
                    <h2 class="accordion-header">
                      <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#control-issues">
                        Control Issues
                      </button>
                    </h2>
                    <div id="control-issues" class="accordion-collapse collapse" data-bs-parent="#troubleshootingAccordion">
                      <div class="accordion-body">
                        <strong>Robot not responding to input:</strong>
                        <ul>
                          <li>Check input device connection</li>
                          <li>Verify teleoperation is active</li>
                          <li>Try increasing safety limits</li>
                          <li>Recalibrate if necessary</li>
                        </ul>
                      </div>
                    </div>
                  </div>
                </div>

                <div class="mt-3 alert alert-info">
                  <strong>Still need help?</strong> 
                  Check the console logs (F12) for detailed error messages, or consult the LeRobot documentation.
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
// This component is purely presentational
</script>

<style scoped>
.teleoperation-docs {
  width: 100%;
}

.nav-pills .nav-link {
  text-align: left;
  margin-bottom: 0.25rem;
}

.table th {
  font-weight: 600;
  font-size: 0.9rem;
}

.table td {
  font-size: 0.85rem;
}

kbd {
  background-color: #212529;
  color: white;
  padding: 0.2rem 0.4rem;
  border-radius: 0.25rem;
  font-size: 0.8rem;
}

.card.bg-danger .card-title,
.card.bg-warning .card-title {
  font-size: 0.9rem;
  margin-bottom: 0.5rem;
}

.card.bg-danger .card-text,
.card.bg-warning .card-text {
  font-size: 0.8rem;
  margin-bottom: 0;
}

.accordion-button {
  font-size: 0.9rem;
  padding: 0.75rem 1rem;
}

.accordion-body {
  font-size: 0.85rem;
}

body.dark-mode .table {
  color: #e4e6eb;
}

body.dark-mode .table th {
  border-color: #444;
}

body.dark-mode .table td {
  border-color: #333;
}

body.dark-mode kbd {
  background-color: #495057;
}
</style>
