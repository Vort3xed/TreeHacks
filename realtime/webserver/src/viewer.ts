import './main.css';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

// ════════════════════════════════════════
// Constants
// ════════════════════════════════════════
const SERVER_HOST = window.location.hostname;
const SERVER_PORT = '5000';
const WS_URL = `wss://${SERVER_HOST}:${SERVER_PORT}/ws/viewer`;
const API_URL = `https://${SERVER_HOST}:${SERVER_PORT}`;

// How many points we pre-allocate in the buffer geometry
const MAX_POINTS = 2_000_000;

// ════════════════════════════════════════
// DOM
// ════════════════════════════════════════
const container = document.getElementById('canvas-container')!;
const statusDot = document.getElementById('status-dot')!;
const statusText = document.getElementById('status-text')!;
const statPoints = document.getElementById('stat-points')!;
const statChunks = document.getElementById('stat-chunks')!;
const statBuffered = document.getElementById('stat-buffered')!;
const statElapsed = document.getElementById('stat-elapsed')!;
const progressContainer = document.getElementById('progress-container')!;
const progressBar = document.getElementById('progress-bar')!;

// Settings
const pointSizeSlider = document.getElementById('point-size') as HTMLInputElement;
const pointSizeVal = document.getElementById('point-size-val')!;
const chunkSizeSlider = document.getElementById('chunk-size') as HTMLInputElement;
const chunkSizeVal = document.getElementById('chunk-size-val')!;
const overlapSlider = document.getElementById('overlap') as HTMLInputElement;
const overlapVal = document.getElementById('overlap-val')!;
const confSlider = document.getElementById('conf-thre') as HTMLInputElement;
const confVal = document.getElementById('conf-thre-val')!;
const toggleGridBtn = document.getElementById('toggle-grid')!;
const toggleRotateBtn = document.getElementById('toggle-rotate')!;
const settingsPanel = document.getElementById('settings-panel')!;
const settingsBtn = document.getElementById('settings-btn')!;

// Upload
const uploadPanel = document.getElementById('upload-panel')!;
const uploadToggleBtn = document.getElementById('upload-toggle-btn')!;
const videoFileInput = document.getElementById('video-file') as HTMLInputElement;
const videoFpsInput = document.getElementById('video-fps') as HTMLInputElement;
const videoFastInput = document.getElementById('video-fast') as HTMLInputElement;
const uploadBtn = document.getElementById('upload-btn') as HTMLButtonElement;
const stopVideoBtn = document.getElementById('stop-video-btn') as HTMLButtonElement;
const uploadStatus = document.getElementById('upload-status')!;

// Controls
const connectBtn = document.getElementById('connect-btn') as HTMLButtonElement;
const disconnectBtn = document.getElementById('disconnect-btn') as HTMLButtonElement;
const resetBtn = document.getElementById('reset-btn') as HTMLButtonElement;
const resetViewBtn = document.getElementById('reset-view-btn')!;

// ════════════════════════════════════════
// Three.js Scene
// ════════════════════════════════════════
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0a0f);

const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.01, 1000);
camera.position.set(0, 2, 5);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
container.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.1;
controls.screenSpacePanning = true;
controls.maxDistance = 200;
controls.target.set(0, 0, 0);

// Grid
const gridHelper = new THREE.GridHelper(20, 40, 0x333333, 0x222222);
scene.add(gridHelper);

// Axes
const axesHelper = new THREE.AxesHelper(1);
scene.add(axesHelper);

// Ambient light (for potential future mesh rendering)
scene.add(new THREE.AmbientLight(0xffffff, 0.5));

// ════════════════════════════════════════
// Point Cloud
// ════════════════════════════════════════

// Pre-allocate buffers
const positions = new Float32Array(MAX_POINTS * 3);
const colors = new Float32Array(MAX_POINTS * 3);
let pointCount = 0;

const geometry = new THREE.BufferGeometry();
geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
geometry.setDrawRange(0, 0);

const material = new THREE.PointsMaterial({
  size: 2,
  vertexColors: true,
  sizeAttenuation: false,
});

const pointCloud = new THREE.Points(geometry, material);

// Wrap point cloud in a group for orientation transforms
const cloudGroup = new THREE.Group();
cloudGroup.add(pointCloud);
// Default: apply OpenCV→OpenGL fix (flip Y and Z)
cloudGroup.scale.set(1, -1, -1);
scene.add(cloudGroup);

// ════════════════════════════════════════
// State
// ════════════════════════════════════════
let ws: WebSocket | null = null;
let autoRotate = false;
let showGrid = true;
let configDebounceTimer: ReturnType<typeof setTimeout> | null = null;
let orientFlip = { x: 1, y: -1, z: -1 }; // default: OpenCV→GL correction
let orientRot = { x: 0, y: 0, z: 0 }; // rotation in degrees

// ════════════════════════════════════════
// Functions
// ════════════════════════════════════════

function setStatus(state: 'connected' | 'disconnected' | 'connecting' | 'error', text?: string) {
  statusDot.className = 'status-dot ' + state;
  statusText.textContent = text ?? ({
    connected: 'Connected',
    disconnected: 'Disconnected',
    connecting: 'Connecting...',
    error: 'Error',
  }[state]);
}

function addPoints(pts: Float32Array, cols: Uint8Array, n: number) {
  if (pointCount + n > MAX_POINTS) {
    console.warn(`Point buffer full (${MAX_POINTS}). Skipping new points.`);
    return;
  }

  // Copy position data
  positions.set(pts, pointCount * 3);

  // Convert uint8 colors to float [0, 1]
  for (let i = 0; i < n * 3; i++) {
    colors[pointCount * 3 + i] = cols[i] / 255;
  }

  pointCount += n;
  geometry.setDrawRange(0, pointCount);

  // Mark attributes as needing update
  (geometry.attributes.position as THREE.BufferAttribute).needsUpdate = true;
  (geometry.attributes.color as THREE.BufferAttribute).needsUpdate = true;

  // Recompute bounding sphere for frustum culling
  geometry.computeBoundingSphere();
}

function clearPoints() {
  pointCount = 0;
  geometry.setDrawRange(0, 0);
}

function parseBinaryPointCloud(buffer: ArrayBuffer) {
  // Header: chunk_id(i32) + total_points(i32) + num_new(i32) + elapsed(f32) = 16 bytes
  const view = new DataView(buffer);
  const chunkId = view.getInt32(0, true);
  const totalPoints = view.getInt32(4, true);
  const numNew = view.getInt32(8, true);
  const elapsed = view.getFloat32(12, true);

  if (numNew <= 0) return;

  // Layout: header(16) + positions(N*12 f32) + colors(N*3 u8)
  const posOffset = 16;
  const colOffset = 16 + numNew * 12;

  const pts = new Float32Array(buffer, posOffset, numNew * 3);
  const cols = new Uint8Array(buffer, colOffset, numNew * 3);

  addPoints(pts, cols, numNew);

  // Update stats
  statPoints.textContent = totalPoints.toLocaleString();
  statChunks.textContent = (chunkId + 1).toString();
  statElapsed.textContent = elapsed.toFixed(1) + 's';
}

function connect() {
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;

  setStatus('connecting');
  ws = new WebSocket(WS_URL);
  ws.binaryType = 'arraybuffer';

  ws.onopen = () => {
    setStatus('connected');
    connectBtn.disabled = true;
    disconnectBtn.disabled = false;
    resetBtn.disabled = false;
  };

  ws.onmessage = (event) => {
    if (event.data instanceof ArrayBuffer) {
      // Binary: point cloud data
      parseBinaryPointCloud(event.data);
      progressContainer.classList.add('hidden');
    } else {
      // JSON message
      const msg = JSON.parse(event.data);
      handleMessage(msg);
    }
  };

  ws.onclose = () => {
    setStatus('disconnected');
    connectBtn.disabled = false;
    disconnectBtn.disabled = true;
    resetBtn.disabled = true;
  };

  ws.onerror = () => {
    setStatus('error', 'Connection failed');
  };
}

function disconnect() {
  if (ws) {
    ws.close();
    ws = null;
  }
}

function handleMessage(msg: any) {
  switch (msg.type) {
    case 'init':
      statPoints.textContent = msg.total_points.toLocaleString();
      statChunks.textContent = msg.chunks_processed.toString();
      // Apply server config to UI
      if (msg.config) {
        chunkSizeSlider.value = msg.config.chunk_size.toString();
        chunkSizeVal.textContent = msg.config.chunk_size.toString();
        overlapSlider.value = msg.config.overlap.toString();
        overlapVal.textContent = msg.config.overlap.toString();
        confSlider.value = msg.config.conf_thre.toString();
        confVal.textContent = msg.config.conf_thre.toString();
      }
      break;

    case 'status':
      statBuffered.textContent = msg.frames_buffered + ' / ' + (msg.frames_buffered + msg.frames_needed);
      statPoints.textContent = msg.total_points.toLocaleString();
      statChunks.textContent = msg.chunks_processed.toString();
      break;

    case 'processing':
      progressContainer.classList.remove('hidden');
      progressBar.style.width = '50%';
      break;

    case 'chunk_done':
      progressContainer.classList.add('hidden');
      statPoints.textContent = msg.total_points.toLocaleString();
      statChunks.textContent = (msg.chunk_id + 1).toString();
      statElapsed.textContent = msg.elapsed.toFixed(1) + 's';
      break;

    case 'reset':
      clearPoints();
      statPoints.textContent = '0';
      statChunks.textContent = '0';
      statBuffered.textContent = '0';
      statElapsed.textContent = '-';
      break;

    case 'configured':
      if (msg.config) {
        chunkSizeSlider.value = msg.config.chunk_size.toString();
        chunkSizeVal.textContent = msg.config.chunk_size.toString();
        overlapSlider.value = msg.config.overlap.toString();
        overlapVal.textContent = msg.config.overlap.toString();
        confSlider.value = msg.config.conf_thre.toString();
        confVal.textContent = msg.config.conf_thre.toString();
      }
      break;

    case 'chunk_error':
      progressContainer.classList.add('hidden');
      setStatus('error', `Chunk ${msg.chunk_id} error: ${msg.error}`);
      // Auto-recover status after 3s
      setTimeout(() => { if (ws?.readyState === WebSocket.OPEN) setStatus('connected'); }, 3000);
      break;

    case 'chunk_skipped':
      progressContainer.classList.add('hidden');
      setStatus('connected', `Chunk skipped: ${msg.reason}`);
      setTimeout(() => { if (ws?.readyState === WebSocket.OPEN) setStatus('connected', 'Connected'); }, 3000);
      break;
  }
}

function sendConfig() {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({
    type: 'configure',
    chunk_size: parseInt(chunkSizeSlider.value),
    overlap: parseInt(overlapSlider.value),
    conf_thre: parseFloat(confSlider.value),
  }));
}

function debouncedSendConfig() {
  if (configDebounceTimer) clearTimeout(configDebounceTimer);
  configDebounceTimer = setTimeout(sendConfig, 500);
}

function resetView() {
  camera.position.set(0, 2, 5);
  controls.target.set(0, 0, 0);
  controls.update();
}

// ════════════════════════════════════════
// Video Upload
// ════════════════════════════════════════

async function uploadVideo() {
  const file = videoFileInput.files?.[0];
  if (!file) return;

  uploadBtn.disabled = true;
  uploadStatus.textContent = 'Uploading...';

  const formData = new FormData();
  formData.append('file', file);

  const fps = parseFloat(videoFpsInput.value) || 2;
  const fast = videoFastInput.checked;

  try {
    const response = await fetch(`${API_URL}/upload-video?fps=${fps}&fast=${fast}`, {
      method: 'POST',
      body: formData,
    });
    const result = await response.json();
    uploadStatus.textContent = `Feeding: ${result.filename} @ ${result.fps} fps`;
    stopVideoBtn.disabled = false;
  } catch (err) {
    uploadStatus.textContent = `Error: ${(err as Error).message}`;
    uploadBtn.disabled = false;
  }
}

async function stopVideo() {
  try {
    await fetch(`${API_URL}/stop-video`, { method: 'POST' });
    uploadStatus.textContent = 'Stopped';
    stopVideoBtn.disabled = true;
    uploadBtn.disabled = false;
  } catch (err) {
    uploadStatus.textContent = `Error: ${(err as Error).message}`;
  }
}

// ════════════════════════════════════════
// Event Listeners
// ════════════════════════════════════════

connectBtn.addEventListener('click', connect);
disconnectBtn.addEventListener('click', disconnect);
resetBtn.addEventListener('click', () => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'reset' }));
  }
});
resetViewBtn.addEventListener('click', resetView);

settingsBtn.addEventListener('click', () => {
  settingsPanel.classList.toggle('hidden');
});

uploadToggleBtn.addEventListener('click', () => {
  uploadPanel.classList.toggle('hidden');
});

// Point size
pointSizeSlider.addEventListener('input', () => {
  const v = parseFloat(pointSizeSlider.value);
  pointSizeVal.textContent = v.toString();
  material.size = v;
});

// Config sliders
chunkSizeSlider.addEventListener('input', () => {
  chunkSizeVal.textContent = chunkSizeSlider.value;
  debouncedSendConfig();
});
overlapSlider.addEventListener('input', () => {
  overlapVal.textContent = overlapSlider.value;
  debouncedSendConfig();
});
confSlider.addEventListener('input', () => {
  confVal.textContent = parseFloat(confSlider.value).toFixed(2);
  debouncedSendConfig();
});

// Grid toggle
toggleGridBtn.addEventListener('click', () => {
  showGrid = !showGrid;
  gridHelper.visible = showGrid;
  axesHelper.visible = showGrid;
  toggleGridBtn.textContent = showGrid ? 'ON' : 'OFF';
});

// Auto-rotate toggle
toggleRotateBtn.addEventListener('click', () => {
  autoRotate = !autoRotate;
  controls.autoRotate = autoRotate;
  controls.autoRotateSpeed = 1.0;
  toggleRotateBtn.textContent = autoRotate ? 'ON' : 'OFF';
});

// Orientation controls
const rotXSlider = document.getElementById('rot-x') as HTMLInputElement;
const rotXVal = document.getElementById('rot-x-val')!;
const rotYSlider = document.getElementById('rot-y') as HTMLInputElement;
const rotYVal = document.getElementById('rot-y-val')!;
const rotZSlider = document.getElementById('rot-z') as HTMLInputElement;
const rotZVal = document.getElementById('rot-z-val')!;

function applyOrientation() {
  cloudGroup.scale.set(orientFlip.x, orientFlip.y, orientFlip.z);
  const deg2rad = Math.PI / 180;
  cloudGroup.rotation.set(
    orientRot.x * deg2rad,
    orientRot.y * deg2rad,
    orientRot.z * deg2rad
  );
}

document.getElementById('flip-x')!.addEventListener('click', () => {
  orientFlip.x *= -1;
  applyOrientation();
});
document.getElementById('flip-y')!.addEventListener('click', () => {
  orientFlip.y *= -1;
  applyOrientation();
});
document.getElementById('flip-z')!.addEventListener('click', () => {
  orientFlip.z *= -1;
  applyOrientation();
});
document.getElementById('fix-opencv')!.addEventListener('click', () => {
  // Toggle between identity and OpenCV→GL fix
  if (orientFlip.y === -1 && orientFlip.z === -1) {
    orientFlip = { x: 1, y: 1, z: 1 };
  } else {
    orientFlip = { x: 1, y: -1, z: -1 };
  }
  applyOrientation();
});

// Rotation sliders
rotXSlider.addEventListener('input', () => {
  orientRot.x = parseInt(rotXSlider.value);
  rotXVal.textContent = orientRot.x + '°';
  applyOrientation();
});
rotYSlider.addEventListener('input', () => {
  orientRot.y = parseInt(rotYSlider.value);
  rotYVal.textContent = orientRot.y + '°';
  applyOrientation();
});
rotZSlider.addEventListener('input', () => {
  orientRot.z = parseInt(rotZSlider.value);
  rotZVal.textContent = orientRot.z + '°';
  applyOrientation();
});
document.getElementById('reset-rotation')!.addEventListener('click', () => {
  orientRot = { x: 0, y: 0, z: 0 };
  rotXSlider.value = '0'; rotXVal.textContent = '0°';
  rotYSlider.value = '0'; rotYVal.textContent = '0°';
  rotZSlider.value = '0'; rotZVal.textContent = '0°';
  applyOrientation();
});

// Video upload
videoFileInput.addEventListener('change', () => {
  uploadBtn.disabled = !videoFileInput.files?.length;
});
uploadBtn.addEventListener('click', uploadVideo);
stopVideoBtn.addEventListener('click', stopVideo);

// Resize
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// ════════════════════════════════════════
// Render Loop
// ════════════════════════════════════════
function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

animate();
