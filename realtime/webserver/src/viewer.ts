import './main.css';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/examples/jsm/renderers/CSS2DRenderer.js';

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

// Label panel
const labelPanel = document.getElementById('label-panel')!;
const labelToggleBtn = document.getElementById('label-toggle-btn')!;
const labelPromptsInput = document.getElementById('label-prompts') as HTMLInputElement;
const labelConfidenceInput = document.getElementById('label-confidence') as HTMLInputElement;
const labelMaxFramesInput = document.getElementById('label-max-frames') as HTMLInputElement;
const labelRunBtn = document.getElementById('label-run-btn') as HTMLButtonElement;
const labelClearBtn = document.getElementById('label-clear-btn') as HTMLButtonElement;
const toggleBoxesBtn = document.getElementById('toggle-boxes')!;
const labelStatus = document.getElementById('label-status')!;
const labelResults = document.getElementById('label-results')!;
const labelResultsList = document.getElementById('label-results-list')!;
const promptModeToggle = document.getElementById('prompt-mode-toggle') as HTMLButtonElement;
const promptModeLabel = document.getElementById('prompt-mode-label')!;

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

// CSS2D renderer for text labels
const labelRenderer = new CSS2DRenderer();
labelRenderer.setSize(window.innerWidth, window.innerHeight);
labelRenderer.domElement.style.position = 'absolute';
labelRenderer.domElement.style.top = '0';
labelRenderer.domElement.style.left = '0';
labelRenderer.domElement.style.pointerEvents = 'none';
container.appendChild(labelRenderer.domElement);

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
let showBoxes = true;
let configDebounceTimer: ReturnType<typeof setTimeout> | null = null;
let orientFlip = { x: 1, y: -1, z: -1 }; // default: OpenCV→GL correction
let orientRot = { x: 0, y: 0, z: 0 }; // rotation in degrees

// Object labeling state
interface DebugImage {
  frame_idx: number;
  mask_b64: string;
  score: number;
}

interface LabeledObject {
  label: string;
  instance_id: number;
  bbox_min: [number, number, number];
  bbox_max: [number, number, number];
  center: [number, number, number];
  extent: [number, number, number];
  rotation: number[][];   // 3x3 rotation matrix
  corners: number[][];    // 8 corner points of OBB
  confidence: number;
  point_count: number;
  debug_images: DebugImage[];
}

const labelGroup = new THREE.Group();
cloudGroup.add(labelGroup); // attach to cloudGroup so orientation transforms apply
let labeledObjects: LabeledObject[] = [];
let isLabeling = false;
let promptMode: 'direct' | 'natural' = 'direct';

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

    case 'labeled_objects':
      isLabeling = false;
      labelRunBtn.disabled = false;
      labeledObjects = msg.objects as LabeledObject[];
      renderLabeledObjects(labeledObjects);
      updateLabelResultsList(labeledObjects);
      labelStatus.textContent = `Found ${labeledObjects.length} object(s)`;
      break;

    case 'labeling_start':
      isLabeling = true;
      labelRunBtn.disabled = true;
      labelStatus.textContent = `Labeling: ${msg.prompts.join(', ')}...`;
      break;

    case 'labeling_error':
      isLabeling = false;
      labelRunBtn.disabled = false;
      labelStatus.textContent = `Error: ${msg.error}`;
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
// Object Labeling
// ════════════════════════════════════════

// Color palette for bounding box classes
const LABEL_COLORS: { [key: string]: string } = {};
const PALETTE = [
  '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
  '#F7DC6F', '#BB8FCE', '#85C1E9', '#F0B27A', '#82E0AA',
  '#D7BDE2', '#F8C471', '#76D7C4', '#F1948A', '#AED6F1',
];
let colorIdx = 0;

function getLabelColor(label: string): string {
  if (!LABEL_COLORS[label]) {
    LABEL_COLORS[label] = PALETTE[colorIdx % PALETTE.length];
    colorIdx++;
  }
  return LABEL_COLORS[label];
}

function clearLabels() {
  // Remove all children from label group
  while (labelGroup.children.length > 0) {
    const child = labelGroup.children[0];
    if (child instanceof CSS2DObject) {
      child.element.remove();
    }
    labelGroup.remove(child);
  }
  labeledObjects = [];
  labelResults.classList.add('hidden');
  labelResultsList.innerHTML = '';
  labelStatus.textContent = '';
}

function renderLabeledObjects(objects: LabeledObject[]) {
  // Clear previous labels
  while (labelGroup.children.length > 0) {
    const child = labelGroup.children[0];
    if (child instanceof CSS2DObject) {
      child.element.remove();
    }
    labelGroup.remove(child);
  }

  for (const obj of objects) {
    const color = getLabelColor(obj.label);
    const threeColor = new THREE.Color(color);

    // Corners are ordered: bottom face 0-1-2-3 (CCW), top face 4-5-6-7 (CCW)
    // Edges: bottom 0-1-2-3-0, top 4-5-6-7-4, verticals 0-4 1-5 2-6 3-7
    if (obj.corners && obj.corners.length === 8) {
      const c = obj.corners.map(p => new THREE.Vector3(p[0], p[1], p[2]));
      const edgePairs = [
        [0, 1], [1, 2], [2, 3], [3, 0],   // bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],   // top face
        [0, 4], [1, 5], [2, 6], [3, 7],   // verticals
      ];

      const linePositions: number[] = [];
      for (const [a, b] of edgePairs) {
        linePositions.push(c[a].x, c[a].y, c[a].z);
        linePositions.push(c[b].x, c[b].y, c[b].z);
      }

      const lineGeo = new THREE.BufferGeometry();
      lineGeo.setAttribute('position', new THREE.Float32BufferAttribute(linePositions, 3));
      const lineMat = new THREE.LineBasicMaterial({ color: threeColor, linewidth: 2 });
      const wireframe = new THREE.LineSegments(lineGeo, lineMat);
      labelGroup.add(wireframe);
    } else {
      // Fallback to AABB if no corners available
      const bboxMin = new THREE.Vector3(...obj.bbox_min);
      const bboxMax = new THREE.Vector3(...obj.bbox_max);
      const box3 = new THREE.Box3(bboxMin, bboxMax);
      const boxHelper = new THREE.Box3Helper(box3, threeColor);
      labelGroup.add(boxHelper);
    }

    // Text label using CSS2DObject
    const labelDiv = document.createElement('div');
    labelDiv.style.cssText = `
      background: ${color}dd;
      color: white;
      font-family: ui-monospace, monospace;
      font-size: 11px;
      font-weight: 600;
      padding: 2px 8px;
      border-radius: 4px;
      white-space: nowrap;
      pointer-events: none;
      text-shadow: 0 1px 2px rgba(0,0,0,0.5);
      box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    `;
    labelDiv.textContent = `${obj.label} (${(obj.confidence * 100).toFixed(0)}%)`;

    const labelObj = new CSS2DObject(labelDiv);
    // Position label at center + half extent upward (or top of AABB)
    labelObj.position.set(
      obj.center[0],
      obj.bbox_max[1] + 0.1,
      obj.center[2],
    );
    labelGroup.add(labelObj);
  }

  // Show debug images panel
  renderDebugImages(objects);

  labelGroup.visible = showBoxes;
}

function updateLabelResultsList(objects: LabeledObject[]) {
  labelResults.classList.remove('hidden');
  labelResultsList.innerHTML = '';

  for (const obj of objects) {
    const color = getLabelColor(obj.label);
    const item = document.createElement('div');
    item.className = 'flex items-center gap-2 text-xs text-white/70 py-0.5';
    item.innerHTML = `
      <span style="background:${color}; width:8px; height:8px; border-radius:50%; flex-shrink:0;"></span>
      <span class="flex-1">${obj.label}</span>
      <span class="text-white/40">${(obj.confidence * 100).toFixed(0)}%</span>
      <span class="text-white/40">${obj.point_count.toLocaleString()} pts</span>
    `;
    labelResultsList.appendChild(item);
  }
}

function renderDebugImages(objects: LabeledObject[]) {
  const debugContainer = document.getElementById('debug-images-container');
  const debugPanel = document.getElementById('debug-panel');
  if (!debugContainer || !debugPanel) return;

  debugContainer.innerHTML = '';

  // Collect all debug images from all objects
  let hasImages = false;
  for (const obj of objects) {
    if (!obj.debug_images || obj.debug_images.length === 0) continue;
    hasImages = true;
    const color = getLabelColor(obj.label);

    const objSection = document.createElement('div');
    objSection.className = 'mb-3';
    objSection.innerHTML = `
      <div class="flex items-center gap-2 mb-1">
        <span style="background:${color}; width:8px; height:8px; border-radius:50%; flex-shrink:0;"></span>
        <span class="text-xs text-white/80 font-semibold">${obj.label}</span>
        <span class="text-xs text-white/40">${(obj.confidence * 100).toFixed(0)}%</span>
      </div>
    `;

    const imgGrid = document.createElement('div');
    imgGrid.className = 'grid grid-cols-2 gap-1';

    for (const dbg of obj.debug_images) {
      const wrapper = document.createElement('div');
      wrapper.className = 'relative';
      wrapper.innerHTML = `
        <img src="data:image/png;base64,${dbg.mask_b64}"
             class="w-full rounded border border-white/10 cursor-pointer hover:border-white/40 transition"
             alt="Frame ${dbg.frame_idx}" />
        <div class="absolute bottom-0 left-0 right-0 bg-black/60 text-[9px] text-white/70 px-1 py-0.5">
          F${dbg.frame_idx} · ${(dbg.score * 100).toFixed(0)}%
        </div>
      `;
      // Click to enlarge
      wrapper.querySelector('img')!.addEventListener('click', () => {
        showFullscreenImage(dbg.mask_b64, `${obj.label} — Frame ${dbg.frame_idx} (${(dbg.score * 100).toFixed(0)}%)`);
      });
      imgGrid.appendChild(wrapper);
    }

    objSection.appendChild(imgGrid);
    debugContainer.appendChild(objSection);
  }

  if (hasImages) {
    debugPanel.classList.remove('hidden');
  } else {
    debugPanel.classList.add('hidden');
  }
}

function showFullscreenImage(base64: string, caption: string) {
  // Create fullscreen overlay for image inspection
  const overlay = document.createElement('div');
  overlay.style.cssText = `
    position: fixed; top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(0,0,0,0.85); z-index: 10000;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    cursor: pointer;
  `;
  overlay.innerHTML = `
    <img src="data:image/png;base64,${base64}" style="max-width: 90vw; max-height: 80vh; border-radius: 8px;" />
    <div style="color: white; font-size: 14px; margin-top: 12px; font-family: monospace;">${caption}</div>
    <div style="color: white/50; font-size: 11px; margin-top: 4px;">Click anywhere to close</div>
  `;
  overlay.addEventListener('click', () => overlay.remove());
  document.body.appendChild(overlay);
}

async function runLabeling() {
  const promptsStr = labelPromptsInput.value.trim();
  if (!promptsStr) {
    labelStatus.textContent = 'Enter object names or description first';
    return;
  }

  labelRunBtn.disabled = true;
  let prompts: string[];

  // If in natural language mode, first parse with OpenAI
  if (promptMode === 'natural') {
    labelStatus.textContent = 'Parsing with AI...';
    try {
      const nlResponse = await fetch(`${API_URL}/parse-nl-prompt`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: promptsStr }),
      });
      if (!nlResponse.ok) {
        const err = await nlResponse.json();
        labelStatus.textContent = `AI parse error: ${err.error || nlResponse.statusText}`;
        labelRunBtn.disabled = false;
        return;
      }
      const nlResult = await nlResponse.json();
      prompts = nlResult.objects as string[];
      if (!prompts || prompts.length === 0) {
        labelStatus.textContent = 'AI could not extract any objects from that description';
        labelRunBtn.disabled = false;
        return;
      }
      labelStatus.textContent = `AI extracted: ${prompts.join(', ')}. Sending labeling request...`;
    } catch (err) {
      labelStatus.textContent = `AI error: ${(err as Error).message}`;
      labelRunBtn.disabled = false;
      return;
    }
  } else {
    prompts = promptsStr.split(',').map(s => s.trim()).filter(Boolean);
  }

  const confidence = parseFloat(labelConfidenceInput.value) || 0.3;
  const maxFrames = parseInt(labelMaxFramesInput.value) || 20;

  labelStatus.textContent = 'Sending labeling request...';

  try {
    const response = await fetch(`${API_URL}/label-objects`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompts,
        confidence,
        max_frames: maxFrames,
      }),
    });

    if (response.status === 202) {
      labelStatus.textContent = `Labeling: ${prompts.join(', ')} (running in background)...`;
      return;
    }

    if (response.status === 409) {
      labelStatus.textContent = 'Labeling already in progress';
      labelRunBtn.disabled = false;
      return;
    }

    if (!response.ok) {
      const err = await response.json();
      labelStatus.textContent = `Error: ${err.error || response.statusText}`;
      labelRunBtn.disabled = false;
      return;
    }

    // Fallback: direct response
    const result = await response.json();
    if (result.objects) {
      labeledObjects = result.objects as LabeledObject[];
      renderLabeledObjects(labeledObjects);
      updateLabelResultsList(labeledObjects);
      labelStatus.textContent = `Found ${labeledObjects.length} object(s)`;
    }
    labelRunBtn.disabled = false;
  } catch (err) {
    labelStatus.textContent = `Error: ${(err as Error).message}`;
    labelRunBtn.disabled = false;
  }
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

// Label panel toggle & controls
labelToggleBtn.addEventListener('click', () => {
  labelPanel.classList.toggle('hidden');
});

labelRunBtn.addEventListener('click', runLabeling);
labelClearBtn.addEventListener('click', clearLabels);

// Prompt mode toggle (Direct vs Natural Language)
promptModeToggle.addEventListener('click', () => {
  if (promptMode === 'direct') {
    promptMode = 'natural';
    promptModeToggle.textContent = 'Natural Language';
    promptModeToggle.classList.remove('btn-secondary');
    promptModeToggle.classList.add('btn-accent');
    promptModeLabel.textContent = 'Describe what you want to find';
    labelPromptsInput.placeholder = 'e.g. I want to find the shelves so we can place boxes into them...';
  } else {
    promptMode = 'direct';
    promptModeToggle.textContent = 'Direct';
    promptModeToggle.classList.remove('btn-accent');
    promptModeToggle.classList.add('btn-secondary');
    promptModeLabel.textContent = 'Object prompts (comma-separated)';
    labelPromptsInput.placeholder = 'chair, table, monitor, person...';
  }
});

// Debug panel close
document.getElementById('debug-close-btn')?.addEventListener('click', () => {
  document.getElementById('debug-panel')?.classList.add('hidden');
});

toggleBoxesBtn.addEventListener('click', () => {
  showBoxes = !showBoxes;
  labelGroup.visible = showBoxes;
  toggleBoxesBtn.textContent = showBoxes ? 'ON' : 'OFF';
});

// Resize
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  labelRenderer.setSize(window.innerWidth, window.innerHeight);
});

// ════════════════════════════════════════
// Render Loop
// ════════════════════════════════════════
function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
  labelRenderer.render(scene, camera);
}

animate();
