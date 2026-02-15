import './main.css';

// ════════════════════════════════════════
// Constants
// ════════════════════════════════════════
const SERVER_HOST = window.location.hostname;
const SERVER_PORT = '5000';
const WS_URL = `wss://${SERVER_HOST}:${SERVER_PORT}/ws/sender`;

// ════════════════════════════════════════
// DOM
// ════════════════════════════════════════
const video = document.getElementById('video') as HTMLVideoElement;
const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const ctx = canvas.getContext('2d')!;
const statusDot = document.getElementById('status-dot')!;
const statusText = document.getElementById('status-text')!;
const fpsText = document.getElementById('fps-text')!;
const frameCount = document.getElementById('frame-count')!;
const recordingIndicator = document.getElementById('recording-indicator')!;
const startBtn = document.getElementById('start-btn') as HTMLButtonElement;
const stopBtn = document.getElementById('stop-btn') as HTMLButtonElement;
const sendFpsSlider = document.getElementById('send-fps') as HTMLInputElement;
const sendFpsVal = document.getElementById('send-fps-val')!;
const fpsControl = document.getElementById('fps-control')!;
const cameraSelectContainer = document.getElementById('camera-select-container')!;
const cameraSelect = document.getElementById('camera-select') as HTMLSelectElement;

// ════════════════════════════════════════
// State
// ════════════════════════════════════════
let ws: WebSocket | null = null;
let streaming = false;
let streamInterval: ReturnType<typeof setInterval> | null = null;
let framesSent = 0;
let fpsCounter = 0;
let lastFpsTime = Date.now();
let currentStream: MediaStream | null = null;

// ════════════════════════════════════════
// Camera
// ════════════════════════════════════════

async function enumerateCameras() {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const cameras = devices.filter(d => d.kind === 'videoinput');
    
    cameraSelect.innerHTML = '';
    cameras.forEach((cam, i) => {
      const opt = document.createElement('option');
      opt.value = cam.deviceId;
      opt.textContent = cam.label || `Camera ${i + 1}`;
      cameraSelect.appendChild(opt);
    });

    if (cameras.length > 1) {
      cameraSelectContainer.classList.remove('hidden');
    }
  } catch (e) {
    console.error('Could not enumerate cameras:', e);
  }
}

async function initCamera(deviceId?: string) {
  try {
    // Stop existing stream
    if (currentStream) {
      currentStream.getTracks().forEach(t => t.stop());
    }

    const constraints: MediaStreamConstraints = {
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        ...(deviceId ? { deviceId: { exact: deviceId } } : { facingMode: 'environment' }),
      },
    };

    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    currentStream = stream;
    video.srcObject = stream;
    await video.play();

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    console.log(`Camera: ${video.videoWidth}x${video.videoHeight}`);
    updateStatus('disconnected', 'Camera ready');
    
    // Re-enumerate after getting permission (labels become available)
    await enumerateCameras();
  } catch (err) {
    console.error('Camera error:', err);
    updateStatus('error', `Camera error: ${(err as Error).message}`);
  }
}

// ════════════════════════════════════════
// Status
// ════════════════════════════════════════

function updateStatus(state: 'connected' | 'disconnected' | 'connecting' | 'error', text?: string) {
  statusDot.className = 'status-dot ' + state;
  statusText.textContent = text ?? ({
    connected: 'Connected',
    disconnected: 'Not Connected',
    connecting: 'Connecting...',
    error: 'Error',
  }[state]);
}

function updateFps() {
  fpsCounter++;
  const now = Date.now();
  const elapsed = now - lastFpsTime;
  if (elapsed >= 1000) {
    const fps = (fpsCounter / (elapsed / 1000)).toFixed(1);
    fpsText.textContent = `${fps} fps`;
    fpsText.classList.remove('hidden');
    fpsCounter = 0;
    lastFpsTime = now;
  }
}

// ════════════════════════════════════════
// Frame Capture
// ════════════════════════════════════════

function sendFrame() {
  if (!streaming || !ws || ws.readyState !== WebSocket.OPEN) return;

  try {
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const base64 = canvas.toDataURL('image/jpeg', 0.75).split(',')[1];

    ws.send(JSON.stringify({
      type: 'frame',
      image: base64,
      timestamp: Date.now() / 1000,
    }));

    framesSent++;
    frameCount.textContent = `${framesSent} frames`;
    updateFps();
  } catch (err) {
    console.error('Frame send error:', err);
  }
}

// ════════════════════════════════════════
// Connection
// ════════════════════════════════════════

function startStreaming() {
  if (streaming) return;
  
  updateStatus('connecting');
  ws = new WebSocket(WS_URL);

  ws.onopen = () => {
    updateStatus('connected', 'Connected - Streaming');
    streaming = true;
    startBtn.disabled = true;
    stopBtn.disabled = false;
    fpsControl.classList.remove('hidden');
    recordingIndicator.classList.remove('hidden');

    const fps = parseInt(sendFpsSlider.value);
    const interval = Math.round(1000 / fps);
    streamInterval = setInterval(sendFrame, interval);
  };

  ws.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);
      if (msg.type === 'frame_ack') {
        if (msg.queue_full) {
          console.warn('Server queue full');
        }
      }
    } catch {}
  };

  ws.onclose = () => {
    updateStatus('disconnected');
    stopStreamingInternal();
  };

  ws.onerror = () => {
    updateStatus('error', 'Connection failed');
    stopStreamingInternal();
  };
}

function stopStreaming() {
  stopStreamingInternal();
  if (ws) {
    ws.close();
    ws = null;
  }
}

function stopStreamingInternal() {
  streaming = false;
  if (streamInterval) {
    clearInterval(streamInterval);
    streamInterval = null;
  }
  startBtn.disabled = false;
  stopBtn.disabled = true;
  fpsText.classList.add('hidden');
  recordingIndicator.classList.add('hidden');
  updateStatus('disconnected', 'Stopped');
}

// ════════════════════════════════════════
// Event Listeners
// ════════════════════════════════════════

startBtn.addEventListener('click', startStreaming);
stopBtn.addEventListener('click', stopStreaming);

sendFpsSlider.addEventListener('input', () => {
  const fps = parseInt(sendFpsSlider.value);
  sendFpsVal.textContent = fps.toString();

  // Update interval if streaming
  if (streaming && streamInterval) {
    clearInterval(streamInterval);
    streamInterval = setInterval(sendFrame, Math.round(1000 / fps));
  }
});

cameraSelect.addEventListener('change', () => {
  initCamera(cameraSelect.value);
});

// ════════════════════════════════════════
// Init
// ════════════════════════════════════════

window.addEventListener('DOMContentLoaded', () => {
  initCamera();
});
