"""
Pi3X Real-Time Streaming Server
FastAPI + WebSocket server for incremental 3D reconstruction.
"""

import asyncio
import base64
import json
import queue
import ssl
import struct
import threading
import time
import os
import sys
import argparse
import tempfile

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from pipeline import IncrementalPi3

# ─────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────
app = FastAPI(title="Pi3X Real-Time Reconstruction")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ─────────────────────────────────────────
# Global State
# ─────────────────────────────────────────
pipeline: IncrementalPi3 = None
frame_queue: queue.Queue = queue.Queue(maxsize=60)
connected_viewers: set[WebSocket] = set()
processing_lock = threading.Lock()
is_processing = False
video_feeder_stop = threading.Event()


# ─────────────────────────────────────────
# WebSocket broadcast helper
# ─────────────────────────────────────────
async def broadcast_to_viewers(message: dict):
    """Send JSON message to all connected viewer WebSockets."""
    dead = set()
    for ws in connected_viewers:
        try:
            await ws.send_json(message)
        except Exception:
            dead.add(ws)
    connected_viewers.difference_update(dead)


async def broadcast_binary(data: bytes):
    """Send binary data to all connected viewer WebSockets."""
    dead = set()
    for ws in connected_viewers:
        try:
            await ws.send_bytes(data)
        except Exception:
            dead.add(ws)
    connected_viewers.difference_update(dead)


def encode_point_cloud_binary(points: np.ndarray, colors: np.ndarray, chunk_id: int, total: int, elapsed: float) -> bytes:
    """
    Encode point cloud as binary for efficient transfer.
    Format:
        Header: chunk_id(i32) + total_points(i32) + num_new(i32) + elapsed(f32) = 16 bytes
        Body:   N * (x f32 + y f32 + z f32 + r u8 + g u8 + b u8) = N * 15 bytes
    """
    n = len(points)
    header = struct.pack('<iiif', chunk_id, total, n, elapsed)

    if n == 0:
        return header

    pts = points.astype(np.float32)
    cols = colors.astype(np.uint8)

    # Interleave: for each point, 12 bytes xyz + 3 bytes rgb
    body = bytearray(n * 15)
    for i in range(n):
        offset = i * 15
        struct.pack_into('<fff', body, offset, pts[i, 0], pts[i, 1], pts[i, 2])
        body[offset + 12] = cols[i, 0]
        body[offset + 13] = cols[i, 1]
        body[offset + 14] = cols[i, 2]

    return header + bytes(body)


def encode_point_cloud_binary_fast(points: np.ndarray, colors: np.ndarray, chunk_id: int, total: int, elapsed: float) -> bytes:
    """Fast version: header + flat xyz floats + flat rgb bytes."""
    n = len(points)
    header = struct.pack('<iiif', chunk_id, total, n, elapsed)

    if n == 0:
        return header

    pts = np.ascontiguousarray(points.astype(np.float32))   # (N, 3)
    cols = np.ascontiguousarray(colors.astype(np.uint8))     # (N, 3)

    # Layout: header(16) + positions(N*12 f32) + colors(N*3 u8)
    return header + pts.tobytes() + cols.tobytes()


# ─────────────────────────────────────────
# Processing Loop (runs in background thread)
# ─────────────────────────────────────────
def processing_loop(loop: asyncio.AbstractEventLoop):
    """Background thread that pulls frames and runs Pi3X inference."""
    global is_processing

    print("[Server] Processing loop started")

    while True:
        try:
            frame_bgr = frame_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        with processing_lock:
            ready = pipeline.add_frame(frame_bgr)

            remaining = pipeline.frames_until_ready()
            status_msg = {
                'type': 'status',
                'frames_buffered': len(pipeline.frame_buffer),
                'frames_needed': remaining,
                'total_points': pipeline.total_points,
                'chunks_processed': pipeline.chunks_processed,
            }
            asyncio.run_coroutine_threadsafe(broadcast_to_viewers(status_msg), loop)

            if ready:
                is_processing = True
                processing_msg = {'type': 'processing', 'chunk_id': pipeline.chunks_processed}
                asyncio.run_coroutine_threadsafe(broadcast_to_viewers(processing_msg), loop)

                try:
                    result = pipeline.process_chunk()
                except Exception as e:
                    print(f"[Server] ERROR in process_chunk: {e}")
                    import traceback; traceback.print_exc()
                    result = None
                    # Send error to viewers
                    err_msg = {'type': 'chunk_error', 'error': str(e), 'chunk_id': pipeline.chunks_processed}
                    asyncio.run_coroutine_threadsafe(broadcast_to_viewers(err_msg), loop)

                is_processing = False

                if result is not None and len(result['points']) > 0:
                    # Auto-save to disk
                    try:
                        pipeline.save_to_disk()
                    except Exception:
                        pass

                    # Downsample for streaming if needed
                    pts = result['points']
                    cols = result['colors']
                    if len(pts) > 100_000:
                        stride = len(pts) // 100_000
                        pts = pts[::stride]
                        cols = cols[::stride]

                    binary_data = encode_point_cloud_binary_fast(
                        pts, cols,
                        result['chunk_id'],
                        result['total_points'],
                        result['elapsed']
                    )
                    asyncio.run_coroutine_threadsafe(broadcast_binary(binary_data), loop)

                    done_msg = {
                        'type': 'chunk_done',
                        'chunk_id': result['chunk_id'],
                        'new_points': len(result['points']),
                        'total_points': result['total_points'],
                        'elapsed': result['elapsed'],
                    }
                    asyncio.run_coroutine_threadsafe(broadcast_to_viewers(done_msg), loop)
                elif result is not None:
                    # Chunk produced 0 valid points — notify viewer
                    skip_msg = {
                        'type': 'chunk_skipped',
                        'chunk_id': result['chunk_id'],
                        'total_points': result['total_points'],
                        'reason': 'No valid points produced (scene may have changed dramatically)',
                    }
                    asyncio.run_coroutine_threadsafe(broadcast_to_viewers(skip_msg), loop)


# ─────────────────────────────────────────
# Video file feeder (testing mode)
# ─────────────────────────────────────────
def feed_video_file(video_path: str, target_fps: float = 2.0, fast: bool = False):
    """Read a video file and push frames into the frame_queue."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[VideoFeeder] Cannot open: {video_path}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip = max(1, int(round(video_fps / target_fps)))
    effective_fps = video_fps / skip
    delay = 0.0 if fast else (1.0 / effective_fps)

    print(f"[VideoFeeder] {video_path}: {total_frames} frames @ {video_fps:.1f} FPS")
    print(f"[VideoFeeder] Feeding every {skip} frame(s) → ~{effective_fps:.1f} effective FPS")

    idx = 0
    fed = 0

    while not video_feeder_stop.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        if idx % skip != 0:
            continue

        if not frame_queue.full():
            frame_queue.put(frame)
            fed += 1
        else:
            pass  # drop frame (backpressure)

        if delay > 0:
            time.sleep(delay)

    cap.release()
    print(f"[VideoFeeder] Done: fed {fed} frames")


# ─────────────────────────────────────────
# HTTP Endpoints
# ─────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "gpu": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "total_points": pipeline.total_points if pipeline else 0,
        "chunks_processed": pipeline.chunks_processed if pipeline else 0,
    }


@app.post("/reset")
async def reset():
    """Reset the pipeline state."""
    with processing_lock:
        # Drain frame queue  
        while not frame_queue.empty():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                break
        pipeline.reset()

    await broadcast_to_viewers({'type': 'reset'})
    return {"status": "reset_complete"}


@app.post("/configure")
async def configure(config: dict):
    """Hot-reconfigure pipeline parameters."""
    with processing_lock:
        pipeline.configure(
            chunk_size=config.get('chunk_size'),
            overlap=config.get('overlap'),
            conf_thre=config.get('conf_thre'),
        )
    return {
        "status": "configured",
        "chunk_size": pipeline.chunk_size,
        "overlap": pipeline.overlap,
        "conf_thre": pipeline.conf_thre,
    }


@app.get("/config")
async def get_config():
    """Get current pipeline configuration."""
    return {
        "chunk_size": pipeline.chunk_size,
        "overlap": pipeline.overlap,
        "conf_thre": pipeline.conf_thre,
    }


@app.get("/cloud")
async def get_full_cloud():
    """Get the full accumulated point cloud (downsampled)."""
    cloud = pipeline.get_global_cloud(max_points=200_000)
    # Return as binary
    pts = cloud['points']
    cols = cloud['colors']
    return JSONResponse({
        "total_points": pipeline.total_points,
        "returned_points": len(pts),
        "chunks_processed": pipeline.chunks_processed,
    })


@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...), fps: float = 2.0, fast: bool = True):
    """Upload a video file for testing (simulates live camera feed)."""
    video_feeder_stop.clear()

    # Save uploaded file
    tmp = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    content = await file.read()
    tmp.write(content)
    tmp.flush()
    tmp_path = tmp.name
    tmp.close()

    # Reset pipeline
    with processing_lock:
        while not frame_queue.empty():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                break
        pipeline.reset()

    await broadcast_to_viewers({'type': 'reset'})

    # Start feeding in background
    t = threading.Thread(target=feed_video_file, args=(tmp_path, fps, fast), daemon=True)
    t.start()

    return {
        "status": "feeding",
        "filename": file.filename,
        "fps": fps,
        "fast": fast,
    }


@app.post("/stop-video")
async def stop_video():
    """Stop the video feeder."""
    video_feeder_stop.set()
    return {"status": "stopped"}


# ─────────────────────────────────────────
# WebSocket Endpoints
# ─────────────────────────────────────────
@app.websocket("/ws/viewer")
async def viewer_ws(ws: WebSocket):
    """Viewer WebSocket: receives point cloud updates."""
    await ws.accept()
    connected_viewers.add(ws)
    print(f"[Server] Viewer connected ({len(connected_viewers)} total)")

    # Send current state
    await ws.send_json({
        'type': 'init',
        'total_points': pipeline.total_points,
        'chunks_processed': pipeline.chunks_processed,
        'config': {
            'chunk_size': pipeline.chunk_size,
            'overlap': pipeline.overlap,
            'conf_thre': pipeline.conf_thre,
        }
    })

    # If we already have a cloud, send it
    if pipeline.total_points > 0:
        cloud = pipeline.get_global_cloud(max_points=100_000)
        binary = encode_point_cloud_binary_fast(
            cloud['points'], cloud['colors'],
            pipeline.chunks_processed - 1,
            pipeline.total_points,
            0.0
        )
        await ws.send_bytes(binary)

    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)

            if msg.get('type') == 'configure':
                with processing_lock:
                    pipeline.configure(
                        chunk_size=msg.get('chunk_size'),
                        overlap=msg.get('overlap'),
                        conf_thre=msg.get('conf_thre'),
                    )
                await ws.send_json({
                    'type': 'configured',
                    'config': {
                        'chunk_size': pipeline.chunk_size,
                        'overlap': pipeline.overlap,
                        'conf_thre': pipeline.conf_thre,
                    }
                })
            elif msg.get('type') == 'reset':
                with processing_lock:
                    while not frame_queue.empty():
                        try:
                            frame_queue.get_nowait()
                        except queue.Empty:
                            break
                    pipeline.reset()
                await broadcast_to_viewers({'type': 'reset'})

            elif msg.get('type') == 'get_cloud':
                cloud = pipeline.get_global_cloud(max_points=100_000)
                binary = encode_point_cloud_binary_fast(
                    cloud['points'], cloud['colors'],
                    pipeline.chunks_processed - 1 if pipeline.chunks_processed > 0 else 0,
                    pipeline.total_points,
                    0.0
                )
                await ws.send_bytes(binary)

    except WebSocketDisconnect:
        pass
    finally:
        connected_viewers.discard(ws)
        print(f"[Server] Viewer disconnected ({len(connected_viewers)} total)")


@app.websocket("/ws/sender")
async def sender_ws(ws: WebSocket):
    """Sender WebSocket: receives camera frames as base64 JPEG."""
    await ws.accept()
    print("[Server] Sender connected")
    frames_received = 0

    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)

            if msg.get('type') == 'frame':
                img_b64 = msg['image']
                img_bytes = base64.b64decode(img_b64)
                img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
                frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

                if frame is not None and not frame_queue.full():
                    frame_queue.put(frame)
                    frames_received += 1

                await ws.send_json({
                    'type': 'frame_ack',
                    'frame': frames_received,
                    'queue_size': frame_queue.qsize(),
                    'queue_full': frame_queue.full(),
                })

    except WebSocketDisconnect:
        pass
    finally:
        print(f"[Server] Sender disconnected (received {frames_received} frames)")


# ─────────────────────────────────────────
# Static file serving
# ─────────────────────────────────────────
webserver_dir = os.path.join(os.path.dirname(__file__), 'webserver', 'dist')
if os.path.exists(webserver_dir):
    app.mount("/", StaticFiles(directory=webserver_dir, html=True), name="static")
else:
    # Fallback: serve from webserver root during dev
    dev_dir = os.path.join(os.path.dirname(__file__), 'webserver')
    if os.path.exists(dev_dir):
        @app.get("/")
        async def serve_index():
            return FileResponse(os.path.join(dev_dir, 'index.html'))

        @app.get("/viewer.html")
        async def serve_viewer():
            return FileResponse(os.path.join(dev_dir, 'viewer.html'))

        @app.get("/sender.html")
        async def serve_sender():
            return FileResponse(os.path.join(dev_dir, 'sender.html'))


# ─────────────────────────────────────────
# Startup
# ─────────────────────────────────────────
@app.on_event("startup")
async def startup():
    loop = asyncio.get_event_loop()
    t = threading.Thread(target=processing_loop, args=(loop,), daemon=True)
    t.start()
    print("[Server] Background processing thread started")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pi3X Real-Time Server')
    parser.add_argument('--port', type=int, default=5000, help='Server port')
    parser.add_argument('--ckpt', type=str, default=None, help='Model checkpoint path')
    parser.add_argument('--video', type=str, default=None, help='Video file for testing')
    parser.add_argument('--video-fps', type=float, default=2.0, help='Effective FPS for video')
    parser.add_argument('--fast', action='store_true', help='Feed video as fast as possible')
    parser.add_argument('--chunk-size', type=int, default=16, help='Chunk size')
    parser.add_argument('--overlap', type=int, default=6, help='Overlap size')
    args = parser.parse_args()

    # Initialize pipeline
    pipeline = IncrementalPi3(device='cuda' if torch.cuda.is_available() else 'cpu')
    pipeline.load_model(ckpt=args.ckpt)
    pipeline.configure(chunk_size=args.chunk_size, overlap=args.overlap)

    # Try to load persisted point cloud
    try:
        if pipeline.load_from_disk():
            print(f"[Server] Loaded {pipeline.total_points} persisted points from disk")
    except Exception as e:
        print(f"[Server] Could not load persisted cloud: {e}")

    # Start video feeder if provided
    if args.video:
        if not os.path.isfile(args.video):
            print(f"Video file not found: {args.video}")
            sys.exit(1)
        t = threading.Thread(target=feed_video_file, args=(args.video, args.video_fps, args.fast), daemon=True)
        t.start()

    # SSL
    cert_file = os.path.join(os.path.dirname(__file__), 'webserver', 'server.cert')
    key_file = os.path.join(os.path.dirname(__file__), 'webserver', 'server.key')

    ssl_config = {}
    if os.path.exists(cert_file) and os.path.exists(key_file):
        ssl_config = {
            'ssl_certfile': cert_file,
            'ssl_keyfile': key_file,
        }
        print(f"[Server] HTTPS enabled")
    else:
        print(f"[Server] No SSL certs found, running HTTP only")

    print("=" * 60)
    print("  Pi3X Real-Time Reconstruction Server")
    print("=" * 60)
    print(f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  Port: {args.port}")
    print(f"  Chunk: {pipeline.chunk_size} frames, Overlap: {pipeline.overlap}")
    if args.video:
        print(f"  Video: {args.video} ({args.video_fps} fps, {'fast' if args.fast else 'realtime'})")
    print(f"  Frontend: https://0.0.0.0:{args.port}")
    print("=" * 60)

    uvicorn.run(app, host='0.0.0.0', port=args.port, **ssl_config)
