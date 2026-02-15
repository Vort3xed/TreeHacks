"""
Incremental Pi3X Pipeline
Wraps the Pi3X model for chunk-at-a-time processing with Sim3 alignment.
Adapted from pi3.pipe.pi3x_vo.Pi3XVO for streaming use.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import time
import traceback
from PIL import Image
from typing import Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pi3.utils.geometry import homogenize_points, depth_edge
from pi3.models.pi3x import Pi3X


PIXEL_LIMIT = 255_000
PATCH_SIZE = 14


def compute_target_size(w: int, h: int) -> tuple[int, int]:
    """Compute target size: multiples of 14, within PIXEL_LIMIT."""
    scale = (PIXEL_LIMIT / (w * h)) ** 0.5
    k = round(w * scale / PATCH_SIZE)
    m = round(h * scale / PATCH_SIZE)
    while k * PATCH_SIZE * m * PATCH_SIZE > PIXEL_LIMIT:
        if k > m:
            k -= 1
        else:
            m -= 1
    return k * PATCH_SIZE, m * PATCH_SIZE


def preprocess_frame(frame_bgr: np.ndarray, target_w: int, target_h: int) -> torch.Tensor:
    """BGR numpy frame -> (3, H, W) float [0,1] tensor."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    pil = pil.resize((target_w, target_h), Image.Resampling.LANCZOS)
    arr = np.array(pil).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)


class IncrementalPi3:
    """
    Streaming incremental reconstruction using Pi3X.
    
    Usage:
        pipe = IncrementalPi3(device='cuda')
        pipe.load_model()
        
        for frame in camera_stream:
            ready = pipe.add_frame(frame)
            if ready:
                delta = pipe.process_chunk()
                # delta has new points + colors
    """

    def __init__(self, device: str = 'cuda'):
        self.device = torch.device(device)
        self.model: Optional[Pi3X] = None
        self.dtype = torch.bfloat16

        # Configuration (hot-reconfigurable)
        self.chunk_size = 16
        self.overlap = 6
        self.conf_thre = 0.05
        self.inject_conditions = ['pose', 'depth', 'ray']

        # Frame buffer
        self.frame_buffer: list[torch.Tensor] = []  # list of (3, H, W) tensors
        self.target_w: Optional[int] = None
        self.target_h: Optional[int] = None

        # Overlap state from previous chunk
        self._prev_overlap_imgs: Optional[torch.Tensor] = None   # (overlap, 3, H, W)
        self._prev_overlap_pts: Optional[torch.Tensor] = None    # (1, overlap, H, W, 3)
        self._prev_overlap_mask: Optional[torch.Tensor] = None   # (1, overlap, H, W)
        self._prev_overlap_poses: Optional[torch.Tensor] = None  # (1, overlap, 4, 4)
        self._prev_overlap_depth: Optional[torch.Tensor] = None  # (1, overlap, H, W)
        self._prev_overlap_conf: Optional[torch.Tensor] = None   # (1, overlap, H, W)
        self._prev_overlap_rays: Optional[torch.Tensor] = None   # (1, overlap, H, W, 3)

        # Global accumulated cloud (CPU)
        self.global_points: list[np.ndarray] = []   # list of (N, 3) arrays
        self.global_colors: list[np.ndarray] = []   # list of (N, 3) arrays (0-255 uint8)
        self.total_points = 0
        self.chunks_processed = 0
        self.is_first_chunk = True

        # Per-frame data for object labeling (stored on CPU)
        self.frame_data: list[dict] = []  # list of {image, point_map, conf_mask}
        self._global_frame_idx = 0

    def load_model(self, ckpt: Optional[str] = None):
        """Load the Pi3X model. Call once at startup."""
        print("[Pipeline] Loading Pi3X model...")
        t0 = time.time()

        if ckpt is not None:
            self.model = Pi3X().to(self.device).eval()
            if ckpt.endswith('.safetensors'):
                from safetensors.torch import load_file
                weight = load_file(ckpt)
            else:
                weight = torch.load(ckpt, map_location=self.device, weights_only=False)
            self.model.load_state_dict(weight, strict=False)
        else:
            self.model = Pi3X.from_pretrained("yyfz233/Pi3X").to(self.device).eval()

        # Determine dtype
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            self.dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
        else:
            self.dtype = torch.float32

        print(f"[Pipeline] Model loaded in {time.time() - t0:.1f}s (dtype={self.dtype})")

    def configure(self, chunk_size: int = None, overlap: int = None,
                  conf_thre: float = None):
        """Hot-reconfigure pipeline parameters."""
        if chunk_size is not None:
            self.chunk_size = max(4, min(20, chunk_size))
        if overlap is not None:
            self.overlap = max(2, min(self.chunk_size // 2, overlap))
        if conf_thre is not None:
            self.conf_thre = max(0.01, min(0.5, conf_thre))
        print(f"[Pipeline] Config: chunk_size={self.chunk_size}, overlap={self.overlap}, conf_thre={self.conf_thre}")

    def add_frame(self, frame_bgr: np.ndarray) -> bool:
        """
        Add a BGR frame. Returns True if we have enough frames for a new chunk.
        """
        h, w = frame_bgr.shape[:2]

        # Determine target size on first frame
        if self.target_w is None:
            self.target_w, self.target_h = compute_target_size(w, h)
            print(f"[Pipeline] Target resolution: {self.target_w}x{self.target_h}")

        tensor = preprocess_frame(frame_bgr, self.target_w, self.target_h)
        self.frame_buffer.append(tensor)

        # How many new frames needed for a chunk?
        needed = self.chunk_size if self.is_first_chunk else (self.chunk_size - self.overlap)
        return len(self.frame_buffer) >= needed

    def frames_until_ready(self) -> int:
        """How many more frames needed before process_chunk can be called."""
        needed = self.chunk_size if self.is_first_chunk else (self.chunk_size - self.overlap)
        return max(0, needed - len(self.frame_buffer))

    @torch.no_grad()
    def process_chunk(self) -> Optional[dict]:
        """
        Process the current frame buffer as a chunk.
        Returns dict with 'points' (N,3), 'colors' (N,3 uint8), 'chunk_id', 'total_points',
        or None if not enough frames.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        needed = self.chunk_size if self.is_first_chunk else (self.chunk_size - self.overlap)
        if len(self.frame_buffer) < needed:
            return None

        t0 = time.time()

        # Build chunk images
        new_frames = self.frame_buffer[:needed]
        self.frame_buffer = self.frame_buffer[needed:]

        if self.is_first_chunk:
            chunk_imgs_list = new_frames
        else:
            # Prepend overlap frames from previous chunk
            overlap_imgs = [self._prev_overlap_imgs[i] for i in range(self._prev_overlap_imgs.shape[0])]
            chunk_imgs_list = overlap_imgs + new_frames

        chunk_imgs = torch.stack(chunk_imgs_list).unsqueeze(0).to(self.device)  # (1, T, 3, H, W)
        B, T, C, H, W = chunk_imgs.shape

        print(f"[Pipeline] Processing chunk {self.chunks_processed}: {T} frames")

        # Build model kwargs with condition injection
        model_kwargs = {'with_prior': False}

        if not self.is_first_chunk and self._prev_overlap_poses is not None:
            overlap = self.overlap

            if 'pose' in self.inject_conditions:
                prior_poses = torch.eye(4, device=self.device).repeat(B, T, 1, 1)
                prior_poses[:, :overlap] = self._prev_overlap_poses.to(self.device)
                mask_pose = torch.zeros((B, T), dtype=torch.bool, device=self.device)
                mask_pose[:, :overlap] = True
                model_kwargs['poses'] = prior_poses
                model_kwargs['mask_add_pose'] = mask_pose
                model_kwargs['with_prior'] = True

            if 'depth' in self.inject_conditions and self._prev_overlap_depth is not None:
                prior_depths = torch.zeros((B, T, H, W), device=self.device)
                prior_depths[:, :overlap] = self._prev_overlap_depth.to(self.device)
                mask_depth = torch.zeros((B, T), dtype=torch.bool, device=self.device)
                mask_depth[:, :overlap] = True
                # Zero out low-confidence depths
                if self._prev_overlap_conf is not None:
                    valid = self._prev_overlap_conf.to(self.device) > self.conf_thre
                    prior_depths[:, :overlap][~valid] = 0
                model_kwargs['depths'] = prior_depths
                model_kwargs['mask_add_depth'] = mask_depth
                model_kwargs['with_prior'] = True

            if 'ray' in self.inject_conditions and self._prev_overlap_rays is not None:
                prior_rays = torch.zeros((B, T, H, W, 3), device=self.device)
                prior_rays[:, :overlap] = self._prev_overlap_rays.to(self.device)
                mask_ray = torch.zeros((B, T), dtype=torch.bool, device=self.device)
                mask_ray[:, :overlap] = True
                model_kwargs['rays'] = prior_rays
                model_kwargs['mask_add_ray'] = mask_ray
                model_kwargs['with_prior'] = True

        # Forward pass
        with torch.amp.autocast('cuda', dtype=self.dtype):
            pred = self.model(chunk_imgs, **model_kwargs)

        # Extract results
        curr_local_depth = pred['local_points'][..., 2]
        curr_pts = pred['points']
        curr_poses = pred['camera_poses']
        curr_conf = torch.sigmoid(pred['conf'])[..., 0]
        curr_rays = pred['rays']

        # Edge filtering
        edge = depth_edge(curr_local_depth, rtol=0.03)
        curr_conf[edge] = 0
        curr_mask = curr_conf > self.conf_thre

        # Fallback if too few valid points
        if curr_mask.sum() < 10:
            flat_conf = curr_conf.view(B, T, -1)
            k = int(flat_conf.shape[-1] * 0.1)
            topk_vals, _ = torch.topk(flat_conf, k, dim=-1)
            min_vals = topk_vals[..., -1].unsqueeze(-1).unsqueeze(-1)
            curr_mask = curr_conf >= min_vals

        # Sim3 alignment
        if self.is_first_chunk:
            aligned_pts = curr_pts
            aligned_poses = curr_poses
        else:
            overlap = self.overlap
            src_pts = curr_pts[:, :overlap]
            src_mask = curr_mask[:, :overlap]
            tgt_pts = self._prev_overlap_pts.to(self.device)
            tgt_mask = self._prev_overlap_mask.to(self.device)

            transform = self._compute_sim3_umeyama_masked(src_pts, tgt_pts, src_mask, tgt_mask)

            # Validate Sim3 transform: reject wildly wrong scales or NaN
            sim3_scale = torch.det(transform[:, :3, :3]).abs().pow(1.0/3.0)
            sim3_has_nan = torch.isnan(transform).any() or torch.isinf(transform).any()
            if sim3_has_nan or sim3_scale.item() < 0.01 or sim3_scale.item() > 100.0:
                print(f"[Pipeline] WARNING: Sim3 alignment failed (scale={sim3_scale.item():.4f}, nan={sim3_has_nan}). Using identity.")
                transform = torch.eye(4, device=self.device).unsqueeze(0)

            aligned_pts = self._apply_sim3_to_points(curr_pts, transform)
            aligned_poses = self._apply_sim3_to_poses(curr_poses, transform)

        # Extract new points (skip overlap region for non-first chunks)
        if self.is_first_chunk:
            new_pts = aligned_pts[0]        # (T, H, W, 3)
            new_conf = curr_mask[0]         # (T, H, W)
            new_imgs = chunk_imgs[0]        # (T, 3, H, W)
            new_start = 0
        else:
            overlap = self.overlap
            new_pts = aligned_pts[0, overlap:]
            new_conf = curr_mask[0, overlap:]
            new_imgs = chunk_imgs[0, overlap:]
            new_start = overlap

        # Store per-frame data for object labeling
        n_new = new_pts.shape[0]
        for fi in range(n_new):
            frame_img = (new_imgs[fi].permute(1, 2, 0) * 255).byte().cpu().numpy()  # (H, W, 3) RGB uint8
            frame_pts = new_pts[fi].cpu().numpy().astype(np.float32)                 # (H, W, 3)
            frame_mask = new_conf[fi].cpu().numpy()                                  # (H, W) bool
            self.frame_data.append({
                'frame_idx': self._global_frame_idx,
                'image': frame_img,
                'point_map': frame_pts,
                'conf_mask': frame_mask,
            })
            self._global_frame_idx += 1

        # Extract valid points and colors
        valid_pts = new_pts[new_conf].cpu().numpy()                          # (N, 3)
        valid_colors = (new_imgs.permute(0, 2, 3, 1)[new_conf] * 255).byte().cpu().numpy()  # (N, 3)

        # Filter NaN/Inf points (can happen when Sim3 alignment drifts)
        finite_mask = np.isfinite(valid_pts).all(axis=1)
        if not finite_mask.all():
            n_bad = (~finite_mask).sum()
            print(f"[Pipeline] WARNING: Filtered {n_bad} NaN/Inf points")
            valid_pts = valid_pts[finite_mask]
            valid_colors = valid_colors[finite_mask]

        # Filter outlier points (> 50 units from median)
        if len(valid_pts) > 100:
            median = np.median(valid_pts, axis=0)
            dists = np.linalg.norm(valid_pts - median, axis=1)
            inlier_mask = dists < 50.0
            n_outlier = (~inlier_mask).sum()
            if n_outlier > 0:
                print(f"[Pipeline] Filtered {n_outlier} outlier points (>50 units from median)")
                valid_pts = valid_pts[inlier_mask]
                valid_colors = valid_colors[inlier_mask]

        # Save overlap state for next chunk
        overlap = self.overlap
        self._prev_overlap_imgs = chunk_imgs[0, -overlap:].cpu()
        self._prev_overlap_pts = aligned_pts[:, -overlap:].cpu()
        self._prev_overlap_mask = curr_mask[:, -overlap:].cpu()
        self._prev_overlap_poses = aligned_poses[:, -overlap:].cpu()
        self._prev_overlap_depth = curr_local_depth[:, -overlap:].cpu()
        self._prev_overlap_conf = curr_conf[:, -overlap:].cpu()
        self._prev_overlap_rays = curr_rays[:, -overlap:].cpu()

        # Accumulate globally
        self.global_points.append(valid_pts)
        self.global_colors.append(valid_colors)
        self.total_points += len(valid_pts)
        self.chunks_processed += 1
        self.is_first_chunk = False

        # Cleanup
        del pred, curr_pts, curr_poses, curr_mask, curr_local_depth, curr_conf, curr_rays
        for k_name in ['poses', 'depths', 'rays']:
            if k_name in model_kwargs:
                del model_kwargs[k_name]
        torch.cuda.empty_cache()

        elapsed = time.time() - t0
        print(f"[Pipeline] Chunk {self.chunks_processed - 1} done: {len(valid_pts)} new points, "
              f"{self.total_points} total, {elapsed:.2f}s")

        # Also extract camera poses for this chunk
        if self.is_first_chunk:
            chunk_poses = aligned_poses[0].cpu().numpy()  # won't reach here after first
        else:
            chunk_poses = aligned_poses[0].cpu().numpy()

        return {
            'points': valid_pts,
            'colors': valid_colors,
            'chunk_id': self.chunks_processed - 1,
            'total_points': self.total_points,
            'elapsed': elapsed,
            'camera_poses': chunk_poses,
        }

    def get_global_cloud(self, max_points: int = 100_000) -> dict:
        """Get the full accumulated cloud, downsampled if needed."""
        if not self.global_points:
            return {'points': np.zeros((0, 3), dtype=np.float32),
                    'colors': np.zeros((0, 3), dtype=np.uint8)}

        all_pts = np.concatenate(self.global_points, axis=0)
        all_colors = np.concatenate(self.global_colors, axis=0)

        if len(all_pts) > max_points:
            stride = len(all_pts) // max_points
            all_pts = all_pts[::stride]
            all_colors = all_colors[::stride]

        return {
            'points': all_pts.astype(np.float32),
            'colors': all_colors.astype(np.uint8),
        }

    def save_to_disk(self, path: str = None):
        """Save the current global point cloud to disk."""
        if path is None:
            path = os.path.join(os.path.dirname(__file__), 'saved_cloud.npz')
        if not self.global_points:
            print("[Pipeline] No points to save")
            return
        all_pts = np.concatenate(self.global_points, axis=0).astype(np.float32)
        all_cols = np.concatenate(self.global_colors, axis=0).astype(np.uint8)
        np.savez_compressed(path, points=all_pts, colors=all_cols,
                            total_points=self.total_points,
                            chunks_processed=self.chunks_processed)
        print(f"[Pipeline] Saved {len(all_pts)} points to {path}")

    def load_from_disk(self, path: str = None) -> bool:
        """Load a saved point cloud. Returns True if loaded."""
        if path is None:
            path = os.path.join(os.path.dirname(__file__), 'saved_cloud.npz')
        if not os.path.exists(path):
            return False
        try:
            data = np.load(path, allow_pickle=True)
            pts = data['points']
            cols = data['colors']
            self.global_points = [pts]
            self.global_colors = [cols]
            self.total_points = int(data['total_points'])
            self.chunks_processed = int(data['chunks_processed'])
            print(f"[Pipeline] Loaded {self.total_points} points from {path}")
            return True
        except Exception as e:
            print(f"[Pipeline] Failed to load cloud: {e}")
            return False

    def reset(self):
        """Clear all state. Model stays loaded."""
        self.frame_buffer.clear()
        self._prev_overlap_imgs = None
        self._prev_overlap_pts = None
        self._prev_overlap_mask = None
        self._prev_overlap_poses = None
        self._prev_overlap_depth = None
        self._prev_overlap_conf = None
        self._prev_overlap_rays = None
        self.global_points.clear()
        self.global_colors.clear()
        self.total_points = 0
        self.chunks_processed = 0
        self.is_first_chunk = True
        self.target_w = None
        self.target_h = None
        self.frame_data.clear()
        self._global_frame_idx = 0
        torch.cuda.empty_cache()
        print("[Pipeline] Reset complete")

    # ─── Sim3 alignment (from Pi3XVO) ────────────────────────────

    def _compute_sim3_umeyama_masked(self, src_points, tgt_points, src_mask, tgt_mask):
        B = src_points.shape[0]
        device = src_points.device

        src = src_points.reshape(B, -1, 3)
        tgt = tgt_points.reshape(B, -1, 3)

        mask = (src_mask.reshape(B, -1) & tgt_mask.reshape(B, -1)).float().unsqueeze(-1)
        valid_cnt = mask.sum(dim=1).squeeze(-1)
        eps = 1e-6

        bad_mask = valid_cnt < 10
        if bad_mask.all():
            return torch.eye(4, device=device).repeat(B, 1, 1)

        src_mean = (src * mask).sum(dim=1, keepdim=True) / (valid_cnt.view(B, 1, 1) + eps)
        tgt_mean = (tgt * mask).sum(dim=1, keepdim=True) / (valid_cnt.view(B, 1, 1) + eps)

        src_centered = (src - src_mean) * mask
        tgt_centered = (tgt - tgt_mean) * mask

        H = torch.bmm(src_centered.transpose(1, 2), tgt_centered)
        U, S, V = torch.svd(H)

        R = torch.bmm(V, U.transpose(1, 2))

        det = torch.det(R)
        diag = torch.ones(B, 3, device=device)
        diag[:, 2] = torch.sign(det)
        R = torch.bmm(torch.bmm(V, torch.diag_embed(diag)), U.transpose(1, 2))

        src_var = (src_centered ** 2).sum(dim=2) * mask.squeeze(-1)
        src_var = src_var.sum(dim=1) / (valid_cnt + eps)

        corrected_S = S.clone()
        corrected_S[:, 2] *= diag[:, 2]
        trace_S = corrected_S.sum(dim=1)

        scale = trace_S / (src_var * valid_cnt + eps)
        scale = scale.view(B, 1, 1)

        t = tgt_mean.transpose(1, 2) - scale * torch.bmm(R, src_mean.transpose(1, 2))

        sim3 = torch.eye(4, device=device).repeat(B, 1, 1)
        sim3[:, :3, :3] = scale * R
        sim3[:, :3, 3] = t.squeeze(2)

        if bad_mask.any():
            identity = torch.eye(4, device=device).repeat(B, 1, 1)
            sim3[bad_mask] = identity[bad_mask]

        return sim3

    def _apply_sim3_to_points(self, points, sim3):
        B, T, H, W, C = points.shape
        flat_pts = points.reshape(B, -1, 3)
        R_s = sim3[:, :3, :3]
        t = sim3[:, :3, 3].unsqueeze(1)
        out_pts = torch.bmm(flat_pts, R_s.transpose(1, 2)) + t
        return out_pts.reshape(B, T, H, W, 3)

    def _apply_sim3_to_poses(self, poses, sim3):
        sim3_expanded = sim3.unsqueeze(1)
        return torch.matmul(sim3_expanded, poses)
