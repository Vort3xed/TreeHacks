"""
Object Labeler — SAM3 text-prompted segmentation → tight 3D bounding boxes.

Given per-frame 3D point maps from Pi3X and text prompts (e.g. "chair"),
runs SAM3 to segment objects in 2D, maps to 3D via the dense point maps,
computes tight oriented bounding boxes (OBB) with Open3D, deduplicates
across frames, and returns labeled 3D bounding boxes with debug images.

Modeled after VGGT-SLAM-NEW/vggt_slam/object_detector.py.
"""

import torch
import numpy as np
import cv2
import base64
import time
from typing import Optional
from dataclasses import dataclass, field
from PIL import Image


@dataclass
class LabeledObject:
    """A detected and labeled 3D object with tight oriented bounding box."""
    label: str
    instance_id: int
    # Oriented bounding box (OBB)
    center: list       # [x, y, z]
    extent: list       # [dx, dy, dz] — tight dimensions
    rotation: list     # 3x3 rotation matrix
    corners: list      # 8 corner points of the OBB
    # AABB fallback (computed from corners for simple rendering)
    bbox_min: list     # [x, y, z] — AABB min
    bbox_max: list     # [x, y, z] — AABB max
    confidence: float
    point_count: int
    # Debug info
    debug_images: list = field(default_factory=list)  # list of {frame_idx, mask_b64, score}


@dataclass
class FrameData:
    """Per-frame data needed for object labeling."""
    frame_idx: int                  # global frame index
    image: np.ndarray               # (H, W, 3) RGB uint8
    point_map: np.ndarray           # (H, W, 3) float32, world coords
    conf_mask: np.ndarray           # (H, W) bool


class ObjectLabeler:
    """
    Runs SAM3 text-conditioned segmentation on stored keyframes,
    maps 2D masks to 3D via Pi3X dense point maps, computes tight
    oriented bounding boxes, deduplicates, and returns results.
    """

    def __init__(self, device: str = 'cuda'):
        self.device = torch.device(device)
        self.sam3_model = None
        self.sam3_processor = None
        self._loaded = False

    def load_model(self):
        """Load SAM3 image model. Call once (lazy on first use)."""
        if self._loaded:
            return

        print("[ObjectLabeler] Loading SAM3 model...")
        t0 = time.time()

        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        self.sam3_model = build_sam3_image_model()
        self.sam3_processor = Sam3Processor(self.sam3_model)
        self._loaded = True

        print(f"[ObjectLabeler] SAM3 loaded in {time.time() - t0:.1f}s")

    def label_objects(
        self,
        frames: list[FrameData],
        prompts: list[str],
        confidence_threshold: float = 0.3,
        max_frames: int = 20,
        include_debug_images: bool = True,
        max_debug_images: int = 5,
    ) -> list[LabeledObject]:
        """
        Run SAM3 on stored frames to detect and label 3D objects.

        Returns list of LabeledObject with tight oriented bounding boxes
        and optional debug images showing the SAM3 segmentation.
        """
        if not self._loaded:
            self.load_model()

        if not frames or not prompts:
            return []

        # Sample frames evenly if too many
        if len(frames) > max_frames:
            indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
            frames = [frames[i] for i in indices]

        print(f"[ObjectLabeler] Labeling {len(prompts)} classes across {len(frames)} frames")
        t0 = time.time()

        # Collect raw detections across all frames
        raw_detections: list[dict] = []

        for prompt in prompts:
            prompt = prompt.strip()
            if not prompt:
                continue

            for fdata in frames:
                try:
                    frame_dets = self._detect_in_frame(
                        fdata, prompt, confidence_threshold, include_debug_images
                    )
                    raw_detections.extend(frame_dets)
                except Exception as e:
                    print(f"[ObjectLabeler] Error on frame {fdata.frame_idx} for '{prompt}': {e}")
                    continue

        if not raw_detections:
            print("[ObjectLabeler] No detections found")
            return []

        print(f"[ObjectLabeler] {len(raw_detections)} raw detections before dedup")

        # Deduplicate overlapping detections (same query, overlapping OBBs)
        deduped = self._deduplicate_detections(raw_detections)

        # Convert to LabeledObject list
        results = []
        for i, det in enumerate(deduped):
            bbox = det['bounding_box']
            corners = np.array(bbox['corners'])
            bbox_min = corners.min(axis=0).tolist()
            bbox_max = corners.max(axis=0).tolist()

            # Limit debug images per object
            debug_imgs = det.get('debug_images', [])
            if len(debug_imgs) > max_debug_images:
                debug_imgs = debug_imgs[:max_debug_images]

            obj = LabeledObject(
                label=det['query'],
                instance_id=i,
                center=bbox['center'],
                extent=bbox['extent'],
                rotation=bbox['rotation'],
                corners=bbox['corners'],
                bbox_min=bbox_min,
                bbox_max=bbox_max,
                confidence=det['confidence'],
                point_count=det.get('point_count', 0),
                debug_images=debug_imgs,
            )
            results.append(obj)

        elapsed = time.time() - t0
        print(f"[ObjectLabeler] Done: {len(results)} objects in {elapsed:.1f}s "
              f"({len(raw_detections)} raw -> {len(deduped)} deduped)")

        return results

    def _detect_in_frame(
        self,
        fdata: FrameData,
        prompt: str,
        confidence_threshold: float,
        include_debug_images: bool = True,
    ) -> list[dict]:
        """
        Run SAM3 on a single frame. Returns list of detection dicts,
        each with a tight OBB and optional debug image.
        """
        pil_img = Image.fromarray(fdata.image)

        # Run SAM3 text-prompted segmentation
        state = self.sam3_processor.set_image(pil_img)
        self.sam3_processor.set_confidence_threshold(confidence_threshold, state)
        output = self.sam3_processor.set_text_prompt(state=state, prompt=prompt)

        masks = output.get('masks')
        scores = output.get('scores')
        boxes = output.get('boxes')

        if masks is None or len(masks) == 0:
            return []

        detections = []
        sam_h, sam_w = masks.shape[-2:]
        pt_h, pt_w = fdata.point_map.shape[:2]

        for i in range(len(masks)):
            mask_2d = masks[i, 0].cpu().numpy()  # (sam_H, sam_W) float/bool
            score = scores[i].item() if torch.is_tensor(scores[i]) else float(scores[i])

            # Resize mask to point map resolution if different
            if sam_h != pt_h or sam_w != pt_w:
                mask_resized = cv2.resize(
                    mask_2d.astype(np.uint8), (pt_w, pt_h),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)
            else:
                mask_resized = mask_2d > 0.5

            # Intersect with confidence mask
            combined_mask = mask_resized & fdata.conf_mask

            if combined_mask.sum() < 20:
                continue

            # Extract 3D points under the mask
            pts_3d = fdata.point_map[combined_mask]  # (K, 3)

            # Filter NaN/Inf
            finite = np.isfinite(pts_3d).all(axis=1)
            pts_3d = pts_3d[finite]

            if len(pts_3d) < 20:
                continue

            # Compute tight oriented bounding box
            bbox = self._compute_obb(pts_3d)
            if bbox is None:
                continue

            # Generate debug image (SAM mask overlay)
            debug_img = None
            if include_debug_images:
                debug_img = self._mask_overlay_to_base64(fdata.image, mask_resized)

            det = {
                'query': prompt,
                'bounding_box': bbox,
                'confidence': score,
                'point_count': len(pts_3d),
                'matched_frame': fdata.frame_idx,
                'debug_images': [],
            }
            if debug_img:
                det['debug_images'].append({
                    'frame_idx': fdata.frame_idx,
                    'mask_b64': debug_img,
                    'score': round(score, 3),
                })

            detections.append(det)

        return detections

    def _compute_obb(self, points: np.ndarray) -> Optional[dict]:
        """
        Compute a tight oriented bounding box using Open3D.
        Applies statistical outlier removal for cleaner bounds.
        """
        import open3d as o3d

        if len(points) < 10:
            return None

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Statistical outlier removal for tighter boxes
        if len(points) > 50:
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

        if len(pcd.points) < 10:
            return None

        try:
            obb = pcd.get_oriented_bounding_box()
            center = obb.center.tolist()
            extent = obb.extent.tolist()
            rotation = np.asarray(obb.R).tolist()
            corners = np.asarray(obb.get_box_points()).tolist()
        except Exception:
            # Fallback to AABB
            aabb = pcd.get_axis_aligned_bounding_box()
            center = aabb.get_center().tolist()
            extent = (aabb.get_max_bound() - aabb.get_min_bound()).tolist()
            rotation = np.eye(3).tolist()
            corners = np.asarray(aabb.get_box_points()).tolist()

        return {
            'center': center,
            'extent': extent,
            'rotation': rotation,
            'corners': corners,
        }

    @staticmethod
    def _deduplicate_detections(detections: list[dict]) -> list[dict]:
        """
        Greedy NMS-style deduplication. Sorts by confidence descending.
        For each candidate, checks overlap with already-kept boxes of the
        same query. Merges debug images from duplicates into the kept one.
        """
        if len(detections) <= 1:
            return [d for d in detections
                    if d.get('bounding_box') is not None]

        keep = []
        for det in sorted(detections, key=lambda d: d.get('confidence', 0), reverse=True):
            if det.get('bounding_box') is None:
                continue

            center = np.array(det['bounding_box']['center'])
            extent = np.array(det['bounding_box']['extent'])
            half_ext = extent / 2.0

            is_dup = False
            merge_into = None
            for kept in keep:
                if kept['query'] != det['query']:
                    continue
                kept_center = np.array(kept['bounding_box']['center'])
                kept_extent = np.array(kept['bounding_box']['extent'])
                kept_half = kept_extent / 2.0

                diff = np.abs(center - kept_center)
                overlap = np.all(diff < (half_ext + kept_half))
                if overlap:
                    is_dup = True
                    merge_into = kept
                    break

            if is_dup and merge_into is not None:
                # Merge debug images from duplicate into the kept detection
                merge_into['debug_images'].extend(det.get('debug_images', []))
            else:
                keep.append(det)

        return keep

    @staticmethod
    def _mask_overlay_to_base64(img_rgb: np.ndarray, mask: np.ndarray) -> str:
        """
        Create a mask overlay image: 50% green tint on masked pixels + contour.
        Returns base64-encoded PNG.
        """
        overlay = img_rgb.copy()
        color = np.array([0, 255, 100], dtype=np.uint8)

        mask_bool = mask.astype(bool) if mask.dtype != bool else mask

        # Resize mask to image dimensions if needed
        if mask_bool.shape[:2] != overlay.shape[:2]:
            mask_bool = cv2.resize(
                mask_bool.astype(np.uint8), (overlay.shape[1], overlay.shape[0]),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)

        overlay[mask_bool] = (overlay[mask_bool].astype(np.float32) * 0.5 +
                              color.astype(np.float32) * 0.5).astype(np.uint8)

        # Draw contour
        mask_uint8 = mask_bool.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.drawContours(overlay_bgr, contours, -1, (0, 255, 100), 2)

        # Resize to reasonable thumbnail size
        h, w = overlay_bgr.shape[:2]
        thumb_w = 320
        thumb_h = int(h * thumb_w / w)
        overlay_bgr = cv2.resize(overlay_bgr, (thumb_w, thumb_h))

        _, buffer = cv2.imencode('.png', overlay_bgr)
        return base64.b64encode(buffer).decode('utf-8')
