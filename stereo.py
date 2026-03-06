"""
Stereo alignment and 3D visualization for the Water Measuring system.

Takes shape snapshot data from two WaterDataRecorders (top + side cameras),
aligns the blobs so that the horizontal center of mass matches, then produces:
  - 3D expansion plot (time × width × height)
  - Combined pixel count comparison
  - Combined spread comparison
  - Aligned contour overlay (top + side merged)
"""

from __future__ import annotations

import os
import cv2  # type: ignore
import numpy as np  # type: ignore
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # type: ignore
import matplotlib.colors as mcolors  # type: ignore
from mpl_toolkits.mplot3d import Axes3D  # type: ignore  # noqa: F401 — needed for 3d projection

from water_data import WaterDataRecorder


class StereoAnalyzer:
    """
    Combines data from two WaterDataRecorder instances (top and side views)
    to produce aligned and 3D visualizations.
    """

    def __init__(
        self,
        top_recorder: WaterDataRecorder,
        side_recorder: WaterDataRecorder,
        output_dir: str = "results/combined",
    ):
        self.top = top_recorder
        self.side = side_recorder
        self.output_dir = output_dir

    # region Public

    def save_all(self) -> None:
        """Generate and save every combined graph."""
        if not self.top.has_data or not self.side.has_data:
            print("[stereo] Need data from both cameras — skipping.")
            return

        os.makedirs(self.output_dir, exist_ok=True)

        self._save_pixel_count_comparison()
        self._save_spread_comparison()
        self._save_aligned_contours()
        self._save_3d_expansion()
        self._save_bounding_box_over_time()
        self._save_debug_video()
        self._save_3d_reconstruction_video()

        print(f"[stereo] Combined graphs saved to '{self.output_dir}/'")

    # endregion

    # region Pixel count comparison

    def _save_pixel_count_comparison(self) -> None:
        from water_data import WaterDataRecorder as _WDR
        fig, ax = plt.subplots(figsize=(10, 5))
        top_clean = _WDR._monotonic_cummax(_WDR._remove_outliers(self.top.pixel_counts))
        side_clean = _WDR._monotonic_cummax(_WDR._remove_outliers(self.side.pixel_counts))
        ax.plot(self.top.timestamps, top_clean,
                label=self.top.cam_label, color="tab:blue", linewidth=1)
        ax.plot(self.side.timestamps, side_clean,
                label=self.side.cam_label, color="tab:orange", linewidth=1)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Pixel Count")
        ax.set_title("Pixel Count — Top vs Side")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, "pixel_count_comparison.png"), dpi=150)
        plt.close(fig)

    # endregion

    # region Spread comparison

    def _save_spread_comparison(self) -> None:
        from water_data import WaterDataRecorder as _WDR
        fig, ax = plt.subplots(figsize=(10, 5))
        top_clean = _WDR._monotonic_cummax(_WDR._remove_outliers(self.top.spread_factors))
        side_clean = _WDR._monotonic_cummax(_WDR._remove_outliers(self.side.spread_factors))
        ax.plot(self.top.timestamps, top_clean,
                label=self.top.cam_label, color="tab:green", linewidth=1)
        ax.plot(self.side.timestamps, side_clean,
                label=self.side.cam_label, color="tab:red", linewidth=1)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Spread Factor (mean σ)")
        ax.set_title("Group Expansion — Top vs Side")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, "spread_comparison.png"), dpi=150)
        plt.close(fig)

    # endregion

    # region Aligned contour overlay

    def _save_aligned_contours(self) -> None:
        """
        Overlay contours from both cameras on a single canvas.

        Alignment strategy:
            For each snapshot pair, compute the horizontal center of mass
            of each mask.  Shift the side-camera contours so that their
            center-of-mass X aligns with the top-camera's.
        """
        top_snaps = self.top.shape_snapshots
        side_snaps = self.side.shape_snapshots
        if not top_snaps or not side_snaps:
            return

        # Canvas size = largest of the two cameras
        th, tw = top_snaps[0][1].shape[:2]
        sh, sw = side_snaps[0][1].shape[:2]
        canvas_h = max(th, sh)
        canvas_w = max(tw, sw)
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        # Pair snapshots by closest timestamp
        pairs = self._pair_snapshots(top_snaps, side_snaps)

        for top_t, top_mask, side_t, side_mask in pairs:
            # Compute horizontal center of mass for alignment
            top_com_x = self._center_of_mass_x(top_mask)
            side_com_x = self._center_of_mass_x(side_mask)

            shift_x = 0
            if top_com_x is not None and side_com_x is not None:
                shift_x = int(top_com_x - side_com_x)

            # Draw top contours in cyan
            top_contours, _ = cv2.findContours(
                top_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(canvas, top_contours, -1, (255, 255, 0), 1)  # cyan (BGR)

            # Draw side contours in magenta, shifted
            side_contours, _ = cv2.findContours(
                side_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for cnt in side_contours:
                cnt_shifted = cnt.copy()
                cnt_shifted[:, :, 0] += shift_x  # shift X
                cv2.drawContours(canvas, [cnt_shifted], -1, (255, 0, 255), 1)  # magenta

        canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.imshow(canvas_rgb, interpolation="nearest")
        ax.set_title("Aligned Contour Overlay (cyan = top, magenta = side)")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, "aligned_contours.png"), dpi=150)
        plt.close(fig)

    # endregion

    # region 3D expansion plot

    def _save_3d_expansion(self) -> None:
        """
        3D plot: X = time, Y = width (from top cam), Z = height (from side cam).

        Width and height are measured as the bounding-box dimensions of the
        detected blob in each camera's mask at each snapshot time.
        """
        top_snaps = self.top.shape_snapshots
        side_snaps = self.side.shape_snapshots
        if not top_snaps or not side_snaps:
            return

        pairs = self._pair_snapshots(top_snaps, side_snaps)

        times = []
        widths = []
        heights = []

        for top_t, top_mask, side_t, side_mask in pairs:
            bbox_top = self._bounding_box(top_mask)
            bbox_side = self._bounding_box(side_mask)
            if bbox_top is None or bbox_side is None:
                continue

            times.append((top_t + side_t) / 2.0)
            widths.append(bbox_top[2])    # width from top view
            heights.append(bbox_side[3])  # height from side view

        if not times:
            return

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")

        # Scatter with time-based color
        sizes = np.full(len(times), 30)
        sc = ax.scatter(  # type: ignore[call-overload]
            times, widths, heights, c=times, cmap="viridis", s=sizes, alpha=0.8
        )
        ax.plot(times, widths, heights, color="gray", alpha=0.4, linewidth=1)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Width (px) — top cam")
        ax.set_zlabel("Height (px) — side cam")  # type: ignore[attr-defined]
        ax.set_title("3D Blob Expansion Over Time")
        fig.colorbar(sc, ax=ax, shrink=0.6, label="Time (s)")
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, "3d_expansion.png"), dpi=150)
        plt.close(fig)

    # endregion

    # region Bounding box over time

    def _save_bounding_box_over_time(self) -> None:
        """Width (top cam) and Height (side cam) on a shared time axis."""
        top_snaps = self.top.shape_snapshots
        side_snaps = self.side.shape_snapshots
        if not top_snaps or not side_snaps:
            return

        pairs = self._pair_snapshots(top_snaps, side_snaps)

        times, widths, heights = [], [], []
        for top_t, top_mask, side_t, side_mask in pairs:
            bbox_top = self._bounding_box(top_mask)
            bbox_side = self._bounding_box(side_mask)
            if bbox_top is None or bbox_side is None:
                continue
            times.append((top_t + side_t) / 2.0)
            widths.append(bbox_top[2])
            heights.append(bbox_side[3])

        if not times:
            return

        fig, ax1 = plt.subplots(figsize=(10, 5))

        # Clean data: remove outliers, then take cumulative max
        from water_data import WaterDataRecorder as _WDR
        w_clean = _WDR._monotonic_cummax(_WDR._remove_outliers(widths))
        h_clean = _WDR._monotonic_cummax(_WDR._remove_outliers(heights))

        ax1.plot(times, w_clean, color="tab:blue", label="Width (top cam)")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Width (px)", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")

        ax2 = ax1.twinx()
        ax2.plot(times, h_clean, color="tab:red", label="Height (side cam)")
        ax2.set_ylabel("Height (px)", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")

        fig.suptitle("Blob Dimensions Over Time")
        fig.legend(loc="upper left", bbox_to_anchor=(0.12, 0.88))
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, "dimensions_over_time.png"), dpi=150)
        plt.close(fig)

    # endregion

    # region Debug video (side-by-side detection snapshots)

    def _save_debug_video(self) -> None:
        """
        Stitch top and side detection masks side-by-side into a debug video.

        Each frame shows:
          left  = top camera mask (colorized green)
          right = side camera mask (colorized magenta)
          timestamp overlay

        Useful for understanding why the 3D reconstruction looks a
        certain way — you can see exactly what each camera detected at
        each timestep.
        """
        top_snaps = self.top.shape_snapshots
        side_snaps = self.side.shape_snapshots
        if not top_snaps or not side_snaps:
            print("[stereo] Not enough snapshots for debug video.")
            return

        pairs = self._pair_snapshots(top_snaps, side_snaps)
        if not pairs:
            return

        vid_path = os.path.join(self.output_dir, "debug_detection.mp4")
        print(f"[stereo] Generating debug detection video ({len(pairs)} frames)…")

        # Determine canvas size: scale both masks to the same height
        sample_top = pairs[0][1]
        sample_side = pairs[0][3]
        target_h = max(sample_top.shape[0], sample_side.shape[0])
        # Scale widths proportionally
        top_w = int(sample_top.shape[1] * target_h / sample_top.shape[0])
        side_w = int(sample_side.shape[1] * target_h / sample_side.shape[0])
        canvas_w = top_w + side_w
        canvas_h = target_h

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
        fps = max(5, min(15, len(pairs) // 3))
        writer = cv2.VideoWriter(vid_path, fourcc, fps, (canvas_w, canvas_h))

        for top_t, top_mask, side_t, side_mask in pairs:
            # Resize masks to target dimensions
            top_resized = cv2.resize(top_mask, (top_w, target_h), interpolation=cv2.INTER_NEAREST)
            side_resized = cv2.resize(side_mask, (side_w, target_h), interpolation=cv2.INTER_NEAREST)

            # Colorize: top = green channel, side = magenta (R+B)
            top_color = np.zeros((target_h, top_w, 3), dtype=np.uint8)
            top_color[:, :, 1] = top_resized  # green

            side_color = np.zeros((target_h, side_w, 3), dtype=np.uint8)
            side_color[:, :, 0] = side_resized  # blue
            side_color[:, :, 2] = side_resized  # red → magenta

            # Stitch side by side
            canvas = np.hstack([top_color, side_color])

            # Draw dividing line
            cv2.line(canvas, (top_w, 0), (top_w, canvas_h), (255, 255, 255), 1)

            # Labels + timestamp
            t = (top_t + side_t) / 2.0
            cv2.putText(canvas, f"{self.top.cam_label}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(canvas, f"{self.side.cam_label}", (top_w + 10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            cv2.putText(canvas, f"t = {t:.1f}s", (canvas_w // 2 - 50, canvas_h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Pixel counts
            top_px = cv2.countNonZero(top_resized)
            side_px = cv2.countNonZero(side_resized)
            cv2.putText(canvas, f"{top_px}px", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
            cv2.putText(canvas, f"{side_px}px", (top_w + 10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 200), 1)

            writer.write(canvas)

        writer.release()
        print(f"[stereo] Debug detection video saved: {vid_path}")

    # endregion

    # region 3D reconstruction video

    def _save_3d_reconstruction_video(self) -> None:
        """
        Generate a video showing a 3D voxel reconstruction of the blob
        evolving over time.

        Method (visual hull):
            - Top camera mask gives the X–Y silhouette (width × depth).
            - Side camera mask gives the X–Z silhouette (width × height).
            - Extrude each silhouette along the missing axis and intersect
              to get a 3D voxel volume.
            - Render each timestep as a rotating 3D matplotlib frame.
            - Combine frames into an MP4 video.
        """
        top_snaps = self.top.shape_snapshots
        side_snaps = self.side.shape_snapshots
        if not top_snaps or not side_snaps:
            print("[stereo] Not enough snapshots for 3D reconstruction video.")
            return

        pairs = self._pair_snapshots(top_snaps, side_snaps)
        if len(pairs) < 2:
            print("[stereo] Need at least 2 paired snapshots for video.")
            return

        # Limit total frames to keep render time reasonable
        MAX_FRAMES = 60
        if len(pairs) > MAX_FRAMES:
            step = len(pairs) / MAX_FRAMES
            pairs = [pairs[int(i * step)] for i in range(MAX_FRAMES)]

        # Downsample masks for performance (32³ is much faster than 64³)
        VOXEL_RES = 32
        vid_path = os.path.join(self.output_dir, "3d_reconstruction.mp4")
        print(f"[stereo] Generating 3D reconstruction video ({len(pairs)} frames @ {VOXEL_RES}³)…")

        # Pre-compute all voxel volumes
        volumes = []
        max_extent = 0
        for top_t, top_mask, side_t, side_mask in pairs:
            vol = self._build_visual_hull(top_mask, side_mask, VOXEL_RES)
            volumes.append(((top_t + side_t) / 2.0, vol))
            if vol is not None:
                max_extent = max(max_extent, VOXEL_RES)

        if not any(v is not None for _, v in volumes):
            print("[stereo] No valid volumes to render.")
            return

        # Render frames
        fig = plt.figure(figsize=(6, 5), dpi=80)
        frames_rendered = []

        # Slow rotation over the video
        total_frames = len(volumes)
        base_elev = 25
        base_azim = -60
        azim_range = 90  # rotate 90° over the whole video

        for idx, (t, vol) in enumerate(volumes):
            if (idx + 1) % 10 == 0 or idx == 0:
                print(f"[stereo]   rendering frame {idx + 1}/{total_frames}…")

            fig.clf()
            ax = fig.add_subplot(111, projection="3d")

            azim = base_azim + (azim_range * idx / max(total_frames - 1, 1))

            if vol is not None and np.any(vol):
                # Color by height (Z axis)
                colors = np.empty(vol.shape + (4,), dtype=np.float32)
                z_size = vol.shape[2]
                cmap = plt.cm.viridis  # type: ignore[attr-defined]
                for z in range(z_size):
                    rgba = cmap(z / max(z_size - 1, 1))
                    colors[:, :, z] = [rgba[0], rgba[1], rgba[2], 0.6]

                ax.voxels(vol, facecolors=colors, edgecolor="none")  # type: ignore[attr-defined]

            ax.set_xlim(0, VOXEL_RES)
            ax.set_ylim(0, VOXEL_RES)
            ax.set_zlim(0, VOXEL_RES)  # type: ignore[attr-defined]
            ax.set_xlabel("X (width)")
            ax.set_ylabel("Y (depth — top)")
            ax.set_zlabel("Z (height — side)")  # type: ignore[attr-defined]
            ax.set_title(f"3D Blob Reconstruction — t = {t:.1f}s")
            ax.view_init(elev=base_elev, azim=azim)  # type: ignore[attr-defined]

            # Render to numpy array
            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            buf = np.asarray(fig.canvas.buffer_rgba())
            buf = buf[:, :, :3]  # RGBA → RGB
            # Convert RGB → BGR for OpenCV
            frames_rendered.append(cv2.cvtColor(buf, cv2.COLOR_RGB2BGR))

        plt.close(fig)

        if not frames_rendered:
            return

        # Write video
        h, w = frames_rendered[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
        # Target ~2 fps per snapshot (since snapshots are 0.1s apart, 10 snapshots = 1s real time)
        # But play at a viewable rate
        fps = max(5, min(15, total_frames // 3))
        writer = cv2.VideoWriter(vid_path, fourcc, fps, (w, h))

        for frame in frames_rendered:
            writer.write(frame)

        writer.release()
        print(f"[stereo] 3D reconstruction video saved: {vid_path}")

    @staticmethod
    def _build_visual_hull(
        top_mask: np.ndarray,
        side_mask: np.ndarray,
        resolution: int = 64,
    ) -> np.ndarray | None:
        """
        Build a 3D voxel volume from top and side 2D masks using visual hull.

        Top mask  → X–Y plane (looking down): columns = X, rows = Y (depth)
        Side mask → X–Z plane (looking from side): columns = X, rows = Z (height)

        Returns a (resolution, resolution, resolution) boolean numpy array,
        or None if both masks are empty.
        """
        # Resize masks to target resolution
        top_small = cv2.resize(top_mask, (resolution, resolution), interpolation=cv2.INTER_NEAREST)
        side_small = cv2.resize(side_mask, (resolution, resolution), interpolation=cv2.INTER_NEAREST)

        top_bool = top_small > 0   # shape: (Y, X)
        side_bool = side_small > 0  # shape: (Z, X)

        if not np.any(top_bool) and not np.any(side_bool):
            return None

        # Extrude top mask along Z axis: at each (x, y), the column is filled for all Z
        # top_bool[y, x] → vol[x, y, z] for all z
        top_extruded = np.broadcast_to(
            top_bool.T[:, :, np.newaxis],  # (X, Y, 1)
            (resolution, resolution, resolution),
        )

        # Extrude side mask along Y axis: at each (x, z), the column is filled for all Y
        # side_bool[z, x] → vol[x, y, z] for all y
        # side_bool is (Z, X), transpose to (X, Z), then broadcast across Y
        side_extruded = np.broadcast_to(
            side_bool.T[:, np.newaxis, :],  # (X, 1, Z)
            (resolution, resolution, resolution),
        )

        # Intersection
        volume = np.logical_and(top_extruded, side_extruded)
        return volume

    # endregion

    # region Helpers

    @staticmethod
    def _center_of_mass_x(mask: np.ndarray) -> float | None:
        """Return the X coordinate of the center of mass, or None if empty."""
        points = np.column_stack(np.where(mask > 0))  # (y, x)
        if len(points) == 0:
            return None
        return float(np.mean(points[:, 1]))

    @staticmethod
    def _bounding_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
        """Return (x, y, w, h) of the bounding box of non-zero pixels, or None."""
        points = np.column_stack(np.where(mask > 0))  # (y, x)
        if len(points) == 0:
            return None
        y_min, x_min = points.min(axis=0)
        y_max, x_max = points.max(axis=0)
        return (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))

    @staticmethod
    def _pair_snapshots(
        top_snaps: list[tuple[float, np.ndarray]],
        side_snaps: list[tuple[float, np.ndarray]],
    ) -> list[tuple[float, np.ndarray, float, np.ndarray]]:
        """
        Pair snapshots from two cameras by closest timestamp.
        Returns list of (top_time, top_mask, side_time, side_mask).
        """
        pairs = []
        side_idx = 0
        for top_t, top_mask in top_snaps:
            # Find the side snapshot with the closest timestamp
            best_idx = side_idx
            best_diff = abs(top_t - side_snaps[best_idx][0])

            while side_idx < len(side_snaps) - 1:
                diff = abs(top_t - side_snaps[side_idx + 1][0])
                if diff < best_diff:
                    best_diff = diff
                    best_idx = side_idx + 1
                    side_idx += 1
                else:
                    break

            # Only pair if within 0.5s of each other
            if best_diff <= 0.5:
                pairs.append((top_t, top_mask, side_snaps[best_idx][0], side_snaps[best_idx][1]))

        return pairs
    # endregion
