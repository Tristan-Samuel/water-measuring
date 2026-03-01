from __future__ import annotations

import time
import os
import cv2  # type: ignore
import numpy as np  # type: ignore
import matplotlib  # type: ignore
matplotlib.use('Agg')  # Non-interactive backend so plots save without a display window
import matplotlib.pyplot as plt  # type: ignore
import matplotlib.colors as mcolors  # type: ignore


class WaterDataRecorder:
    """
    Records frame-by-frame color analysis data during a live webcam session.

    Tracks:
        - Pixel count per frame
        - Time since first color detection
        - Group expansion (spread factor over time)
        - Shape snapshots every 0.1 seconds
        - Final detected shape

    After the session, call save_graphs() to produce:
        - pixel_count_over_time.png
        - group_expansion_over_time.png
        - final_shape.png
        - incremental_shape_layers.png
    """

    SNAPSHOT_INTERVAL = 0.1  # seconds between shape snapshots

    def __init__(self, output_dir="results", cam_label: str = "Camera"):
        self.cam_label = cam_label

        # --- Timing ---
        self._first_detection_time = None  # Set once, on first non-zero pixel frame
        self._last_snapshot_time = None    # Tracks when the last 0.1s snapshot was taken

        # --- Per-frame time-series data ---
        self.timestamps = []       # Seconds since first detection
        self.pixel_counts = []     # Total grouped pixel count per sample
        self.spread_factors = []   # Spread factor per sample

        # --- Shape snapshots (binary masks captured every 0.1s) ---
        self.shape_snapshots = []      # List of (timestamp, mask) tuples
        self.final_shape_mask = None   # The last valid mask recorded

        # --- Output ---
        self.output_dir = output_dir

    # region Record frame
    
    def record_frame(self, filtered_mask, total_pixels, spread_factor):
        """
        Record data for one frame.

        Args:
            filtered_mask: The binary mask of valid-group pixels (from bitwise_and).
            total_pixels:  Number of non-zero pixels in filtered_mask.
            spread_factor: Mean std-dev of pixel positions.
        """
        now = time.time()

        # Nothing detected yet — skip
        if total_pixels == 0 and self._first_detection_time is None:
            return

        # --- First detection ever ---
        if self._first_detection_time is None:
            self._first_detection_time = now
            self._last_snapshot_time = now

        elapsed = now - self._first_detection_time

        # --- Record time-series data ---
        self.timestamps.append(elapsed)
        self.pixel_counts.append(total_pixels)
        self.spread_factors.append(spread_factor)

        # --- Keep the mask as the running "final" shape ---
        if total_pixels > 0:
            self.final_shape_mask = filtered_mask.copy()

        # --- Shape snapshot every SNAPSHOT_INTERVAL ---
        if self._last_snapshot_time is not None and now - self._last_snapshot_time >= self.SNAPSHOT_INTERVAL:
            self._last_snapshot_time = now
            if total_pixels > 0:
                self.shape_snapshots.append((elapsed, filtered_mask.copy()))

    # endregion

    # region Helpers

    @property
    def has_data(self):
        return len(self.timestamps) > 0

    @property
    def duration(self):
        """Total recording duration in seconds (0 if no data)."""
        return self.timestamps[-1] if self.timestamps else 0.0

    # endregion

    # region Graph generation
    
    def save_graphs(self):
        """Generate and save all graphs to output_dir."""
        if not self.has_data:
            print("WaterDataRecorder: No data recorded — nothing to save.")
            return

        os.makedirs(self.output_dir, exist_ok=True)

        self._save_pixel_count_graph()
        self._save_expansion_graph()
        self._save_final_shape()
        self._save_incremental_layers()
        self._save_overlayed_contours()

        print(f"Graphs saved to '{self.output_dir}/'")

    # region Pixel count graph
    def _save_pixel_count_graph(self):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.timestamps, self.pixel_counts, color='tab:blue', linewidth=1)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Pixel Count")
        ax.set_title(f"Detected Pixel Count Over Time — {self.cam_label}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, "pixel_count_over_time.png"), dpi=150)
        plt.close(fig)

    # endregion

    # region Expansion graph
    def _save_expansion_graph(self):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.timestamps, self.spread_factors, color='tab:green', linewidth=1)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Spread Factor (mean σ)")
        ax.set_title(f"Group Expansion Over Time — {self.cam_label}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, "group_expansion_over_time.png"), dpi=150)
        plt.close(fig)

    # endregion

    # region Final shape graph
    def _save_final_shape(self):
        if self.final_shape_mask is None:
            return
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(self.final_shape_mask, cmap='Greens', interpolation='nearest')
        ax.set_title(f"Final Detected Shape — {self.cam_label}")
        ax.axis('off')
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, "final_shape.png"), dpi=150)
        plt.close(fig)

    # endregion

    # region Incremental layers graph
    def _save_incremental_layers(self):
        if not self.shape_snapshots:
            return

        # Pick up to 12 snapshots evenly spaced for a readable grid
        total = len(self.shape_snapshots)
        max_panels = 12
        if total <= max_panels:
            chosen = self.shape_snapshots
        else:
            indices = np.linspace(0, total - 1, max_panels, dtype=int)
            chosen = [self.shape_snapshots[i] for i in indices]

        cols = min(4, len(chosen))
        rows = (len(chosen) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
        axes = np.atleast_2d(axes)  # Ensure 2D indexing works for 1-row case

        for idx, (t, mask) in enumerate(chosen):
            r, c = divmod(idx, cols)
            axes[r, c].imshow(mask, cmap='Greens', interpolation='nearest')
            axes[r, c].set_title(f"t = {t:.1f}s")
            axes[r, c].axis('off')

        # Hide any unused subplots
        for idx in range(len(chosen), rows * cols):
            r, c = divmod(idx, cols)
            axes[r, c].axis('off')

        fig.suptitle(f"Shape Snapshots (0.1s increments) — {self.cam_label}", fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, "incremental_shape_layers.png"), dpi=150)
        plt.close(fig)

    # endregion

    # region Overlayed contours graph
    def _save_overlayed_contours(self):
        if not self.shape_snapshots:
            return

        # Use the first snapshot to determine the canvas size
        h, w = self.shape_snapshots[0][1].shape[:2]

        # Create an RGB canvas (black background)
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

        # Build a colormap: early snapshots are cool (blue), late ones are warm (red)
        n = len(self.shape_snapshots)
        cmap = plt.cm.plasma  # type: ignore[attr-defined]

        for idx, (t, mask) in enumerate(self.shape_snapshots):
            # Find contours of this snapshot's mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Map index to a color from the colormap (BGR for OpenCV)
            rgba = cmap(idx / max(n - 1, 1))
            color = (int(rgba[2] * 255), int(rgba[1] * 255), int(rgba[0] * 255))

            cv2.drawContours(canvas, contours, -1, color, 1)

        # Convert BGR canvas to RGB for matplotlib
        canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(canvas_rgb, interpolation='nearest')
        ax.set_title(f"Overlayed Shape Contours Over Time — {self.cam_label}")
        ax.axis('off')

        # Add a colorbar to show the time mapping
        sm = plt.cm.ScalarMappable(
            cmap=cmap,
            norm=mcolors.Normalize(
                vmin=self.shape_snapshots[0][0],
                vmax=self.shape_snapshots[-1][0]
            )
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Time (s)")

        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, "overlayed_contours.png"), dpi=150)
        plt.close(fig)
    # endregion

    # endregion

    # region Serialization

    def save_data(self, path: str | None = None) -> str:
        """
        Save all recorded data to a .npz file so it can be loaded later
        for offline analysis.

        Returns the path to the saved file.
        """
        path = path or os.path.join(self.output_dir, "recording_data.npz")
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Build snapshot arrays: timestamps + masks stored separately
        snap_times = np.array([t for t, _ in self.shape_snapshots], dtype=np.float64)
        snap_masks = np.array([m for _, m in self.shape_snapshots], dtype=np.uint8) if self.shape_snapshots else np.array([])

        np.savez_compressed(
            path,
            timestamps=np.array(self.timestamps, dtype=np.float64),
            pixel_counts=np.array(self.pixel_counts, dtype=np.int64),
            spread_factors=np.array(self.spread_factors, dtype=np.float64),
            snap_times=snap_times,
            snap_masks=snap_masks,
            final_shape_mask=self.final_shape_mask if self.final_shape_mask is not None else np.array([]),
            cam_label=np.array([self.cam_label]),
        )
        print(f"[recorder] Data saved to '{path}'")
        return path

    @classmethod
    def load_data(cls, path: str) -> "WaterDataRecorder":
        """
        Load a WaterDataRecorder from a .npz file saved by save_data().
        """
        data = np.load(path, allow_pickle=False)

        cam_label = str(data["cam_label"][0])
        out_dir = os.path.dirname(path)

        rec = cls(output_dir=out_dir, cam_label=cam_label)
        rec.timestamps = data["timestamps"].tolist()
        rec.pixel_counts = data["pixel_counts"].tolist()
        rec.spread_factors = data["spread_factors"].tolist()

        final = data["final_shape_mask"]
        rec.final_shape_mask = final if final.ndim == 2 else None

        snap_times = data["snap_times"]
        snap_masks = data["snap_masks"]
        if snap_times.size > 0 and snap_masks.ndim == 3:
            rec.shape_snapshots = [(float(snap_times[i]), snap_masks[i]) for i in range(len(snap_times))]

        return rec

    # endregion
