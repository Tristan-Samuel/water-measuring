import time
import os
import numpy as np  # type: ignore
import matplotlib  # type: ignore
matplotlib.use('Agg')  # Non-interactive backend so plots save without a display window
import matplotlib.pyplot as plt  # type: ignore


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

    def __init__(self, output_dir="results"):
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
    
    # MARK: Main loop

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

    # MARK: Helpers
    
    @property
    def has_data(self):
        return len(self.timestamps) > 0

    @property
    def duration(self):
        """Total recording duration in seconds (0 if no data)."""
        return self.timestamps[-1] if self.timestamps else 0.0

    # MARK: Graph generation

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

        print(f"Graphs saved to '{self.output_dir}/'")

    # --- 1. Pixel count over time ---
    def _save_pixel_count_graph(self):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.timestamps, self.pixel_counts, color='tab:blue', linewidth=1)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Pixel Count")
        ax.set_title("Detected Pixel Count Over Time")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, "pixel_count_over_time.png"), dpi=150)
        plt.close(fig)

    # --- 2. Group expansion (spread) over time ---
    def _save_expansion_graph(self):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.timestamps, self.spread_factors, color='tab:green', linewidth=1)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Spread Factor (mean σ)")
        ax.set_title("Group Expansion Over Time")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, "group_expansion_over_time.png"), dpi=150)
        plt.close(fig)

    # --- 3. Final shape ---
    def _save_final_shape(self):
        if self.final_shape_mask is None:
            return
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(self.final_shape_mask, cmap='Greens', interpolation='nearest')
        ax.set_title("Final Detected Shape")
        ax.axis('off')
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, "final_shape.png"), dpi=150)
        plt.close(fig)

    # --- 4. Incremental shape layers ---
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

        fig.suptitle("Shape Snapshots (0.1s increments)", fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, "incremental_shape_layers.png"), dpi=150)
        plt.close(fig)
