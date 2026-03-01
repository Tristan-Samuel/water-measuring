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

        print(f"[stereo] Combined graphs saved to '{self.output_dir}/'")

    # endregion

    # region Pixel count comparison

    def _save_pixel_count_comparison(self) -> None:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.top.timestamps, self.top.pixel_counts,
                label=self.top.cam_label, color="tab:blue", linewidth=1)
        ax.plot(self.side.timestamps, self.side.pixel_counts,
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
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.top.timestamps, self.top.spread_factors,
                label=self.top.cam_label, color="tab:green", linewidth=1)
        ax.plot(self.side.timestamps, self.side.spread_factors,
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
        ax1.plot(times, widths, color="tab:blue", label="Width (top cam)")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Width (px)", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")

        ax2 = ax1.twinx()
        ax2.plot(times, heights, color="tab:red", label="Height (side cam)")
        ax2.set_ylabel("Height (px)", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")

        fig.suptitle("Blob Dimensions Over Time")
        fig.legend(loc="upper left", bbox_to_anchor=(0.12, 0.88))
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, "dimensions_over_time.png"), dpi=150)
        plt.close(fig)

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
