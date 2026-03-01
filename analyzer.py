"""
Color analysis engine for the Water Measuring system.

Two modes:
  1. RECORD — capture from camera, save video + data to recordings/<timestamp>/
     Stops by: duration, target time, or external stop-trigger file.
     Starts saving only when color is first detected.

  2. ANALYZE — offline, loads saved data from a recording folder and generates graphs.
"""

from __future__ import annotations

import os
import signal
import time
from datetime import datetime, timedelta
import cv2  # type: ignore
import numpy as np  # type: ignore

from camera import CameraBase
from water_data import WaterDataRecorder


# region Record

class Recorder:
    """
    Captures frames from a camera, runs color detection, and saves
    video + serialized data to a recording folder.

    Recording starts when color is first detected.
    Stops when one of these conditions is met:
      - duration seconds have elapsed (since color detection)
      - the wall-clock reaches until_time
      - a stop-trigger file appears on disk
      - SIGINT / SIGTERM is received
    """

    STOP_FILE = ".stop_recording"

    def __init__(
        self,
        camera: CameraBase,
        *,
        color_lower: list[int],
        color_upper: list[int],
        use_roi: bool = False,
        roi_size: int = 500,
        min_contour_area: int = 500,
        save_video: bool = True,
        fps: int = 20,
        codec: str = "mp4v",
        snapshot_interval: float = 0.1,
        recording_dir: str = "recordings",
        cam_label: str = "Camera",
        duration: float | None = None,
        until_time: str | None = None,
    ):
        self.camera = camera
        self.color_lower = np.array(color_lower)
        self.color_upper = np.array(color_upper)
        self.use_roi = use_roi
        self.roi_size = roi_size
        self.min_contour_area = min_contour_area
        self.save_video = save_video
        self.fps = fps
        self.codec = codec
        self.snapshot_interval = snapshot_interval
        self.cam_label = cam_label
        self.duration = duration
        self.until_time = until_time

        # Create timestamped output folder
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_dir = os.path.join(recording_dir, cam_label.lower().replace(" ", "_"), ts)
        os.makedirs(self.output_dir, exist_ok=True)

        # Recorder
        self.recorder = WaterDataRecorder(
            output_dir=self.output_dir,
            cam_label=cam_label,
        )
        self.recorder.SNAPSHOT_INTERVAL = snapshot_interval

        # Video writer (created on first frame so we know the resolution)
        self._writer: cv2.VideoWriter | None = None
        self._stop_requested = False

        # Parse --until into a datetime
        self._until_dt: datetime | None = None
        if until_time:
            today = datetime.now().date()
            t = datetime.strptime(until_time, "%H:%M").time()
            self._until_dt = datetime.combine(today, t)
            # If that time already passed today, assume tomorrow
            if self._until_dt <= datetime.now():
                self._until_dt += timedelta(days=1)

    # region Run

    def run(self) -> str:
        """
        Main recording loop. Returns the path to the recording folder.
        """
        # Install signal handlers for clean shutdown
        original_sigint = signal.getsignal(signal.SIGINT)
        original_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        print(f"[record] '{self.cam_label}' — waiting for color detection…")
        if self.duration:
            print(f"[record] Will record for {self.duration}s after first detection.")
        if self._until_dt:
            print(f"[record] Will stop at {self._until_dt.strftime('%H:%M')}.")
        print(f"[record] Output: {self.output_dir}")
        print(f"[record] To stop remotely: python3 cli.py stop")

        detection_start: float | None = None

        try:
            while not self._stop_requested:
                frame = self.camera.read()
                if frame is None:
                    print("[record] Camera returned None — stopping.")
                    break

                result = self._process_frame(frame)
                total_pixels = result["total_pixels"]

                # Start recording on first color detection
                if total_pixels > 0 and detection_start is None:
                    detection_start = time.time()
                    print(f"[record] Color detected! Recording started.")

                # Only save frames after detection
                if detection_start is not None:
                    if self._writer is None and self.save_video:
                        h, w = frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*self.codec)  # type: ignore[attr-defined]
                        vid_path = os.path.join(self.output_dir, "video.mp4")
                        self._writer = cv2.VideoWriter(vid_path, fourcc, self.fps, (w, h))

                    if self._writer is not None:
                        self._writer.write(result["frame"])

                    self.recorder.record_frame(
                        result["filtered_mask"],
                        total_pixels,
                        result["spread_factor"],
                    )

                # Check stop conditions
                if self._should_stop(detection_start):
                    break

        finally:
            # Restore signal handlers
            signal.signal(signal.SIGINT, original_sigint)
            signal.signal(signal.SIGTERM, original_sigterm)

        self._finish()
        return self.output_dir

    # endregion

    # region Stop conditions

    def _should_stop(self, detection_start: float | None) -> bool:
        """Check all stop conditions."""
        # Duration limit (relative to first detection)
        if self.duration and detection_start is not None:
            elapsed = time.time() - detection_start
            if elapsed >= self.duration:
                print(f"[record] Duration reached ({self.duration}s).")
                return True

        # Wall-clock time limit
        if self._until_dt and datetime.now() >= self._until_dt:
            print(f"[record] Target time reached ({self._until_dt.strftime('%H:%M')}).")
            return True

        # External stop-trigger file
        stop_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.STOP_FILE)
        if os.path.exists(stop_file):
            print("[record] Stop trigger file detected.")
            try:
                os.remove(stop_file)
            except OSError:
                pass
            return True

        return False

    def _signal_handler(self, signum, frame):
        print(f"\n[record] Signal {signum} received — stopping…")
        self._stop_requested = True

    # endregion

    # region Frame processing

    def _process_frame(self, frame: np.ndarray) -> dict:
        """Run color detection and contour analysis on one frame."""
        h, w = frame.shape[:2]

        if self.use_roi:
            roi_x1 = w // 2 - self.roi_size // 2
            roi_y1 = h // 2 - self.roi_size // 2
            roi_x2 = roi_x1 + self.roi_size
            roi_y2 = roi_y1 + self.roi_size
            roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        else:
            roi_x1, roi_y1 = 0, 0
            roi = frame

        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        mask = cv2.inRange(lab, self.color_lower, self.color_upper)

        total_pixels = 0
        spread_factor = 0.0

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_mask = np.zeros_like(mask)

        for cnt in contours:
            if cv2.contourArea(cnt) < self.min_contour_area:
                continue
            cv2.drawContours(valid_mask, [cnt], -1, 255, thickness=-1)
            cnt_offset = cnt + np.array([roi_x1, roi_y1])
            cv2.drawContours(frame, [cnt_offset], -1, (0, 255, 0), 2)

        filtered_mask = cv2.bitwise_and(mask, valid_mask)
        points = np.column_stack(np.where(filtered_mask > 0))

        if len(points) > 0:
            total_pixels = len(points)
            spread_factor = float(np.mean(np.std(points, axis=0)))

        return {
            "total_pixels": total_pixels,
            "spread_factor": spread_factor,
            "filtered_mask": filtered_mask,
            "frame": frame,
        }

    # endregion

    # region Cleanup

    def _finish(self) -> None:
        """Release resources and save recorded data."""
        self.camera.release()
        if self._writer is not None:
            self._writer.release()

        if self.recorder.has_data:
            self.recorder.save_data()
            print(f"[record] '{self.cam_label}' finished. Data saved to: {self.output_dir}")
        else:
            print(f"[record] '{self.cam_label}' finished. No color was detected.")

    # endregion

# endregion


# region Analyze

def analyze_recording(recording_path: str, output_dir: str | None = None) -> WaterDataRecorder | None:
    """
    Load saved recording data and generate all graphs.

    Args:
        recording_path: Path to a recording folder (contains recording_data.npz)
                        or directly to a .npz file.
        output_dir:     Where to save graphs. Defaults to a 'graphs/' subfolder
                        inside the recording folder.
    """
    if os.path.isdir(recording_path):
        npz_path = os.path.join(recording_path, "recording_data.npz")
    else:
        npz_path = recording_path
        recording_path = os.path.dirname(npz_path)

    if not os.path.exists(npz_path):
        print(f"[analyze] No recording data found at: {npz_path}")
        return None

    out = output_dir or os.path.join(recording_path, "graphs")
    os.makedirs(out, exist_ok=True)

    rec = WaterDataRecorder.load_data(npz_path)
    rec.output_dir = out
    rec.save_graphs()

    print(f"[analyze] Graphs saved to: {out}")
    return rec


def analyze_stereo(top_path: str, side_path: str, output_dir: str = "recordings/combined") -> None:
    """
    Load two recordings (top + side) and generate combined stereo graphs.
    """
    from stereo import StereoAnalyzer

    top_rec = _load_recorder(top_path)
    side_rec = _load_recorder(side_path)

    if top_rec is None or side_rec is None:
        print("[analyze] Could not load both recordings for stereo analysis.")
        return

    os.makedirs(output_dir, exist_ok=True)
    stereo = StereoAnalyzer(top_rec, side_rec, output_dir=output_dir)
    stereo.save_all()


def _load_recorder(path: str) -> WaterDataRecorder | None:
    """Load a WaterDataRecorder from a folder or .npz path."""
    if os.path.isdir(path):
        npz = os.path.join(path, "recording_data.npz")
    else:
        npz = path
    if not os.path.exists(npz):
        print(f"[analyze] Not found: {npz}")
        return None
    return WaterDataRecorder.load_data(npz)

# endregion
