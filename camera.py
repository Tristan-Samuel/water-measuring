"""
Camera abstraction for the Water Measuring system.

Supports:
  - Raspberry Pi Camera Module 3 via Picamera2
  - USB webcams via OpenCV (fallback / dev on non-Pi machines)

Usage:
    cam = create_camera(cam_id=0, resolution=(1920, 1080))
    frame = cam.read()       # returns a BGR numpy array or None
    cam.release()
"""

from __future__ import annotations

import sys
import numpy as np

# ── Try to import Picamera2 (only available on Raspberry Pi OS) ──
_PICAMERA2_AVAILABLE = False
try:
    from picamera2 import Picamera2  # type: ignore
    _PICAMERA2_AVAILABLE = True
except ImportError:
    # On Raspberry Pi, picamera2 is a system apt package.
    # The venv must be created with --system-site-packages to access it.
    import platform
    _is_arm = platform.machine().startswith("aarch64") or platform.machine().startswith("arm")
    if _is_arm:
        print("[camera] WARNING: picamera2 not importable on this ARM system.")
        print("         Make sure it's installed:  sudo apt install -y python3-picamera2")
        print("         And that your venv uses --system-site-packages (re-run: bash setup.sh)")
    else:
        print("[camera] Picamera2 not available, falling back to OpenCV")
    pass

import cv2  # type: ignore


# region CameraBase
class CameraBase:
    """Minimal camera interface."""

    def __init__(self, cam_id: int, resolution: tuple[int, int]):
        self.cam_id = cam_id
        self.resolution = resolution  # (width, height)

    def read(self) -> np.ndarray | None:
        """Return a BGR frame, or None on failure."""
        raise NotImplementedError

    def release(self) -> None:
        raise NotImplementedError

    @property
    def width(self) -> int:
        return self.resolution[0]

    @property
    def height(self) -> int:
        return self.resolution[1]


# endregion


# region PiCamera
class PiCamera(CameraBase):
    """Wraps Picamera2 to present the same read()/release() API."""

    def __init__(self, cam_id: int, resolution: tuple[int, int]):
        super().__init__(cam_id, resolution)

        # Check how many cameras libcamera can see
        cameras = Picamera2.global_camera_info()
        if cam_id >= len(cameras):
            available = len(cameras)
            msg = (
                f"Camera index {cam_id} requested but libcamera only sees "
                f"{available} camera(s).\n"
            )
            if available > 0:
                for i, info in enumerate(cameras):
                    msg += f"  [{i}] {info.get('Model', 'unknown')} — {info.get('Location', 'unknown')} ({info.get('Id', '')})\n"
            msg += (
                "\nTips:\n"
                "  • Run: libcamera-hello --list-cameras\n"
                "  • Check both ribbon cables are seated firmly\n"
                "  • Make sure both camera ports are enabled in raspi-config\n"
                "  • Try: sudo dmesg | grep -i imx"
            )
            raise RuntimeError(msg)

        self._cam = Picamera2(cam_id)
        config = self._cam.create_preview_configuration(
            main={"size": resolution, "format": "BGR888"}
        )
        self._cam.configure(config)
        self._cam.start()

    def read(self) -> np.ndarray | None:
        try:
            frame = self._cam.capture_array("main")
            return frame
        except Exception:
            return None

    def release(self) -> None:
        try:
            self._cam.stop()
            self._cam.close()
        except Exception:
            pass


# endregion


# region CVCamera
class CVCamera(CameraBase):
    """Wraps cv2.VideoCapture."""

    def __init__(self, cam_id: int, resolution: tuple[int, int]):
        super().__init__(cam_id, resolution)
        self._cap = cv2.VideoCapture(cam_id)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera {cam_id}")

    def read(self) -> np.ndarray | None:
        ret, frame = self._cap.read()
        return frame if ret else None

    def release(self) -> None:
        self._cap.release()


# endregion


# region Factory
def create_camera(cam_id: int = 0,
                  resolution: tuple[int, int] = (1920, 1080),
                  force_opencv: bool = False) -> CameraBase:
    """
    Create a camera instance.

    On Raspberry Pi (picamera2 available) → PiCamera.
    Otherwise (or if force_opencv=True)   → CVCamera.
    """
    if _PICAMERA2_AVAILABLE and not force_opencv:
        print(f"[camera] Using Picamera2 (id={cam_id}, {resolution[0]}x{resolution[1]})")
        return PiCamera(cam_id, resolution)
    else:
        print(f"[camera] Using OpenCV (id={cam_id}, {resolution[0]}x{resolution[1]})")
        return CVCamera(cam_id, resolution)
# endregion


# region Diagnostics
def list_cameras() -> None:
    """Print all cameras detected by libcamera / OpenCV."""
    if _PICAMERA2_AVAILABLE:
        cameras = Picamera2.global_camera_info()
        print(f"Picamera2 sees {len(cameras)} camera(s):")
        if not cameras:
            print("  (none)")
        for i, info in enumerate(cameras):
            model = info.get("Model", "unknown")
            loc = info.get("Location", "?")
            dev_id = info.get("Id", "")
            print(f"  [{i}] {model} — location={loc}  ({dev_id})")
    else:
        print("Picamera2 not available. Checking OpenCV devices:")
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"  [{i}] OpenCV camera — {w}x{h}")
                cap.release()

    print("\nFor more detail run: libcamera-hello --list-cameras")
# endregion
