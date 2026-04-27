"""
Time-based scheduler for the solenoid valve and for recordings.

Checks the current time against a list of HH:MM triggers from config.
Runs in a background thread so it doesn't block the main loop.
"""

from __future__ import annotations

import subprocess
import threading
import time
from datetime import datetime

from solenoid import SolenoidController


def _count_pi_cameras() -> int:
    """Return the number of cameras libcamera can see, or -1 if unavailable."""
    try:
        from picamera2 import Picamera2  # type: ignore
        return len(Picamera2.global_camera_info())
    except Exception:
        return -1  # not on Pi, or libcamera unavailable — don't reboot


def _minutes_until_next(times: list[str]) -> float | None:
    """Return minutes until the nearest upcoming HH:MM UTC trigger today or tomorrow."""
    if not times:
        return None
    now = datetime.utcnow()
    today = now.strftime("%Y-%m-%d")
    min_delta = None
    for t in times:
        try:
            trigger = datetime.strptime(f"{today} {t}", "%Y-%m-%d %H:%M")
        except ValueError:
            continue
        delta = (trigger - now).total_seconds() / 60
        if delta < 0:
            delta += 1440  # trigger already passed today, check tomorrow
        if min_delta is None or delta < min_delta:
            min_delta = delta
    return min_delta


class SolenoidScheduler:
    """
    Triggers a SolenoidController at configured times of day.

    Usage:
        sched = SolenoidScheduler(controller, times=["08:00", "14:00"])
        sched.start()    # runs in background
        ...
        sched.stop()
    """

    def __init__(self, controller: SolenoidController, times: list[str]):
        """
        Args:
            controller: The SolenoidController to trigger.
            times:      List of "HH:MM" strings (24-hour format).
        """
        self.controller = controller
        self.times = times  # e.g. ["08:00", "14:00", "20:00"]
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._triggered_today: set[str] = set()
        self._last_date: str = ""

    def start(self) -> None:
        """Start the scheduler in a background daemon thread."""
        if self._thread is not None and self._thread.is_alive():
            print("[scheduler] Already running")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print(f"[scheduler] Started — triggers at {self.times}")

    def stop(self) -> None:
        """Stop the scheduler."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        print("[scheduler] Stopped")

    def _run(self) -> None:
        while not self._stop_event.is_set():
            now = datetime.utcnow()
            today_str = now.strftime("%Y-%m-%d")

            # Reset triggered set when the day rolls over
            if today_str != self._last_date:
                self._triggered_today.clear()
                self._last_date = today_str

            current_hhmm = now.strftime("%H:%M")

            for t in self.times:
                if t == current_hhmm and t not in self._triggered_today:
                    print(f"[scheduler] Triggering solenoid (scheduled UTC: {t})")
                    self.controller.trigger()
                    self._triggered_today.add(t)

            # Check every 30 seconds (good enough for minute-level scheduling)
            self._stop_event.wait(timeout=30)


class RecordingScheduler:
    """
    Starts recordings at configured times of day.

    Usage:
        sched = RecordingScheduler(
            build_recorder_fn=...,  # callable(camera, duration, until) → Recorder
            times=["08:00", "14:00"],
            camera="both",
            duration=120,
            until=None,
        )
        sched.start()
        ...
        sched.stop()
    """

    def __init__(
        self,
        build_recorder_fn,
        times: list[str],
        camera: str = "both",
        duration: float | None = None,
        until: str | None = None,
    ):
        """
        Args:
            build_recorder_fn: callable(cam_name, duration, until) that
                               creates and returns a Recorder instance.
            times:    List of "HH:MM" strings (24-hour format).
            camera:   "top", "side", or "both".
            duration: Recording duration in seconds (None = no limit).
            until:    Wall-clock stop time "HH:MM" (None = use duration).
        """
        self.build_recorder_fn = build_recorder_fn
        self.times = times
        self.camera = camera
        self.duration = duration
        self.until = until
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._triggered_today: set[str] = set()
        self._last_date: str = ""
        self._active_threads: list[threading.Thread] = []

    def start(self) -> None:
        """Start the scheduler in a background daemon thread."""
        if self._thread is not None and self._thread.is_alive():
            print("[rec-scheduler] Already running")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        cam_label = self.camera
        dur_label = f"{self.duration}s" if self.duration else "no limit"
        until_label = self.until or "—"
        print(f"[rec-scheduler] Started — camera={cam_label}, "
              f"duration={dur_label}, until={until_label}, times={self.times}")

    def stop(self) -> None:
        """Stop the scheduler (does NOT stop active recordings)."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        print("[rec-scheduler] Stopped")

    def _run(self) -> None:
        while not self._stop_event.is_set():
            now = datetime.utcnow()
            today_str = now.strftime("%Y-%m-%d")

            if today_str != self._last_date:
                self._triggered_today.clear()
                self._last_date = today_str

            current_hhmm = now.strftime("%H:%M")

            for t in self.times:
                if t == current_hhmm and t not in self._triggered_today:
                    self._triggered_today.add(t)
                    print(f"[rec-scheduler] Starting recording (scheduled UTC: {t})")
                    self._start_recording()

            # Clean up finished threads
            self._active_threads = [th for th in self._active_threads if th.is_alive()]

            # Failsafe: if a recording is due within 5 minutes and fewer than
            # 2 cameras are detected, reboot so the Pi re-initialises the camera stack.
            minutes = _minutes_until_next(self.times)
            if minutes is not None and minutes <= 5:
                cam_count = _count_pi_cameras()
                if cam_count != -1 and cam_count < 2:
                    print(
                        f"[rec-scheduler] FAILSAFE: only {cam_count} camera(s) detected "
                        f"with a trigger in {minutes:.1f} min — rebooting."
                    )
                    subprocess.run(["sudo", "reboot"], check=False)

            self._stop_event.wait(timeout=30)

    def _start_recording(self) -> None:
        """Launch a recording in a new thread."""
        cams = [self.camera] if self.camera in ("top", "side") else ["top", "side"]

        for cam_name in cams:
            def _record(cn=cam_name):
                try:
                    rec = self.build_recorder_fn(cn, self.duration, self.until)
                    rec.run()
                except Exception as e:
                    print(f"[rec-scheduler] Error recording {cn}: {e}")

            th = threading.Thread(target=_record, daemon=True)
            th.start()
            self._active_threads.append(th)
