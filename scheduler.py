"""
Time-based scheduler for the solenoid valve.

Checks the current time against a list of HH:MM triggers from config.
Runs in a background thread so it doesn't block the main loop.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime

from solenoid import SolenoidController


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
            now = datetime.now()
            today_str = now.strftime("%Y-%m-%d")

            # Reset triggered set when the day rolls over
            if today_str != self._last_date:
                self._triggered_today.clear()
                self._last_date = today_str

            current_hhmm = now.strftime("%H:%M")

            for t in self.times:
                if t == current_hhmm and t not in self._triggered_today:
                    print(f"[scheduler] Triggering solenoid (scheduled: {t})")
                    self.controller.trigger()
                    self._triggered_today.add(t)

            # Check every 30 seconds (good enough for minute-level scheduling)
            self._stop_event.wait(timeout=30)
