"""
Solenoid valve controller for the Water Measuring system.

Controls a solenoid valve connected to a Raspberry Pi GPIO pin.
On non-Pi machines, operates in simulation mode (prints actions).

Usage:
    valve = SolenoidController(gpio_pin=17)
    valve.open(duration=5.0)   # opens for 5 seconds, then closes
    valve.trigger()            # alias using configured default duration
    valve.close()              # force-close immediately
    valve.cleanup()            # release GPIO resources
"""

from __future__ import annotations

import time
import threading

# ── Try to import RPi.GPIO (only on Raspberry Pi) ──
_GPIO_AVAILABLE = False
try:
    import RPi.GPIO as GPIO  # type: ignore
    _GPIO_AVAILABLE = True
except ImportError:
    pass


class SolenoidController:
    """
    Drives a solenoid valve via a GPIO pin.

    The pin is driven HIGH to open and LOW to close (active-high).
    Falls back to simulation mode on non-Pi hardware.
    """

    def __init__(self, gpio_pin: int = 17, default_duration: float = 5.0):
        self.gpio_pin = gpio_pin
        self.default_duration = default_duration
        self._is_open = False
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None
        self._simulation = not _GPIO_AVAILABLE

        if not self._simulation:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.gpio_pin, GPIO.OUT, initial=GPIO.LOW)
            print(f"[solenoid] GPIO {self.gpio_pin} configured (BCM mode)")
        else:
            print(f"[solenoid] Simulation mode (GPIO not available)")

    # region Public API

    def open(self, duration: float | None = None) -> None:
        """
        Open the valve for *duration* seconds, then automatically close.
        If duration is None, uses self.default_duration.
        """
        duration = duration if duration is not None else self.default_duration

        with self._lock:
            self._cancel_pending_timer()
            self._set_pin(True)
            print(f"[solenoid] OPENED — will close in {duration:.1f}s")

            # Schedule automatic close
            self._timer = threading.Timer(duration, self._auto_close)
            self._timer.daemon = True
            self._timer.start()

    def trigger(self) -> None:
        """Open the valve for the default duration."""
        self.open(self.default_duration)

    def close(self) -> None:
        """Immediately close the valve."""
        with self._lock:
            self._cancel_pending_timer()
            self._set_pin(False)
            print("[solenoid] CLOSED")

    @property
    def is_open(self) -> bool:
        return self._is_open

    def cleanup(self) -> None:
        """Release GPIO resources."""
        self.close()
        if not self._simulation:
            try:
                GPIO.cleanup(self.gpio_pin)
            except Exception:
                pass
        print("[solenoid] Cleaned up")

    # endregion

    # region Internal

    def _set_pin(self, high: bool) -> None:
        self._is_open = high
        if not self._simulation:
            GPIO.output(self.gpio_pin, GPIO.HIGH if high else GPIO.LOW)

    def _auto_close(self) -> None:
        with self._lock:
            self._set_pin(False)
            print("[solenoid] AUTO-CLOSED (timer expired)")

    def _cancel_pending_timer(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
    # endregion
