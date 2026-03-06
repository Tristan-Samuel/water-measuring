#!/usr/bin/env python3
"""
Water Measuring System — CLI

Recording (headless, designed for Raspberry Pi over SSH):
    python3 cli.py record --camera top --duration 60
    python3 cli.py record --camera side --until 14:30
    python3 cli.py record --camera both --duration 120

Stopping a recording remotely (from another SSH session):
    python3 cli.py stop

Offline analysis (generate graphs from saved recordings):
    python3 cli.py analyze recordings/top_camera/2025-01-15_10-30-00
    python3 cli.py analyze --latest top
    python3 cli.py analyze --stereo recordings/top/... recordings/side/...

Solenoid control:
    python3 cli.py solenoid open
    python3 cli.py solenoid open --duration 3.0
    python3 cli.py solenoid close
    python3 cli.py solenoid test

Color detection (per-camera or global):
    python3 cli.py color '#C8C800' --tolerance 30
    python3 cli.py color '#C8C800' --camera top --tolerance 30
    python3 cli.py color '#C8C800' --camera side --tolerance 50
    python3 cli.py color --show
    python3 cli.py color --show --camera side

Scheduling:
    python3 cli.py schedule start

Utilities:
    python3 cli.py config                    # show config
    python3 cli.py update                    # git pull latest code
    python3 cli.py recordings                # list saved recordings
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import threading
import time

from config_loader import (
    load_config,
    camera_cfg,
    camera_crop,
    color_range,
    analysis_cfg,
    solenoid_cfg,
    schedule_cfg,
    recording_cfg,
    clahe_cfg,
)
from camera import create_camera, list_cameras
from analyzer import Recorder, analyze_recording, analyze_stereo
from solenoid import SolenoidController
from scheduler import SolenoidScheduler, RecordingScheduler


# region Helpers

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def _build_recorder(cfg: dict, cam_name: str, *, duration: float | None, until: str | None) -> Recorder:
    """Create a Recorder for a given camera name."""
    cam_c = camera_cfg(cfg, cam_name)
    ana_c = analysis_cfg(cfg)
    rec_c = recording_cfg(cfg)
    lower, upper = color_range(cfg, cam_name)
    cl = clahe_cfg(cfg)

    camera = create_camera(
        cam_id=cam_c["id"],
        resolution=tuple(cam_c["resolution"]),
    )

    return Recorder(
        camera=camera,
        color_lower=lower,
        color_upper=upper,
        crop=camera_crop(cfg, cam_name),
        use_roi=ana_c["use_roi"],
        roi_size=ana_c["roi_size"],
        min_contour_area=ana_c["min_contour_area"],
        save_video=rec_c["save_video"],
        fps=rec_c["fps"],
        codec=rec_c["codec"],
        snapshot_interval=rec_c.get("snapshot_interval", 0.1),
        recording_dir=os.path.join(PROJECT_DIR, "recordings"),
        cam_label=cam_c["label"],
        duration=duration,
        until_time=until,
        clahe_enabled=cl["enabled"],
        clahe_clip_limit=cl["clip_limit"],
        clahe_grid_size=tuple(cl["grid_size"]),
    )


def _find_latest_recording(cam_name: str) -> str | None:
    """Find the most recent recording folder for a camera."""
    base = os.path.join(PROJECT_DIR, "recordings", cam_name.lower().replace(" ", "_"))
    if not os.path.isdir(base):
        return None
    folders = sorted(glob.glob(os.path.join(base, "*")))
    return folders[-1] if folders else None


# endregion


# region Commands

def cmd_record(args, cfg):
    """Start recording from one or both cameras."""
    import signal as _signal

    cam = args.camera
    duration = args.duration
    until = args.until

    if cam in ("top", "side"):
        rec = _build_recorder(cfg, cam, duration=duration, until=until)

        # Install signal handlers on the main thread
        def _stop_handler(signum, frame):
            print(f"\n[cli] Signal {signum} received — stopping…")
            rec.request_stop()
        _signal.signal(_signal.SIGINT, _stop_handler)
        _signal.signal(_signal.SIGTERM, _stop_handler)

        rec.run()

    elif cam == "both":
        top_rec = _build_recorder(cfg, "top", duration=duration, until=until)
        side_rec = _build_recorder(cfg, "side", duration=duration, until=until)

        # Install signal handlers on the main thread to stop both recorders
        def _stop_handler(signum, frame):
            print(f"\n[cli] Signal {signum} received — stopping both cameras…")
            top_rec.request_stop()
            side_rec.request_stop()
        _signal.signal(_signal.SIGINT, _stop_handler)
        _signal.signal(_signal.SIGTERM, _stop_handler)

        top_thread = threading.Thread(target=top_rec.run, daemon=True)
        side_thread = threading.Thread(target=side_rec.run, daemon=True)

        top_thread.start()
        side_thread.start()

        top_thread.join()
        side_thread.join()

    else:
        print(f"Unknown camera: {cam}. Use 'top', 'side', or 'both'.")
        sys.exit(1)


def cmd_stop(args, cfg):
    """Create a stop-trigger file so an active recording stops."""
    stop_path = os.path.join(PROJECT_DIR, Recorder.STOP_FILE)
    with open(stop_path, "w") as f:
        f.write("stop")
    print(f"[cli] Stop signal written. Active recording will stop shortly.")


def cmd_analyze(args, cfg):
    """Run offline analysis on a saved recording."""
    # --latest-stereo: auto-find latest top + side recordings
    if args.latest_stereo:
        top_path = _find_latest_recording("top_camera")
        side_path = _find_latest_recording("side_camera")
        if top_path is None or side_path is None:
            # Try label variants from config
            for name in ("top", "side"):
                cam_c = camera_cfg(cfg, name)
                label_dir = cam_c["label"].lower().replace(" ", "_")
                path = _find_latest_recording(label_dir)
                if name == "top" and path:
                    top_path = path
                elif name == "side" and path:
                    side_path = path
        if top_path is None or side_path is None:
            missing = []
            if top_path is None:
                missing.append("top")
            if side_path is None:
                missing.append("side")
            print(f"[cli] No recordings found for: {', '.join(missing)}")
            print("      Run 'python3 cli.py recordings' to see what's available.")
            sys.exit(1)
        print(f"[cli] Top:  {top_path}")
        print(f"[cli] Side: {side_path}")
        out = args.output or os.path.join(PROJECT_DIR, "recordings", "combined")
        analyze_stereo(top_path, side_path, output_dir=out)
        return

    if args.stereo:
        if len(args.stereo) != 2:
            print("[cli] --stereo requires exactly 2 paths: <top> <side>")
            sys.exit(1)
        out = args.output or os.path.join(PROJECT_DIR, "recordings", "combined")
        analyze_stereo(args.stereo[0], args.stereo[1], output_dir=out)
        return

    # Determine recording path
    if args.path:
        rec_path = args.path
    elif args.latest:
        rec_path = _find_latest_recording(args.latest)
        if rec_path is None:
            # Try with config label
            cam_c = camera_cfg(cfg, args.latest)
            label_dir = cam_c["label"].lower().replace(" ", "_")
            rec_path = _find_latest_recording(label_dir)
        if rec_path is None:
            print(f"[cli] No recordings found for '{args.latest}'.")
            sys.exit(1)
        print(f"[cli] Using latest recording: {rec_path}")
    else:
        print("[cli] Provide a recording path, --latest <camera>, or --latest-stereo.")
        sys.exit(1)

    out = args.output
    result = analyze_recording(rec_path, output_dir=out)
    if result is None:
        sys.exit(1)


def cmd_solenoid(args, cfg):
    """Control the solenoid valve."""
    sol_c = solenoid_cfg(cfg)
    controller = SolenoidController(
        gpio_pin=sol_c["gpio_pin"],
        default_duration=sol_c["open_duration"],
    )

    action = args.action

    if action == "open":
        duration = args.duration if args.duration is not None else sol_c["open_duration"]
        controller.open(duration)
        time.sleep(duration + 0.5)
    elif action == "close":
        controller.close()
    elif action == "test":
        print("[cli] Test pulse: 1 second")
        controller.open(1.0)
        time.sleep(1.5)
    else:
        print(f"Unknown action: {action}. Use 'open', 'close', or 'test'.")
        sys.exit(1)

    controller.cleanup()


def cmd_schedule(args, cfg):
    """Start the solenoid and/or recording schedule."""
    sch_c = schedule_cfg(cfg)

    schedulers = []

    # --- Solenoid schedule ---
    run_solenoid = sch_c.get("enabled", False)
    if args.mode in (None, "solenoid"):
        run_solenoid = True
    if args.mode == "recording":
        run_solenoid = False

    if run_solenoid:
        sol_c = solenoid_cfg(cfg)
        controller = SolenoidController(
            gpio_pin=sol_c["gpio_pin"],
            default_duration=sol_c["open_duration"],
        )
        sol_sched = SolenoidScheduler(controller, sch_c["times"])
        sol_sched.start()
        schedulers.append(("solenoid", sol_sched, controller))

    # --- Recording schedule ---
    rec_sc = sch_c.get("recording", {})
    run_recording = rec_sc.get("enabled", False)
    if args.mode in (None, "recording"):
        run_recording = True
    if args.mode == "solenoid":
        run_recording = False

    if run_recording:
        camera = args.rec_camera or rec_sc.get("camera", "both")
        duration = args.rec_duration if args.rec_duration is not None else rec_sc.get("duration")
        until = args.rec_until or rec_sc.get("until")
        times = rec_sc.get("times", sch_c.get("times", []))

        def _build(cam_name, dur, unt):
            return _build_recorder(cfg, cam_name, duration=dur, until=unt)

        rec_sched = RecordingScheduler(
            build_recorder_fn=_build,
            times=times,
            camera=camera,
            duration=duration,
            until=until,
        )
        rec_sched.start()
        schedulers.append(("recording", rec_sched, None))

    if not schedulers:
        print("[cli] Nothing to schedule. Enable schedule in config or use --mode.")
        return

    print("[cli] Scheduler running. Press Ctrl-C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass

    for label, sched, extra in schedulers:
        sched.stop()
        if extra is not None and hasattr(extra, "cleanup"):
            extra.cleanup()


def cmd_recordings(args, cfg):
    """List all saved recordings."""
    base = os.path.join(PROJECT_DIR, "recordings")
    if not os.path.isdir(base):
        print("[cli] No recordings folder found.")
        return

    found = False
    for cam_dir in sorted(os.listdir(base)):
        cam_path = os.path.join(base, cam_dir)
        if not os.path.isdir(cam_path):
            continue
        sessions = sorted(os.listdir(cam_path))
        if sessions:
            found = True
            print(f"\n  {cam_dir}/")
            for s in sessions:
                full = os.path.join(cam_path, s)
                files = os.listdir(full) if os.path.isdir(full) else []
                has_data = "recording_data.npz" in files
                has_video = "video.mp4" in files
                has_graphs = "graphs" in files
                markers = []
                if has_data:
                    markers.append("data")
                if has_video:
                    markers.append("video")
                if has_graphs:
                    markers.append("graphs")
                print(f"    {s}  [{', '.join(markers) or 'empty'}]")

    if not found:
        print("[cli] No recordings found.")


def cmd_config(args, cfg):
    """Show the current configuration."""
    import yaml  # type: ignore
    print(yaml.dump(cfg, default_flow_style=False, sort_keys=False))


def cmd_update(args, cfg):
    """Pull the latest code from git."""
    print("[cli] Pulling latest code…")
    result = subprocess.run(
        ["git", "pull"],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        sys.exit(1)
    print("[cli] Update complete.")


def cmd_cameras(args, cfg):
    """List detected cameras."""
    list_cameras()


def cmd_color(args, cfg):
    """Set detection color from hex, RGB, or show current."""
    import yaml as _yaml  # type: ignore

    if args.show or (args.value is None):
        lower = cfg["color"]["lower"]
        upper = cfg["color"]["upper"]
        print(f"Current CIELAB range:")
        print(f"  Lower: {lower}")
        print(f"  Upper: {upper}")
        return

    value = args.value.strip()
    tolerance = args.tolerance

    try:
        L, a, b = _parse_color_to_lab(value)
    except ValueError as e:
        print(f"[cli] {e}")
        sys.exit(1)

    # Build CIELAB bounds with tolerance
    lower = [
        max(0, int(L - tolerance * 1.0)),    # L tolerance (wider)
        max(0, int(a - tolerance)),
        max(0, int(b - tolerance)),
    ]
    upper = [
        min(255, int(L + tolerance * 1.0)),
        min(255, int(a + tolerance)),
        min(255, int(b + tolerance)),
    ]

    # Update config.yaml
    cfg["color"]["lower"] = lower
    cfg["color"]["upper"] = upper

    config_path = args.config or os.path.join(PROJECT_DIR, "config.yaml")
    with open(config_path, "r") as f:
        raw = f.read()

    # Replace the color section in-place
    import re
    lower_str = f"{lower[0]}, {lower[1]}, {lower[2]}"
    upper_str = f"{upper[0]}, {upper[1]}, {upper[2]}"
    raw = re.sub(
        r'(lower:\s*\[)[^\]]*(\])',
        lambda m: m.group(1) + lower_str + m.group(2),
        raw,
    )
    raw = re.sub(
        r'(upper:\s*\[)[^\]]*(\])',
        lambda m: m.group(1) + upper_str + m.group(2),
        raw,
    )
    with open(config_path, "w") as f:
        f.write(raw)

    print(f"[cli] Color set from: {value}")
    print(f"  CIELAB center: L={L:.0f}, a={a:.0f}, b={b:.0f}")
    print(f"  Lower bound:   {lower}")
    print(f"  Upper bound:   {upper}")
    print(f"  Tolerance:     ±{tolerance}")
    print(f"  Saved to:      {config_path}")


def _parse_color_to_lab(value: str) -> tuple[float, float, float]:
    """
    Parse a color string and return CIELAB (L, a, b) values.

    Accepts:
      - Hex:      '#FF5733' or 'FF5733'
      - RGB:      'rgb(255, 87, 51)' or '255,87,51'
      - Lab:      'lab(50, 30, 40)'
    """
    import cv2 as _cv2
    import numpy as _np

    value = value.strip()

    # Lab passthrough
    if value.lower().startswith("lab("):
        nums = value[4:].rstrip(")").split(",")
        if len(nums) != 3:
            raise ValueError("lab() needs 3 values: lab(L, a, b)")
        return float(nums[0]), float(nums[1]), float(nums[2])

    # Hex
    if value.startswith("#"):
        value = value[1:]
    if len(value) == 6 and all(c in "0123456789abcdefABCDEF" for c in value):
        r = int(value[0:2], 16)
        g = int(value[2:4], 16)
        b = int(value[4:6], 16)
    # RGB string
    elif value.lower().startswith("rgb("):
        nums = value[4:].rstrip(")").split(",")
        r, g, b = int(nums[0]), int(nums[1]), int(nums[2])
    elif "," in value:
        parts = value.split(",")
        if len(parts) == 3:
            r, g, b = int(parts[0]), int(parts[1]), int(parts[2])
        else:
            raise ValueError(f"Can't parse color: {value}")
    else:
        raise ValueError(
            f"Can't parse '{value}'. Use hex (#FF5733), "
            f"rgb(255,87,51), or lab(50,30,40)"
        )

    # Convert RGB → CIELAB via OpenCV
    pixel = _np.uint8([[[b, g, r]]])  # OpenCV uses BGR
    lab = _cv2.cvtColor(pixel, _cv2.COLOR_BGR2LAB)
    L, a_val, b_val = int(lab[0, 0, 0]), int(lab[0, 0, 1]), int(lab[0, 0, 2])
    return float(L), float(a_val), float(b_val)


def cmd_live(args, cfg):
    """Start the Flask debug viewer."""
    # Import here so Flask isn't required for normal operation
    try:
        from web_viewer import create_app
    except ImportError:
        print("[cli] Flask is required for live viewer.")
        print("      Install it: pip install flask")
        sys.exit(1)

    host = args.host
    port = args.port
    print(f"[cli] Starting live viewer at http://{host}:{port}")
    print(f"      Open this URL in your browser (same network as the Pi).")
    app = create_app(cfg)
    app.run(host=host, port=port, threaded=True)


# endregion


# region Argument parser

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="water",
        description="Water Measuring System — Raspberry Pi CLI",
        epilog=(
            "Examples:\n"
            "  python3 cli.py record --camera both --duration 60\n"
            "  python3 cli.py analyze --latest-stereo\n"
            "  python3 cli.py color '#3AB5E6' --tolerance 40\n"
            "  python3 cli.py live --port 8080\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to config.yaml (default: ./config.yaml)",
    )
    sub = parser.add_subparsers(dest="command", required=True,
                                metavar="COMMAND")

    # ── record ──
    rec_p = sub.add_parser("record",
        help="Start recording from camera(s)",
        description=(
            "Start recording from one or both cameras.\n\n"
            "Recording begins when the target color is first detected in frame.\n"
            "Stops when --duration expires, --until time is reached, or\n"
            "'python3 cli.py stop' is run from another terminal."
        ),
        epilog=(
            "Examples:\n"
            "  python3 cli.py record --camera top --duration 60\n"
            "  python3 cli.py record --camera side --until 14:30\n"
            "  python3 cli.py record --camera both --duration 120\n"
            "  python3 cli.py record                  # both cameras, no time limit\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    rec_p.add_argument(
        "--camera", "-c", choices=["top", "side", "both"], default="both",
        help="Which camera(s) to record from (default: both)",
    )
    rec_p.add_argument(
        "--duration", "-d", type=float, default=None,
        help="Stop after N seconds (counted from first color detection)",
    )
    rec_p.add_argument(
        "--until", "-u", type=str, default=None,
        metavar="HH:MM",
        help="Stop at a wall-clock time, e.g. '14:30' or '09:00'",
    )

    # ── stop ──
    sub.add_parser("stop",
        help="Signal an active recording to stop",
        description=(
            "Creates a .stop_recording trigger file that an active\n"
            "recording watches for. Run this from a second SSH session\n"
            "to gracefully stop a recording that's in progress."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── analyze ──
    ana_p = sub.add_parser("analyze",
        help="Generate graphs from a saved recording",
        description=(
            "Run offline analysis on saved recording data.\n\n"
            "Can analyze a single camera recording or combine top + side\n"
            "cameras for stereo analysis (spread comparison, 3D reconstruction)."
        ),
        epilog=(
            "Examples:\n"
            "  python3 cli.py analyze recordings/top_camera/2025-01-15_10-30-00\n"
            "  python3 cli.py analyze --latest top\n"
            "  python3 cli.py analyze --latest-stereo\n"
            "  python3 cli.py analyze --stereo recordings/top/... recordings/side/...\n"
            "  python3 cli.py analyze --latest top --output ~/Desktop/results\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ana_p.add_argument(
        "path", nargs="?", default=None,
        help="Path to a recording folder (containing recording_data.npz)",
    )
    ana_p.add_argument(
        "--latest", "-l", type=str, default=None,
        metavar="CAMERA",
        help="Auto-find the latest recording for CAMERA ('top' or 'side')",
    )
    ana_p.add_argument(
        "--latest-stereo", action="store_true", default=False,
        help="Auto-find the latest top + side recordings and run stereo analysis",
    )
    ana_p.add_argument(
        "--stereo", "-s", nargs=2, default=None,
        metavar=("TOP_PATH", "SIDE_PATH"),
        help="Run stereo analysis on two recording folders (top first, side second)",
    )
    ana_p.add_argument(
        "--output", "-o", type=str, default=None,
        metavar="DIR",
        help="Output directory for graphs (default: graphs/ inside the recording folder)",
    )

    # ── solenoid ──
    sol_p = sub.add_parser("solenoid",
        help="Control the solenoid valve",
        description=(
            "Manually open, close, or test the solenoid valve.\n\n"
            "Uses the GPIO pin and default duration from config.yaml.\n"
            "On non-Pi systems, runs in simulation mode (no GPIO)."
        ),
        epilog=(
            "Examples:\n"
            "  python3 cli.py solenoid open\n"
            "  python3 cli.py solenoid open --duration 5.0\n"
            "  python3 cli.py solenoid close\n"
            "  python3 cli.py solenoid test              # 1-second pulse\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sol_p.add_argument(
        "action", choices=["open", "close", "test"],
        help="'open' = energise valve, 'close' = de-energise, 'test' = 1 s pulse",
    )
    sol_p.add_argument(
        "--duration", "-d", type=float, default=None,
        help="Override the open duration in seconds (only for 'open')",
    )

    # ── schedule ──
    sched_p = sub.add_parser("schedule",
        help="Start the solenoid and/or recording schedule (Ctrl-C to stop)",
        description=(
            "Runs background daemons that trigger the solenoid and/or\n"
            "start recordings at the times listed in config.yaml.\n\n"
            "By default, both solenoid and recording schedules are started\n"
            "if their 'enabled' flag is true in config.yaml. Use --mode to\n"
            "run only one type.\n\n"
            "Press Ctrl-C to stop."
        ),
        epilog=(
            "Examples:\n"
            "  python3 cli.py schedule                        # run all enabled schedules\n"
            "  python3 cli.py schedule --mode solenoid         # solenoid only\n"
            "  python3 cli.py schedule --mode recording         # recording only\n"
            "  python3 cli.py schedule --mode recording --camera top --duration 60\n"
            "  python3 cli.py schedule --mode recording --until 14:30\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sched_p.add_argument(
        "--mode", choices=["solenoid", "recording"],
        default=None,
        help="Run only the solenoid or recording schedule (default: both if enabled)",
    )
    sched_p.add_argument(
        "--camera", dest="rec_camera", choices=["top", "side", "both"],
        default=None,
        help="Camera for recording schedule (overrides config)",
    )
    sched_p.add_argument(
        "--duration", dest="rec_duration", type=float, default=None,
        help="Recording duration in seconds (overrides config)",
    )
    sched_p.add_argument(
        "--until", dest="rec_until", default=None,
        help="Wall-clock stop time HH:MM for recordings (overrides config)",
    )

    # ── recordings ──
    sub.add_parser("recordings",
        help="List all saved recordings",
        description=(
            "Scans the recordings/ folder and prints every saved session\n"
            "with markers showing which files are present (data, video, graphs)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── config ──
    sub.add_parser("config",
        help="Show current configuration",
        description="Prints the full contents of config.yaml as YAML.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── update ──
    sub.add_parser("update",
        help="Pull the latest code from git",
        description="Runs 'git pull' in the project directory to fetch updates.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── cameras ──
    sub.add_parser("cameras",
        help="List detected cameras (diagnostic)",
        description=(
            "Prints all cameras detected by Picamera2 (libcamera) or OpenCV.\n"
            "Useful for verifying both cameras are connected and their indices."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── color ──
    col_p = sub.add_parser("color",
        help="Set or show the detection color",
        description=(
            "View or change the CIELAB color detection range.\n\n"
            "Accepts colors as hex, RGB, or direct CIELAB values.\n"
            "Automatically converts to CIELAB and writes bounds\n"
            "(center ± tolerance) into config.yaml."
        ),
        epilog=(
            "Examples:\n"
            "  python3 cli.py color '#FF5733'\n"
            "  python3 cli.py color 'rgb(255, 87, 51)'\n"
            "  python3 cli.py color '255,87,51' --tolerance 40\n"
            "  python3 cli.py color 'lab(50, 160, 200)'\n"
            "  python3 cli.py color '#C8C800' --camera top --tolerance 30\n"
            "  python3 cli.py color '#C8C800' --camera side --tolerance 50\n"
            "  python3 cli.py color --show\n"
            "  python3 cli.py color --show --camera side\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    col_p.add_argument(
        "value", nargs="?", default=None,
        help=(
            "Color value in any of these formats:\n"
            "  Hex:  '#FF5733' or 'FF5733'\n"
            "  RGB:  'rgb(255,87,51)' or '255,87,51'\n"
            "  Lab:  'lab(50,30,40)'  (direct CIELAB, no conversion)"
        ),
    )
    col_p.add_argument(
        "--tolerance", "-t", type=int, default=50,
        help="Detection tolerance (±) around the center color (default: 50)",
    )
    col_p.add_argument(
        "--camera", dest="cam_target", choices=["top", "side"], default=None,
        help="Set color for a specific camera (default: global)",
    )
    col_p.add_argument(
        "--show", action="store_true",
        help="Show the current color range without changing it",
    )

    # ── live ──
    live_p = sub.add_parser("live",
        help="Start the live camera viewer in your browser",
        description=(
            "Launches a Flask web server that streams live camera feeds\n"
            "with color detection overlays. Open the URL in any browser\n"
            "on the same network as the Pi.\n\n"
            "Requires Flask (pip install flask)."
        ),
        epilog=(
            "Examples:\n"
            "  python3 cli.py live\n"
            "  python3 cli.py live --port 8080\n"
            "  python3 cli.py live --host 127.0.0.1    # localhost only\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    live_p.add_argument(
        "--host", default="0.0.0.0",
        help="Network interface to bind to (default: 0.0.0.0 = all interfaces)",
    )
    live_p.add_argument(
        "--port", "-p", type=int, default=5000,
        help="Port number (default: 5000)",
    )

    return parser


# endregion


# region Entry point

def main():
    parser = build_parser()
    args = parser.parse_args()
    cfg = load_config(args.config)

    handlers = {
        "record": cmd_record,
        "stop": cmd_stop,
        "analyze": cmd_analyze,
        "solenoid": cmd_solenoid,
        "schedule": cmd_schedule,
        "recordings": cmd_recordings,
        "config": cmd_config,
        "update": cmd_update,
        "cameras": cmd_cameras,
        "color": cmd_color,
        "live": cmd_live,
    }

    handler = handlers.get(args.command)
    if handler:
        handler(args, cfg)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

# endregion
