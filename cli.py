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
    color_range,
    analysis_cfg,
    solenoid_cfg,
    schedule_cfg,
    recording_cfg,
)
from camera import create_camera
from analyzer import Recorder, analyze_recording, analyze_stereo
from solenoid import SolenoidController
from scheduler import SolenoidScheduler


# region Helpers

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def _build_recorder(cfg: dict, cam_name: str, *, duration: float | None, until: str | None) -> Recorder:
    """Create a Recorder for a given camera name."""
    cam_c = camera_cfg(cfg, cam_name)
    ana_c = analysis_cfg(cfg)
    rec_c = recording_cfg(cfg)
    lower, upper = color_range(cfg)

    camera = create_camera(
        cam_id=cam_c["id"],
        resolution=tuple(cam_c["resolution"]),
    )

    return Recorder(
        camera=camera,
        color_lower=lower,
        color_upper=upper,
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
    cam = args.camera
    duration = args.duration
    until = args.until

    if cam in ("top", "side"):
        rec = _build_recorder(cfg, cam, duration=duration, until=until)
        rec.run()

    elif cam == "both":
        top_rec = _build_recorder(cfg, "top", duration=duration, until=until)
        side_rec = _build_recorder(cfg, "side", duration=duration, until=until)

        top_thread = threading.Thread(target=top_rec.run, daemon=True)
        side_thread = threading.Thread(target=side_rec.run, daemon=True)

        top_thread.start()
        side_thread.start()

        try:
            top_thread.join()
            side_thread.join()
        except KeyboardInterrupt:
            print("\n[cli] Interrupted — recordings will be saved…")

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
            print(f"[cli] No recordings found for '{args.latest}'.")
            sys.exit(1)
        print(f"[cli] Using latest recording: {rec_path}")
    else:
        print("[cli] Provide a recording path or use --latest <camera>.")
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
    """Start the solenoid schedule."""
    sol_c = solenoid_cfg(cfg)
    sch_c = schedule_cfg(cfg)

    controller = SolenoidController(
        gpio_pin=sol_c["gpio_pin"],
        default_duration=sol_c["open_duration"],
    )
    scheduler = SolenoidScheduler(controller, sch_c["times"])

    scheduler.start()
    print("[cli] Scheduler running. Press Ctrl-C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    scheduler.stop()
    controller.cleanup()


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


# endregion


# region Argument parser

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="water",
        description="Water Measuring System — Raspberry Pi CLI",
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to config.yaml (default: ./config.yaml)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── record ──
    rec_p = sub.add_parser("record", help="Start recording from camera(s)")
    rec_p.add_argument(
        "--camera", "-c", choices=["top", "side", "both"], default="both",
        help="Which camera(s) to record (default: both)",
    )
    rec_p.add_argument(
        "--duration", "-d", type=float, default=None,
        help="Stop after this many seconds (from first color detection)",
    )
    rec_p.add_argument(
        "--until", "-u", type=str, default=None,
        help="Stop at this wall-clock time, e.g. '14:30'",
    )

    # ── stop ──
    sub.add_parser("stop", help="Signal an active recording to stop")

    # ── analyze ──
    ana_p = sub.add_parser("analyze", help="Generate graphs from a saved recording")
    ana_p.add_argument(
        "path", nargs="?", default=None,
        help="Path to a recording folder (or .npz file)",
    )
    ana_p.add_argument(
        "--latest", "-l", type=str, default=None,
        metavar="CAMERA",
        help="Use the latest recording for this camera (e.g. 'top', 'side')",
    )
    ana_p.add_argument(
        "--stereo", "-s", nargs=2, default=None,
        metavar=("TOP_PATH", "SIDE_PATH"),
        help="Run stereo analysis on two recording folders",
    )
    ana_p.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output directory for graphs (default: graphs/ inside recording)",
    )

    # ── solenoid ──
    sol_p = sub.add_parser("solenoid", help="Control the solenoid valve")
    sol_p.add_argument("action", choices=["open", "close", "test"])
    sol_p.add_argument(
        "--duration", "-d", type=float, default=None,
        help="Override open duration in seconds",
    )

    # ── schedule ──
    sub.add_parser("schedule", help="Start the solenoid schedule (Ctrl-C to stop)")

    # ── recordings ──
    sub.add_parser("recordings", help="List all saved recordings")

    # ── config ──
    sub.add_parser("config", help="Show current configuration")

    # ── update ──
    sub.add_parser("update", help="Pull the latest code from git")

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
    }

    handler = handlers.get(args.command)
    if handler:
        handler(args, cfg)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

# endregion
