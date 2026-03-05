# Water Measuring System

A Raspberry Pi 5-based system for color detection and water spread analysis using dual cameras, with automated solenoid valve control and comprehensive data visualization.

## Overview

This system uses one or two Raspberry Pi Camera Module 3s to detect colored water as it spreads across a surface. It tracks pixel counts, spread patterns, and shape evolution over time.

**Recording and analysis are separate steps** — you start a headless recording session over SSH, then run offline analysis whenever you like to generate graphs.

## Architecture

```
cli.py                  ← Main entry point (SSH-friendly CLI)
├── config_loader.py    ← Reads config.yaml
├── camera.py           ← Camera abstraction (PiCamera2 / OpenCV)
├── analyzer.py         ← Recorder (capture) + offline analysis engine
├── water_data.py       ← Per-camera data recording, serialization + graph generation
├── solenoid.py         ← GPIO solenoid valve controller
├── scheduler.py        ← Time-based solenoid trigger
└── stereo.py           ← Dual-camera alignment + 3D visualization
```

### Workflow

```
1. RECORD  →  Camera captures frames, waits for color detection
              Saves video.mp4 + recording_data.npz to recordings/<cam>/<timestamp>/
              Stops by: duration, wall-clock time, remote trigger, or Ctrl-C

2. ANALYZE →  Loads saved .npz data offline
              Generates all graphs to a graphs/ subfolder
              Can also run stereo analysis on two recordings
```

## Hardware

| Component | Purpose |
|---|---|
| Raspberry Pi 5 | Main compute |
| Camera Module 3 (×1 or ×2) | Top and/or side view capture |
| Solenoid valve | Water release control |
| Relay module or MOSFET | Drives solenoid from GPIO |
| 12V power supply | Powers the solenoid |

### Wiring

- **Solenoid signal** → GPIO 17 (BCM) via relay/MOSFET
- **Camera(s)** → CSI ribbon cable(s) on the Pi

> The GPIO pin is configurable in `config.yaml`.

## Setup

### 1. Clone and run setup

```bash
git clone https://github.com/Tristan-Samuel/water-measuring.git
cd water-measuring
bash setup.sh
```

`setup.sh` will:
- Create a `.venv` virtual environment (with `--system-site-packages` on Raspberry Pi so `picamera2` and `RPi.GPIO` are accessible)
- Install all Python dependencies from `requirements.txt`
- Verify Pi-specific packages are available (on Raspberry Pi)

### 2. Activate the environment

```bash
source .venv/bin/activate
```

### 3. Configure

Edit `config.yaml` to match your setup (camera IDs, color thresholds, solenoid pin, etc.).

Key color-related settings:

```yaml
cameras:
  top:
    color:                    # per-camera CIELAB overrides (optional)
      lower: [20, 100, 140]   # [L_min, a_min, b_min]
      upper: [255, 130, 200]  # [L_max, a_max, b_max]
  side:
    color:                    # wider tolerance for less light
      lower: [20, 85, 120]
      upper: [255, 145, 220]

color:                        # global fallback
  lower: [20, 100, 140]
  upper: [255, 130, 200]

clahe:                        # brightness normalization
  enabled: true
  clip_limit: 2.0
  grid_size: [8, 8]
```

- **L\*** range is wide (20–255) — acts as a noise floor only; CLAHE handles brightness.
- **a\*** controls green–red axis. Keeping it near 128 (neutral) targets yellow-green and excludes orange.
- **b\*** controls blue–yellow axis. High values (>140) select yellow hues.

### Dual Camera Setup (Pi 5)

The Pi 5 has two CSI camera connectors (CAM0 and CAM1). To use both Camera Module 3s, edit `/boot/firmware/config.txt`:

```bash
sudo nano /boot/firmware/config.txt
```

Make sure `camera_auto_detect=1` is set, then add an overlay for the second port:

```
camera_auto_detect=1
dtoverlay=imx708,cam1
```

Reboot (`sudo reboot`), then verify both cameras are detected:

```bash
rpicam-hello --list-cameras    # should show 2 cameras
python3 cli.py cameras         # same check from the CLI
```

> On non-Pi machines, the system auto-detects and falls back to OpenCV webcam
> capture and solenoid simulation mode. No extra setup needed.

## CLI Usage

All commands are run through `cli.py`. SSH into your Pi and use:

### Record

```bash
# Record from top camera for 60 seconds after color is detected
python3 cli.py record --camera top --duration 60

# Record from side camera until 14:30
python3 cli.py record --camera side --until 14:30

# Record both cameras for 2 minutes
python3 cli.py record --camera both --duration 120

# Record indefinitely (stop manually)
python3 cli.py record --camera top
```

Recording starts capturing when color is first detected. The video and data are saved to `recordings/<camera_label>/<timestamp>/`.

### Stop a Recording Remotely

From another SSH session (or any terminal on the Pi):

```bash
python3 cli.py stop
```

This creates a trigger file that the active recording detects and stops cleanly.

### Analyze (Offline)

```bash
# Analyze a specific recording
python3 cli.py analyze recordings/top_camera/2025-01-15_10-30-00

# Analyze the latest recording for a camera
python3 cli.py analyze --latest top

# Auto-find latest top + side recordings and run stereo analysis
python3 cli.py analyze --latest-stereo

# Stereo analysis from two specific recordings
python3 cli.py analyze --stereo recordings/top_camera/2025-01-15_10-30-00 recordings/side_camera/2025-01-15_10-30-00

# Save graphs to a custom folder
python3 cli.py analyze --latest top --output ~/my-graphs
```

### List Recordings

```bash
python3 cli.py recordings
```

### Solenoid Control

```bash
python3 cli.py solenoid open              # Open for configured duration
python3 cli.py solenoid open --duration 3  # Open for 3 seconds
python3 cli.py solenoid close             # Close immediately
python3 cli.py solenoid test              # Quick 1-second test pulse
```

### Scheduled Triggers

```bash
python3 cli.py schedule    # Start scheduler (Ctrl-C to stop)
```

### View Config

```bash
python3 cli.py config
```

### Update Code

Pull the latest code from git without needing SCP:

```bash
python3 cli.py update
```

### Set Detection Color

Paste a color value directly — hex, RGB, or CIELAB — and it auto-converts and saves to `config.yaml`.
Each camera can have its own color bounds (useful when lighting differs between top and side views):

```bash
# From a hex color picker (sets global fallback)
python3 cli.py color '#FF5733'

# Set color for a specific camera
python3 cli.py color '#C8C800' --camera top --tolerance 30
python3 cli.py color '#C8C800' --camera side --tolerance 50

# From RGB values
python3 cli.py color 'rgb(255, 87, 51)'
python3 cli.py color '255,87,51'

# Direct CIELAB values
python3 cli.py color 'lab(50, 160, 180)'

# Adjust detection tolerance (± around the color, default: 50)
python3 cli.py color '#FF5733' --tolerance 30

# Show current color config (global + per-camera)
python3 cli.py color --show
python3 cli.py color --show --camera side
```

When `--camera` is omitted, the global `color:` section in config.yaml is updated.
When `--camera top` or `--camera side` is given, only that camera's color block is changed.
During recording, each camera uses its own color bounds if present, otherwise the global fallback.

### Live Debug Viewer

Start a web-based viewer to see camera feeds and color detection in real time:

```bash
python3 cli.py live
```

Then open `http://<pi-ip>:5000` in your browser (any device on the same network). Shows:
- Raw camera feeds (top + side)
- Detection overlay with contours, pixel counts, and color mask

```bash
# Custom port
python3 cli.py live --port 8080
```

## Copying Files to Your Computer

After running analysis, copy results to your local machine:

```bash
# Copy a specific recording's graphs
scp -r pi@<pi-ip>:~/water-measuring/recordings/top_camera/2025-01-15_10-30-00/graphs/ ./graphs/

# Copy all recordings
scp -r pi@<pi-ip>:~/water-measuring/recordings/ ./recordings/

# Copy combined stereo results
scp -r pi@<pi-ip>:~/water-measuring/recordings/combined/ ./combined/

# Using rsync (faster for repeated transfers, only copies changes)
rsync -avz pi@<pi-ip>:~/water-measuring/recordings/ ./recordings/
```

> Replace `pi@<pi-ip>` with your Pi's username and IP address.
> Find your Pi's IP with: `hostname -I` (on the Pi).

## Output

Recordings are saved to the `recordings/` directory:

```
recordings/
├── top_camera/
│   └── 2025-01-15_10-30-00/
│       ├── video.mp4                     # Recorded video with contour overlays
│       ├── recording_data.npz            # Serialized analysis data
│       └── graphs/                       # Generated by 'analyze' command
│           ├── pixel_count_over_time.png
│           ├── group_expansion_over_time.png
│           ├── final_shape.png
│           ├── incremental_shape_layers.png
│           └── overlayed_contours.png
├── side_camera/
│   └── ...
└── combined/                             # Stereo analysis output
    ├── pixel_count_comparison.png
    ├── spread_comparison.png
    ├── aligned_contours.png
    ├── 3d_expansion.png
    ├── dimensions_over_time.png
    └── 3d_reconstruction.mp4             # Animated 3D voxel blob over time
```

### Per-Camera Graphs

| Graph | Description |
|---|---|
| `pixel_count_over_time.png` | Line chart of detected pixels vs time |
| `group_expansion_over_time.png` | Spread factor (mean σ of pixel positions) vs time |
| `final_shape.png` | The last detected shape as a heatmap |
| `incremental_shape_layers.png` | Grid of shape snapshots at 0.1s intervals |
| `overlayed_contours.png` | All snapshot contours overlaid, color-coded by time |

### Combined Stereo Graphs

| Graph | Description |
|---|---|
| `pixel_count_comparison.png` | Top vs side pixel counts on one plot |
| `spread_comparison.png` | Top vs side spread factors on one plot |
| `aligned_contours.png` | Contours from both cameras overlaid with horizontal alignment |
| `3d_expansion.png` | 3D scatter: time × width (top) × height (side) |
| `dimensions_over_time.png` | Blob width and height on a dual Y-axis plot |
| `3d_reconstruction.mp4` | Animated video of 3D voxel blob evolving over time (visual hull from top + side) |

## How It Works

### Color Detection

Frames are converted to **CIELAB** color space, which separates lightness (L\*) from chromaticity (a\*, b\*), making detection robust to lighting changes. A threshold range produces a binary mask of matching pixels.

**CLAHE brightness normalization** — Before thresholding, the L\* channel is equalized using Contrast Limited Adaptive Histogram Equalization (CLAHE). This evens out uneven illumination (e.g., backlighting through a substrate) so the chromaticity channels remain consistent regardless of brightness. CLAHE settings (`clip_limit`, `grid_size`) are configurable in `config.yaml` and can be disabled.

**Per-camera color bounds** — The top and side cameras can have independent CIELAB detection ranges. The side camera typically needs wider tolerance because it receives less direct light. Each camera checks for its own `color:` block under `cameras.<name>`; if absent, the global `color:` section is used as a fallback.

### Group Filtering

`cv2.findContours` finds connected regions. Only contours with area ≥ `min_contour_area` are kept — this filters noise. Pixel counts and spread are computed using **only** these valid groups.

### Recording Triggers

Recording starts automatically when color is first detected. It stops when any of these conditions is met:

| Condition | How |
|---|---|
| Duration | `--duration 60` — stops 60s after first detection |
| Wall-clock time | `--until 14:30` — stops at that time |
| Remote trigger | `python3 cli.py stop` from another SSH session |
| Signal | `Ctrl-C` (SIGINT) or `kill <pid>` (SIGTERM) |

### Stereo Alignment

When using two cameras, the system pairs snapshots by closest timestamp. For each pair, it computes the horizontal center-of-mass of each blob and shifts the side camera's contours to align with the top camera's. This allows the 3D expansion graph to combine width (from top view) with height (from side view).

### Auto-Detection

| Feature | Raspberry Pi | Desktop/Mac |
|---|---|---|
| Camera | Picamera2 (CSI) | OpenCV (USB webcam) |
| Solenoid | RPi.GPIO (real) | Simulation (prints actions) |

No code changes needed — the system detects the platform automatically.

## File Reference

| File | Purpose |
|---|---|
| `cli.py` | CLI entry point with subcommands |
| `config.yaml` | All configurable settings |
| `config_loader.py` | YAML parser + typed accessors |
| `camera.py` | Camera abstraction (PiCamera / OpenCV) |
| `analyzer.py` | Recorder (capture) + offline analysis functions |
| `water_data.py` | Data recorder, serialization + graph generator |
| `solenoid.py` | GPIO solenoid valve controller |
| `scheduler.py` | Time-based solenoid scheduler |
| `stereo.py` | Dual-camera alignment + 3D visualization |
| `web_viewer.py` | Flask live debug viewer for camera feeds |
