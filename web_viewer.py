"""
Flask web app — live viewer + full control panel.

Serves a single-page tabbed UI that mirrors every CLI command.
API routes under /api/* handle all actions.
Camera feed routes remain at /frame/<cam>/<kind>.
"""

from __future__ import annotations

import glob
import io
import os
import re
import subprocess
import threading
import time
from datetime import datetime, timedelta

import cv2  # type: ignore
import numpy as np  # type: ignore
import yaml  # type: ignore
from flask import Flask, Response, render_template_string, request, jsonify, send_file

from camera import create_camera, list_cameras
from config_loader import (
    camera_cfg, camera_crop, color_range, analysis_cfg, clahe_cfg,
    load_config, solenoid_cfg,
)
from analyzer import Recorder, analyze_recording, analyze_stereo
from solenoid import SolenoidController


# ---------------------------------------------------------------------------
# Camera capture thread
# ---------------------------------------------------------------------------

class CameraLoop:
    """Grabs frames in a thread, encodes two JPEGs (raw + detection)."""

    def __init__(self, name: str, cfg: dict):
        cam_c = camera_cfg(cfg, name)
        self.name = name
        self.label = cam_c["label"]
        self.camera = create_camera(
            cam_id=cam_c["id"],
            resolution=tuple(cam_c["resolution"]),
        )
        lo, hi = color_range(cfg, name)
        self.lo = np.array(lo)
        self.hi = np.array(hi)
        self.min_area = analysis_cfg(cfg)["min_contour_area"]

        # Per-camera crop [x, y, w, h] or None
        self._crop = camera_crop(cfg, name)

        # CLAHE brightness normalisation
        cl = clahe_cfg(cfg)
        self._clahe = cv2.createCLAHE(
            clipLimit=cl["clip_limit"],
            tileGridSize=tuple(cl["grid_size"]),
        ) if cl["enabled"] else None

        self.raw_jpg: bytes = b""
        self.det_jpg: bytes = b""
        self.seq: int = 0
        self._last_frame: np.ndarray | None = None   # raw BGR (post-crop)
        self._last_lab: np.ndarray | None = None     # CIELAB (post-CLAHE)
        self.lock = threading.Lock()
        self._stop = False

        t = threading.Thread(target=self._run, daemon=True)
        t.start()

    # ---- background loop ---------------------------------------------------
    def _run(self):
        n = 0
        while not self._stop:
            frame = self.camera.read()
            if frame is None:
                time.sleep(0.1)
                continue

            n += 1
            # Make an independent copy so Picamera2 can't recycle the buffer
            frame = frame.copy()

            # Apply per-camera crop if configured
            if self._crop is not None:
                cx, cy, cw, ch = self._crop
                fh, fw = frame.shape[:2]
                cx = max(0, min(cx, fw - 1))
                cy = max(0, min(cy, fh - 1))
                cw = min(cw, fw - cx)
                ch = min(ch, fh - cy)
                frame = frame[cy:cy + ch, cx:cx + cw]

            # --- raw jpeg ---
            ok1, buf1 = cv2.imencode(".jpg", frame,
                                     [cv2.IMWRITE_JPEG_QUALITY, 70])

            # --- detection jpeg ---
            det = frame.copy()
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

            # CLAHE: normalise L channel for brightness consistency
            if self._clahe is not None:
                l_ch, a_ch, b_ch = cv2.split(lab)
                l_ch = self._clahe.apply(l_ch)
                lab = cv2.merge([l_ch, a_ch, b_ch])

            mask = cv2.inRange(lab, self.lo, self.hi)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            total = 0
            for c in cnts:
                if cv2.contourArea(c) < self.min_area:
                    continue
                cv2.drawContours(det, [c], -1, (0, 255, 0), 2)
                single = np.zeros_like(mask)
                cv2.drawContours(single, [c], -1, 255, -1)
                px = cv2.countNonZero(cv2.bitwise_and(mask, single))
                total += px
                M = cv2.moments(c)
                if M["m00"]:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(det, f"{px}px", (cx - 30, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 2)
            cv2.putText(det, f"Total: {total}px", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
            cv2.putText(det, f"frame {n}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 200, 200), 2)
            # red overlay
            red = np.zeros_like(det)
            red[:, :, 2] = mask
            det = cv2.addWeighted(det, 1.0, red, 0.3, 0)
            ok2, buf2 = cv2.imencode(".jpg", det,
                                     [cv2.IMWRITE_JPEG_QUALITY, 70])

            if ok1 and ok2:
                with self.lock:
                    self.raw_jpg = buf1.tobytes()
                    self.det_jpg = buf2.tobytes()
                    self.seq = n
                    self._last_frame = frame
                    self._last_lab = lab

            if n % 200 == 0:
                print(f"[live:{self.name}] frame {n}")

            time.sleep(0.033)  # cap ~30 fps

    def stop(self):
        self._stop = True
        self.camera.release()


# ---------------------------------------------------------------------------
# Recording state (one active recording at a time)
# ---------------------------------------------------------------------------

_record_state: dict = {
    "running": False,
    "camera": None,
    "started_at": None,
    "thread": None,
    "recorders": [],
}
_record_lock = threading.Lock()

# Analysis state
_analyze_state: dict = {"running": False, "last_result": None, "error": None}
_analyze_lock = threading.Lock()

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Helpers shared with API
# ---------------------------------------------------------------------------

def _lab_to_hex(lo: list, hi: list) -> str:
    mid = np.array([[(lo[0]+hi[0])//2, (lo[1]+hi[1])//2, (lo[2]+hi[2])//2]], dtype=np.uint8).reshape(1,1,3)
    bgr = cv2.cvtColor(mid, cv2.COLOR_LAB2BGR)[0, 0]
    return "#{:02x}{:02x}{:02x}".format(int(bgr[2]), int(bgr[1]), int(bgr[0]))


def _parse_color_to_lab(value: str):
    """Parse hex/rgb/lab string → (L, a, b)."""
    value = value.strip()
    if value.lower().startswith("lab("):
        nums = value[4:].rstrip(")").split(",")
        if len(nums) != 3:
            raise ValueError("lab() needs 3 values")
        return float(nums[0]), float(nums[1]), float(nums[2])
    if value.startswith("#"):
        value = value[1:]
    if len(value) == 6 and all(c in "0123456789abcdefABCDEF" for c in value):
        r, g, b = int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)
    elif value.lower().startswith("rgb("):
        nums = value[4:].rstrip(")").split(",")
        r, g, b = int(nums[0]), int(nums[1]), int(nums[2])
    elif "," in value:
        parts = value.split(",")
        if len(parts) == 3:
            r, g, b = int(parts[0]), int(parts[1]), int(parts[2])
        else:
            raise ValueError(f"Can't parse: {value}")
    else:
        raise ValueError(f"Can't parse '{value}'. Use hex, rgb(), or lab().")
    pixel = np.uint8([[[b, g, r]]])
    lab = cv2.cvtColor(pixel, cv2.COLOR_BGR2LAB)
    return float(lab[0,0,0]), float(lab[0,0,1]), float(lab[0,0,2])


def _build_recorder_from_cfg(cfg: dict, cam_name: str, duration, until):
    from config_loader import recording_cfg
    cam_c = camera_cfg(cfg, cam_name)
    ana_c = analysis_cfg(cfg)
    rec_c = recording_cfg(cfg)
    cl = clahe_cfg(cfg)
    lower, upper = color_range(cfg, cam_name)
    camera = create_camera(cam_id=cam_c["id"], resolution=tuple(cam_c["resolution"]))
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


def _find_latest_recording(cam_label_dir: str) -> str | None:
    base = os.path.join(PROJECT_DIR, "recordings", cam_label_dir)
    if not os.path.isdir(base):
        return None
    folders = sorted(glob.glob(os.path.join(base, "*")))
    return folders[-1] if folders else None


# ---------------------------------------------------------------------------
# HTML — single-page tabbed app (Tailwind CDN, dark theme)
# ---------------------------------------------------------------------------

PAGE = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Water Measuring</title>
<script src="https://cdn.tailwindcss.com"></script>
<script>
tailwind.config = {
  darkMode: 'class',
  theme: { extend: { colors: { accent: '#58a6ff', ok: '#3fb950', warn: '#d29922', danger: '#f85149' } } }
}
</script>
<style>
  body { background: #0d1117; color: #c9d1d9; font-family: ui-monospace, 'Cascadia Code', monospace; }
  .panel { background: #161b22; border: 1px solid #30363d; border-radius: 8px; }
  .tab-btn { transition: background .15s, color .15s; }
  .tab-btn.active { background: #1f6feb; color: #fff; }
  .tab-btn:not(.active):hover { background: #21262d; }
  .tab-panel { display: none; }
  .tab-panel.active { display: block; }
  input, select, textarea { background: #0d1117; border: 1px solid #30363d; color: #c9d1d9; border-radius: 6px; padding: 6px 10px; width: 100%; box-sizing: border-box; }
  input:focus, select:focus, textarea:focus { outline: none; border-color: #58a6ff; }
  label { font-size: 12px; color: #8b949e; display: block; margin-bottom: 4px; }
  .btn { padding: 7px 16px; border-radius: 6px; font-size: 13px; cursor: pointer; border: 1px solid transparent; transition: opacity .15s; }
  .btn:hover { opacity: .85; }
  .btn:disabled { opacity: .4; cursor: not-allowed; }
  .btn-primary { background: #1f6feb; color: #fff; border-color: #388bfd; }
  .btn-danger  { background: #da3633; color: #fff; border-color: #f85149; }
  .btn-ok      { background: #238636; color: #fff; border-color: #3fb950; }
  .btn-ghost   { background: transparent; color: #58a6ff; border-color: #30363d; }
  .status-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }
  .dot-ok     { background: #3fb950; box-shadow: 0 0 6px #3fb950; }
  .dot-off    { background: #484f58; }
  .dot-warn   { background: #d29922; }
  pre { background: #010409; border: 1px solid #30363d; border-radius: 6px; padding: 12px; font-size: 12px; overflow-x: auto; white-space: pre-wrap; }
  table { border-collapse: collapse; width: 100%; font-size: 13px; }
  th { text-align: left; padding: 8px 12px; border-bottom: 1px solid #30363d; color: #8b949e; font-weight: 500; }
  td { padding: 7px 12px; border-bottom: 1px solid #21262d; }
  tr:hover td { background: #161b22; }
  .swatch { display: inline-block; width: 22px; height: 14px; border-radius: 3px; border: 1px solid #484f58; vertical-align: middle; margin-right: 6px; }
  .cam-feed { position: relative; background: #010409; border-radius: 6px; overflow: hidden; }
  .cam-feed img { width: 100%; display: block; cursor: crosshair; }
  .cam-feed h3 { font-size: 12px; padding: 6px 10px; margin: 0; background: rgba(0,0,0,.5); position: absolute; top: 0; left: 0; right: 0; }
  #toast { position: fixed; bottom: 20px; right: 20px; z-index: 9999; min-width: 260px;
           background: #161b22; border: 1px solid #30363d; border-radius: 8px;
           padding: 12px 16px; font-size: 13px; line-height: 1.5;
           display: none; box-shadow: 0 8px 30px rgba(0,0,0,.6); }
  .signal-bar { display: inline-flex; align-items: flex-end; gap: 2px; height: 14px; }
  .signal-bar span { width: 3px; background: #484f58; border-radius: 1px; }
  .signal-bar span.lit { background: #3fb950; }
  input[type=range] { padding: 0; height: 6px; cursor: pointer; }
</style>
</head>
<body class="min-h-screen">

<!-- Header -->
<header class="flex items-center gap-3 px-4 py-3 border-b border-gray-800 sticky top-0 z-50" style="background:#161b22">
  <span style="font-size:20px">💧</span>
  <span class="font-bold text-base tracking-wide" style="color:#58a6ff">Water Measuring</span>
  <span id="hdr-status" class="ml-auto text-xs" style="color:#484f58">connecting…</span>
</header>

<!-- Tab bar -->
<nav class="flex gap-1 px-4 py-2 border-b border-gray-800 overflow-x-auto" style="background:#0d1117">
  {% for t in [['live','📷 Live'],['record','⏺ Record'],['solenoid','💧 Solenoid'],['recordings','📁 Recordings'],['config','⚙️ Config'],['wifi','📶 WiFi'],['system','🔧 System']] %}
  <button class="tab-btn px-3 py-1.5 rounded text-sm {% if loop.first %}active{% endif %}" data-tab="{{ t[0] }}">{{ t[1] }}</button>
  {% endfor %}
</nav>

<!-- Toast -->
<div id="toast"><span id="toast-body"></span><span style="float:right;cursor:pointer;opacity:.5" onclick="hideToast()">✕</span></div>

<!-- ═══════════════ TAB: LIVE ═══════════════ -->
<div class="tab-panel active p-4" id="tab-live">
  <div class="flex gap-4 flex-wrap mb-4">
    {% for c in cams %}
    <div class="cam-feed flex-1" style="min-width:280px">
      <img id="r{{ c.name }}" data-cam="{{ c.name }}" alt="{{ c.label }} raw" style="padding-top:26px">
      <h3>{{ c.label }} — Raw</h3>
    </div>
    <div class="cam-feed flex-1" style="min-width:280px">
      <img id="d{{ c.name }}" data-cam="{{ c.name }}" alt="{{ c.label }} detection" style="padding-top:26px">
      <h3>{{ c.label }} — Detection</h3>
    </div>
    {% endfor %}
  </div>
  <!-- Color picker sidebar -->
  <div class="panel p-4 max-w-md">
    <h3 class="text-sm font-semibold mb-3" style="color:#58a6ff">Color Detection</h3>
    {% for c in cams %}
    <div class="mb-5" id="cb-{{ c.name }}">
      <div class="flex items-center gap-2 mb-1">
        <span class="swatch" id="sw-{{ c.name }}"></span>
        <b class="text-sm">{{ c.label }}</b>
      </div>
      <div class="font-mono text-xs mb-2" style="color:#8b949e">
        L: <span id="lo-{{ c.name }}">{{ c.lo }}</span><br>
        U: <span id="hi-{{ c.name }}">{{ c.hi }}</span>
      </div>
      <label>Tolerance: <span id="tv-{{ c.name }}" class="font-mono">{{ c.tolerance }}</span></label>
      <input type="range" min="5" max="120" value="{{ c.tolerance }}" id="tol-{{ c.name }}" data-cam="{{ c.name }}" style="width:100%">
    </div>
    {% endfor %}
    <p class="text-xs mt-3" style="color:#484f58">Click any camera image to sample a color. Changes are saved to config.yaml.</p>
  </div>
</div>

<!-- ═══════════════ TAB: RECORD ═══════════════ -->
<div class="tab-panel p-4" id="tab-record">
  <div class="panel p-5 max-w-lg">
    <h2 class="text-sm font-semibold mb-4" style="color:#58a6ff">Start Recording</h2>
    <div class="grid grid-cols-2 gap-3 mb-3">
      <div>
        <label>Camera</label>
        <select id="rec-camera">
          <option value="both">Both</option>
          <option value="top">Top only</option>
          <option value="side">Side only</option>
        </select>
      </div>
      <div>
        <label>Duration (seconds, blank = no limit)</label>
        <input type="number" id="rec-duration" placeholder="e.g. 120" min="1">
      </div>
      <div>
        <label>Stop at time (e.g. 14:30 or 2:30PM)</label>
        <input type="text" id="rec-until" placeholder="optional">
      </div>
      <div>
        <label>Start at time (optional wait)</label>
        <input type="text" id="rec-at" placeholder="e.g. 08:00">
      </div>
      <div>
        <label>Solenoid pulse before recording (seconds)</label>
        <input type="number" id="rec-solenoid" placeholder="blank = no pulse" min="0.1" step="0.5">
      </div>
      <div class="flex items-center gap-2 pt-4">
        <input type="checkbox" id="rec-analyze" style="width:auto">
        <label for="rec-analyze" class="mb-0" style="color:#c9d1d9">Auto-analyze when done</label>
      </div>
    </div>
    <div class="flex gap-2 mt-4">
      <button class="btn btn-ok" id="rec-start-btn" onclick="recordStart()">▶ Start Recording</button>
      <button class="btn btn-danger" id="rec-stop-btn" disabled onclick="recordStop()">■ Stop</button>
    </div>
    <div id="rec-status" class="mt-4 text-sm" style="color:#8b949e"></div>
  </div>
</div>

<!-- ═══════════════ TAB: SOLENOID ═══════════════ -->
<div class="tab-panel p-4" id="tab-solenoid">
  <div class="panel p-5 max-w-sm">
    <h2 class="text-sm font-semibold mb-4" style="color:#58a6ff">Solenoid Control</h2>
    <div class="flex items-center gap-3 mb-5">
      <span class="status-dot dot-off" id="sol-dot"></span>
      <span id="sol-status-txt" class="text-sm">Unknown</span>
    </div>
    <div class="mb-4">
      <label>Open duration (seconds)</label>
      <input type="number" id="sol-duration" value="5" min="0.1" step="0.5">
    </div>
    <div class="flex gap-2">
      <button class="btn btn-ok" onclick="solenoidAction('open')">Open</button>
      <button class="btn btn-danger" onclick="solenoidAction('close')">Close</button>
      <button class="btn btn-ghost" onclick="solenoidAction('test')">Test (1s)</button>
    </div>
    <div id="sol-msg" class="mt-3 text-sm" style="color:#8b949e"></div>
  </div>
</div>

<!-- ═══════════════ TAB: RECORDINGS ═══════════════ -->
<div class="tab-panel p-4" id="tab-recordings">
  <div class="flex gap-3 mb-4 flex-wrap">
    <button class="btn btn-ghost" onclick="loadRecordings()">↻ Refresh</button>
    <div class="flex items-center gap-2 text-sm" style="color:#8b949e">
      Stereo analyze:
      <select id="stereo-top" style="width:auto;padding:4px 8px"></select>
      +
      <select id="stereo-side" style="width:auto;padding:4px 8px"></select>
      <button class="btn btn-primary" onclick="analyzeStero()">Analyze Stereo</button>
    </div>
  </div>
  <div id="analyze-status" class="mb-3 text-sm" style="color:#8b949e"></div>
  <div class="panel overflow-hidden">
    <table>
      <thead><tr><th>Camera</th><th>Timestamp</th><th>Files</th><th>Actions</th></tr></thead>
      <tbody id="rec-table-body"><tr><td colspan="4" style="color:#484f58;text-align:center;padding:20px">Loading…</td></tr></tbody>
    </table>
  </div>
  <!-- Graph viewer -->
  <div id="graph-viewer" class="mt-4 panel p-4" style="display:none">
    <div class="flex items-center justify-between mb-3">
      <span id="graph-title" class="text-sm font-semibold" style="color:#58a6ff"></span>
      <button class="btn btn-ghost" onclick="document.getElementById('graph-viewer').style.display='none'">✕ Close</button>
    </div>
    <div id="graph-grid" class="flex flex-wrap gap-3"></div>
  </div>
</div>

<!-- ═══════════════ TAB: CONFIG ═══════════════ -->
<div class="tab-panel p-4" id="tab-config">
  <div class="flex gap-3 mb-4">
    <button class="btn btn-primary" onclick="saveConfig()">💾 Save Config</button>
    <button class="btn btn-ghost" onclick="loadConfig()">↻ Reload</button>
    <span id="cfg-msg" class="text-sm self-center" style="color:#8b949e"></span>
  </div>
  <div class="panel p-4">
    <label>config.yaml (edit directly)</label>
    <textarea id="cfg-yaml" rows="30" style="font-family:monospace;font-size:12px;resize:vertical"></textarea>
  </div>
</div>

<!-- ═══════════════ TAB: WIFI ═══════════════ -->
<div class="tab-panel p-4" id="tab-wifi">
  <div class="panel p-4 max-w-2xl">
    <div class="flex items-center justify-between mb-4">
      <h2 class="text-sm font-semibold" style="color:#58a6ff">WiFi</h2>
      <button class="btn btn-ghost" onclick="wifiScan()">↻ Scan networks</button>
    </div>
    <div id="wifi-current" class="mb-4 text-sm p-3 rounded" style="background:#010409;border:1px solid #30363d">
      <span style="color:#484f58">Checking current connection…</span>
    </div>
    <div id="wifi-scan-status" class="text-xs mb-2" style="color:#8b949e"></div>
    <table id="wifi-table" style="display:none">
      <thead><tr><th>SSID</th><th>Signal</th><th>Security</th><th></th></tr></thead>
      <tbody id="wifi-table-body"></tbody>
    </table>
    <!-- Connect modal -->
    <div id="wifi-connect-panel" class="mt-4 p-4 rounded" style="display:none;background:#010409;border:1px solid #30363d">
      <div class="text-sm font-semibold mb-3">Connect to: <span id="wifi-connecting-ssid" style="color:#58a6ff"></span></div>
      <label>Password (leave blank for open networks)</label>
      <input type="password" id="wifi-password" placeholder="WiFi password" autocomplete="off">
      <div class="flex gap-2 mt-3">
        <button class="btn btn-ok" onclick="wifiConnect()">Connect</button>
        <button class="btn btn-ghost" onclick="document.getElementById('wifi-connect-panel').style.display='none'">Cancel</button>
      </div>
      <div id="wifi-connect-msg" class="mt-2 text-sm" style="color:#8b949e"></div>
    </div>
  </div>
</div>

<!-- ═══════════════ TAB: SYSTEM ═══════════════ -->
<div class="tab-panel p-4" id="tab-system">
  <div class="grid gap-4 max-w-2xl">
    <div class="panel p-4">
      <h2 class="text-sm font-semibold mb-3" style="color:#58a6ff">Software Update</h2>
      <button class="btn btn-primary mb-3" onclick="runUpdate()">⬆ Pull latest from git</button>
      <pre id="update-output" style="display:none;min-height:80px"></pre>
    </div>
    <div class="panel p-4">
      <div class="flex items-center justify-between mb-3">
        <h2 class="text-sm font-semibold" style="color:#58a6ff">Detected Cameras</h2>
        <button class="btn btn-ghost" onclick="loadCameras()">↻ Refresh</button>
      </div>
      <pre id="cameras-output" style="color:#8b949e">Click refresh to check cameras.</pre>
    </div>
  </div>
</div>

<script>
// ═══════════════════════════════════════════════
//  Utility
// ═══════════════════════════════════════════════
function toast(msg, ok) {
  var el = document.getElementById('toast');
  document.getElementById('toast-body').innerHTML = msg;
  el.style.borderColor = ok === false ? '#f85149' : ok === true ? '#3fb950' : '#30363d';
  el.style.display = 'block';
  clearTimeout(el._t);
  el._t = setTimeout(hideToast, 5000);
}
function hideToast() { document.getElementById('toast').style.display = 'none'; }

function api(url, opts) {
  return fetch(url, Object.assign({ headers: {'Content-Type': 'application/json'} }, opts))
    .then(function(r) { return r.json(); });
}

// ═══════════════════════════════════════════════
//  Tab switching
// ═══════════════════════════════════════════════
document.querySelectorAll('.tab-btn').forEach(function(btn) {
  btn.addEventListener('click', function() {
    document.querySelectorAll('.tab-btn').forEach(function(b) { b.classList.remove('active'); });
    document.querySelectorAll('.tab-panel').forEach(function(p) { p.classList.remove('active'); });
    btn.classList.add('active');
    var panel = document.getElementById('tab-' + btn.dataset.tab);
    if (panel) panel.classList.add('active');
    if (btn.dataset.tab === 'recordings') loadRecordings();
    if (btn.dataset.tab === 'config') loadConfig();
    if (btn.dataset.tab === 'wifi') { wifiStatus(); }
    if (btn.dataset.tab === 'system') loadCameras();
  });
});

// ═══════════════════════════════════════════════
//  Live camera feeds
// ═══════════════════════════════════════════════
var CAMS = [{% for c in cams %}"{{ c.name }}",{% endfor %}];
var MIN_INTERVAL = 80;

function startFeedLoop(img, url) {
  var pending = false;
  function next() {
    if (pending) return;
    pending = true;
    var t0 = Date.now();
    var tmp = new Image();
    tmp.onload = function() {
      img.src = tmp.src; pending = false;
      setTimeout(next, Math.max(0, MIN_INTERVAL - (Date.now() - t0)));
    };
    tmp.onerror = function() { pending = false; setTimeout(next, 500); };
    tmp.src = url + '?_=' + Date.now();
  }
  next();
}

CAMS.forEach(function(n) {
  startFeedLoop(document.getElementById('r' + n), '/frame/' + n + '/raw');
  startFeedLoop(document.getElementById('d' + n), '/frame/' + n + '/det');
});

function updateColorSidebar(cam, d) {
  var sw = document.getElementById('sw-' + cam);
  var lo = document.getElementById('lo-' + cam);
  var hi = document.getElementById('hi-' + cam);
  if (sw && d.hex) sw.style.background = d.hex;
  if (lo) lo.textContent = '[' + d.lower.join(', ') + ']';
  if (hi) hi.textContent = '[' + d.upper.join(', ') + ']';
}

// Click-to-pick color
document.querySelectorAll('.cam-feed img').forEach(function(img) {
  img.addEventListener('click', function(e) {
    var rect = img.getBoundingClientRect();
    var cam = img.dataset.cam;
    if (!cam) return;
    var tol = parseInt(document.getElementById('tol-' + cam).value || '50');
    api('/pick_color/' + cam, {
      method: 'POST',
      body: JSON.stringify({ fx: (e.clientX - rect.left) / rect.width,
                             fy: (e.clientY - rect.top) / rect.height,
                             tolerance: tol })
    }).then(function(d) {
      if (d.error) { toast(d.error, false); return; }
      updateColorSidebar(cam, d);
      toast('<span class="swatch" style="background:' + d.hex + '"></span>'
        + cam + ' color picked — L=' + d.L + ' a=' + d.a + ' b=' + d.b, true);
    });
  });
});

// Tolerance slider
document.querySelectorAll('input[type=range][data-cam]').forEach(function(sl) {
  sl.addEventListener('input', function() {
    document.getElementById('tv-' + sl.dataset.cam).textContent = sl.value;
  });
  var db = null;
  sl.addEventListener('change', function() {
    clearTimeout(db);
    var cam = sl.dataset.cam;
    var tol = parseInt(sl.value);
    db = setTimeout(function() {
      api('/set_tolerance/' + cam, { method: 'POST', body: JSON.stringify({ tolerance: tol }) })
        .then(function(d) { if (!d.error) updateColorSidebar(cam, d); });
    }, 200);
  });
});

// Poll color info every 3s
function refreshColors() {
  CAMS.forEach(function(cam) {
    api('/color_info/' + cam).then(function(d) { if (!d.error) updateColorSidebar(cam, d); }).catch(function(){});
  });
}
refreshColors();
setInterval(refreshColors, 3000);

// ═══════════════════════════════════════════════
//  Recording
// ═══════════════════════════════════════════════
var recPoll = null;

function recordStart() {
  var body = {
    camera: document.getElementById('rec-camera').value,
    duration: parseFloat(document.getElementById('rec-duration').value) || null,
    until: document.getElementById('rec-until').value.trim() || null,
    at: document.getElementById('rec-at').value.trim() || null,
    solenoid: parseFloat(document.getElementById('rec-solenoid').value) || null,
    analyze: document.getElementById('rec-analyze').checked
  };
  api('/api/record/start', { method: 'POST', body: JSON.stringify(body) })
    .then(function(d) {
      if (d.error) { toast(d.error, false); return; }
      toast('Recording started', true);
      setRecUI(true);
      startRecPoll();
    });
}

function recordStop() {
  api('/api/record/stop', { method: 'POST' }).then(function(d) {
    toast(d.message || 'Stop signal sent');
    setRecUI(false);
  });
}

function setRecUI(running) {
  document.getElementById('rec-start-btn').disabled = running;
  document.getElementById('rec-stop-btn').disabled = !running;
}

function startRecPoll() {
  clearInterval(recPoll);
  recPoll = setInterval(function() {
    api('/api/record/status').then(function(d) {
      var el = document.getElementById('rec-status');
      if (d.running) {
        var elapsed = d.started_at ? Math.floor((Date.now()/1000) - d.started_at) : 0;
        el.innerHTML = '<span class="status-dot dot-ok"></span>Recording ' + (d.camera || '') + ' — ' + elapsed + 's elapsed';
        el.style.color = '#3fb950';
        setRecUI(true);
      } else {
        el.innerHTML = '<span class="status-dot dot-off"></span>Idle';
        el.style.color = '#8b949e';
        setRecUI(false);
        clearInterval(recPoll);
      }
    }).catch(function(){});
  }, 2000);
}

// Poll on load to sync UI with any in-progress recording
startRecPoll();

// ═══════════════════════════════════════════════
//  Solenoid
// ═══════════════════════════════════════════════
function solenoidAction(action) {
  var dur = parseFloat(document.getElementById('sol-duration').value) || 5;
  api('/api/solenoid/' + action, { method: 'POST', body: JSON.stringify({ duration: dur }) })
    .then(function(d) {
      var msg = d.error || d.message || action;
      document.getElementById('sol-msg').textContent = msg;
      if (action === 'open' || action === 'test') {
        setSolDot(true);
        setTimeout(function() { setSolDot(false); }, (action === 'test' ? 1 : dur) * 1000 + 500);
      } else {
        setSolDot(false);
      }
      toast(msg, !d.error);
    });
}

function setSolDot(open) {
  var dot = document.getElementById('sol-dot');
  var txt = document.getElementById('sol-status-txt');
  dot.className = 'status-dot ' + (open ? 'dot-ok' : 'dot-off');
  txt.textContent = open ? 'Open (energised)' : 'Closed';
}

// ═══════════════════════════════════════════════
//  Recordings browser
// ═══════════════════════════════════════════════
function loadRecordings() {
  api('/api/recordings').then(function(d) {
    if (d.error) { toast(d.error, false); return; }
    var tbody = document.getElementById('rec-table-body');
    var stopSelTop = document.getElementById('stereo-top');
    var stopSelSide = document.getElementById('stereo-side');
    tbody.innerHTML = '';
    stopSelTop.innerHTML = '';
    stopSelSide.innerHTML = '';
    if (!d.recordings || !d.recordings.length) {
      tbody.innerHTML = '<tr><td colspan="4" style="color:#484f58;text-align:center;padding:20px">No recordings found.</td></tr>';
      return;
    }
    d.recordings.forEach(function(r) {
      var files = (r.has_video ? 'video ' : '') + (r.has_data ? 'data ' : '') + (r.has_graphs ? 'graphs' : '');
      var row = document.createElement('tr');
      row.innerHTML = '<td>' + r.camera + '</td>'
        + '<td class="font-mono">' + r.timestamp + '</td>'
        + '<td style="color:#8b949e">' + files.trim() + '</td>'
        + '<td class="flex gap-1 flex-wrap">'
        + '<button class="btn btn-ghost" style="padding:3px 8px;font-size:12px" onclick="analyzeOne(\'' + r.path.replace(/\\/g,'\\\\') + '\')">Analyze</button>'
        + (r.has_graphs ? '<button class="btn btn-ghost" style="padding:3px 8px;font-size:12px" onclick="showGraphs(\'' + r.path.replace(/\\/g,'\\\\') + '\',\'' + r.camera + ' ' + r.timestamp + '\')">Graphs</button>' : '')
        + '</td>';
      tbody.appendChild(row);
      var opt = document.createElement('option');
      opt.value = r.path; opt.textContent = r.camera + ' / ' + r.timestamp;
      stopSelTop.appendChild(opt.cloneNode(true));
      stopSelSide.appendChild(opt);
    });
  });
}

function analyzeOne(path) {
  document.getElementById('analyze-status').textContent = 'Analyzing ' + path + '…';
  api('/api/analyze', { method: 'POST', body: JSON.stringify({ path: path }) })
    .then(function(d) {
      if (d.error) { toast(d.error, false); document.getElementById('analyze-status').textContent = 'Error: ' + d.error; return; }
      toast('Analysis complete', true);
      document.getElementById('analyze-status').textContent = 'Done: ' + (d.output_dir || '');
      loadRecordings();
    });
}

function analyzeStero() {
  var top = document.getElementById('stereo-top').value;
  var side = document.getElementById('stereo-side').value;
  if (!top || !side || top === side) { toast('Select different top and side recordings', false); return; }
  document.getElementById('analyze-status').textContent = 'Running stereo analysis…';
  api('/api/analyze/stereo', { method: 'POST', body: JSON.stringify({ top: top, side: side }) })
    .then(function(d) {
      if (d.error) { toast(d.error, false); document.getElementById('analyze-status').textContent = 'Error: ' + d.error; return; }
      toast('Stereo analysis complete', true);
      document.getElementById('analyze-status').textContent = 'Done: ' + (d.output_dir || '');
    });
}

function showGraphs(path, label) {
  api('/api/recordings/graphs?path=' + encodeURIComponent(path)).then(function(d) {
    if (!d.graphs || !d.graphs.length) { toast('No graphs found', false); return; }
    document.getElementById('graph-title').textContent = label;
    var grid = document.getElementById('graph-grid');
    grid.innerHTML = '';
    d.graphs.forEach(function(g) {
      var img = document.createElement('img');
      img.src = '/api/recordings/image?path=' + encodeURIComponent(g);
      img.style.cssText = 'max-width:320px;border-radius:6px;border:1px solid #30363d;cursor:pointer';
      img.onclick = function() { window.open(img.src, '_blank'); };
      grid.appendChild(img);
    });
    document.getElementById('graph-viewer').style.display = 'block';
  });
}

// ═══════════════════════════════════════════════
//  Config
// ═══════════════════════════════════════════════
function loadConfig() {
  fetch('/api/config/raw').then(function(r) { return r.text(); }).then(function(t) {
    document.getElementById('cfg-yaml').value = t;
    document.getElementById('cfg-msg').textContent = '';
  });
}

function saveConfig() {
  var raw = document.getElementById('cfg-yaml').value;
  api('/api/config', { method: 'POST', body: JSON.stringify({ yaml: raw }) })
    .then(function(d) {
      if (d.error) { document.getElementById('cfg-msg').textContent = 'Error: ' + d.error; document.getElementById('cfg-msg').style.color='#f85149'; toast(d.error, false); return; }
      document.getElementById('cfg-msg').textContent = 'Saved ✓';
      document.getElementById('cfg-msg').style.color = '#3fb950';
      toast('Config saved', true);
      setTimeout(function() { document.getElementById('cfg-msg').textContent=''; }, 3000);
    });
}

// ═══════════════════════════════════════════════
//  WiFi
// ═══════════════════════════════════════════════
function wifiStatus() {
  api('/api/wifi/status').then(function(d) {
    var el = document.getElementById('wifi-current');
    if (d.error) { el.innerHTML = '<span style="color:#f85149">nmcli unavailable: ' + d.error + '</span>'; return; }
    if (d.connections && d.connections.length) {
      el.innerHTML = d.connections.map(function(c) {
        return '<span class="status-dot dot-ok"></span><b>' + c.name + '</b>'
          + ' <span style="color:#8b949e">on ' + c.device + ' — ' + c.state + '</span>';
      }).join('<br>');
    } else {
      el.innerHTML = '<span class="status-dot dot-off"></span><span style="color:#8b949e">Not connected</span>';
    }
  }).catch(function() {
    document.getElementById('wifi-current').innerHTML = '<span style="color:#8b949e">Status unavailable</span>';
  });
}

function wifiScan() {
  var statusEl = document.getElementById('wifi-scan-status');
  statusEl.textContent = 'Scanning…';
  api('/api/wifi/scan').then(function(d) {
    statusEl.textContent = '';
    if (d.error) { toast(d.error, false); return; }
    var tbody = document.getElementById('wifi-table-body');
    tbody.innerHTML = '';
    (d.networks || []).forEach(function(n) {
      var bars = signalBars(parseInt(n.signal || 0));
      var row = document.createElement('tr');
      row.innerHTML = '<td class="font-semibold">' + escHtml(n.ssid) + '</td>'
        + '<td>' + bars + '</td>'
        + '<td style="color:#8b949e">' + escHtml(n.security || 'open') + '</td>'
        + '<td><button class="btn btn-ghost" style="padding:3px 10px;font-size:12px" onclick="wifiPrompt(\'' + escHtml(n.ssid).replace(/'/g, "\\'") + '\')">Connect</button></td>';
      tbody.appendChild(row);
    });
    document.getElementById('wifi-table').style.display = d.networks && d.networks.length ? '' : 'none';
    if (!d.networks || !d.networks.length) statusEl.textContent = 'No networks found.';
  }).catch(function() { toast('Scan failed', false); statusEl.textContent = ''; });
}

function signalBars(pct) {
  var lit = Math.round(pct / 25);
  var html = '<div class="signal-bar">';
  for (var i = 1; i <= 4; i++) {
    html += '<span style="height:' + (i * 3 + 2) + 'px" class="' + (i <= lit ? 'lit' : '') + '"></span>';
  }
  return html + '</div>';
}

function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

var wifiTargetSSID = '';
function wifiPrompt(ssid) {
  wifiTargetSSID = ssid;
  document.getElementById('wifi-connecting-ssid').textContent = ssid;
  document.getElementById('wifi-password').value = '';
  document.getElementById('wifi-connect-msg').textContent = '';
  document.getElementById('wifi-connect-panel').style.display = 'block';
  document.getElementById('wifi-password').focus();
}

function wifiConnect() {
  var pw = document.getElementById('wifi-password').value;
  var msg = document.getElementById('wifi-connect-msg');
  msg.textContent = 'Connecting…';
  api('/api/wifi/connect', {
    method: 'POST',
    body: JSON.stringify({ ssid: wifiTargetSSID, password: pw })
  }).then(function(d) {
    if (d.error) { msg.textContent = 'Error: ' + d.error; msg.style.color = '#f85149'; toast(d.error, false); return; }
    msg.textContent = d.message || 'Connected';
    msg.style.color = '#3fb950';
    toast('Connected to ' + wifiTargetSSID, true);
    setTimeout(function() { document.getElementById('wifi-connect-panel').style.display = 'none'; wifiStatus(); }, 2000);
  }).catch(function() { msg.textContent = 'Request failed'; msg.style.color = '#f85149'; });
}

// Password field: Enter key submits
document.getElementById('wifi-password').addEventListener('keydown', function(e) {
  if (e.key === 'Enter') wifiConnect();
});

// ═══════════════════════════════════════════════
//  System
// ═══════════════════════════════════════════════
function runUpdate() {
  var el = document.getElementById('update-output');
  el.style.display = 'block';
  el.textContent = 'Running git pull…\n';
  api('/api/update', { method: 'POST' }).then(function(d) {
    el.textContent = d.output || d.error || 'Done';
    toast(d.error ? 'Update failed' : 'Update complete', !d.error);
  });
}

function loadCameras() {
  var el = document.getElementById('cameras-output');
  el.textContent = 'Detecting cameras…';
  api('/api/cameras').then(function(d) {
    el.textContent = d.output || d.error || 'No cameras detected';
  });
}

// ═══════════════════════════════════════════════
//  Health poll → header status
// ═══════════════════════════════════════════════
function checkHealth() {
  api('/health').then(function(d) {
    var el = document.getElementById('hdr-status');
    el.textContent = 'connected';
    el.style.color = '#3fb950';
  }).catch(function() {
    var el = document.getElementById('hdr-status');
    el.textContent = 'offline';
    el.style.color = '#f85149';
  });
}
checkHealth();
setInterval(checkHealth, 10000);
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Flask app factory
# ---------------------------------------------------------------------------

def create_app(cfg: dict) -> Flask:
    app = Flask(__name__)
    loops: dict[str, CameraLoop] = {}
    cam_info = []

    for name in ("top", "side"):
        try:
            cam_c = camera_cfg(cfg, name)
            loops[name] = CameraLoop(name, cfg)
            cam_info.append({"name": name, "label": cam_c["label"]})
        except Exception as e:
            print(f"[live] skip {name}: {e}")

    # Per-camera tolerance state (in-memory)
    tolerances: dict[str, int] = {name: 50 for name in loops}

    # ── Config path ──
    config_path = os.path.join(PROJECT_DIR, "config.yaml")

    # ──────────────────────────────────────────
    #  Main page
    # ──────────────────────────────────────────
    @app.route("/")
    def index():
        for ci in cam_info:
            n = ci["name"]
            if n in loops:
                ci["lo"] = loops[n].lo.tolist()
                ci["hi"] = loops[n].hi.tolist()
                ci["tolerance"] = tolerances.get(n, 50)
            else:
                lo, hi = color_range(cfg, n)
                ci["lo"] = lo
                ci["hi"] = hi
                ci["tolerance"] = 50
        return render_template_string(PAGE, cams=cam_info)

    # ──────────────────────────────────────────
    #  Camera frames
    # ──────────────────────────────────────────
    @app.route("/frame/<cam>/<kind>")
    def frame(cam: str, kind: str):
        if cam not in loops:
            return "no such camera", 404
        lp = loops[cam]
        with lp.lock:
            data = lp.raw_jpg if kind == "raw" else lp.det_jpg
        if not data:
            return Response(
                b"GIF89a\x01\x00\x01\x00\x80\x00\x00\xff\xff\xff"
                b"\x00\x00\x00!\xf9\x04\x00\x00\x00\x00\x00,"
                b"\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;",
                mimetype="image/gif",
            )
        return Response(data, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})

    # ──────────────────────────────────────────
    #  Color picker (existing API, kept)
    # ──────────────────────────────────────────
    @app.route("/color_info/<cam>")
    def color_info(cam: str):
        if cam not in loops:
            return jsonify(error="no such camera"), 404
        lp = loops[cam]
        lo, hi = lp.lo.tolist(), lp.hi.tolist()
        return jsonify(lower=lo, upper=hi, hex=_lab_to_hex(lo, hi), tolerance=tolerances.get(cam, 50))

    @app.route("/pick_color/<cam>", methods=["POST"])
    def pick_color(cam: str):
        if cam not in loops:
            return jsonify(error="no such camera"), 404
        data = request.get_json(silent=True) or {}
        fx = float(data.get("fx", 0.5))
        fy = float(data.get("fy", 0.5))
        tol = max(5, min(120, int(data.get("tolerance", 50))))
        tolerances[cam] = tol
        lp = loops[cam]
        with lp.lock:
            lab_img = lp._last_lab
            bgr_img = lp._last_frame
        if lab_img is None:
            return jsonify(error="no frame yet"), 503
        h, w = lab_img.shape[:2]
        px = max(0, min(int(fx * w), w - 1))
        py = max(0, min(int(fy * h), h - 1))
        r = 5
        patch = lab_img[max(0,py-r):py+r+1, max(0,px-r):px+r+1].reshape(-1, 3).astype(float)
        L, a, b = float(np.mean(patch[:,0])), float(np.mean(patch[:,1])), float(np.mean(patch[:,2]))
        lower = [max(0,int(L-tol)), max(0,int(a-tol)), max(0,int(b-tol))]
        upper = [min(255,int(L+tol)), min(255,int(a+tol)), min(255,int(b+tol))]
        lp.lo = np.array(lower)
        lp.hi = np.array(upper)
        _update_config_color(cfg, cam, lower, upper, config_path)
        bgr_px = bgr_img[py, px]
        hex_col = "#{:02x}{:02x}{:02x}".format(int(bgr_px[2]), int(bgr_px[1]), int(bgr_px[0]))
        return jsonify(L=int(L), a=int(a), b=int(b), lower=lower, upper=upper, tolerance=tol, hex=hex_col)

    @app.route("/set_tolerance/<cam>", methods=["POST"])
    def set_tolerance(cam: str):
        if cam not in loops:
            return jsonify(error="no such camera"), 404
        data = request.get_json(silent=True) or {}
        tol = max(5, min(120, int(data.get("tolerance", 50))))
        tolerances[cam] = tol
        lp = loops[cam]
        lo, hi = lp.lo.tolist(), lp.hi.tolist()
        L, a, b = (lo[0]+hi[0])/2, (lo[1]+hi[1])/2, (lo[2]+hi[2])/2
        lower = [max(0,int(L-tol)), max(0,int(a-tol)), max(0,int(b-tol))]
        upper = [min(255,int(L+tol)), min(255,int(a+tol)), min(255,int(b+tol))]
        lp.lo = np.array(lower)
        lp.hi = np.array(upper)
        _update_config_color(cfg, cam, lower, upper, config_path)
        return jsonify(lower=lower, upper=upper, tolerance=tol, hex=_lab_to_hex(lower, upper))

    @app.route("/health")
    def health():
        return jsonify(ok=True)

    # ──────────────────────────────────────────
    #  API: Recording
    # ──────────────────────────────────────────
    @app.route("/api/record/start", methods=["POST"])
    def api_record_start():
        with _record_lock:
            if _record_state["running"]:
                return jsonify(error="A recording is already running"), 409
        data = request.get_json(silent=True) or {}
        cam = data.get("camera", "both")
        duration = data.get("duration")
        until = data.get("until")
        sol_dur = data.get("solenoid")
        do_analyze = bool(data.get("analyze", False))

        def _run():
            with _record_lock:
                _record_state["running"] = True
                _record_state["camera"] = cam
                _record_state["started_at"] = time.time()
            try:
                if sol_dur:
                    sol_c = solenoid_cfg(cfg)
                    ctrl = SolenoidController(gpio_pin=sol_c["gpio_pin"], default_duration=sol_c["open_duration"])
                    ctrl.open(float(sol_dur))
                    time.sleep(float(sol_dur) + 0.2)
                    ctrl.cleanup()

                cams_to_run = ["top", "side"] if cam == "both" else [cam]
                recorders = [_build_recorder_from_cfg(cfg, c, duration, until) for c in cams_to_run]
                with _record_lock:
                    _record_state["recorders"] = recorders

                threads = [threading.Thread(target=r.run, daemon=True) for r in recorders]
                for t in threads: t.start()
                for t in threads: t.join()

                if do_analyze:
                    if cam == "both":
                        top_p = _find_latest_recording(camera_cfg(cfg, "top")["label"].lower().replace(" ","_"))
                        side_p = _find_latest_recording(camera_cfg(cfg, "side")["label"].lower().replace(" ","_"))
                        if top_p and side_p:
                            analyze_stereo(top_p, side_p, output_dir=os.path.join(PROJECT_DIR, "recordings", "combined"))
                    else:
                        lbl = camera_cfg(cfg, cam)["label"].lower().replace(" ","_")
                        p = _find_latest_recording(lbl)
                        if p:
                            analyze_recording(p)
            except Exception as e:
                print(f"[web] recording error: {e}")
            finally:
                with _record_lock:
                    _record_state["running"] = False
                    _record_state["recorders"] = []

        threading.Thread(target=_run, daemon=True).start()
        return jsonify(ok=True, camera=cam)

    @app.route("/api/record/stop", methods=["POST"])
    def api_record_stop():
        stop_path = os.path.join(PROJECT_DIR, Recorder.STOP_FILE)
        with open(stop_path, "w") as f:
            f.write("stop")
        return jsonify(ok=True, message="Stop signal sent")

    @app.route("/api/record/status")
    def api_record_status():
        with _record_lock:
            return jsonify(
                running=_record_state["running"],
                camera=_record_state["camera"],
                started_at=_record_state["started_at"],
            )

    # ──────────────────────────────────────────
    #  API: Solenoid
    # ──────────────────────────────────────────
    @app.route("/api/solenoid/<action>", methods=["POST"])
    def api_solenoid(action: str):
        if action not in ("open", "close", "test"):
            return jsonify(error="Unknown action"), 400
        data = request.get_json(silent=True) or {}
        dur = float(data.get("duration", 5))
        try:
            sol_c = solenoid_cfg(cfg)
            ctrl = SolenoidController(gpio_pin=sol_c["gpio_pin"], default_duration=sol_c["open_duration"])
            if action == "open":
                ctrl.open(dur)
                threading.Timer(dur + 0.1, ctrl.cleanup).start()
                return jsonify(ok=True, message=f"Solenoid opened for {dur}s")
            elif action == "close":
                ctrl.close()
                ctrl.cleanup()
                return jsonify(ok=True, message="Solenoid closed")
            elif action == "test":
                ctrl.open(1.0)
                threading.Timer(1.5, ctrl.cleanup).start()
                return jsonify(ok=True, message="Test pulse (1s)")
        except Exception as e:
            return jsonify(error=str(e)), 500

    # ──────────────────────────────────────────
    #  API: Recordings browser
    # ──────────────────────────────────────────
    @app.route("/api/recordings")
    def api_recordings():
        base = os.path.join(PROJECT_DIR, "recordings")
        recs = []
        if os.path.isdir(base):
            for cam_dir in sorted(os.listdir(base)):
                cam_path = os.path.join(base, cam_dir)
                if not os.path.isdir(cam_path):
                    continue
                for session in sorted(os.listdir(cam_path), reverse=True):
                    full = os.path.join(cam_path, session)
                    if not os.path.isdir(full):
                        continue
                    files = set(os.listdir(full))
                    recs.append({
                        "camera": cam_dir,
                        "timestamp": session,
                        "path": full,
                        "has_data": "recording_data.npz" in files,
                        "has_video": "video.mp4" in files,
                        "has_graphs": "graphs" in files,
                    })
        return jsonify(recordings=recs)

    @app.route("/api/recordings/graphs")
    def api_recordings_graphs():
        path = request.args.get("path", "")
        if not path or not os.path.isdir(path):
            return jsonify(error="path not found"), 404
        # Security: must be inside recordings dir
        rec_base = os.path.realpath(os.path.join(PROJECT_DIR, "recordings"))
        if not os.path.realpath(path).startswith(rec_base):
            return jsonify(error="invalid path"), 403
        graphs_dir = os.path.join(path, "graphs")
        graphs = sorted(glob.glob(os.path.join(graphs_dir, "*.png"))) if os.path.isdir(graphs_dir) else []
        return jsonify(graphs=graphs)

    @app.route("/api/recordings/image")
    def api_recordings_image():
        path = request.args.get("path", "")
        if not path or not os.path.isfile(path):
            return "not found", 404
        # Security: must be inside recordings dir
        rec_base = os.path.realpath(os.path.join(PROJECT_DIR, "recordings"))
        if not os.path.realpath(path).startswith(rec_base):
            return "forbidden", 403
        return send_file(path, mimetype="image/png")

    @app.route("/api/analyze", methods=["POST"])
    def api_analyze():
        data = request.get_json(silent=True) or {}
        path = data.get("path", "")
        if not path or not os.path.isdir(path):
            return jsonify(error="Recording path not found"), 404
        rec_base = os.path.realpath(os.path.join(PROJECT_DIR, "recordings"))
        if not os.path.realpath(path).startswith(rec_base):
            return jsonify(error="invalid path"), 403
        with _analyze_lock:
            if _analyze_state["running"]:
                return jsonify(error="Analysis already running"), 409
            _analyze_state["running"] = True

        def _run():
            try:
                result = analyze_recording(path)
                with _analyze_lock:
                    _analyze_state["last_result"] = result
                    _analyze_state["error"] = None
            except Exception as e:
                with _analyze_lock:
                    _analyze_state["error"] = str(e)
            finally:
                with _analyze_lock:
                    _analyze_state["running"] = False

        threading.Thread(target=_run, daemon=True).start()
        out_dir = os.path.join(path, "graphs")
        return jsonify(ok=True, output_dir=out_dir)

    @app.route("/api/analyze/stereo", methods=["POST"])
    def api_analyze_stereo():
        data = request.get_json(silent=True) or {}
        top, side = data.get("top", ""), data.get("side", "")
        rec_base = os.path.realpath(os.path.join(PROJECT_DIR, "recordings"))
        for p in (top, side):
            if not p or not os.path.isdir(p):
                return jsonify(error=f"Path not found: {p}"), 404
            if not os.path.realpath(p).startswith(rec_base):
                return jsonify(error="invalid path"), 403
        with _analyze_lock:
            if _analyze_state["running"]:
                return jsonify(error="Analysis already running"), 409
            _analyze_state["running"] = True
        out = os.path.join(PROJECT_DIR, "recordings", "combined")

        def _run():
            try:
                analyze_stereo(top, side, output_dir=out)
                with _analyze_lock:
                    _analyze_state["error"] = None
            except Exception as e:
                with _analyze_lock:
                    _analyze_state["error"] = str(e)
            finally:
                with _analyze_lock:
                    _analyze_state["running"] = False

        threading.Thread(target=_run, daemon=True).start()
        return jsonify(ok=True, output_dir=out)

    # ──────────────────────────────────────────
    #  API: Config
    # ──────────────────────────────────────────
    @app.route("/api/config/raw")
    def api_config_raw():
        if not os.path.exists(config_path):
            return "# config.yaml not found", 200, {"Content-Type": "text/plain"}
        with open(config_path, "r") as f:
            return f.read(), 200, {"Content-Type": "text/plain"}

    @app.route("/api/config", methods=["POST"])
    def api_config_save():
        data = request.get_json(silent=True) or {}
        raw = data.get("yaml", "")
        try:
            parsed = yaml.safe_load(raw)
            if not isinstance(parsed, dict):
                raise ValueError("Config must be a YAML mapping")
        except Exception as e:
            return jsonify(error=f"Invalid YAML: {e}"), 400
        with open(config_path, "w") as f:
            f.write(raw)
        # Reload in-memory cfg
        cfg.clear()
        cfg.update(parsed)
        return jsonify(ok=True)

    # ──────────────────────────────────────────
    #  API: WiFi (nmcli — Raspberry Pi OS Bookworm+)
    # ──────────────────────────────────────────
    @app.route("/api/wifi/status")
    def api_wifi_status():
        try:
            r = subprocess.run(
                ["nmcli", "-t", "-f", "NAME,DEVICE,STATE", "connection", "show", "--active"],
                capture_output=True, text=True, timeout=5,
            )
            conns = []
            for line in r.stdout.strip().splitlines():
                parts = line.split(":")
                if len(parts) >= 3:
                    conns.append({"name": parts[0], "device": parts[1], "state": parts[2]})
            return jsonify(connections=conns)
        except FileNotFoundError:
            return jsonify(error="nmcli not found — is NetworkManager installed?"), 503
        except Exception as e:
            return jsonify(error=str(e)), 500

    @app.route("/api/wifi/scan")
    def api_wifi_scan():
        try:
            # Trigger a rescan first (best-effort, may require sudo on some Pi configs)
            subprocess.run(["nmcli", "device", "wifi", "rescan"], capture_output=True, timeout=8)
            r = subprocess.run(
                ["nmcli", "-t", "-f", "SSID,SIGNAL,SECURITY", "device", "wifi", "list"],
                capture_output=True, text=True, timeout=8,
            )
            networks = []
            seen = set()
            for line in r.stdout.strip().splitlines():
                parts = line.split(":")
                if len(parts) >= 2:
                    ssid = parts[0].strip()
                    if not ssid or ssid in seen:
                        continue
                    seen.add(ssid)
                    networks.append({
                        "ssid": ssid,
                        "signal": parts[1] if len(parts) > 1 else "0",
                        "security": parts[2] if len(parts) > 2 else "",
                    })
            networks.sort(key=lambda x: int(x["signal"] or 0), reverse=True)
            return jsonify(networks=networks)
        except FileNotFoundError:
            return jsonify(error="nmcli not found"), 503
        except Exception as e:
            return jsonify(error=str(e)), 500

    @app.route("/api/wifi/connect", methods=["POST"])
    def api_wifi_connect():
        data = request.get_json(silent=True) or {}
        ssid = data.get("ssid", "").strip()
        password = data.get("password", "")
        if not ssid:
            return jsonify(error="SSID is required"), 400
        # Validate SSID — alphanumeric + common WiFi name chars only
        if not re.match(r'^[\w\s\-_.@#!+]+$', ssid):
            return jsonify(error="Invalid SSID"), 400
        try:
            # Use list args — no shell=True, no string interpolation
            cmd = ["nmcli", "device", "wifi", "connect", ssid]
            if password:
                cmd += ["password", password]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if r.returncode == 0:
                return jsonify(ok=True, message=f"Connected to {ssid}")
            else:
                # Sanitize output — don't echo the password back
                err = r.stderr.strip() or r.stdout.strip()
                err = re.sub(r'password\s+\S+', 'password ***', err, flags=re.IGNORECASE)
                return jsonify(error=err), 400
        except FileNotFoundError:
            return jsonify(error="nmcli not found"), 503
        except subprocess.TimeoutExpired:
            return jsonify(error="Connection timed out"), 504
        except Exception as e:
            return jsonify(error=str(e)), 500

    # ──────────────────────────────────────────
    #  API: System
    # ──────────────────────────────────────────
    @app.route("/api/update", methods=["POST"])
    def api_update():
        lines = []
        try:
            stash = subprocess.run(
                ["git", "stash", "--include-untracked"],
                cwd=PROJECT_DIR, capture_output=True, text=True,
            )
            stashed = "No local changes" not in stash.stdout
            pull = subprocess.run(
                ["git", "pull"], cwd=PROJECT_DIR, capture_output=True, text=True,
            )
            lines.append(pull.stdout)
            if pull.returncode != 0:
                lines.append(pull.stderr)
            if stashed:
                pop = subprocess.run(
                    ["git", "stash", "pop"], cwd=PROJECT_DIR, capture_output=True, text=True,
                )
                lines.append(pop.stdout)
            return jsonify(ok=pull.returncode == 0, output="\n".join(lines))
        except Exception as e:
            return jsonify(error=str(e)), 500

    @app.route("/api/cameras")
    def api_cameras():
        try:
            import io as _io
            import contextlib
            buf = _io.StringIO()
            with contextlib.redirect_stdout(buf):
                list_cameras()
            return jsonify(output=buf.getvalue())
        except Exception as e:
            return jsonify(error=str(e)), 500

    return app


# ---------------------------------------------------------------------------
# Config color persistence helper
# ---------------------------------------------------------------------------

def _update_config_color(cfg: dict, cam: str, lower: list, upper: list, config_path: str) -> None:
    if not os.path.exists(config_path):
        return
    with open(config_path, "r") as f:
        raw = f.read()
    lower_str = f"{lower[0]}, {lower[1]}, {lower[2]}"
    upper_str = f"{upper[0]}, {upper[1]}, {upper[2]}"
    pat = re.compile(
        rf'(  {re.escape(cam)}:.*?color:\s*\n\s*lower:\s*\[)[^\]]*(\].*?upper:\s*\[)[^\]]*(\])',
        re.DOTALL,
    )
    new_raw = pat.sub(lambda m: m.group(1) + lower_str + m.group(2) + upper_str + m.group(3), raw)
    if new_raw != raw:
        with open(config_path, "w") as f:
            f.write(new_raw)
    cfg.setdefault("cameras", {}).setdefault(cam, {}).setdefault("color", {})
    cfg["cameras"][cam]["color"]["lower"] = lower
    cfg["cameras"][cam]["color"]["upper"] = upper
