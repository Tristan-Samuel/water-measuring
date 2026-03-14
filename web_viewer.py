"""
Flask live viewer — keeps it simple.

Uses a background thread per camera to grab frames.
Serves each frame as a fresh JPEG via /stream/<cam>/<kind>.
The page uses plain <img> tags that reload themselves via JS setInterval.
"""

from __future__ import annotations

import threading
import time
import cv2  # type: ignore
import numpy as np  # type: ignore
from flask import Flask, Response, render_template_string, request, jsonify

import os
from camera import create_camera
from config_loader import camera_cfg, camera_crop, color_range, analysis_cfg, clahe_cfg


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
# HTML — deliberately minimal
# ---------------------------------------------------------------------------

PAGE = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Water Measuring — Live</title>
<style>
body{margin:0;background:#111;color:#eee;font-family:sans-serif}
h1{font-size:16px;padding:10px 16px;margin:0;background:#1a1a2e}
.g{display:flex;flex-wrap:wrap;gap:6px;padding:6px}
.f{flex:1 1 48%;min-width:300px;background:#0f3460;border-radius:6px;overflow:hidden;position:relative}
.f h2{font-size:13px;padding:6px 10px;margin:0;background:rgba(0,0,0,.3)}
.f img{width:100%;display:block;background:#000;cursor:crosshair}
#toast{position:fixed;top:12px;right:12px;background:#1a1a2e;border:1px solid #444;
  border-radius:8px;padding:12px 16px;font-size:13px;line-height:1.5;
  display:none;z-index:999;min-width:220px;box-shadow:0 4px 20px rgba(0,0,0,.6)}
#toast .swatch{display:inline-block;width:18px;height:18px;border-radius:3px;
  vertical-align:middle;margin-right:6px;border:1px solid #666}
#toast .close{float:right;cursor:pointer;opacity:.6;margin-left:12px}
#toast .close:hover{opacity:1}
</style>
</head>
<body>
<h1>💧 Live <span style="font-size:12px;opacity:.6">(click any image to pick color)</span></h1>
<div id="toast"><span class="close" onclick="this.parentNode.style.display='none'">&times;</span><div id="toast-body"></div></div>
<div class="g">
{% for c in cams %}
  <div class="f">
    <h2>{{ c.label }} — Raw</h2>
    <img id="r{{ c.name }}" data-cam="{{ c.name }}">
  </div>
  <div class="f">
    <h2>{{ c.label }} — Detection</h2>
    <img id="d{{ c.name }}" data-cam="{{ c.name }}">
  </div>
{% endfor %}
</div>
<script>
// Each image runs its own load loop: request next frame only after
// the current one finishes (or errors).  This prevents connection
// pile-up that causes the feed to stall / freeze.
var cams = [{% for c in cams %}"{{ c.name }}",{% endfor %}];
var MIN_INTERVAL = 80;  // ms — minimum gap between requests per image

function startLoop(img, url) {
    var pending = false;

    function next() {
        if (pending) return;
        pending = true;
        var start = Date.now();
        var tmp = new Image();
        tmp.onload = function() {
            img.src = tmp.src;           // swap only when fully loaded
            pending = false;
            var elapsed = Date.now() - start;
            setTimeout(next, Math.max(0, MIN_INTERVAL - elapsed));
        };
        tmp.onerror = function() {
            pending = false;
            setTimeout(next, 500);       // back off on error
        };
        tmp.src = url + "?_=" + Date.now();
    }
    next();
}

cams.forEach(function(n) {
    startLoop(document.getElementById("r"+n), "/frame/"+n+"/raw");
    startLoop(document.getElementById("d"+n), "/frame/"+n+"/det");
});

// ---- click-to-pick-color ----
document.querySelectorAll('.f img').forEach(function(img) {
    img.addEventListener('click', function(e) {
        var rect = img.getBoundingClientRect();
        var fx = (e.clientX - rect.left) / rect.width;
        var fy = (e.clientY - rect.top) / rect.height;
        var cam = img.getAttribute('data-cam');
        if (!cam) return;
        fetch('/pick_color/' + cam, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({fx: fx, fy: fy, tolerance: 50})
        })
        .then(function(r){ return r.json(); })
        .then(function(d) {
            if (d.error) { alert(d.error); return; }
            var t = document.getElementById('toast');
            var b = document.getElementById('toast-body');
            b.innerHTML = '<span class="swatch" style="background:'+d.hex+'"></span>'
                + '<b>' + cam + '</b> color updated<br>'
                + 'LAB center: ' + d.L + ', ' + d.a + ', ' + d.b + '<br>'
                + 'Lower: [' + d.lower.join(', ') + ']<br>'
                + 'Upper: [' + d.upper.join(', ') + ']<br>'
                + '<span style="opacity:.6">tolerance &plusmn;' + d.tolerance + '</span>';
            t.style.display = 'block';
            setTimeout(function(){ t.style.display='none'; }, 8000);
        })
        .catch(function(err){ console.error(err); });
    });
});
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Flask app
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

    @app.route("/")
    def index():
        lo, hi = color_range(cfg)  # global defaults for display
        return render_template_string(PAGE, cams=cam_info, lo=lo, hi=hi)

    @app.route("/frame/<cam>/<kind>")
    def frame(cam: str, kind: str):
        if cam not in loops:
            return "no such camera", 404
        lp = loops[cam]
        with lp.lock:
            data = lp.raw_jpg if kind == "raw" else lp.det_jpg
        if not data:
            # 1×1 transparent gif so the img tag doesn't break
            return Response(
                b"GIF89a\x01\x00\x01\x00\x80\x00\x00\xff\xff\xff"
                b"\x00\x00\x00!\xf9\x04\x00\x00\x00\x00\x00,"
                b"\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;",
                mimetype="image/gif",
            )
        return Response(data, mimetype="image/jpeg", headers={
            "Cache-Control": "no-store",
        })

    @app.route("/pick_color/<cam>", methods=["POST"])
    def pick_color(cam: str):
        """Sample the colour at the clicked point and update detection bounds."""
        if cam not in loops:
            return jsonify(error="no such camera"), 404

        data = request.get_json(silent=True) or {}
        fx = float(data.get("fx", 0.5))
        fy = float(data.get("fy", 0.5))
        tolerance = int(data.get("tolerance", 50))

        lp = loops[cam]
        with lp.lock:
            lab = lp._last_lab
            bgr = lp._last_frame

        if lab is None or bgr is None:
            return jsonify(error="no frame yet"), 503

        h, w = lab.shape[:2]
        px = max(0, min(int(fx * w), w - 1))
        py = max(0, min(int(fy * h), h - 1))

        # Sample a small patch (11×11) for a stable colour
        r = 5
        y0, y1 = max(0, py - r), min(h, py + r + 1)
        x0, x1 = max(0, px - r), min(w, px + r + 1)
        patch = lab[y0:y1, x0:x1].reshape(-1, 3).astype(float)
        L, a, b = float(np.mean(patch[:, 0])), float(np.mean(patch[:, 1])), float(np.mean(patch[:, 2]))

        lower = [
            max(0, int(L - tolerance)),
            max(0, int(a - tolerance)),
            max(0, int(b - tolerance)),
        ]
        upper = [
            min(255, int(L + tolerance)),
            min(255, int(a + tolerance)),
            min(255, int(b + tolerance)),
        ]

        # Update the live loop immediately
        lp.lo = np.array(lower)
        lp.hi = np.array(upper)

        # Also persist to config.yaml
        _update_config_color(cfg, cam, lower, upper)

        # Approximate hex for the UI swatch (LAB → BGR → hex)
        bgr_px = bgr[py, px]
        hex_col = "#{:02x}{:02x}{:02x}".format(int(bgr_px[2]), int(bgr_px[1]), int(bgr_px[0]))

        print(f"[live] pick_color {cam} @ ({px},{py}): L={L:.0f} a={a:.0f} b={b:.0f}  "
              f"lower={lower} upper={upper}")

        return jsonify(
            L=int(L), a=int(a), b=int(b),
            lower=lower, upper=upper,
            tolerance=tolerance, hex=hex_col,
        )

    @app.route("/health")
    def health():
        return {"ok": True}

    return app


def _update_config_color(cfg: dict, cam: str, lower: list[int], upper: list[int]) -> None:
    """Persist picked colour bounds to config.yaml (per-camera section)."""
    import re
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    if not os.path.exists(config_path):
        return
    with open(config_path, "r") as f:
        raw = f.read()

    lower_str = f"{lower[0]}, {lower[1]}, {lower[2]}"
    upper_str = f"{upper[0]}, {upper[1]}, {upper[2]}"

    cam_pattern = re.compile(
        rf'(  {re.escape(cam)}:.*?color:\s*\n\s*lower:\s*\[)[^\]]*(\].*?upper:\s*\[)[^\]]*(\])',
        re.DOTALL,
    )
    new_raw = cam_pattern.sub(
        lambda m: m.group(1) + lower_str + m.group(2) + upper_str + m.group(3),
        raw,
    )
    if new_raw != raw:
        with open(config_path, "w") as f:
            f.write(new_raw)
        # Also update the in-memory cfg so future cameras/restarts see it
        cfg.setdefault("cameras", {}).setdefault(cam, {}).setdefault("color", {})
        cfg["cameras"][cam]["color"]["lower"] = lower
        cfg["cameras"][cam]["color"]["upper"] = upper
