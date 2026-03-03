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
from flask import Flask, Response, render_template_string

from camera import create_camera
from config_loader import camera_cfg, color_range, analysis_cfg


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
        lo, hi = color_range(cfg)
        self.lo = np.array(lo)
        self.hi = np.array(hi)
        self.min_area = analysis_cfg(cfg)["min_contour_area"]

        self.raw_jpg: bytes = b""
        self.det_jpg: bytes = b""
        self.seq: int = 0
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

            # --- raw jpeg ---
            ok1, buf1 = cv2.imencode(".jpg", frame,
                                     [cv2.IMWRITE_JPEG_QUALITY, 70])

            # --- detection jpeg ---
            det = frame.copy()
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
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
.f{flex:1 1 48%;min-width:300px;background:#0f3460;border-radius:6px;overflow:hidden}
.f h2{font-size:13px;padding:6px 10px;margin:0;background:rgba(0,0,0,.3)}
.f img{width:100%;display:block;background:#000}
</style>
</head>
<body>
<h1>💧 Live</h1>
<div class="g">
{% for c in cams %}
  <div class="f">
    <h2>{{ c.label }} — Raw</h2>
    <img id="r{{ c.name }}">
  </div>
  <div class="f">
    <h2>{{ c.label }} — Detection</h2>
    <img id="d{{ c.name }}">
  </div>
{% endfor %}
</div>
<script>
// Dead-simple: every 100 ms, set each img.src to a new URL.
// The ?_= cache-buster forces a real HTTP request every time.
var cams = [{% for c in cams %}"{{ c.name }}",{% endfor %}];
function reload() {
    var t = Date.now();
    cams.forEach(function(n) {
        document.getElementById("r"+n).src = "/frame/"+n+"/raw?_="+t;
        document.getElementById("d"+n).src = "/frame/"+n+"/det?_="+t;
    });
}
setInterval(reload, 100);
reload();
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
        lo, hi = color_range(cfg)
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

    @app.route("/health")
    def health():
        return {"ok": True}

    return app
