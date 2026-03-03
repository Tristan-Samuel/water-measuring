"""
Flask debug viewer for the Water Measuring system.

Shows live camera feeds with color detection overlay in a browser.
Access from any device on the same network as the Pi.

Usage:
    python3 cli.py live              # http://<pi-ip>:5000
    python3 cli.py live --port 8080  # http://<pi-ip>:8080
"""

from __future__ import annotations

import threading
import time
import cv2  # type: ignore
import numpy as np  # type: ignore
from flask import Flask, Response, render_template_string  # type: ignore

from camera import create_camera
from config_loader import camera_cfg, color_range, analysis_cfg


# region HTML Template

PAGE_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Water Measuring — Live Viewer</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #1a1a2e; color: #eee;
        }
        header {
            background: #16213e; padding: 12px 20px;
            display: flex; align-items: center; gap: 12px;
        }
        header h1 { font-size: 18px; font-weight: 600; }
        .status { font-size: 12px; }
        .status.ok { color: #4ecca3; }
        .status.err { color: #e74c3c; }
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px; padding: 8px;
        }
        @media (max-width: 800px) {
            .grid { grid-template-columns: 1fr; }
        }
        .feed {
            background: #0f3460; border-radius: 8px;
            overflow: hidden;
        }
        .feed-header {
            font-size: 14px; padding: 8px 12px;
            background: rgba(0,0,0,0.3);
            display: flex; justify-content: space-between; align-items: center;
        }
        .feed-header .fps { font-size: 11px; color: #4ecca3; font-weight: normal; }
        .feed img {
            width: 100%; display: block;
            min-height: 120px; background: #0a0a1a;
        }
        .info {
            padding: 12px 20px; font-size: 13px;
            color: #888; text-align: center;
        }
    </style>
</head>
<body>
    <header>
        <h1>💧 Water Measuring — Live</h1>
        <span id="global-status" class="status ok">● connecting…</span>
    </header>
    <div class="grid">
        {% for cam in cameras %}
        <div class="feed">
            <div class="feed-header">
                <span>{{ cam.label }} — Raw</span>
                <span class="fps" id="fps-{{ cam.name }}-raw"></span>
            </div>
            <img id="img-{{ cam.name }}-raw" alt="{{ cam.label }} raw">
        </div>
        <div class="feed">
            <div class="feed-header">
                <span>{{ cam.label }} — Detection</span>
                <span class="fps" id="fps-{{ cam.name }}-detection"></span>
            </div>
            <img id="img-{{ cam.name }}-detection" alt="{{ cam.label }} detection">
        </div>
        {% endfor %}
    </div>
    <div class="info">
        Color range (CIELAB): {{ lower }} → {{ upper }}<br>
        Live-updating streams. Open on any device on the same network.
    </div>
    <script>
    (function() {
        var feeds = [
            {% for cam in cameras %}
            {cam: "{{ cam.name }}", kind: "raw"},
            {cam: "{{ cam.name }}", kind: "detection"},
            {% endfor %}
        ];

        feeds.forEach(function(f) {
            var img = document.getElementById("img-" + f.cam + "-" + f.kind);
            var fpsEl = document.getElementById("fps-" + f.cam + "-" + f.kind);
            var frames = 0, lastCount = 0, lastCheck = performance.now();
            var inflight = false;

            // FPS counter — update display every second
            setInterval(function() {
                var now = performance.now();
                var elapsed = (now - lastCheck) / 1000;
                if (elapsed > 0) {
                    var fps = ((frames - lastCount) / elapsed).toFixed(1);
                    fpsEl.textContent = fps + " fps";
                }
                lastCount = frames;
                lastCheck = now;
            }, 1000);

            // Pure snapshot polling — works in every browser
            function poll() {
                if (inflight) return;  // don't stack requests
                inflight = true;
                var url = "/snapshot/" + f.cam + "/" + f.kind + "?t=" + Date.now();

                // Create a new Image to decode the JPEG off-screen,
                // then swap it in.  This avoids flicker and ensures
                // the browser actually fetches a new image each time.
                var tmp = new Image();
                tmp.onload = function() {
                    img.src = tmp.src;
                    frames++;
                    inflight = false;
                    // Request next frame on next tick
                    setTimeout(poll, 33);  // ~30 fps target
                };
                tmp.onerror = function() {
                    inflight = false;
                    setTimeout(poll, 1000);  // retry after 1 s on error
                };
                tmp.src = url;
            }
            // Kick off polling
            poll();
        });

        // Global status indicator
        var statusEl = document.getElementById("global-status");
        function checkAlive() {
            fetch("/health").then(function(r) {
                if (r.ok) {
                    statusEl.textContent = "● streaming";
                    statusEl.className = "status ok";
                } else { throw 0; }
            }).catch(function() {
                statusEl.textContent = "● disconnected";
                statusEl.className = "status err";
            });
        }
        setInterval(checkAlive, 3000);
        checkAlive();
    })();
    </script>
</body>
</html>
"""

# endregion


# region Stream Manager

class CameraStream:
    """Manages a camera and produces JPEG frames for streaming."""

    def __init__(self, name: str, cfg: dict):
        cam_c = camera_cfg(cfg, name)
        self.name = name
        self.label = cam_c["label"]
        self.camera = create_camera(
            cam_id=cam_c["id"],
            resolution=tuple(cam_c["resolution"]),
        )
        lower, upper = color_range(cfg)
        self.color_lower = np.array(lower)
        self.color_upper = np.array(upper)

        ana_c = analysis_cfg(cfg)
        self.min_contour_area = ana_c["min_contour_area"]

        self._raw_frame: bytes = b""
        self._det_frame: bytes = b""
        self._frame_num: int = 0
        self._lock = threading.Lock()
        self._running = True

        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self):
        frame_count = 0
        t0 = time.time()
        while self._running:
            frame = self.camera.read()
            if frame is None:
                time.sleep(0.05)
                continue

            frame_count += 1
            if frame_count % 100 == 0:
                elapsed = time.time() - t0
                fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"[live:{self.name}] captured {frame_count} frames "
                      f"({fps:.1f} fps avg)")

            # Encode raw frame
            _, raw_jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])

            # Run detection
            det_frame = frame.copy()
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            mask = cv2.inRange(lab, self.color_lower, self.color_upper)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            total_px = 0
            for cnt in contours:
                if cv2.contourArea(cnt) < self.min_contour_area:
                    continue
                cv2.drawContours(det_frame, [cnt], -1, (0, 255, 0), 2)
                # Per-group pixel count
                single = np.zeros_like(mask)
                cv2.drawContours(single, [cnt], -1, 255, thickness=-1)
                px = cv2.countNonZero(cv2.bitwise_and(mask, single))
                total_px += px
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(det_frame, f"{px}px", (cx - 30, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # HUD
            cv2.putText(det_frame, f"Total: {total_px}px", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            # Frame counter — visual proof that stream is live
            cv2.putText(det_frame, f"#{frame_count}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2)

            # Overlay mask as semi-transparent red
            red_overlay = np.zeros_like(det_frame)
            red_overlay[:, :, 2] = mask  # red channel
            det_frame = cv2.addWeighted(det_frame, 1.0, red_overlay, 0.3, 0)

            _, det_jpg = cv2.imencode(".jpg", det_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])

            with self._lock:
                self._raw_frame = raw_jpg.tobytes()
                self._det_frame = det_jpg.tobytes()
                self._frame_num = frame_count

            time.sleep(0.03)  # ~30 fps cap

    def get_raw(self) -> bytes:
        with self._lock:
            return self._raw_frame

    def get_detection(self) -> bytes:
        with self._lock:
            return self._det_frame

    def stop(self):
        self._running = False
        self.camera.release()

# endregion


# region App Factory

def create_app(cfg: dict) -> Flask:
    """Create and configure the Flask app."""
    app = Flask(__name__)

    streams: dict[str, CameraStream] = {}

    # Figure out which cameras to start
    camera_info = []
    for name in ("top", "side"):
        try:
            cam_c = camera_cfg(cfg, name)
            stream = CameraStream(name, cfg)
            streams[name] = stream
            camera_info.append({"name": name, "label": cam_c["label"]})
        except Exception as e:
            print(f"[live] Could not start {name} camera: {e}")

    lower, upper = color_range(cfg)

    @app.route("/")
    def index():
        return render_template_string(
            PAGE_HTML,
            cameras=camera_info,
            lower=lower,
            upper=upper,
        )

    @app.route("/snapshot/<cam_name>/<kind>")
    def snapshot(cam_name: str, kind: str):
        """Return a single JPEG frame (used by JS polling)."""
        if cam_name not in streams:
            return "Camera not found", 404
        if kind not in ("raw", "detection"):
            return "Use /raw or /detection", 400
        s = streams[cam_name]
        with s._lock:
            frame = s._raw_frame if kind == "raw" else s._det_frame
            num = s._frame_num
        if not frame:
            return "", 204
        return Response(frame, mimetype="image/jpeg", headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
            "X-Frame-Number": str(num),
        })

    @app.route("/health")
    def health():
        return {"ok": True, "cameras": list(streams.keys())}

    @app.teardown_appcontext
    def cleanup(exception):
        for s in streams.values():
            s.stop()

    return app

# endregion
