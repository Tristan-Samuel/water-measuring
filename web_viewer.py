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
        const feeds = [
            {% for cam in cameras %}
            {cam: "{{ cam.name }}", kind: "raw"},
            {cam: "{{ cam.name }}", kind: "detection"},
            {% endfor %}
        ];

        // Try MJPEG first; fall back to snapshot polling if the browser
        // doesn't push updates (detected by watching naturalWidth changes).
        feeds.forEach(function(f) {
            const img = document.getElementById("img-" + f.cam + "-" + f.kind);
            const fpsEl = document.getElementById("fps-" + f.cam + "-" + f.kind);
            let frames = 0, lastCount = 0, lastCheck = performance.now();
            let usingMjpeg = true;
            let abortCtrl = null;

            // FPS counter
            setInterval(function() {
                const now = performance.now();
                const elapsed = (now - lastCheck) / 1000;
                const fps = ((frames - lastCount) / elapsed).toFixed(1);
                fpsEl.textContent = fps + " fps";
                lastCount = frames;
                lastCheck = now;
            }, 1000);

            function startMjpeg() {
                usingMjpeg = true;
                img.src = "/feed/" + f.cam + "/" + f.kind + "?t=" + Date.now();
                img.onload = function() { frames++; };

                // Detect stale: if no new frame in 3 s switch to polling
                let stalePrev = frames;
                setTimeout(function checkStale() {
                    if (!usingMjpeg) return;
                    if (frames === stalePrev) {
                        console.log(f.cam + "/" + f.kind + ": MJPEG stale, switching to polling");
                        usingMjpeg = false;
                        startPolling();
                        return;
                    }
                    stalePrev = frames;
                    setTimeout(checkStale, 3000);
                }, 4000);  // give MJPEG 4 s to produce first frame
            }

            function startPolling() {
                function fetchFrame() {
                    const url = "/snapshot/" + f.cam + "/" + f.kind + "?t=" + Date.now();
                    fetch(url).then(function(r) { return r.blob(); }).then(function(blob) {
                        const objectUrl = URL.createObjectURL(blob);
                        const prev = img.src;
                        img.src = objectUrl;
                        // Revoke previous blob URL to avoid memory leak
                        if (prev && prev.startsWith("blob:")) URL.revokeObjectURL(prev);
                        frames++;
                        setTimeout(fetchFrame, 50);  // ~20 fps
                    }).catch(function() {
                        setTimeout(fetchFrame, 1000);  // retry on error
                    });
                }
                fetchFrame();
            }

            startMjpeg();
        });

        // Global status indicator
        const statusEl = document.getElementById("global-status");
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
        self._lock = threading.Lock()
        self._running = True

        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self):
        while self._running:
            frame = self.camera.read()
            if frame is None:
                time.sleep(0.05)
                continue

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

            # Overlay mask as semi-transparent red
            red_overlay = np.zeros_like(det_frame)
            red_overlay[:, :, 2] = mask  # red channel
            det_frame = cv2.addWeighted(det_frame, 1.0, red_overlay, 0.3, 0)

            _, det_jpg = cv2.imencode(".jpg", det_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])

            with self._lock:
                self._raw_frame = raw_jpg.tobytes()
                self._det_frame = det_jpg.tobytes()

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

    def _mjpeg_stream(stream: CameraStream, kind: str):
        """Generate MJPEG frames for a stream."""
        while True:
            if kind == "raw":
                frame = stream.get_raw()
            else:
                frame = stream.get_detection()

            if frame:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )
            time.sleep(0.05)  # ~20 fps delivery

    @app.route("/feed/<cam_name>/<kind>")
    def feed(cam_name: str, kind: str):
        if cam_name not in streams:
            return "Camera not found", 404
        if kind not in ("raw", "detection"):
            return "Use /raw or /detection", 400
        return Response(
            _mjpeg_stream(streams[cam_name], kind),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/snapshot/<cam_name>/<kind>")
    def snapshot(cam_name: str, kind: str):
        """Return a single JPEG frame (used by JS polling fallback)."""
        if cam_name not in streams:
            return "Camera not found", 404
        if kind not in ("raw", "detection"):
            return "Use /raw or /detection", 400
        s = streams[cam_name]
        frame = s.get_raw() if kind == "raw" else s.get_detection()
        if not frame:
            return "", 204
        return Response(frame, mimetype="image/jpeg",
                        headers={"Cache-Control": "no-store"})

    @app.route("/health")
    def health():
        return {"ok": True, "cameras": list(streams.keys())}

    @app.teardown_appcontext
    def cleanup(exception):
        for s in streams.values():
            s.stop()

    return app

# endregion
