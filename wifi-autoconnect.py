#!/usr/bin/env python3
"""
wifi-autoconnect.py — Connect to the strongest open (no-password) WiFi on boot.

Run as a systemd oneshot service before water-web.service.
Requires NetworkManager (nmcli) — available by default on Pi OS Bookworm.

Usage:
    python3 wifi-autoconnect.py
"""

import subprocess
import sys
import time


def rescan():
    subprocess.run(["nmcli", "device", "wifi", "rescan"],
                   capture_output=True, timeout=10)
    time.sleep(3)  # allow scan to populate


def list_open_networks():
    """Return list of (signal_strength, ssid) for open networks, sorted strongest first."""
    r = subprocess.run(
        ["nmcli", "-t", "-f", "SSID,SIGNAL,SECURITY", "device", "wifi", "list"],
        capture_output=True, text=True, timeout=15,
    )
    networks = []
    for line in r.stdout.splitlines():
        # nmcli -t format: SSID:SIGNAL:SECURITY  (colons in SSID are escaped as \:)
        # Split from right so we reliably get SIGNAL and SECURITY
        parts = line.rsplit(":", 2)
        if len(parts) != 3:
            continue
        ssid_raw, signal_str, security = parts
        # Unescape nmcli's backslash-escaped colons in SSIDs
        ssid = ssid_raw.replace("\\:", ":").strip()
        # Open networks have security == '--'
        if security.strip() != "--" or not ssid:
            continue
        try:
            networks.append((int(signal_str), ssid))
        except ValueError:
            pass
    # Deduplicate (same SSID can appear multiple times), keep highest signal
    seen = {}
    for signal, ssid in networks:
        if ssid not in seen or signal > seen[ssid]:
            seen[ssid] = signal
    return sorted(((s, n) for n, s in seen.items()), reverse=True)


def already_connected():
    r = subprocess.run(
        ["nmcli", "-t", "-f", "STATE", "networking"],
        capture_output=True, text=True, timeout=5,
    )
    return "connected" in r.stdout.lower()


def main():
    print("[wifi-autoconnect] Starting …")

    if already_connected():
        print("[wifi-autoconnect] Already connected, nothing to do.")
        sys.exit(0)

    print("[wifi-autoconnect] Rescanning for open networks …")
    try:
        rescan()
    except Exception as e:
        print(f"[wifi-autoconnect] Rescan failed: {e}")

    try:
        networks = list_open_networks()
    except Exception as e:
        print(f"[wifi-autoconnect] Could not list networks: {e}")
        sys.exit(1)

    if not networks:
        print("[wifi-autoconnect] No open networks found.")
        sys.exit(1)

    print(f"[wifi-autoconnect] Found {len(networks)} open network(s):")
    for sig, ssid in networks:
        print(f"  {sig:3d}%  {ssid}")

    for signal, ssid in networks:
        print(f"[wifi-autoconnect] Trying '{ssid}' (signal {signal}%) …")
        try:
            r = subprocess.run(
                ["nmcli", "device", "wifi", "connect", ssid],
                capture_output=True, text=True, timeout=30,
            )
            if r.returncode == 0:
                print(f"[wifi-autoconnect] Connected to '{ssid}'")
                sys.exit(0)
            else:
                err = (r.stderr or r.stdout).strip()
                print(f"[wifi-autoconnect] Failed: {err}")
        except subprocess.TimeoutExpired:
            print(f"[wifi-autoconnect] Timed out connecting to '{ssid}'")
        except Exception as e:
            print(f"[wifi-autoconnect] Error: {e}")

    print("[wifi-autoconnect] Could not connect to any open network.")
    sys.exit(1)


if __name__ == "__main__":
    main()
