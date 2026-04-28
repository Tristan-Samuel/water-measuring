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

# SSIDs to always try first (case-insensitive), even if password-protected
PRIORITY_SSIDS = {"tristan", "shawn"}


def rescan():
    subprocess.run(["nmcli", "device", "wifi", "rescan"],
                   capture_output=True, timeout=10)
    time.sleep(3)  # allow scan to populate


def list_priority_networks():
    """Return (signal, ssid) for PRIORITY_SSIDS visible in scan, strongest first."""
    r = subprocess.run(
        ["nmcli", "-t", "-f", "SSID,SIGNAL", "device", "wifi", "list"],
        capture_output=True, text=True, timeout=15,
    )
    seen = {}
    for line in r.stdout.splitlines():
        parts = line.rsplit(":", 1)
        if len(parts) != 2:
            continue
        ssid_raw, signal_str = parts
        ssid = ssid_raw.replace("\\:", ":").strip()
        if ssid.lower() not in PRIORITY_SSIDS or not ssid:
            continue
        try:
            signal = int(signal_str)
        except ValueError:
            continue
        if ssid not in seen or signal > seen[ssid]:
            seen[ssid] = signal
    return sorted(((s, n) for n, s in seen.items()), reverse=True)


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
        # Open networks appear as '--' in human mode or empty string in terse (-t) mode
        sec = security.strip()
        if sec not in ("--", "") or not ssid:
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


def has_internet(timeout=8):
    """Return True if we can reach a public IP (bypasses DNS issues)."""
    for host in ("8.8.8.8", "1.1.1.1"):
        try:
            r = subprocess.run(
                ["ping", "-c", "2", "-W", str(timeout), host],
                capture_output=True, timeout=timeout + 2,
            )
            if r.returncode == 0:
                return True
        except Exception:
            pass
    return False


def disconnect_wifi():
    """Disconnect from any active WiFi connection."""
    r = subprocess.run(
        ["nmcli", "-t", "-f", "DEVICE,TYPE,STATE", "device"],
        capture_output=True, text=True, timeout=5,
    )
    for line in r.stdout.splitlines():
        parts = line.split(":")
        if len(parts) >= 3 and parts[1] == "wifi" and parts[2] == "connected":
            subprocess.run(["nmcli", "device", "disconnect", parts[0]],
                           capture_output=True, timeout=10)
            break



def wait_for_wifi_device(max_wait=60):
    """Wait until a WiFi device is managed and available, up to max_wait seconds."""
    print("[wifi-autoconnect] Waiting for WiFi device to be ready …")
    for _ in range(max_wait):
        r = subprocess.run(
            ["nmcli", "-t", "-f", "DEVICE,TYPE,STATE", "device"],
            capture_output=True, text=True, timeout=5,
        )
        for line in r.stdout.splitlines():
            parts = line.split(":")
            if len(parts) >= 3 and parts[1] == "wifi" and parts[2] in ("disconnected", "connected"):
                print(f"[wifi-autoconnect] WiFi device ready: {parts[0]}")
                return True
        time.sleep(1)
    print("[wifi-autoconnect] WiFi device not ready after timeout.")
    return False


def already_connected():
    """Return True only if a WiFi interface is currently active (not just any network)."""
    r = subprocess.run(
        ["nmcli", "-t", "-f", "TYPE,STATE", "connection", "show", "--active"],
        capture_output=True, text=True, timeout=5,
    )
    for line in r.stdout.strip().splitlines():
        parts = line.split(":")
        if len(parts) >= 2 and parts[0] == "802-11-wireless" and "activated" in parts[1]:
            return True
    return False


def main():
    print("[wifi-autoconnect] Starting …")

    if already_connected():
        print("[wifi-autoconnect] Already connected, nothing to do.")
        sys.exit(0)

    # Wait for WiFi hardware to be ready before scanning
    if not wait_for_wifi_device(max_wait=30):
        print("[wifi-autoconnect] No WiFi device found — skipping.")
        sys.exit(0)

    print("[wifi-autoconnect] Rescanning …")
    try:
        rescan()
    except Exception as e:
        print(f"[wifi-autoconnect] Rescan failed: {e}")

    # --- Priority networks first (tristan / shawn) ---
    try:
        priority = list_priority_networks()
    except Exception as e:
        print(f"[wifi-autoconnect] Could not list priority networks: {e}")
        priority = []

    if priority:
        print(f"[wifi-autoconnect] Found {len(priority)} priority network(s) — trying first:")
        for sig, ssid in priority:
            print(f"  {sig:3d}%  {ssid}")
        for signal, ssid in priority:
            print(f"[wifi-autoconnect] Trying priority '{ssid}' (signal {signal}%) …")
            try:
                r = subprocess.run(
                    ["nmcli", "device", "wifi", "connect", ssid],
                    capture_output=True, text=True, timeout=30,
                )
                if r.returncode != 0:
                    err = (r.stderr or r.stdout).strip()
                    print(f"[wifi-autoconnect] Failed to connect: {err}")
                    continue
                print(f"[wifi-autoconnect] Joined '{ssid}', checking internet …")
                time.sleep(3)
                if has_internet():
                    print(f"[wifi-autoconnect] Internet confirmed on '{ssid}'")
                    sys.exit(0)
                else:
                    print(f"[wifi-autoconnect] '{ssid}' has no internet, trying next …")
                    disconnect_wifi()
            except subprocess.TimeoutExpired:
                print(f"[wifi-autoconnect] Timed out connecting to '{ssid}'")
            except Exception as e:
                print(f"[wifi-autoconnect] Error: {e}")

    # --- Fallback: open networks ---
    networks = []
    for attempt in range(3):
        try:
            networks = list_open_networks()
        except Exception as e:
            print(f"[wifi-autoconnect] Could not list networks: {e}")
            sys.exit(0)
        if networks:
            break
        if attempt < 2:
            print(f"[wifi-autoconnect] No open networks found, retrying ({attempt + 2}/3) …")
            time.sleep(5)

    if not networks:
        print("[wifi-autoconnect] No open networks found — skipping.")
        sys.exit(0)  # normal on password-protected or no-wifi environments

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
            if r.returncode != 0:
                err = (r.stderr or r.stdout).strip()
                print(f"[wifi-autoconnect] Failed to connect: {err}")
                continue
            print(f"[wifi-autoconnect] Joined '{ssid}', checking internet …")
            time.sleep(3)  # give DHCP a moment
            if has_internet():
                print(f"[wifi-autoconnect] Internet confirmed on '{ssid}'")
                sys.exit(0)
            else:
                print(f"[wifi-autoconnect] '{ssid}' has no internet (captive portal?), trying next …")
                disconnect_wifi()
        except subprocess.TimeoutExpired:
            print(f"[wifi-autoconnect] Timed out connecting to '{ssid}'")
        except Exception as e:
            print(f"[wifi-autoconnect] Error: {e}")

    print("[wifi-autoconnect] Could not connect to any open network — continuing boot anyway.")
    sys.exit(0)  # best-effort: don't block boot


if __name__ == "__main__":
    main()
