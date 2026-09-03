import collections
import datetime
import os
import time

import numpy as np

from backend.core.config import BIT_DURATION, FREQ_0, FREQ_1, FORENSIC_CAPTURE_DIR, SAMPLE_RATE
from backend.services.payload_service import decode_signal_to_text, save_signal_to_wav


def list_audio_devices() -> None:
    import sounddevice as sd

    print("\n[+] Available Audio Input Devices:")
    devices = sd.query_devices()
    default_in = sd.default.device[0]
    for idx, d in enumerate(devices):
        if d.get("max_input_channels", 0) > 0:
            prefix = " [*] " if idx == default_in else " [ ] "
            print(
                f"{prefix}Device #{idx}: {d.get('name')} "
                f"(Max Channels: {d.get('max_input_channels')}, "
                f"Default SR: {d.get('default_samplerate')}Hz)"
            )
    print()


def run_listener(
    device_id=None,
    buffer_sec=5.0,
    confidence_threshold=0.35,
    save_captures=True,
    capture_dir: str | None = None,
):
    import sounddevice as sd

    capture_dir = capture_dir or FORENSIC_CAPTURE_DIR

    print("=" * 70)
    print(" ULTRASONIC COVERT CHANNEL ACOUSTIC RECEIVER & FORENSIC MONITOR")
    print(f" Target Frequencies : Freq 0 = {FREQ_0} Hz | Freq 1 = {FREQ_1} Hz")
    print(f" Sample Rate        : {SAMPLE_RATE} Hz")
    print(f" Audio Device ID     : {device_id if device_id is not None else 'Default Microphone'}")
    print("=" * 70)
    print("[*] Listening on microphone for ultrasonic air-gap transmissions...")
    print("[*] Press Ctrl+C to stop.\n")

    if save_captures:
        os.makedirs(capture_dir, exist_ok=True)

    buffer_len = int(SAMPLE_RATE * buffer_sec)
    audio_ringbuffer = collections.deque(maxlen=buffer_len)
    last_decoded_text = ""
    last_decode_time = 0.0
    block_size = int(SAMPLE_RATE * 0.25)

    def audio_callback(indata, frames, time_info, status):
        if status:
            pass
        audio_ringbuffer.extend(indata[:, 0].copy())

    try:
        with sd.InputStream(
            channels=1,
            samplerate=SAMPLE_RATE,
            blocksize=block_size,
            device=device_id,
            callback=audio_callback,
        ):
            capture_count = 0
            while True:
                time.sleep(0.15)
                if len(audio_ringbuffer) < int(SAMPLE_RATE * 1.5):
                    continue

                signal_snapshot = np.array(audio_ringbuffer, dtype=np.float32)
                result = decode_signal_to_text(signal_snapshot, sample_rate=SAMPLE_RATE)

                now = time.time()
                if result.success and result.text and result.confidence >= confidence_threshold:
                    if result.text == last_decoded_text and (now - last_decode_time) < 4.0:
                        continue

                    last_decoded_text = result.text
                    last_decode_time = now
                    capture_count += 1
                    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    print("\n" + "!" * 70)
                    print(f" [FORENSIC ALERT #{capture_count}] ULTRASONIC TRANSMISSION DETECTED!")
                    print(f" Timestamp     : {timestamp_str}")
                    print(f" Confidence    : {result.confidence:.2%}")
                    print(f" Bits Decoded  : {result.bit_count} bits")
                    print(f" RECOVERED PAYLOAD : >>> {result.text} <<<")

                    if save_captures:
                        cap_file = os.path.join(capture_dir, f"capture_{int(now)}_{capture_count}.wav")
                        save_signal_to_wav(cap_file, signal_snapshot, SAMPLE_RATE)
                        print(f" Audio Saved   : {cap_file}")
                    print("!" * 70 + "\n")

    except KeyboardInterrupt:
        print("\n[*] Stopping ultrasonic acoustic listener.")
    except Exception as e:
        print(f"\n[!] Listener error: {e}")
