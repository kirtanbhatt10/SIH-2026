import sounddevice as sd

hostapis = sd.query_hostapis()
print("=== HOST APIs ===")
for i, api in enumerate(hostapis):
    print(f"API {i}: {api['name']}")

print("\n=== ALL SOUND DEVICES ===")
for i, dev in enumerate(sd.query_devices()):
    api_name = hostapis[dev['hostapi']]['name']
    print(f"[{i:2d}] {dev['name']:<50s} | HostAPI: {api_name:<18s} | In: {dev['max_input_channels']} | Out: {dev['max_output_channels']} | Default SR: {dev['default_samplerate']} Hz")

print(f"\nsd.default.device: {sd.default.device}")
