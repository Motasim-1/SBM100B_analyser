
import sounddevice as sd

def list_input_devices():
    out = []
    for idx, dev in enumerate(sd.query_devices()):
        if int(dev.get("max_input_channels", 0)) > 0:
            out.append({
                "index": idx,
                "name": str(dev.get("name", f"Device {idx}")),
                "channels": int(dev.get("max_input_channels", 0)),
                "default_samplerate": float(dev.get("default_samplerate", 0)),
            })
    return out
