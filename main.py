
"""
Wake-Word Backend (FastAPI + WebSocket + OpenWakeWord)
- Robust Windows probing: auto-select a working mic configuration.
- Strategy:
  * Prefer WASAPI, then DirectSound, then MME
  * Try device candidates: env MIC_DEVICE_INDEX first, then common Realtek indices
  * Try (channels,dtype) in [(2,'float32'), (1,'float32'), (1,'int16')]
  * Adaptive blocksize (driver default). Capture @48k, mix to mono, downsample to 16k frames of 1280.
"""

import os
import time
import json
import queue
import asyncio
import threading
from typing import Dict, Optional, Tuple

import numpy as np
import sounddevice as sd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from openwakeword.model import Model as OWWModel

DEVICE_RATE = 48000
TARGET_RATE = 16000
DECIM = DEVICE_RATE // TARGET_RATE  # 3

ENV_IDX = os.getenv("MIC_DEVICE_INDEX")
ENV_IDX = int(ENV_IDX) if (ENV_IDX is not None and ENV_IDX.strip() != "") else None

OWW_INFERENCE = os.getenv("OWW_INFERENCE", "onnx")
OWW_MODEL_PATH = os.getenv("OWW_MODEL_PATH", "models/hei_buddy.onnx")
OWW_THRESHOLD = float(os.getenv("OWW_THRESHOLD", "0.60"))
TRIGGER_LEVEL = int(os.getenv("TRIGGER_LEVEL", "3"))
REFRACTORY_SEC = float(os.getenv("REFRACTORY_SEC", "1.0"))

app = FastAPI(title="WakeWord Backend (auto-probe mic)", version="1.3")
_event_queue: "queue.Queue[Dict]" = queue.Queue()

class Hub:
    def __init__(self) -> None:
        self._clients = []
        self._lock = asyncio.Lock()
    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._clients.append(ws)
    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            if ws in self._clients:
                self._clients.remove(ws)
    async def broadcast(self, data: Dict) -> None:
        async with self._lock:
            dead = []
            for ws in self._clients:
                try:
                    await ws.send_json(data)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                if ws in self._clients:
                    self._clients.remove(ws)

hub = Hub()

HOST_PRIORITY = ["WASAPI", "DIRECTSOUND", "MME"]
CHANNEL_DTYPE_TRIES = [(2, 'float32'), (1, 'float32'), (1, 'int16')]
CANDIDATE_IDXS = []
if ENV_IDX is not None:
    CANDIDATE_IDXS.append(ENV_IDX)
CANDIDATE_IDXS += [9, 1, 5, 14, 15, 0]


def _try_open(device: int, channels: int, dtype: str) -> bool:
    try:
        with sd.InputStream(channels=channels,
                            samplerate=DEVICE_RATE,
                            blocksize=None,
                            dtype=dtype,
                            device=device,
                            callback=lambda *a, **k: None):
            return True
    except Exception as e:
        return False


def probe_input() -> Tuple[int, int, str, int]:
    """Return (device_index, channels, dtype, hostapi_index). Raise on failure."""
    hostapis = sd.query_hostapis()
    # build list of (idx,name)
    ordered_hostapis = []
    for pri in HOST_PRIORITY:
        for i, h in enumerate(hostapis):
            n = (h.get('name') or '').upper()
            if pri in n and (i, n) not in ordered_hostapis:
                ordered_hostapis.append((i, n))
    # append the rest
    for i, h in enumerate(hostapis):
        n = (h.get('name') or '').upper()
        if (i, n) not in ordered_hostapis:
            ordered_hostapis.append((i, n))

    devices = sd.query_devices()
    present_idxs = {i for i in range(len(devices))}
    candidate_idxs = [i for i in CANDIDATE_IDXS if i in present_idxs]
    if not candidate_idxs:
        candidate_idxs = list(present_idxs)

    for hap_idx, hap_name in ordered_hostapis:
        try:
            sd.default.hostapi = hap_idx
            print(f"[Audio] Trying host API: {hap_name} (index={hap_idx})")
        except Exception:
            pass
        for dev in candidate_idxs:
            # skip outputs
            try:
                info = sd.query_devices(dev)
                if info.get('max_input_channels', 0) <= 0:
                    continue
            except Exception:
                continue
            for ch, dt in CHANNEL_DTYPE_TRIES:
                ok = _try_open(dev, ch, dt)
                print(f"[Audio] Probe device={dev} ch={ch} dtype={dt} -> {'OK' if ok else 'FAIL'}")
                if ok:
                    return dev, ch, dt, hap_idx
    raise RuntimeError("No working audio input configuration found")


def _wakeword_thread():
    # model
    custom_model = OWW_MODEL_PATH if (OWW_MODEL_PATH and os.path.isfile(OWW_MODEL_PATH)) else None
    if custom_model:
        print(f"[OWW] Using custom model: {custom_model}")
        oww = OWWModel(wakeword_models=[custom_model], inference_framework=OWW_INFERENCE)
    else:
        print("[OWW] No custom model found. Using OpenWakeWord built-in models.")
        oww = OWWModel(inference_framework=OWW_INFERENCE)
    print("[OWW] Models loaded:", list(oww.models.keys()))

    # probe audio
    try:
        dev, channels, dtype, hap = probe_input()
        print(f"[Audio] Selected device={dev}, channels={channels}, dtype={dtype}, hostapi_index={hap}")
    except Exception as e:
        print(f"[Audio] Probe failed: {e}")
        return

    last_trigger_ts = 0.0
    consecutive_hits = 0
    buf48 = np.zeros(0, dtype=np.float32)
    buf16 = np.zeros(0, dtype=np.float32)

    def flush_oww_frames():
        nonlocal buf16, last_trigger_ts, consecutive_hits
        while buf16.size >= 1280:
            frame16 = buf16[:1280]
            buf16 = buf16[1280:]
            pcm16 = (frame16 * 32767.0).astype(np.int16)
            scores: Dict[str, float] = oww.predict(pcm16)
            max_model: Optional[str] = None
            max_score = 0.0
            for name, score in scores.items():
                if score > max_score:
                    max_score = float(score)
                    max_model = name
            now = time.time()
            refractory_ok = (now - last_trigger_ts) >= REFRACTORY_SEC
            if max_score >= OWW_THRESHOLD and refractory_ok:
                consecutive_hits += 1
            else:
                if consecutive_hits > 0:
                    consecutive_hits -= 1
            if consecutive_hits >= TRIGGER_LEVEL and refractory_ok:
                last_trigger_ts = now
                consecutive_hits = 0
                _event_queue.put({
                    "type": "wakeword",
                    "model": max_model,
                    "score": max_score,
                    "ts": now
                })

    def audio_callback(indata, frames, time_info, status):
        nonlocal buf48, buf16
        if status:
            # print(status)
            pass
        if indata.ndim == 2:
            mono48 = indata.mean(axis=1).astype(np.float32)
        else:
            mono48 = indata.astype(np.float32)
        buf48 = np.concatenate((buf48, mono48))
        n = (buf48.size // DECIM) * DECIM
        if n:
            chunk48 = buf48[:n]
            buf48 = buf48[n:]
            converted16 = chunk48[::DECIM]
            buf16 = np.concatenate((buf16, converted16))
            flush_oww_frames()

    print(f"[Audio] Opening input stream @ {DEVICE_RATE} Hz (adaptive), device={dev}, channels={channels}, dtype={dtype}")
    try:
        with sd.InputStream(channels=channels,
                            samplerate=DEVICE_RATE,
                            blocksize=None,
                            dtype=dtype,
                            device=dev,
                            callback=audio_callback):
            print("[WakeWord] Listening for 'hei buddy' (auto-probed)…")
            while True:
                time.sleep(0.1)
    except Exception as e:
        print(f"[Audio] Failed to open input stream: {e}")


@app.on_event("startup")
async def on_startup():
    threading.Thread(target=_wakeword_thread, daemon=True).start()
    async def pump():
        loop = asyncio.get_running_loop()
        while True:
            ev = await loop.run_in_executor(None, _event_queue.get)
            await hub.broadcast(ev)
    asyncio.create_task(pump())


@app.websocket("/ws")
async def ws(ws: WebSocket):
    await hub.connect(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        await hub.disconnect(ws)


@app.get("/debug/trigger")
async def debug_trigger():
    payload = {"type": "wakeword", "debug": True, "ts": time.time()}
    await hub.broadcast(payload)
    return JSONResponse(payload)


@app.get("/health")
def health():
    return {"status": "ok"}
