"""
Remepy — Sustained Phonation Demo Server
Runs on port 8766. Same Praat analysis as the main server.
Stores recordings to demo-samples/ prefix in the same S3 bucket.
"""

import os, base64, secrets, tempfile, json
from datetime import datetime, timezone

import numpy as np
import boto3
import parselmouth
from parselmouth.praat import call
from scipy.fft import fft, ifft
from fastapi import FastAPI, File, Form, UploadFile, Request, Response
from fastapi.responses import FileResponse

# ── Auth ──────────────────────────────────────────────────────────────────
_APP_PASSWORD = os.environ.get("APP_PASSWORD", "demo")


def _check_auth(auth_header: str) -> bool:
    if not auth_header.startswith("Basic "):
        return False
    try:
        decoded = base64.b64decode(auth_header[6:]).decode("utf-8", errors="replace")
        _, pw = decoded.split(":", 1)
        return secrets.compare_digest(pw.encode(), _APP_PASSWORD.encode())
    except Exception:
        return False


# ── S3 ────────────────────────────────────────────────────────────────────
_S3_BUCKET = "speech-metrics-storage"
_S3_PREFIX = "demo-samples/sustained-phonation/"
_s3 = None


def _get_s3():
    global _s3
    if _s3 is None:
        _s3 = boto3.client("s3")
    return _s3


# ── Praat helpers (same logic as main server) ─────────────────────────────
def safe(fn):
    try:
        v = fn()
        if v is None:
            return None
        f = float(v)
        return f if np.isfinite(f) else None
    except Exception:
        return None


def wav_bytes_to_sound(wav_bytes: bytes) -> parselmouth.Sound:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        tmp = f.name
    try:
        return parselmouth.Sound(tmp)
    finally:
        os.unlink(tmp)


def voiced_times_from_pitch(pitch_obj) -> np.ndarray:
    f0 = pitch_obj.selected_array["frequency"]
    ts = pitch_obj.ts()
    return ts[f0 > 0]


def make_voiced_checker(pitch_obj):
    f0 = pitch_obj.selected_array["frequency"]
    ts = pitch_obj.ts()
    voiced = f0 > 0

    def check(t: float) -> bool:
        idx = int(np.searchsorted(ts, t))
        idx = max(0, min(len(ts) - 1, idx))
        return bool(voiced[idx])

    return check


def compute_spl_voiced(sound: parselmouth.Sound, pitch_obj) -> tuple:
    try:
        intensity = sound.to_intensity(minimum_pitch=75.0, time_step=0.01)
        i_vals = intensity.values[0]
        noise_floor = float(np.percentile(i_vals, 20)) if len(i_vals) > 0 else None
        vtimes = voiced_times_from_pitch(pitch_obj)
        v_spls = np.array([intensity.get_value(t) for t in vtimes])
        v_spls = v_spls[np.isfinite(v_spls) & (v_spls > 0)]
        if len(v_spls) == 0:
            return None, None, noise_floor
        return float(np.mean(v_spls)), None, noise_floor
    except Exception:
        return None, None, None


def compute_cpp(sound: parselmouth.Sound, pitch_obj=None) -> float | None:
    samples = sound.values[0]
    sr = int(sound.sampling_frequency)
    frame_size = int(0.025 * sr)
    hop_size = int(0.010 * sr)
    q_min = max(1, int(sr / 600))
    q_max = min(int(sr / 75), frame_size // 2 - 1)
    if q_min >= q_max or frame_size > len(samples):
        return None
    is_voiced = make_voiced_checker(pitch_obj) if pitch_obj is not None else None
    cpp_vals = []
    for start in range(0, len(samples) - frame_size, hop_size):
        fc = (start + frame_size / 2) / sr
        if is_voiced is not None:
            if not is_voiced(fc):
                continue
        elif np.sqrt(np.mean(samples[start : start + frame_size] ** 2)) < 1e-6:
            continue
        frame = samples[start : start + frame_size]
        spec = np.abs(fft(frame * np.hanning(frame_size), n=frame_size))
        cep = np.abs(ifft(np.log(spec**2 + 1e-12))).real
        if q_max >= len(cep):
            continue
        region = cep[q_min : q_max + 1]
        peak_val = region.max()
        q = np.arange(q_min, q_max + 1, dtype=float)
        baseline = np.polyval(np.polyfit(q, region, 1), q[region.argmax()])
        cpp_val = (peak_val - baseline) * 10
        if np.isfinite(cpp_val) and cpp_val > 0:
            cpp_vals.append(cpp_val)
    return float(np.mean(cpp_vals)) if cpp_vals else None


def analyze_sustained_phonation(sound: parselmouth.Sound) -> dict:
    pitch_obj = None
    f0_mean = mpt = None
    try:
        pitch_obj = sound.to_pitch_ac(time_step=0.01, pitch_floor=75.0, pitch_ceiling=600.0)
        f0_arr = pitch_obj.selected_array["frequency"]
        f0_voiced = f0_arr[f0_arr > 0]
        f0_mean = float(np.mean(f0_voiced)) if len(f0_voiced) > 5 else None
        mpt = float(len(f0_voiced) * pitch_obj.time_step) if len(f0_voiced) > 0 else None
    except Exception:
        pass

    jitter_pct = shimmer_pct = hnr = None
    try:
        pp = call(sound, "To PointProcess (periodic, cc)", 75.0, 600.0)
        jitter_pct = safe(lambda: call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3) * 100)
        shimmer_pct = safe(
            lambda: call([sound, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6) * 100
        )
    except Exception:
        pass

    try:
        harm = call(sound, "To Harmonicity (cc)", 0.01, 75.0, 0.1, 1.0)
        hnr = safe(lambda: call(harm, "Get mean", 0, 0))
    except Exception:
        pass

    spl_mean, _, _ = (
        compute_spl_voiced(sound, pitch_obj) if pitch_obj is not None else (None, None, None)
    )
    cpp = compute_cpp(sound, pitch_obj)

    return {
        "mpt_s": mpt,
        "f0_mean_hz": f0_mean,
        "spl_mean_db": spl_mean,
        "jitter_pct": jitter_pct,
        "shimmer_pct": shimmer_pct,
        "hnr_db": hnr,
        "cpp_db": cpp,
        "duration_s": float(sound.duration),
    }


# ── App ───────────────────────────────────────────────────────────────────
app = FastAPI()


@app.middleware("http")
async def basic_auth(request: Request, call_next):
    if request.url.path == "/health":
        return await call_next(request)
    if _check_auth(request.headers.get("Authorization", "")):
        return await call_next(request)
    return Response(
        status_code=401,
        headers={"WWW-Authenticate": 'Basic realm="Remepy Voice Exercise"'},
        content="Unauthorized",
    )


@app.get("/")
def serve_index():
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "index.html")
    return FileResponse(path, media_type="text/html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    wav_bytes = await file.read()
    sound = wav_bytes_to_sound(wav_bytes)
    return analyze_sustained_phonation(sound)


@app.post("/store")
async def store(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    round_num: int = Form(...),
    metrics: str = Form("{}"),
):
    wav_bytes = await file.read()
    key = f"{_S3_PREFIX}{session_id}/round_{round_num}.wav"
    meta_key = f"{_S3_PREFIX}{session_id}/round_{round_num}_metrics.json"
    try:
        s3 = _get_s3()
        s3.put_object(Bucket=_S3_BUCKET, Key=key, Body=wav_bytes, ContentType="audio/wav")
        s3.put_object(
            Bucket=_S3_BUCKET,
            Key=meta_key,
            Body=json.dumps(
                {
                    "session_id": session_id,
                    "round": round_num,
                    "stored_at": datetime.now(timezone.utc).isoformat(),
                    "metrics": json.loads(metrics),
                }
            ).encode(),
            ContentType="application/json",
        )
    except Exception as exc:
        print(f"S3 store warning: {exc}")
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "key": key}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("DEMO_PORT", 8766))
    _dir        = os.path.dirname(os.path.realpath(__file__))
    ssl_keyfile  = os.path.join(_dir, "key.pem")
    ssl_certfile = os.path.join(_dir, "cert.pem")
    use_ssl      = os.path.exists(ssl_keyfile) and os.path.exists(ssl_certfile)
    proto        = "https" if use_ssl else "http"
    print(f"Starting Remepy demo at {proto}://0.0.0.0:{port}")
    uvicorn.run(
        app, host="0.0.0.0", port=port, log_level="info",
        ssl_keyfile=ssl_keyfile   if use_ssl else None,
        ssl_certfile=ssl_certfile if use_ssl else None,
    )
