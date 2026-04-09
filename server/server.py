"""
Speech metrics server — clinical-grade analysis via Praat (parselmouth) + Whisper.

Endpoints:
  POST /analyze/task1   — Sustained Phonation
  POST /analyze/task3   — Reading Passage
  GET  /health
"""

import os
import ssl
import tempfile
import traceback

import numpy as np

# Allow Whisper model download through corporate proxies with self-signed certs.
# This only affects the one-time model download; no external data is sent.
ssl._create_default_https_context = ssl._create_unverified_context  # noqa: SLF001
from fastapi import FastAPI, File, Form, UploadFile
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from scipy.fft import fft, ifft
from scipy.spatial import ConvexHull

import parselmouth
from parselmouth.praat import call

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
import subprocess
try:
    _GIT_VERSION = subprocess.check_output(
        ['git', 'rev-parse', '--short', 'HEAD'],
        cwd=os.path.dirname(__file__),
        stderr=subprocess.DEVNULL
    ).decode().strip()
except Exception:
    _GIT_VERSION = 'unknown'

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_STATIC_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "static"))
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

# Whisper loaded lazily on first Task-3 request to avoid delaying startup
_whisper_model = None


def get_whisper():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        print("Loading Whisper base model (one-time) …", flush=True)
        _whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
        print("Whisper ready.", flush=True)
    return _whisper_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def safe(fn):
    """Call fn(), return None on any error or non-finite result."""
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


def sound_to_tmpfile(sound: parselmouth.Sound) -> str:
    fd, tmp = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sound.save(tmp, "WAV")
    return tmp


# ---------------------------------------------------------------------------
# Voice gate helpers
# ---------------------------------------------------------------------------
def voiced_times_from_pitch(pitch_obj) -> np.ndarray:
    """Return array of frame-centre times where Praat detects voicing."""
    f0  = pitch_obj.selected_array["frequency"]
    ts  = pitch_obj.ts()
    return ts[f0 > 0]


def make_voiced_checker(pitch_obj):
    """
    Return a fast O(log N) checker: is_voiced(t) -> bool.
    Uses the nearest Praat pitch frame to decide voicing at time t.
    """
    f0 = pitch_obj.selected_array["frequency"]
    ts = pitch_obj.ts()
    voiced = f0 > 0

    def check(t: float) -> bool:
        idx = int(np.searchsorted(ts, t))
        idx = max(0, min(len(ts) - 1, idx))
        return bool(voiced[idx])

    return check


# ---------------------------------------------------------------------------
# SPL on voiced frames
# ---------------------------------------------------------------------------
def compute_spl_voiced(sound: parselmouth.Sound, pitch_obj) -> tuple:
    """
    Return (spl_mean_db, spl_p10_p90_range, noise_floor_db) using only
    voiced frames (from pitch_obj) for SPL and all frames for noise floor.
    """
    try:
        intensity   = sound.to_intensity(minimum_pitch=75.0, time_step=0.01)
        i_vals      = intensity.values[0]
        noise_floor = float(np.percentile(i_vals, 20)) if len(i_vals) > 0 else None

        vtimes = voiced_times_from_pitch(pitch_obj)
        v_spls = np.array([intensity.get_value(t) for t in vtimes])
        v_spls = v_spls[np.isfinite(v_spls) & (v_spls > 0)]

        if len(v_spls) == 0:
            return None, None, noise_floor

        spl_mean = float(np.mean(v_spls))
        spl_var  = float(np.percentile(v_spls, 90) - np.percentile(v_spls, 10))
        return spl_mean, spl_var, noise_floor
    except Exception:
        return None, None, None


# ---------------------------------------------------------------------------
# CPP (Cepstral Peak Prominence) via scipy — voiced frames only
# ---------------------------------------------------------------------------
def compute_cpp(sound: parselmouth.Sound, pitch_obj=None) -> float | None:
    """
    CPP averaged over voiced frames.
    If pitch_obj is supplied, only frames where Praat detects voicing are used.
    Falls back to RMS > 1e-6 gate when pitch_obj is None.
    """
    samples = sound.values[0]
    sr = int(sound.sampling_frequency)
    frame_size = int(0.025 * sr)  # 25 ms
    hop_size = int(0.010 * sr)    # 10 ms
    q_min = max(1, int(sr / 600))
    q_max = min(int(sr / 75), frame_size // 2 - 1)

    if q_min >= q_max or frame_size > len(samples):
        return None

    is_voiced = make_voiced_checker(pitch_obj) if pitch_obj is not None else None

    cpp_vals = []
    for start in range(0, len(samples) - frame_size, hop_size):
        frame_centre = (start + frame_size / 2) / sr

        if is_voiced is not None:
            if not is_voiced(frame_centre):
                continue
        else:
            if np.sqrt(np.mean(samples[start : start + frame_size] ** 2)) < 1e-6:
                continue

        frame = samples[start : start + frame_size]
        win      = np.hanning(frame_size)
        spec     = np.abs(fft(frame * win, n=frame_size))
        log_spec = np.log(spec ** 2 + 1e-12)
        cep      = np.abs(ifft(log_spec)).real

        if q_max >= len(cep):
            continue

        region    = cep[q_min : q_max + 1]
        peak_val  = region.max()
        q         = np.arange(q_min, q_max + 1, dtype=float)
        coeffs    = np.polyfit(q, region, 1)
        baseline  = np.polyval(coeffs, q[region.argmax()])
        cpp_val   = (peak_val - baseline) * 10
        if np.isfinite(cpp_val) and cpp_val > 0:
            cpp_vals.append(cpp_val)

    return float(np.mean(cpp_vals)) if cpp_vals else None


# ---------------------------------------------------------------------------
# Formants + VSA via Praat Burg LPC — voiced frames only
# ---------------------------------------------------------------------------
def compute_formants_vsa(sound: parselmouth.Sound, pitch_obj=None) -> dict:
    """
    F1, F2, F3 (Hz) and VSA (Hz²) via Praat Burg LPC.
    If pitch_obj is supplied, only voiced frame times are sampled.
    """
    try:
        formants = sound.to_formant_burg(
            time_step=0.01,
            max_number_of_formants=5,
            maximum_formant=5500.0,
            window_length=0.025,
            pre_emphasis_from=50.0,
        )
    except Exception:
        return {"f1": None, "f2": None, "f3": None, "vsa_hz2": None}

    # Sample only voiced times when pitch is available
    if pitch_obj is not None:
        sample_times = voiced_times_from_pitch(pitch_obj)
    else:
        sample_times = np.arange(0.05, sound.duration - 0.05, 0.01)

    f1_vals, f2_vals, f3_vals = [], [], []
    for t in sample_times:
        f1 = safe(lambda t=t: call(formants, "Get value at time", 1, t, "hertz", "linear"))
        f2 = safe(lambda t=t: call(formants, "Get value at time", 2, t, "hertz", "linear"))
        f3 = safe(lambda t=t: call(formants, "Get value at time", 3, t, "hertz", "linear"))
        if f1 and f2 and f1 > 200 and f2 > 500 and f2 > f1:
            f1_vals.append(f1)
            f2_vals.append(f2)
            if f3 and f3 > f2:
                f3_vals.append(f3)

    vsa = None
    if len(f1_vals) >= 5:
        try:
            hull = ConvexHull(np.column_stack([f1_vals, f2_vals]))
            vsa  = float(hull.volume)
        except Exception:
            pass

    return {
        "f1":     float(np.mean(f1_vals)) if f1_vals else None,
        "f2":     float(np.mean(f2_vals)) if f2_vals else None,
        "f3":     float(np.mean(f3_vals)) if f3_vals else None,
        "vsa_hz2": vsa,
    }


# ---------------------------------------------------------------------------
# Task 1 — Sustained Phonation
# ---------------------------------------------------------------------------
def analyze_task1(sound: parselmouth.Sound) -> dict:
    # ── Pitch (computed once; reused for voice-gating below) ───────────────
    pitch_obj = None
    f0_mean = mpt = None
    try:
        pitch_obj = sound.to_pitch_ac(
            time_step=0.01, pitch_floor=75.0, pitch_ceiling=600.0
        )
        f0_arr    = pitch_obj.selected_array["frequency"]
        f0_voiced = f0_arr[f0_arr > 0]
        f0_mean   = float(np.mean(f0_voiced)) if len(f0_voiced) > 5 else None
        mpt       = float(len(f0_voiced) * pitch_obj.time_step) if len(f0_voiced) > 0 else None
    except Exception:
        pass

    # ── Jitter & Shimmer via PointProcess ──────────────────────────────────
    jitter_pct = jitter_rap = shimmer_pct = shimmer_apq3 = shimmer_apq11 = None
    try:
        pp = call(sound, "To PointProcess (periodic, cc)", 75.0, 600.0)
        jitter_pct   = safe(lambda: call(pp, "Get jitter (local)",   0, 0, 0.0001, 0.02, 1.3) * 100)
        jitter_rap   = safe(lambda: call(pp, "Get jitter (rap)",     0, 0, 0.0001, 0.02, 1.3) * 100)
        shimmer_pct  = safe(lambda: call([sound, pp], "Get shimmer (local)",  0, 0, 0.0001, 0.02, 1.3, 1.6) * 100)
        shimmer_apq3 = safe(lambda: call([sound, pp], "Get shimmer (apq3)",   0, 0, 0.0001, 0.02, 1.3, 1.6) * 100)
        shimmer_apq11= safe(lambda: call([sound, pp], "Get shimmer (apq11)",  0, 0, 0.0001, 0.02, 1.3, 1.6) * 100)
    except Exception:
        pass

    # ── HNR (Harmonicity) ──────────────────────────────────────────────────
    hnr = None
    try:
        harm = call(sound, "To Harmonicity (cc)", 0.01, 75.0, 0.1, 1.0)
        hnr  = safe(lambda: call(harm, "Get mean", 0, 0))
    except Exception:
        pass

    # ── SPL on voiced frames only ───────────────────────────────────────────
    spl_mean, _, noise_floor = (
        compute_spl_voiced(sound, pitch_obj) if pitch_obj is not None
        else (None, None, None)
    )

    # ── CPP on voiced frames only ───────────────────────────────────────────
    cpp = compute_cpp(sound, pitch_obj)

    # ── Formants / VSA on voiced frames only ───────────────────────────────
    fmts = compute_formants_vsa(sound, pitch_obj)

    return {
        "f0_mean_hz":    f0_mean,
        "spl_mean_db":   spl_mean,
        "mpt_s":         mpt,
        "jitter_pct":    jitter_pct,
        "jitter_rap":    jitter_rap,
        "shimmer_pct":   shimmer_pct,
        "shimmer_apq3":  shimmer_apq3,
        "shimmer_apq11": shimmer_apq11,
        "cpp_db":        cpp,
        "hnr_db":        hnr,
        "noise_floor_db": noise_floor,
        "vsa_hz2":       fmts["vsa_hz2"],
    }


# ---------------------------------------------------------------------------
# Task 2 — Pitch Glides
# ---------------------------------------------------------------------------
def analyze_task2(sound: parselmouth.Sound) -> dict:
    pitch_obj = None
    f0_min = f0_max = f0_range = None
    try:
        pitch_obj = sound.to_pitch_ac(
            time_step=0.01, pitch_floor=75.0, pitch_ceiling=600.0
        )
        f0_arr    = pitch_obj.selected_array["frequency"]
        f0_voiced = f0_arr[f0_arr > 0]
        if len(f0_voiced) > 5:
            f0_min   = float(f0_voiced.min())
            f0_max   = float(f0_voiced.max())
            f0_range = float(12 * np.log2(f0_max / f0_min)) if f0_min > 0 else None
    except Exception:
        pass

    spl_mean, _, noise_floor = (
        compute_spl_voiced(sound, pitch_obj) if pitch_obj is not None
        else (None, None, None)
    )

    return {
        "f0_min_hz":      f0_min,
        "f0_max_hz":      f0_max,
        "f0_range_st":    f0_range,
        "spl_mean_db":    spl_mean,
        "noise_floor_db": noise_floor,
    }


# ---------------------------------------------------------------------------
# Task 3 — Reading Passage
# ---------------------------------------------------------------------------
def analyze_task3(sound: parselmouth.Sound) -> dict:
    duration = sound.duration

    # ── Pitch (computed once; reused for voice-gating below) ───────────────
    pitch_obj = None
    f0_mean = f0_std = prosody_st = None
    try:
        pitch_obj = sound.to_pitch_ac(
            time_step=0.01, pitch_floor=75.0, pitch_ceiling=600.0
        )
        f0_arr    = pitch_obj.selected_array["frequency"]
        f0_voiced = f0_arr[f0_arr > 0]
        if len(f0_voiced) > 5:
            f0_mean    = float(np.mean(f0_voiced))
            f0_std     = float(np.std(f0_voiced))
            f0_st      = 12 * np.log2(f0_voiced / np.mean(f0_voiced))
            prosody_st = float(np.std(f0_st))
    except Exception:
        pass

    # ── SPL on voiced frames only ───────────────────────────────────────────
    spl_mean, spl_var, noise_floor = (
        compute_spl_voiced(sound, pitch_obj) if pitch_obj is not None
        else (None, None, None)
    )

    # ── CPP on voiced frames only ───────────────────────────────────────────
    cpp = compute_cpp(sound, pitch_obj)

    # ── Formants / VSA on voiced frames only ───────────────────────────────
    fmts = compute_formants_vsa(sound, pitch_obj)

    # ── Speech rate + pauses via faster-whisper ────────────────────────────
    speech_rate_wpm = avg_pause_s = None
    transcript_text = ""
    try:
        model = get_whisper()
        tmp = sound_to_tmpfile(sound)
        try:
            segments_gen, _info = model.transcribe(tmp, beam_size=1)
            segments = list(segments_gen)
        finally:
            os.unlink(tmp)

        transcript_text = " ".join(seg.text for seg in segments).strip()
        if segments:
            all_words = transcript_text.split()
            speech_time = sum(seg.end - seg.start for seg in segments)
            if speech_time > 1.0 and len(all_words) > 0:
                speech_rate_wpm = float(len(all_words) / speech_time * 60)

            pauses = [
                segments[i].start - segments[i - 1].end
                for i in range(1, len(segments))
                if segments[i].start - segments[i - 1].end > 0.25
            ]
            avg_pause_s = float(np.mean(pauses)) if pauses else 0.0
    except Exception:
        traceback.print_exc()

    return {
        "f0_mean_hz":        f0_mean,
        "f0_std_hz":         f0_std,
        "cpp_db":            cpp,
        "spl_mean_db":       spl_mean,
        "spl_variability_db": spl_var,
        "speech_rate_wpm":   speech_rate_wpm,
        "prosody_st":        prosody_st,
        "noise_floor_db":    noise_floor,
        "avg_pause_s":       avg_pause_s,
        "vsa_hz2":           fmts["vsa_hz2"],
        "transcript":        transcript_text,
    }


# ---------------------------------------------------------------------------
# Articulation scoring (word accuracy + consonant precision)
# ---------------------------------------------------------------------------
def _levenshtein(a: list, b: list) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]; dp[0] = i
        for j in range(1, n + 1):
            tmp = dp[j]
            dp[j] = prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = tmp
    return dp[n]


def compute_articulation(transcript: str, reference: str) -> dict:
    """Word accuracy and consonant precision vs. reference passage."""
    norm  = lambda s: [w for w in __import__('re').sub(r'[^a-z\s]', '', s.lower()).split() if w]
    ref_w = norm(reference)
    hyp_w = norm(transcript)
    artic = max(0.0, min(100.0, (1 - _levenshtein(ref_w, hyp_w) / max(len(ref_w), 1)) * 100))

    vowels = set('aeiou')
    cons   = lambda s: [c for c in __import__('re').sub(r'[^a-z]', '', s.lower()) if c not in vowels]
    ref_c  = cons(reference)
    hyp_c  = cons(transcript)
    cp     = max(0.0, min(100.0, (1 - _levenshtein(ref_c, hyp_c) / max(len(ref_c), 1)) * 100)) if ref_c else None

    return {"articulation_pct": artic, "consonant_precision_pct": cp}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


def _html(name):
    path = os.path.join(os.path.dirname(__file__), "..", name)
    return FileResponse(os.path.realpath(path), media_type="text/html")


@app.get("/")
def serve_menu():
    return _html("index.html")


@app.get("/task1")
def serve_task1():
    return _html("task1.html")


@app.get("/task2")
def serve_task2():
    return _html("task2.html")


@app.get("/task3")
def serve_task3():
    return _html("task3.html")


@app.post("/analyze/task1")
async def endpoint_task1(file: UploadFile = File(...)):
    wav_bytes = await file.read()
    sound = wav_bytes_to_sound(wav_bytes)
    return {**analyze_task1(sound), "version": _GIT_VERSION}


@app.post("/analyze/task2")
async def endpoint_task2(file: UploadFile = File(...)):
    wav_bytes = await file.read()
    sound = wav_bytes_to_sound(wav_bytes)
    return {**analyze_task2(sound), "version": _GIT_VERSION}


@app.post("/analyze/task3")
async def endpoint_task3(file: UploadFile = File(...),
                         reference_text: Optional[str] = Form(None)):
    wav_bytes = await file.read()
    sound = wav_bytes_to_sound(wav_bytes)
    result = analyze_task3(sound)
    if reference_text and result.get("transcript"):
        result.update(compute_articulation(result["transcript"], reference_text))
    return {**result, "version": _GIT_VERSION}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    # Pre-load Whisper at startup so the first request isn't slow
    get_whisper()
    port = int(os.environ.get("PORT", 8765))
    _dir = os.path.dirname(__file__)
    ssl_keyfile  = os.path.join(_dir, "key.pem")
    ssl_certfile = os.path.join(_dir, "cert.pem")
    use_ssl = os.path.exists(ssl_keyfile) and os.path.exists(ssl_certfile)
    proto = "https" if use_ssl else "http"
    print(f"Starting speech metrics server on {proto}://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info",
                ssl_keyfile=ssl_keyfile  if use_ssl else None,
                ssl_certfile=ssl_certfile if use_ssl else None)
