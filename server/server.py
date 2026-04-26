"""
Speech metrics server — clinical-grade analysis via Praat (parselmouth) + Whisper.

Endpoints:
  POST /analyze/sustained-phonation   — Sustained Phonation
  POST /analyze/pitch-glides          — Pitch Glides
  POST /analyze/reading-passage       — Reading Passage
  GET  /health
"""

import os
import re
import ssl
import base64
import difflib
import secrets
import tempfile
import traceback
import json
import sqlite3
import uuid as _uuid_mod
import urllib.request
from datetime import datetime, timezone

import numpy as np

# Allow Whisper model download through corporate proxies with self-signed certs.
# This only affects the one-time model download; no external data is sent.
ssl._create_default_https_context = ssl._create_unverified_context  # noqa: SLF001
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import boto3
from botocore.exceptions import ClientError
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


@app.middleware("http")
async def no_cache_static(request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


# ---------------------------------------------------------------------------
# Basic Auth
# Password is loaded from APP_PASSWORD env var (never hardcoded here).
# ---------------------------------------------------------------------------
_APP_PASSWORD = os.environ.get("APP_PASSWORD", "")
if not _APP_PASSWORD:
    print("WARNING: APP_PASSWORD env var not set — all requests will be rejected")


def _check_auth(authorization: str) -> bool:
    if not authorization.startswith("Basic "):
        return False
    try:
        decoded = base64.b64decode(authorization[6:]).decode("utf-8", errors="replace")
        _, password = decoded.split(":", 1)
        return secrets.compare_digest(password.encode("utf-8"), _APP_PASSWORD.encode("utf-8"))
    except Exception:
        return False


@app.middleware("http")
async def basic_auth(request: Request, call_next):
    if request.url.path == "/health":
        return await call_next(request)
    if _APP_PASSWORD and _check_auth(request.headers.get("Authorization", "")):
        return await call_next(request)
    return Response(
        status_code=401,
        headers={"WWW-Authenticate": 'Basic realm="Speech Assessment"'},
        content="Unauthorized",
    )


# ---------------------------------------------------------------------------
# Publish infrastructure — S3 + SQLite
# ---------------------------------------------------------------------------
def _detect_aws_region() -> str:
    try:
        token = urllib.request.urlopen(
            urllib.request.Request(
                'http://169.254.169.254/latest/api/token', method='PUT',
                headers={'X-aws-ec2-metadata-token-ttl-seconds': '10'},
            ), timeout=1,
        ).read().decode()
        req = urllib.request.Request(
            'http://169.254.169.254/latest/meta-data/placement/region',
            headers={'X-aws-ec2-metadata-token': token},
        )
        return urllib.request.urlopen(req, timeout=1).read().decode()
    except Exception:
        return boto3.session.Session().region_name or 'us-east-1'


_AWS_REGION = _detect_aws_region()
_S3_BUCKET  = 'speech-metrics-storage'
_S3_PREFIX  = 'public-samples/'
_s3_client  = None


def _get_s3():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client('s3', region_name=_AWS_REGION)
    return _s3_client


def _init_s3():
    print(f'S3 configured: {_S3_BUCKET}/{_S3_PREFIX}')


_DB_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'published.db')


def _db():
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db():
    with _db() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS published (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                task           INTEGER NOT NULL,
                user_id        TEXT,
                username       TEXT,
                filename       TEXT,
                published_at   TEXT NOT NULL,
                s3_key         TEXT NOT NULL,
                metrics        TEXT,
                reference_text TEXT,
                audio_sr       INTEGER,
                duration_s     REAL,
                version        TEXT
            )
        ''')
        # Migrate existing table (column added after initial deploy)
        try:
            conn.execute('ALTER TABLE published ADD COLUMN username TEXT')
        except Exception:
            pass
        conn.commit()


# Initialise on import (runs whether started via __main__ or uvicorn)
try:
    _init_db()
    _init_s3()
except Exception as _pub_init_err:
    print(f'WARNING: publish infrastructure unavailable: {_pub_init_err}')

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
    jitter_pct = jitter_rap = shimmer_pct = shimmer_apq3 = shimmer_apq11 = shimmer_db = None
    try:
        pp = call(sound, "To PointProcess (periodic, cc)", 75.0, 600.0)
        jitter_pct   = safe(lambda: call(pp, "Get jitter (local)",   0, 0, 0.0001, 0.02, 1.3) * 100)
        jitter_rap   = safe(lambda: call(pp, "Get jitter (rap)",     0, 0, 0.0001, 0.02, 1.3) * 100)
        shimmer_pct  = safe(lambda: call([sound, pp], "Get shimmer (local)",     0, 0, 0.0001, 0.02, 1.3, 1.6) * 100)
        shimmer_apq3 = safe(lambda: call([sound, pp], "Get shimmer (apq3)",      0, 0, 0.0001, 0.02, 1.3, 1.6) * 100)
        shimmer_apq11= safe(lambda: call([sound, pp], "Get shimmer (apq11)",     0, 0, 0.0001, 0.02, 1.3, 1.6) * 100)
        shimmer_db   = safe(lambda: call([sound, pp], "Get shimmer (local, dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6))
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

    # ── Softest intensity (10th-percentile) — used as I_low proxy for DSI ──
    spl_min_db = None
    try:
        intens_obj = call(sound, "To Intensity", 100.0, 0.0, "yes")
        spl_min_db = safe(lambda: call(intens_obj, "Get quantile", 0, 0, 0.10))
    except Exception:
        pass

    # ── CPP on voiced frames only ───────────────────────────────────────────
    cpp = compute_cpp(sound, pitch_obj)

    # ── Noam's Metric ───────────────────────────────────────────────────────
    # jitter/shimmer thresholds are raw ratios (0.01 = 1 %, 0.038 = 3.8 %)
    noam_score = 0
    if cpp        is not None and cpp        < 12:  noam_score += 2   # high weight
    if hnr        is not None and hnr        < 20:  noam_score += 1
    if jitter_pct is not None and jitter_pct > 1.0: noam_score += 1   # ratio > 0.01
    if shimmer_pct is not None and shimmer_pct > 3.8: noam_score += 1 # ratio > 0.038
    noam_strained = noam_score >= 3

    # ── AVQI (Maryn & Weenink 2015) ─────────────────────────────────────────
    # Uses shimmer(local,dB), shimmer APQ3%, shimmer APQ11%, CPP, HNR
    avqi = None
    if all(v is not None for v in [shimmer_db, shimmer_apq3, shimmer_apq11, cpp, hnr]):
        avqi = round(float(
            3.528
            + 0.214  * shimmer_db
            - 0.221  * shimmer_apq3
            + 0.207  * shimmer_apq11
            - 0.213  * cpp
            - 0.259  * hnr
        ), 2)

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
        "avqi":          avqi,
        "shimmer_db":    shimmer_db,
        "spl_min_db":    spl_min_db,
        "noam_score":    noam_score,
        "noam_strained": noam_strained,
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
# ---------------------------------------------------------------------------
# Publish endpoints
# ---------------------------------------------------------------------------
_TASK_SLUGS = {'sustained-phonation': 1, 'pitch-glides': 2, 'reading-passage': 3}


@app.post('/publish/{task_slug}')
async def publish_entry(task_slug: str, file: UploadFile = File(...), metadata: str = Form(...)):
    task_n = _TASK_SLUGS.get(task_slug)
    if task_n is None:
        raise HTTPException(status_code=404, detail='Unknown task')
    data    = json.loads(metadata)
    audio   = await file.read()
    s3_key  = f'{_S3_PREFIX}{task_slug}/{_uuid_mod.uuid4()}.wav'
    _get_s3().put_object(Bucket=_S3_BUCKET, Key=s3_key, Body=audio, ContentType='audio/wav')
    now = datetime.now(timezone.utc).isoformat()
    with _db() as conn:
        cur = conn.execute(
            'INSERT INTO published '
            '(task,user_id,username,filename,published_at,s3_key,metrics,reference_text,audio_sr,duration_s,version) '
            'VALUES (?,?,?,?,?,?,?,?,?,?,?)',
            (task_n, data.get('user_id'), data.get('username'), data.get('filename'), now, s3_key,
             json.dumps(data.get('metrics', {})), data.get('reference_text'),
             data.get('audio_sr'), data.get('duration_s'), data.get('version')),
        )
        conn.commit()
        entry_id = cur.lastrowid
    return {'id': entry_id, 'published_at': now}


@app.get('/published/{task_slug}')
def get_published(task_slug: str):
    task_n = _TASK_SLUGS.get(task_slug)
    if task_n is None:
        raise HTTPException(status_code=404, detail='Unknown task')
    with _db() as conn:
        rows = conn.execute(
            'SELECT id,task,user_id,username,filename,published_at,metrics,audio_sr,duration_s,version '
            'FROM published WHERE task=? ORDER BY published_at DESC', (task_n,),
        ).fetchall()
    result = []
    for r in rows:
        e = dict(r)
        e['metrics'] = json.loads(e['metrics'] or '{}')
        result.append(e)
    return result


@app.delete('/published/{entry_id}')
def delete_published(entry_id: int):
    with _db() as conn:
        row = conn.execute('SELECT s3_key FROM published WHERE id=?', (entry_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail='Not found')
        try:
            _get_s3().delete_object(Bucket=_S3_BUCKET, Key=row['s3_key'])
        except Exception as exc:
            print(f'S3 delete warning: {exc}')
        conn.execute('DELETE FROM published WHERE id=?', (entry_id,))
        conn.commit()
    return {'ok': True}


@app.get('/audio/{entry_id}')
def stream_audio(entry_id: int):
    with _db() as conn:
        row = conn.execute('SELECT s3_key, filename FROM published WHERE id=?', (entry_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail='Not found')
    obj  = _get_s3().get_object(Bucket=_S3_BUCKET, Key=row['s3_key'])
    fname = (row['filename'] or 'recording').rstrip('.zip') + '.wav'
    return StreamingResponse(
        obj['Body'],
        media_type='audio/wav',
        headers={'Content-Disposition': f'attachment; filename="{fname}"'},
    )


# ---------------------------------------------------------------------------
# Verbal Fluency — word lists, matching, endpoints
# ---------------------------------------------------------------------------

_VF_WORD_LISTS = {
    'fruits': [
        'apple','apricot','avocado','banana','blackberry','blackcurrant','blueberry',
        'cantaloupe','cherry','clementine','coconut','cranberry','currant','damson',
        'date','dragon fruit','durian','elderberry','feijoa','fig','gooseberry',
        'grape','grapefruit','guava','honeydew','jackfruit','kiwi','kumquat',
        'lemon','lime','lychee','mandarin','mango','melon','mulberry','nectarine',
        'olive','orange','papaya','passion fruit','peach','pear','persimmon',
        'pineapple','plum','pomegranate','pomelo','quince','raspberry','rhubarb',
        'satsuma','soursop','star fruit','strawberry','tangerine','tomato',
        'ugli fruit','watermelon',
    ],
    'animals': [
        'alligator','ant','antelope','ape','armadillo','baboon','badger','bat',
        'bear','beaver','bee','bison','boar','buffalo','butterfly','camel','cat',
        'cheetah','chicken','chimpanzee','cobra','cow','crab','crocodile','deer',
        'dog','dolphin','donkey','duck','eagle','elephant','elk','falcon','ferret',
        'fish','flamingo','fox','frog','giraffe','gnu','goat','gorilla','grizzly bear',
        'hamster','hare','hawk','hedgehog','hippo','hippopotamus','horse','hyena',
        'ibex','ibis','jackal','jaguar','jellyfish','kangaroo','kestrel','koala',
        'lemur','leopard','lion','lizard','llama','lobster','lynx','manatee',
        'monkey','moose','mouse','narwhal','newt','ocelot','octopus','ostrich',
        'otter','owl','panda','panther','parrot','peacock','penguin','pig','pigeon',
        'platypus','polar bear','porcupine','puma','rabbit','raccoon','rat','rhino',
        'rhinoceros','salmon','seahorse','seal','shark','sheep','skunk','snow leopard',
        'snake','spider','squirrel','swan','tapir','tiger','toad','turkey','turtle',
        'whale','wolf','wombat','zebra','guinea pig','killer whale','mountain lion',
    ],
    'vegetables': [
        'artichoke','asparagus','aubergine','beetroot','bok choy','broccoli',
        'brussels sprout','cabbage','carrot','cauliflower','celery','celeriac',
        'chickpea','chilli','chicory','courgette','corn','cucumber','eggplant',
        'endive','fennel','garlic','ginger','green bean','kale','leek','lentil',
        'lettuce','mushroom','okra','onion','parsnip','pea','pepper','potato',
        'pumpkin','radish','rocket','shallot','spinach','spring onion','squash',
        'swede','sweet potato','sweetcorn','turnip','watercress','yam','zucchini',
    ],
    'colors': [
        'amber','beige','black','blue','bronze','brown','chartreuse','coral',
        'cream','crimson','cyan','emerald','fuchsia','gold','gray','grey','green',
        'indigo','jade','khaki','lavender','lilac','magenta','maroon','mauve',
        'navy','ochre','olive','orange','periwinkle','pink','purple','red','ruby',
        'rust','salmon','sapphire','scarlet','silver','tan','taupe','teal',
        'turquoise','violet','white','yellow',
    ],
    'furniture': [
        'armchair','bar stool','bean bag','bed','bed frame','bench','bookcase',
        'bookshelf','bunk bed','cabinet','chair','chest','chest of drawers',
        'closet','coffee table','cot','couch','cupboard','curtain','daybed',
        'desk','dining chair','dining table','drawer','dresser','end table',
        'filing cabinet','footstool','futon','hammock','lamp','loveseat',
        'mirror','nightstand','office chair','ottoman','recliner','rug',
        'settee','shelf','sideboard','sofa','standing desk','stool','table',
        'tv stand','wardrobe',
    ],
    'jobs': [
        'accountant','actor','actress','architect','artist','astronaut','baker',
        'banker','barber','biologist','builder','butcher','carpenter','cashier',
        'chef','chemist','cleaner','clerk','coach','consultant','courier',
        'decorator','dentist','designer','detective','diplomat','doctor','driver',
        'editor','electrician','engineer','farmer','firefighter','fireman',
        'gardener','geologist','historian','inspector','interpreter','inventor',
        'journalist','judge','lawyer','librarian','lifeguard','locksmith',
        'manager','mechanic','midwife','minister','model','musician','nurse',
        'optician','painter','paramedic','pharmacist','photographer','pilot',
        'plumber','poet','police','postman','programmer','psychologist',
        'receptionist','referee','sailor','salesman','scientist','sculptor',
        'secretary','social worker','software engineer','soldier','solicitor',
        'surgeon','surveyor','taxi driver','teacher','technician','therapist',
        'trainer','translator','undertaker','vet','veterinarian','waiter','writer',
    ],
}

_VF_FILLERS = {
    'um','uh','er','ah','hmm','like','the','a','an','and','or','so',
    'well','i','its','my','ok','okay','let','me','see','think','say',
    'maybe','also','too','very','really','just','some','any',
}

def _vf_norm(w: str) -> str:
    return re.sub(r"[^a-z]", "", w.lower())

def _vf_stem(w: str) -> str:
    for s in ('ies', 'es', 's'):
        if w.endswith(s) and len(w) - len(s) >= 3:
            return w[:-len(s)]
    return w

def _vf_match_word(word: str, word_list: list) -> str | None:
    """Return canonical list entry matching word, or None."""
    w = _vf_norm(word)
    if not w or len(w) < 2 or w in _VF_FILLERS:
        return None
    norm_list = [_vf_norm(x) for x in word_list]
    canon = dict(zip(norm_list, word_list))
    # Exact
    if w in canon:
        return canon[w]
    # Stem
    ws = _vf_stem(w)
    if ws in canon:
        return canon[ws]
    for nw in norm_list:
        if _vf_stem(nw) == ws:
            return canon[nw]
    # Fuzzy
    m = difflib.get_close_matches(w, norm_list, n=1, cutoff=0.82)
    if m:
        return canon[m[0]]
    return None

def _vf_match_transcript(transcript: str, category: str) -> dict:
    tokens = re.sub(r"[^\w\s]", "", transcript.lower()).split()
    tokens = [t for t in tokens if _vf_norm(t) not in _VF_FILLERS and len(_vf_norm(t)) >= 2]

    is_letter = category.startswith('letter:')
    letter    = category.split(':')[1] if is_letter else ''
    word_list = [] if is_letter else _VF_WORD_LISTS.get(category, [])

    matched, unmatched = [], []
    seen_matched = set()
    i = 0
    while i < len(tokens):
        canonical = None
        # Try bigram first (semantic categories only)
        if not is_letter and i + 1 < len(tokens):
            bigram = tokens[i] + ' ' + tokens[i + 1]
            canonical = _vf_match_word(bigram, word_list)
            if canonical:
                if canonical not in seen_matched:
                    matched.append(canonical)
                    seen_matched.add(canonical)
                i += 2
                continue
        # Unigram
        w = _vf_norm(tokens[i])
        if is_letter:
            if w.startswith(letter) and len(w) >= 2:
                if w not in seen_matched:
                    matched.append(tokens[i])
                    seen_matched.add(w)
            else:
                unmatched.append(tokens[i])
        else:
            canonical = _vf_match_word(tokens[i], word_list)
            if canonical:
                if canonical not in seen_matched:
                    matched.append(canonical)
                    seen_matched.add(canonical)
            else:
                unmatched.append(tokens[i])
        i += 1

    return {'matched': matched, 'unmatched': unmatched}


@app.get('/verbal-fluency')
def serve_verbal_fluency():
    return _html('verbal-fluency.html')


@app.post('/verbal-fluency/transcribe')
async def vf_transcribe(file: UploadFile = File(...), category: str = Form(...)):
    audio_bytes = await file.read()
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name
    try:
        segs, _ = get_whisper().transcribe(tmp_path, language='en', beam_size=1)
        transcript = ' '.join(s.text for s in segs).strip()
    finally:
        os.unlink(tmp_path)
    result = _vf_match_transcript(transcript, category)
    return {'transcript': transcript, **result}


@app.get("/hangman")
def serve_hangman():
    return _html("hangman.html")


@app.post("/hangman/guess")
async def hangman_guess(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name
    try:
        segs, _ = get_whisper().transcribe(tmp_path, language='en', beam_size=1)
        transcript = ' '.join(s.text for s in segs).strip()
    finally:
        os.unlink(tmp_path)
    return {'transcript': transcript}


# ---------------------------------------------------------------------------
# Sentence Completion — sentences, matching, endpoints
# ---------------------------------------------------------------------------

_SC_SENTENCES = [
    # Proverbs
    {'id':1,  'cat':'Proverb',       'text':"The ___ doesn't fall far from the tree.",              'answers':['apple']},
    {'id':2,  'cat':'Proverb',       'text':"A ___ in time saves nine.",                            'answers':['stitch']},
    {'id':3,  'cat':'Proverb',       'text':"Actions speak louder than ___.",                       'answers':['words']},
    {'id':4,  'cat':'Proverb',       'text':"All that glitters is not ___.",                        'answers':['gold']},
    {'id':5,  'cat':'Proverb',       'text':"Don't judge a book by its ___.",                       'answers':['cover']},
    {'id':6,  'cat':'Proverb',       'text':"Every ___ has a silver lining.",                       'answers':['cloud']},
    {'id':7,  'cat':'Proverb',       'text':"The early ___ catches the worm.",                      'answers':['bird']},
    {'id':8,  'cat':'Proverb',       'text':"Better late than ___.",                                'answers':['never']},
    {'id':9,  'cat':'Proverb',       'text':"A rolling ___ gathers no moss.",                       'answers':['stone']},
    {'id':10, 'cat':'Proverb',       'text':"Don't bite the ___ that feeds you.",                   'answers':['hand']},
    {'id':11, 'cat':'Proverb',       'text':"The ___ is always greener on the other side.",         'answers':['grass']},
    {'id':12, 'cat':'Proverb',       'text':"Where there's smoke, there's ___.",                    'answers':['fire']},
    {'id':13, 'cat':'Proverb',       'text':"Two wrongs don't make a ___.",                         'answers':['right']},
    {'id':14, 'cat':'Proverb',       'text':"The pen is mightier than the ___.",                    'answers':['sword']},
    {'id':15, 'cat':'Proverb',       'text':"Honesty is the best ___.",                             'answers':['policy']},
    {'id':16, 'cat':'Proverb',       'text':"Absence makes the heart grow ___.",                    'answers':['fonder']},
    {'id':17, 'cat':'Proverb',       'text':"Birds of a feather flock ___.",                        'answers':['together']},
    {'id':18, 'cat':'Proverb',       'text':"Don't count your chickens before they ___.",           'answers':['hatch']},
    {'id':19, 'cat':'Proverb',       'text':"A picture is worth a thousand ___.",                   'answers':['words']},
    {'id':20, 'cat':'Proverb',       'text':"Curiosity killed the ___.",                            'answers':['cat']},
    {'id':21, 'cat':'Proverb',       'text':"Every ___ has its day.",                               'answers':['dog']},
    {'id':22, 'cat':'Proverb',       'text':"Fortune favors the ___.",                              'answers':['brave', 'bold']},
    {'id':23, 'cat':'Proverb',       'text':"Laughter is the best ___.",                            'answers':['medicine']},
    {'id':24, 'cat':'Proverb',       'text':"Look before you ___.",                                 'answers':['leap']},
    {'id':25, 'cat':'Proverb',       'text':"Necessity is the mother of ___.",                      'answers':['invention']},
    {'id':26, 'cat':'Proverb',       'text':"Practice makes ___.",                                  'answers':['perfect']},
    {'id':27, 'cat':'Proverb',       'text':"Time flies when you're having ___.",                   'answers':['fun']},
    {'id':28, 'cat':'Proverb',       'text':"United we stand, divided we ___.",                     'answers':['fall']},
    {'id':29, 'cat':'Proverb',       'text':"When in Rome, do as the Romans ___.",                  'answers':['do']},
    {'id':30, 'cat':'Proverb',       'text':"You can't make an omelette without breaking ___.",     'answers':['eggs']},
    # Idioms
    {'id':31, 'cat':'Idiom',         'text':"Break a ___!",                                         'answers':['leg']},
    {'id':32, 'cat':'Idiom',         'text':"Bite the ___.",                                        'answers':['bullet']},
    {'id':33, 'cat':'Idiom',         'text':"Beat around the ___.",                                 'answers':['bush']},
    {'id':34, 'cat':'Idiom',         'text':"Spill the ___.",                                       'answers':['beans']},
    {'id':35, 'cat':'Idiom',         'text':"Hit the nail on the ___.",                             'answers':['head']},
    {'id':36, 'cat':'Idiom',         'text':"Kill two birds with one ___.",                         'answers':['stone']},
    {'id':37, 'cat':'Idiom',         'text':"Let the cat out of the ___.",                          'answers':['bag']},
    {'id':38, 'cat':'Idiom',         'text':"Once in a blue ___.",                                  'answers':['moon']},
    {'id':39, 'cat':'Idiom',         'text':"Burn the midnight ___.",                               'answers':['oil']},
    {'id':40, 'cat':'Idiom',         'text':"Cost an arm and a ___.",                               'answers':['leg']},
    {'id':41, 'cat':'Idiom',         'text':"Kick the ___.",                                        'answers':['bucket']},
    {'id':42, 'cat':'Idiom',         'text':"Miss the ___.",                                        'answers':['boat']},
    {'id':43, 'cat':'Idiom',         'text':"Pull someone's ___.",                                  'answers':['leg']},
    {'id':44, 'cat':'Idiom',         'text':"The tip of the ___.",                                  'answers':['iceberg']},
    {'id':45, 'cat':'Idiom',         'text':"Wrap your ___ around it.",                             'answers':['head']},
    {'id':46, 'cat':'Idiom',         'text':"Under the ___.",                                       'answers':['weather']},
    {'id':47, 'cat':'Idiom',         'text':"Jump on the ___.",                                     'answers':['bandwagon']},
    {'id':48, 'cat':'Idiom',         'text':"Hit the ___ running.",                                 'answers':['ground']},
    {'id':49, 'cat':'Idiom',         'text':"Steal someone's ___.",                                 'answers':['thunder']},
    {'id':50, 'cat':'Idiom',         'text':"On the tip of my ___.",                                'answers':['tongue']},
    {'id':51, 'cat':'Idiom',         'text':"The ___ is in your court.",                            'answers':['ball']},
    {'id':52, 'cat':'Idiom',         'text':"Add fuel to the ___.",                                 'answers':['fire']},
    {'id':53, 'cat':'Idiom',         'text':"Bite off more than you can ___.",                      'answers':['chew']},
    {'id':54, 'cat':'Idiom',         'text':"The ___ of the storm.",                                'answers':['eye']},
    {'id':55, 'cat':'Idiom',         'text':"Hit the ___ (go to sleep).",                           'answers':['sack', 'hay']},
    # Nursery Rhymes
    {'id':56, 'cat':'Nursery Rhyme', 'text':"Jack and Jill went up the ___.",                       'answers':['hill']},
    {'id':57, 'cat':'Nursery Rhyme', 'text':"Humpty Dumpty sat on a ___.",                          'answers':['wall']},
    {'id':58, 'cat':'Nursery Rhyme', 'text':"Twinkle, twinkle, little ___.",                        'answers':['star']},
    {'id':59, 'cat':'Nursery Rhyme', 'text':"Mary had a little ___.",                               'answers':['lamb']},
    {'id':60, 'cat':'Nursery Rhyme', 'text':"Baa, baa, black ___, have you any wool?",              'answers':['sheep']},
    {'id':61, 'cat':'Nursery Rhyme', 'text':"Little Bo Peep has lost her ___.",                     'answers':['sheep']},
    {'id':62, 'cat':'Nursery Rhyme', 'text':"Old MacDonald had a ___.",                             'answers':['farm']},
    {'id':63, 'cat':'Nursery Rhyme', 'text':"Row, row, row your ___.",                              'answers':['boat']},
    {'id':64, 'cat':'Nursery Rhyme', 'text':"Hickory dickory dock, the mouse ran up the ___.",      'answers':['clock']},
    {'id':65, 'cat':'Nursery Rhyme', 'text':"London Bridge is falling ___.",                        'answers':['down']},
    {'id':66, 'cat':'Nursery Rhyme', 'text':"Ring around the ___.",                                 'answers':['rosie', 'rosy']},
    {'id':67, 'cat':'Nursery Rhyme', 'text':"Jack be nimble, Jack be ___.",                         'answers':['quick']},
    {'id':68, 'cat':'Nursery Rhyme', 'text':"Little Miss Muffet sat on a ___.",                     'answers':['tuffet']},
    {'id':69, 'cat':'Nursery Rhyme', 'text':"Georgie Porgie, pudding and ___.",                     'answers':['pie']},
    {'id':70, 'cat':'Nursery Rhyme', 'text':"This little piggy went to ___.",                       'answers':['market']},
    {'id':71, 'cat':'Nursery Rhyme', 'text':"The itsy bitsy spider climbed up the water ___.",      'answers':['spout']},
    {'id':72, 'cat':'Nursery Rhyme', 'text':"The wheels on the bus go round and ___.",              'answers':['round']},
    {'id':73, 'cat':'Nursery Rhyme', 'text':"Head, shoulders, knees and ___.",                      'answers':['toes']},
    {'id':74, 'cat':'Nursery Rhyme', 'text':"One, two, buckle my ___.",                             'answers':['shoe']},
    {'id':75, 'cat':'Nursery Rhyme', 'text':"Three blind ___.",                                     'answers':['mice']},
    {'id':76, 'cat':'Nursery Rhyme', 'text':"I'm a little teapot, short and ___.",                  'answers':['stout']},
    {'id':77, 'cat':'Nursery Rhyme', 'text':"Five little ducks went out one ___.",                  'answers':['day']},
    {'id':78, 'cat':'Nursery Rhyme', 'text':"Pat-a-cake, pat-a-cake, baker's ___.",                 'answers':['man']},
    {'id':79, 'cat':'Nursery Rhyme', 'text':"Rock-a-bye baby, on the treetop, when the wind blows the cradle will ___.", 'answers':['rock']},
    {'id':80, 'cat':'Nursery Rhyme', 'text':"Pease porridge hot, pease porridge ___.",              'answers':['cold']},
]


def _sc_check_answer(transcript: str, answers: list) -> bool:
    raw = re.sub(r"[^\w\s']", ' ', transcript.lower()).strip()
    tokens = raw.split()
    for answer in answers:
        norm_ans = answer.lower().strip()
        ans_toks = norm_ans.split()
        if len(ans_toks) == 1:
            for tok in tokens:
                if tok == norm_ans:
                    return True
                if difflib.SequenceMatcher(None, tok, norm_ans).ratio() >= 0.80:
                    return True
        else:
            for i in range(len(tokens) - len(ans_toks) + 1):
                phrase = ' '.join(tokens[i:i + len(ans_toks)])
                if difflib.SequenceMatcher(None, phrase, norm_ans).ratio() >= 0.85:
                    return True
    return False


@app.get('/sentence-completion')
def serve_sentence_completion():
    return _html('sentence-completion.html')


@app.get('/sentence-completion/sentences')
def sc_get_sentences():
    return [{'id': s['id'], 'cat': s['cat'], 'text': s['text']} for s in _SC_SENTENCES]


@app.post('/sentence-completion/check')
async def sc_check(file: UploadFile = File(...), sentence_id: int = Form(...)):
    audio_bytes = await file.read()
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name
    try:
        segs, _ = get_whisper().transcribe(tmp_path, language='en', beam_size=1)
        transcript = ' '.join(s.text for s in segs).strip()
    finally:
        os.unlink(tmp_path)
    sentence = next((s for s in _SC_SENTENCES if s['id'] == sentence_id), None)
    if not sentence:
        raise HTTPException(status_code=404, detail='Unknown sentence')
    correct = _sc_check_answer(transcript, sentence['answers'])
    return {'transcript': transcript, 'correct': correct, 'canonical': sentence['answers'][0]}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/me")
def get_me(request: Request):
    auth = request.headers.get("Authorization", "")
    username = ""
    if auth.startswith("Basic "):
        try:
            decoded = base64.b64decode(auth[6:]).decode("utf-8", errors="replace")
            username = decoded.split(":", 1)[0]
        except Exception:
            pass
    # Sanitize: printable ASCII only, max 40 chars, strip whitespace
    username = ''.join(c for c in username if 32 <= ord(c) < 127)[:40].strip()
    return {"username": username or "User"}


def _html(name):
    path = os.path.join(os.path.dirname(__file__), "..", name)
    return FileResponse(os.path.realpath(path), media_type="text/html")


@app.get("/")
def serve_menu():
    return _html("index.html")


@app.get("/sustained-phonation")
def serve_sustained_phonation():
    return _html("sustained-phonation.html")


@app.get("/pitch-glides")
def serve_pitch_glides():
    return _html("pitch-glides.html")


@app.get("/reading-passage")
def serve_reading_passage():
    return _html("reading-passage.html")


@app.post("/analyze/sustained-phonation")
async def endpoint_sustained_phonation(file: UploadFile = File(...)):
    wav_bytes = await file.read()
    sound = wav_bytes_to_sound(wav_bytes)
    return {**analyze_task1(sound), "version": _GIT_VERSION}


@app.post("/analyze/pitch-glides")
async def endpoint_pitch_glides(file: UploadFile = File(...)):
    wav_bytes = await file.read()
    sound = wav_bytes_to_sound(wav_bytes)
    return {**analyze_task2(sound), "version": _GIT_VERSION}


@app.post("/analyze/reading-passage")
async def endpoint_reading_passage(file: UploadFile = File(...),
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
