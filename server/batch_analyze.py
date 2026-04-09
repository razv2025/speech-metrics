"""
Batch analysis using Praat (parselmouth) — clinical-grade metrics for all tasks.

Produces:
  ../results_amplitude.tsv   (Task 1 — Sustained Phonation)
  ../results_pitch.tsv       (Task 2 — Pitch Glides)
  ../results_reading.tsv     (Task 3 — Reading Passage)

Usage:
  python3 batch_analyze.py [--workers N]   (default: cpu_count)
"""

import os, sys, ssl, argparse, traceback
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np

# Allow corporate-proxy SSL for any lazy model fetches
ssl._create_default_https_context = ssl._create_unverified_context  # noqa: SLF001

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT    = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"
TASKS   = [
    ("amplitude", 1, ROOT / "results_amplitude.tsv"),
    ("pitch",     2, ROOT / "results_pitch.tsv"),
    ("reading",   3, ROOT / "results_reading.tsv"),
]

HEADERS = {
    1: ["filename","f0_mean_hz","spl_mean_db","mpt_s","jitter_pct","shimmer_pct",
        "cpp_db","hnr_db","noise_floor_db"],
    2: ["filename","f0_min_hz","f0_max_hz","f0_range_st","spl_mean_db","noise_floor_db"],
    3: ["filename","f0_mean_hz","f0_std_hz","cpp_db","spl_mean_db","spl_variability_db",
        "speech_rate_syl_min","prosody_st","noise_floor_db","avg_pause_s","vsa_khz2"],
}

# ---------------------------------------------------------------------------
# Import shared helpers from server.py
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from server import (safe, compute_cpp, compute_formants_vsa,
                    compute_spl_voiced, voiced_times_from_pitch)

import parselmouth
from parselmouth.praat import call
from scipy.fft import fft, ifft
from scipy.spatial import ConvexHull


# ---------------------------------------------------------------------------
# Per-task analysers
# ---------------------------------------------------------------------------
def _pitch(sound):
    """Compute Praat pitch object; returns None on failure."""
    try:
        return sound.to_pitch_ac(time_step=0.01, pitch_floor=75.0, pitch_ceiling=600.0)
    except Exception:
        return None


def analyze_t1(sound):
    """Task 1 — Sustained Phonation. All metrics gated on Praat-voiced frames."""
    pitch_obj = _pitch(sound)
    f0_mean = mpt = None
    if pitch_obj is not None:
        f0_arr    = pitch_obj.selected_array["frequency"]
        f0_voiced = f0_arr[f0_arr > 0]
        if len(f0_voiced) > 5:
            f0_mean = float(np.mean(f0_voiced))
            mpt     = float(len(f0_voiced) * pitch_obj.time_step)

    jitter = shimmer = None
    try:
        pp      = call(sound, "To PointProcess (periodic, cc)", 75.0, 600.0)
        jitter  = safe(lambda: call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3) * 100)
        shimmer = safe(lambda: call([sound, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6) * 100)
    except Exception:
        pass

    hnr = None
    try:
        harm = call(sound, "To Harmonicity (cc)", 0.01, 75.0, 0.1, 1.0)
        hnr  = safe(lambda: call(harm, "Get mean", 0, 0))
    except Exception:
        pass

    spl_mean, _, noise_floor = (
        compute_spl_voiced(sound, pitch_obj) if pitch_obj is not None else (None, None, None)
    )
    cpp  = compute_cpp(sound, pitch_obj)

    return {
        "f0_mean_hz":    f0_mean,
        "spl_mean_db":   spl_mean,
        "mpt_s":         mpt,
        "jitter_pct":    jitter,
        "shimmer_pct":   shimmer,
        "cpp_db":        cpp,
        "hnr_db":        hnr,
        "noise_floor_db": noise_floor,
    }


def analyze_t2(sound):
    """Task 2 — Pitch Glides. SPL gated on voiced frames."""
    pitch_obj = _pitch(sound)
    f0_min = f0_max = f0_range = None
    if pitch_obj is not None:
        f0_arr    = pitch_obj.selected_array["frequency"]
        f0_voiced = f0_arr[f0_arr > 0]
        if len(f0_voiced) > 5:
            f0_min   = float(f0_voiced.min())
            f0_max   = float(f0_voiced.max())
            f0_range = float(12 * np.log2(f0_max / f0_min)) if f0_min > 0 else None

    spl_mean, _, noise_floor = (
        compute_spl_voiced(sound, pitch_obj) if pitch_obj is not None else (None, None, None)
    )

    return {
        "f0_min_hz":     f0_min,
        "f0_max_hz":     f0_max,
        "f0_range_st":   f0_range,
        "spl_mean_db":   spl_mean,
        "noise_floor_db": noise_floor,
    }


def analyze_t3(sound):
    """Task 3 — Reading Passage. CPP, SPL, formants all gated on voiced frames."""
    duration  = sound.duration
    pitch_obj = _pitch(sound)

    f0_mean = f0_std = prosody_st = None
    if pitch_obj is not None:
        f0_arr    = pitch_obj.selected_array["frequency"]
        f0_voiced = f0_arr[f0_arr > 0]
        if len(f0_voiced) > 5:
            f0_mean    = float(np.mean(f0_voiced))
            f0_std     = float(np.std(f0_voiced))
            f0_st      = 12 * np.log2(f0_voiced / np.mean(f0_voiced))
            prosody_st = float(np.std(f0_st))

    spl_mean, spl_var, noise_floor = (
        compute_spl_voiced(sound, pitch_obj) if pitch_obj is not None else (None, None, None)
    )

    # Speech rate: intensity-peak counting on full intensity contour (syllable proxy)
    speech_rate = None
    try:
        intensity = sound.to_intensity(minimum_pitch=75.0, time_step=0.01)
        i_vals    = intensity.values[0]
        if len(i_vals) >= 5 and duration > 1.0:
            sm     = np.convolve(i_vals, np.ones(3) / 3, mode="same")
            thresh = np.mean(sm) - 5
            peaks  = sum(
                1 for i in range(1, len(sm) - 1)
                if sm[i] > thresh and sm[i] > sm[i - 1] and sm[i] > sm[i + 1]
            )
            speech_rate = float(peaks / duration * 60)
    except Exception:
        pass

    cpp = compute_cpp(sound, pitch_obj)

    avg_pause = None
    try:
        tg = call(sound, "To TextGrid (silences)", 100, 0, -25, 0.1, 0.05, "silent", "sounding")
        n_intervals = call(tg, "Get number of intervals", 1)
        pauses = []
        for i in range(1, n_intervals + 1):
            label = call(tg, "Get label of interval", 1, i)
            if label == "silent":
                t0  = call(tg, "Get start time of interval", 1, i)
                t1  = call(tg, "Get end time of interval",   1, i)
                dur = t1 - t0
                if 0.25 <= dur <= 5.0:
                    pauses.append(dur)
        avg_pause = float(np.mean(pauses)) if pauses else 0.0
    except Exception:
        pass

    fmts     = compute_formants_vsa(sound, pitch_obj)
    vsa_khz2 = (fmts["vsa_hz2"] / 1e6) if fmts["vsa_hz2"] is not None else None

    return {
        "f0_mean_hz":          f0_mean,
        "f0_std_hz":           f0_std,
        "cpp_db":              cpp,
        "spl_mean_db":         spl_mean,
        "spl_variability_db":  spl_var,
        "speech_rate_syl_min": speech_rate,
        "prosody_st":          prosody_st,
        "noise_floor_db":      noise_floor,
        "avg_pause_s":         avg_pause,
        "vsa_khz2":            vsa_khz2,
    }


ANALYSERS = {1: analyze_t1, 2: analyze_t2, 3: analyze_t3}


# ---------------------------------------------------------------------------
# Worker (runs in a subprocess)
# ---------------------------------------------------------------------------
def process_file(args):
    wav_path, task = args
    stem = Path(wav_path).stem
    try:
        sound    = parselmouth.Sound(str(wav_path))
        metrics  = ANALYSERS[task](sound)
        return stem, metrics, None
    except Exception as exc:
        return stem, None, str(exc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, cpu_count() - 1))
    args = parser.parse_args()

    for subfolder, task, out_path in TASKS:
        wav_dir = SAMPLES / subfolder
        wavs    = sorted(wav_dir.glob("*.wav"))
        n       = len(wavs)
        print(f"\nTask {task} ({subfolder}): {n} files — {args.workers} workers …")

        jobs    = [(str(w), task) for w in wavs]
        rows    = []
        errors  = 0

        with Pool(processes=args.workers) as pool:
            for i, (stem, metrics, err) in enumerate(pool.imap_unordered(process_file, jobs), 1):
                if i % 100 == 0 or i == n:
                    print(f"  {i}/{n}", flush=True)
                if err:
                    errors += 1
                    rows.append([stem] + [None] * (len(HEADERS[task]) - 1))
                else:
                    rows.append([stem] + [metrics.get(h) for h in HEADERS[task][1:]])

        # Sort by filename for deterministic output
        rows.sort(key=lambda r: r[0])

        def fmt(v):
            if v is None:
                return ""
            if isinstance(v, float):
                return f"{v:.6g}"
            return str(v)

        with open(out_path, "w") as f:
            f.write("\t".join(HEADERS[task]) + "\n")
            for row in rows:
                f.write("\t".join(fmt(v) for v in row) + "\n")

        print(f"  → {out_path}  (errors: {errors})")

    print("\nDone. Regenerating rankings …")
    os.system(f"node {ROOT / 'rankings.js'}")


if __name__ == "__main__":
    main()
