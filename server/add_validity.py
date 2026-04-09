"""
Post-process results TSVs to add a 'valid' column.

A file is marked valid=0 if it appears to be silence, noise, or non-human:
  - SNR < 5 dB  (noise_floor within 5 dB of spl_mean — noise-dominated)
    (Praat returns -300 for noise_floor when no quiet frames exist → treated as clean)
  - F0 not detected (null f0_mean_hz / f0_min_hz)
  - Task 1: MPT < 0.5 s
  - Task 2: F0 range < 0.5 semitones  (no real glide)

Re-writes each TSV in-place with the extra column appended.
"""

import csv, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def fv(row, key):
    v = row.get(key, "")
    try:
        return float(v) if v else None
    except ValueError:
        return None


def compute_snr(row):
    """SNR = spl_mean - noise_floor.  Praat returns -300 for 'no quiet frames' → clean."""
    spl   = fv(row, "spl_mean_db")
    noise = fv(row, "noise_floor_db")
    if spl is None:
        return None
    if noise is None or noise < -100:   # -300 sentinel → treat as clean
        return 999.0
    return spl - noise


# ── Task 1 — Sustained Phonation ────────────────────────────────────────────
# Validity is based on voice quality indicators, NOT SNR:
# sustained phonation has no silence frames so SNR is meaningless here.
#   • F0 detected                        → periodic voice present
#   • MPT ≥ 0.5 s                        → actual phonation, not a click
#   • HNR ≥ 0 dB                         → more harmonic than noisy
#   • CPP ≥ 1.0 dB                       → clear cepstral peak (periodic source)
#   • Jitter < 5 %                       → Praat's limit; beyond this = noise artifact
#   • Shimmer < 15 %                     → same — turbulent airflow / breathing noise
def is_valid_t1(row):
    if fv(row, "f0_mean_hz") is None:
        return False
    if (fv(row, "mpt_s") or 0) < 0.5:
        return False
    if (fv(row, "hnr_db") or -99) < 0:
        return False
    if (fv(row, "cpp_db") or 0) < 1.0:
        return False
    if (fv(row, "jitter_pct") or 999) >= 5.0:
        return False
    if (fv(row, "shimmer_pct") or 999) >= 15.0:
        return False
    return True


# ── Task 2 — Pitch Glides ────────────────────────────────────────────────────
# SNR is equally unreliable here (continuous voice, no silences).
#   • F0 detected                        → periodic voice present
#   • F0 range ≥ 2 ST                    → actual glide performed (not monotone noise)
def is_valid_t2(row):
    if fv(row, "f0_min_hz") is None:
        return False
    if (fv(row, "f0_range_st") or 0) < 2.0:
        return False
    return True


# ── Task 3 — Reading Passage ─────────────────────────────────────────────────
# Reading has natural pauses so SNR is reliable.
#   • SNR ≥ 5 dB                         → speech audible above background noise
#   • F0 detected                        → at least some voiced speech
def is_valid_t3(row):
    snr = compute_snr(row)
    if snr is None or snr < 5.0:
        return False
    if fv(row, "f0_mean_hz") is None:
        return False
    return True


VALIDATORS = {
    "results_amplitude.tsv": is_valid_t1,
    "results_pitch.tsv":     is_valid_t2,
    "results_reading.tsv":   is_valid_t3,
}


def process(tsv_name, validator):
    path = ROOT / tsv_name
    rows = list(csv.DictReader(open(path), delimiter="\t"))
    headers = list(rows[0].keys())
    if "valid" not in headers:
        headers.append("valid")

    total, n_valid = 0, 0
    out_rows = []
    for row in rows:
        v = 1 if validator(row) else 0
        row["valid"] = v
        out_rows.append(row)
        total  += 1
        n_valid += v

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(out_rows)

    n_invalid = total - n_valid
    print(f"{tsv_name}: {n_valid}/{total} valid  ({n_invalid} flagged as invalid)")


if __name__ == "__main__":
    for name, fn in VALIDATORS.items():
        process(name, fn)
    print("Done.")
