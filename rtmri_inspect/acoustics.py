"""Spectrogram, formant tracking and a simple tube model.

Formant tracker (replaces ``fmt_track.m``)
------------------------------------------
Same ingredients as the MATLAB version (pre-emphasis, windowed LPC, pole angles ->
frequencies, pole radii -> bandwidths) but candidate selection is a Viterbi search
over the whole segment instead of a greedy frame-by-frame pick, and silent frames
are masked out instead of producing junk formants.
"""
from __future__ import annotations

from fractions import Fraction
from itertools import combinations

import numpy as np
from scipy import signal
from scipy.linalg import solve_toeplitz

EPS = 1e-10


# --------------------------------------------------------------------------- framing
def _frames(y: np.ndarray, centers: np.ndarray, n: int) -> np.ndarray:
    """Return (len(centers), n) frames of ``y`` centred on sample indices ``centers`` (zero padded)."""
    half = n // 2
    pad = np.pad(y, (n, n))
    starts = centers.astype(np.int64) - half + n
    idx = starts[:, None] + np.arange(n)[None, :]
    return pad[idx]


def preemphasis(y: np.ndarray, coef: float = 0.97) -> np.ndarray:
    return np.r_[y[:1], y[1:] - coef * y[:-1]]


# --------------------------------------------------------------------------- spectrogram
def spectrogram_db(y: np.ndarray, fs: int, t0: float, t1: float, win_ms: float = 20.0, step_ms: float = 5.0,
                   fmax: float = 5000.0, preemph: bool = True, nfft: int | None = None, off: float = 0.0):
    """Magnitude spectrogram in dB (re. full-scale sine amplitude) for audio times t0..t1.

    ``off`` is the A/V offset: timeline time = audio time + off.
    Returns (dB[freq, time], freqs_hz, times_timeline).
    """
    if preemph:
        y = preemphasis(y)
    n = int(round(win_ms * fs / 1000.0))
    n += n % 2
    step = step_ms / 1000.0
    times = np.arange(t0, t1 + 1e-9, step)
    centers = np.round((times - off) * fs)
    w = signal.get_window("hann", n, fftbins=False)
    if nfft is None:
        nfft = 1 << int(np.ceil(np.log2(max(n * 2, 256))))
    fr = _frames(y, centers, n) * w[None, :]
    spec = np.abs(np.fft.rfft(fr, nfft, axis=1)) / (w.sum() / 2.0)
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    keep = freqs <= min(fmax, fs / 2)
    db = 20 * np.log10(spec[:, keep] + EPS)
    return db.T.astype(np.float32), freqs[keep], times


def spectrogram_ref_db(y: np.ndarray, fs: int, win_ms: float = 20.0, preemph: bool = True) -> float:
    """A stable top-of-scale level: the 99.9th percentile of dB values over the whole file."""
    dur = len(y) / fs
    db, _, _ = spectrogram_db(y, fs, 0.0, dur, win_ms=win_ms, step_ms=20.0, fmax=fs / 2, preemph=preemph)
    return float(np.percentile(db, 99.9))


def db_to_gray(db: np.ndarray, ref_db: float, dyn_range: float) -> np.ndarray:
    """Map dB to uint8 (255 = loudest). Row 0 is the lowest frequency."""
    x = np.clip((db - (ref_db - dyn_range)) / dyn_range, 0, 1)
    return (x * 255).astype(np.uint8)


# --------------------------------------------------------------------------- LPC
def _lpc(frame: np.ndarray, order: int) -> np.ndarray | None:
    """Autocorrelation-method LPC coefficients (a[0]=1)."""
    r = np.correlate(frame, frame, mode="full")[len(frame) - 1: len(frame) + order]
    if r[0] <= 1e-12:
        return None
    r = r.copy()
    r[0] *= 1.0 + 1e-9  # tiny white-noise correction for stability
    try:
        a = solve_toeplitz(r[:order], r[1: order + 1])
    except Exception:
        return None
    return np.r_[1.0, -a]


def _candidates(a: np.ndarray, fs: float, fmin: float, fmax: float, max_bw: float):
    rts = np.roots(a)
    rts = rts[np.imag(rts) > 1e-6]
    f = np.angle(rts) * fs / (2 * np.pi)
    bw = -np.log(np.maximum(np.abs(rts), 1e-6)) * fs / np.pi
    ok = (f > fmin) & (f < fmax) & (bw < max_bw) & (bw > 0)
    f, bw = f[ok], bw[ok]
    o = np.argsort(f)
    return f[o], bw[o]


def track_formants(y: np.ndarray, fs: int, t0: float, t1: float, n_formants: int = 4, max_freq: float = 5500.0,
                   win_ms: float = 25.0, step_ms: float = 5.0, order: int | None = None, max_bw: float = 600.0,
                   preemph_coef: float = 0.97, min_level_db: float = -40.0, speaker_scale: float = 1.0,
                   off: float = 0.0, keep_candidates: int = 7):
    """Track ``n_formants`` formants between timeline times t0 and t1.

    Returns dict(t=[T], F=[T, n], B=[T, n], level=[T]) with NaN where silent / unresolved.
    """
    # resample so that Nyquist ~ max_freq (+ headroom); fewer poles needed, fewer spurious ones
    target = 2.0 * max_freq
    if fs > target * 1.05:
        fr = Fraction(target / fs).limit_denominator(100)
        ys = signal.resample_poly(y, fr.numerator, fr.denominator)
        fs2 = fs * fr.numerator / fr.denominator
    else:
        ys, fs2 = y, float(fs)
    ys = preemphasis(ys, preemph_coef)
    if order is None:
        order = 2 * n_formants + 2
    n = int(round(win_ms * fs2 / 1000.0))
    times = np.arange(t0, t1 + 1e-9, step_ms / 1000.0)
    centers = np.round((times - off) * fs2)
    w = np.hamming(n)
    fr_all = _frames(ys, centers, n) * w[None, :]
    # silence gate: frame level (original-rate signal) relative to loud speech in the whole file
    lvl, ref = _frame_level_db(y, fs, times - off, win_ms)
    valid = lvl > (ref + min_level_db)

    nom = np.array([500, 1500, 2500, 3500, 4500, 5500][:n_formants], float) * speaker_scale
    T = len(times)
    F = np.full((T, n_formants), np.nan)
    B = np.full((T, n_formants), np.nan)

    cands: list[tuple[np.ndarray, np.ndarray] | None] = []
    for k in range(T):
        if not valid[k]:
            cands.append(None)
            continue
        a = _lpc(fr_all[k], order)
        if a is None:
            cands.append(None)
            continue
        f, bw = _candidates(a, fs2, 90.0, max_freq, max_bw)
        if len(f) > keep_candidates:  # keep the sharpest resonances
            sel = np.sort(np.argsort(bw)[:keep_candidates])
            f, bw = f[sel], bw[sel]
        cands.append((f, bw) if len(f) else None)

    # Viterbi over contiguous runs of frames that have candidates
    k = 0
    while k < T:
        if cands[k] is None:
            k += 1
            continue
        e = k
        while e < T and cands[e] is not None:
            e += 1
        _viterbi_run(cands[k:e], nom, F, B, k, n_formants)
        k = e
    return {"t": times, "F": F, "B": B, "level": lvl - ref}


_REF_CACHE: dict = {}


def _frame_level_db(y: np.ndarray, fs: int, audio_times: np.ndarray, win_ms: float):
    """RMS level (dB) of frames centred on ``audio_times`` and the file's 95th-percentile reference level."""
    n = max(int(round(win_ms * fs / 1000.0)), 8)
    yp = preemphasis(y, 0.97)
    fr = _frames(yp, np.round(audio_times * fs), n)
    lvl = 20 * np.log10(np.sqrt((fr ** 2).mean(axis=1)) + 1e-12)
    key = (id(y), len(y), n)
    if key not in _REF_CACHE:
        m = len(yp) // n
        seg = yp[: m * n].reshape(m, n)
        lv = 20 * np.log10(np.sqrt((seg ** 2).mean(axis=1)) + 1e-12)
        if len(_REF_CACHE) > 8:
            _REF_CACHE.clear()
        _REF_CACHE[key] = float(np.percentile(lv, 95)) if m > 1 else float(lvl.max())
    return lvl, _REF_CACHE[key]


def _states(f: np.ndarray, bw: np.ndarray, nf: int):
    """All ordered choices of nf candidates (padded with NaN when there are too few)."""
    if len(f) >= nf:
        idx = np.array(list(combinations(range(len(f)), nf)))
        return f[idx], bw[idx]
    pf = np.full((1, nf), np.nan)
    pb = np.full((1, nf), np.nan)
    pf[0, : len(f)] = f
    pb[0, : len(f)] = bw
    return pf, pb


def _viterbi_run(cs, nom, F, B, k0, nf):
    MISS_LOCAL, MISS_TRANS = 1.0, 0.6
    W_BW, W_NOM, W_TR = 1.0, 0.9, 3.0
    lognom = np.log(nom)
    prev_f = prev_cost = None
    back = []
    sts = []
    for f, bw in cs:
        sf, sb = _states(f, bw, nf)
        miss = np.isnan(sf)
        loc = np.where(miss, MISS_LOCAL, W_BW * np.nan_to_num(sb) / 500.0 + W_NOM * np.abs(np.log(np.nan_to_num(sf, nan=1.0)) - lognom[None, :]))
        loc = loc.sum(axis=1)
        if prev_f is None:
            cost = loc
            bp = np.zeros(len(loc), int)
        else:
            lf = np.log(np.where(np.isnan(sf), 1.0, sf))[:, None, :]
            lp = np.log(np.where(np.isnan(prev_f), 1.0, prev_f))[None, :, :]
            d = W_TR * np.abs(lf - lp)
            anym = np.isnan(sf)[:, None, :] | np.isnan(prev_f)[None, :, :]
            d = np.where(anym, MISS_TRANS, d).sum(axis=2)
            tot = d + prev_cost[None, :]
            bp = tot.argmin(axis=1)
            cost = loc + tot[np.arange(len(loc)), bp]
        sts.append((sf, sb))
        back.append(bp)
        prev_f, prev_cost = sf, cost
    s = int(np.argmin(prev_cost))
    for i in range(len(cs) - 1, -1, -1):
        sf, sb = sts[i]
        F[k0 + i] = sf[s]
        B[k0 + i] = sb[s]
        s = back[i][s] if i > 0 else s


# --------------------------------------------------------------------------- tube model
def tube_formants(areas_cm2: np.ndarray, lengths_cm: np.ndarray, n_formants: int = 4, fmax: float = 5500.0,
                  c: float = 35000.0) -> np.ndarray:
    """Resonances of a closed(glottis)-open(lips) chain of lossless tubes.

    ``areas_cm2`` and ``lengths_cm`` are ordered glottis -> lips. Formants are the zeros of the
    A-element of the chain matrix, found by sign change on a 2 Hz grid and refined linearly.
    This is a first-order sanity check on an area function, not a substitute for a full model.
    """
    areas = np.maximum(np.asarray(areas_cm2, float), 0.02)
    lens = np.asarray(lengths_cm, float)
    f = np.arange(20.0, fmax, 2.0)
    k = 2 * np.pi * f / c
    A = np.ones_like(f, dtype=complex)
    B = np.zeros_like(f, dtype=complex)
    C = np.zeros_like(f, dtype=complex)
    D = np.ones_like(f, dtype=complex)
    # chain from glottis (index 0) to lips; Z = rho*c / A (rho*c folded out: it cancels for the A element)
    for a_i, l_i in zip(areas, lens):
        kl = k * l_i
        z = 1.0 / a_i
        ca, sa = np.cos(kl), np.sin(kl)
        a2, b2, c2, d2 = ca, 1j * z * sa, 1j * sa / z, ca
        A, B, C, D = A * a2 + B * c2, A * b2 + B * d2, C * a2 + D * c2, C * b2 + D * d2
    r = np.real(D)  # closed glottis (U=0) + open lips (P=0): resonance when K22 = 0
    s = np.sign(r)
    idx = np.where(s[:-1] * s[1:] < 0)[0]
    out = []
    for i in idx:
        x0, x1 = f[i], f[i + 1]
        y0, y1 = r[i], r[i + 1]
        out.append(x0 - y0 * (x1 - x0) / (y1 - y0))
    return np.array(out[:n_formants])


# --------------------------------------------------------------------------- tube-model synthesis
def rosenberg_train(fs: int, dur: float, f0_start: float, f0_end: float) -> np.ndarray:
    """Glottal flow: train of Rosenberg pulses (open phase 40 %, closing phase 16 % of the period)
    with f0 gliding linearly from f0_start to f0_end."""
    n = int(fs * dur)
    out = np.zeros(n)
    t = 0.0
    while t < dur:
        f0 = f0_start + (f0_end - f0_start) * t / dur
        T0 = 1.0 / f0
        Tp, Tn = 0.4 * T0, 0.16 * T0
        i0 = int(round(t * fs))
        L = int(T0 * fs) + 2
        tt = np.arange(L) / fs
        g = np.zeros(L)
        r = tt < Tp
        g[r] = 0.5 * (1 - np.cos(np.pi * tt[r] / Tp))
        f = (tt >= Tp) & (tt < Tp + Tn)
        g[f] = np.cos(np.pi * (tt[f] - Tp) / (2 * Tn))
        end = min(n, i0 + L)
        if i0 < n:
            out[i0:end] += g[: end - i0]
        t += T0
    return out


def tube_transfer(areas_cm2, lengths_cm, freqs, c=35000.0, wall_loss=3.5e-4):
    """U_lips / U_glottis of a chain of tubes (glottis -> lips) with frequency-dependent
    propagation loss (alpha = wall_loss*sqrt(f) per cm) and a piston radiation load at the lips."""
    areas = np.maximum(np.asarray(areas_cm2, float), 0.05)
    lens = np.maximum(np.asarray(lengths_cm, float), 0.02)
    f = np.maximum(freqs, 1.0)
    w = 2 * np.pi * f
    gam = wall_loss * np.sqrt(f) + 1j * w / c
    K11 = np.ones_like(f, complex); K12 = np.zeros_like(K11); K21 = np.zeros_like(K11); K22 = np.ones_like(K11)
    for A, l in zip(areas, lens):
        Z = 1.0 / A
        ch, sh = np.cosh(gam * l), np.sinh(gam * l)
        m11, m12, m21, m22 = ch, Z * sh, sh / Z, ch
        K11, K12, K21, K22 = K11 * m11 + K12 * m21, K11 * m12 + K12 * m22, K21 * m11 + K22 * m21, K21 * m12 + K22 * m22
    a = np.sqrt(areas[-1] / np.pi)
    ka = w / c * a
    Zr = (1.0 / areas[-1]) * (ka ** 2 / 4 + 1j * 0.6133 * ka)
    return 1.0 / (K21 * Zr + K22)


def synthesize_vowel(areas_cm2, lengths_cm, fs: int = 16000, dur: float = 2.0, f0: float = 120.0,
                     f0_end: float | None = None) -> np.ndarray:
    """Source-filter synthesis: Rosenberg glottal flow -> tube transfer function -> lip radiation (derivative)."""
    f0_end = f0 * 0.85 if f0_end is None else f0_end
    src = rosenberg_train(fs, dur, f0, f0_end)
    n = len(src)
    nfft = 1 << int(np.ceil(np.log2(n * 2)))
    H = tube_transfer(areas_cm2, lengths_cm, np.fft.rfftfreq(nfft, 1.0 / fs))
    y = np.fft.irfft(np.fft.rfft(src, nfft) * H, nfft)[:n]
    y = np.diff(y, prepend=0.0)
    fade = int(0.03 * fs)
    ramp = np.linspace(0, 1, fade)
    y[:fade] *= ramp
    y[-fade:] *= ramp[::-1]
    return (0.8 * y / (np.abs(y).max() + 1e-12)).astype(np.float32)
