"""Loading and synchronising rtMRI video + audio.

Design notes (why this is more robust than the old MATLAB tool)
---------------------------------------------------------------
* One timeline, in seconds. Every frame has its own timestamp (taken from the
  container when ffprobe is available, otherwise uniform from the average fps).
  Time -> frame uses those timestamps, never ``floor(t * FrameRate)``.
* Audio and video lengths are allowed to differ (they often do). Nothing is
  stretched; an explicit ``av_offset`` lets you slide the audio against the
  video, and ``estimate_av_offset`` can suggest one from motion vs. audio energy.
* Audio can come from a .wav, from the audio track inside the video, or be absent.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy import signal
from scipy.io import wavfile

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}


# --------------------------------------------------------------------------- discovery
def _subdir(root: Path, names: tuple[str, ...]) -> Path | None:
    """Return the sub-folder of ``root`` whose name matches one of ``names`` (case-insensitive)."""
    for d in sorted(root.iterdir()):
        if d.is_dir() and d.name.lower() in names:
            return d
    return None


def _discover_root(root: Path) -> list[dict] | None:
    """Root-folder layout: ``root/mp4/*.mp4`` and ``root/wav/*.wav``.

    Returns one entry per base name found in either folder (sorted), paired by name,
    or None if ``root`` does not have both sub-folders (the generic search is used instead).
    A name missing its video is listed with ``video`` = None (it cannot be opened).
    """
    vdir = _subdir(root, ("mp4", "video", "videos"))
    adir = _subdir(root, ("wav", "audio"))
    if vdir is None or adir is None:
        return None
    vids = {p.stem: p for p in sorted(vdir.iterdir()) if p.is_file() and p.suffix.lower() in VIDEO_EXTS and not p.name.startswith(".")}
    auds = {p.stem: p for p in sorted(adir.iterdir()) if p.is_file() and p.suffix.lower() in AUDIO_EXTS and not p.name.startswith(".")}
    out = []
    for name in sorted(set(vids) | set(auds)):
        v, a = vids.get(name), auds.get(name)
        out.append({"token": name, "video": str(v) if v else None, "wav": str(a) if a else None, "rel": name})
    return out


def discover(root: Path) -> list[dict]:
    """Find video files under ``root`` and pair each with a matching audio file.

    A wav is matched by base name, looked for (in order) next to the video,
    in ``wav``/``audio`` folders beside or above it, then anywhere under root.
    """
    root = Path(root)
    if root.is_dir():
        structured = _discover_root(root)
        if structured is not None:
            return structured
    if root.is_file():
        vids = [root]
        base = root.parent
    else:
        base = root
        vids = sorted(p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXTS and not p.name.startswith("."))
    auds: dict[str, list[Path]] = {}
    for p in base.rglob("*"):
        if p.suffix.lower() in AUDIO_EXTS and not p.name.startswith("."):
            auds.setdefault(p.stem, []).append(p)
    out = []
    for v in vids:
        wav = None
        cands = []
        for d in (v.parent, v.parent / "wav", v.parent / "audio", v.parent.parent / "wav", v.parent.parent / "audio", base / "wav", base / "audio"):
            for ext in (".wav", ".flac"):
                cands.append(d / (v.stem + ext))
        for c in cands:
            if c.exists():
                wav = c
                break
        if wav is None and v.stem in auds:
            wav = sorted(auds[v.stem], key=lambda p: len(p.parts))[0]
        out.append({"token": v.stem, "video": str(v), "wav": str(wav) if wav else None,
                    "rel": str(v.relative_to(base)) if base in v.parents else v.name})
    return out


# --------------------------------------------------------------------------- helpers
def _ffmpeg() -> str | None:
    return shutil.which("ffmpeg")


def _ffprobe() -> str | None:
    return shutil.which("ffprobe")


def _read_audio_file(path: Path) -> tuple[np.ndarray, int]:
    """Return mono float32 audio in [-1, 1] and the sample rate."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # "Chunk (non-data) not understood" is harmless
            fs, y = wavfile.read(str(path))
        y = np.asarray(y)
        if y.dtype.kind == "i":
            y = y.astype(np.float32) / float(np.iinfo(y.dtype).max + 1)
        elif y.dtype.kind == "u":
            y = (y.astype(np.float32) - 128.0) / 128.0
        else:
            y = y.astype(np.float32)
        if y.ndim > 1:
            y = y.mean(axis=1)
        return y, int(fs)
    except Exception:
        ff = _ffmpeg()
        if ff is None:
            raise
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "a.wav"
            subprocess.run([ff, "-v", "error", "-y", "-i", str(path), "-ac", "1", "-c:a", "pcm_s16le", str(tmp)], check=True)
            return _read_audio_file(tmp)


def _extract_audio_from_video(path: Path) -> tuple[np.ndarray, int] | None:
    ff, fp = _ffmpeg(), _ffprobe()
    if not ff or not fp:
        return None
    r = subprocess.run([fp, "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True)
    if not r.stdout.strip():
        return None
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "a.wav"
        p = subprocess.run([ff, "-v", "error", "-y", "-i", str(path), "-vn", "-ac", "1", "-c:a", "pcm_s16le", str(tmp)])
        if p.returncode != 0 or not tmp.exists():
            return None
        return _read_audio_file(tmp)


def _frame_times(path: Path, n: int, fps: float) -> tuple[np.ndarray, str]:
    """Per-frame presentation times (s). Uses ffprobe packet timestamps when they
    are consistent with the decoded frame count, else a uniform grid."""
    fp = _ffprobe()
    if fp:
        try:
            r = subprocess.run([fp, "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time",
                                "-of", "csv=p=0", str(path)], capture_output=True, text=True, timeout=120)
            t = np.array([float(x) for x in r.stdout.split() if x.strip() not in ("", "N/A")])
            t.sort()
            if len(t) == n and np.all(np.isfinite(t)):
                return t, "container timestamps"
        except Exception:
            pass
    return np.arange(n) / fps, "uniform (average fps)"


# --------------------------------------------------------------------------- Media
@dataclass
class Media:
    token: str
    video_path: str
    audio_path: str | None
    frames: np.ndarray                 # (n, h, w) uint8 grey
    frame_times: np.ndarray            # (n,) seconds
    fps: float
    audio: np.ndarray | None           # mono float32
    fs: int | None
    audio_source: str = "none"
    time_source: str = ""
    warnings: list[str] = field(default_factory=list)
    av_offset: float = 0.0             # seconds added to audio time to land on the video timeline
    _gain: np.ndarray | None = None

    # ---- basic properties
    @property
    def n(self): return int(self.frames.shape[0])
    @property
    def h(self): return int(self.frames.shape[1])
    @property
    def w(self): return int(self.frames.shape[2])
    @property
    def video_dur(self): return float(self.frame_times[-1] - self.frame_times[0] + 1.0 / self.fps)
    @property
    def audio_dur(self): return 0.0 if self.audio is None else len(self.audio) / self.fs
    @property
    def t_end(self):
        """End of the common timeline (the longer of audio / video)."""
        return max(float(self.frame_times[-1]) + 1.0 / self.fps, self.audio_dur + self.av_offset)

    def frame_at(self, t: float) -> int:
        i = int(np.searchsorted(self.frame_times, t))
        if i <= 0: return 0
        if i >= self.n: return self.n - 1
        return i if abs(self.frame_times[i] - t) < abs(self.frame_times[i - 1] - t) else i - 1

    # ---- loading
    @classmethod
    def load(cls, video: str, wav: str | None = None, flip_h: bool = False) -> "Media":
        vp = Path(video)
        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            raise IOError(f"Cannot open video {vp}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if fr.ndim == 3:
                fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            frames.append(fr)
        cap.release()
        if not frames:
            raise IOError(f"No frames could be decoded from {vp}")
        arr = np.ascontiguousarray(np.stack(frames))
        if flip_h:
            arr = np.ascontiguousarray(arr[:, :, ::-1])
        n = arr.shape[0]
        warns: list[str] = []
        times, tsrc = _frame_times(vp, n, fps if fps > 1 else 25.0)
        dt = np.diff(times)
        fps_eff = float(1.0 / np.median(dt)) if len(dt) else (fps or 25.0)
        if len(dt) and (dt.max() > 1.5 * np.median(dt)):
            warns.append(f"Video frame timing is irregular (max gap {dt.max()*1000:.1f} ms vs median {np.median(dt)*1000:.1f} ms); "
                         "per-frame timestamps are used for sync.")

        audio, fs, asrc, apath = None, None, "none", None
        if wav:
            audio, fs = _read_audio_file(Path(wav))
            asrc, apath = "wav file", wav
        else:
            got = _extract_audio_from_video(vp)
            if got:
                audio, fs = got
                asrc = "audio track in video"
        m = cls(token=vp.stem, video_path=str(vp), audio_path=apath, frames=arr, frame_times=times, fps=fps_eff,
                audio=audio, fs=fs, audio_source=asrc, time_source=tsrc, warnings=warns)
        if audio is None:
            m.warnings.append("No audio found - spectrogram and formant tools are disabled for this file.")
        else:
            d = m.audio_dur - m.video_dur
            if abs(d) > 0.05:
                who = "longer" if d > 0 else "shorter"
                m.warnings.append(f"Audio is {abs(d):.2f} s {who} than the video. Both are placed on the same timeline starting at 0; "
                                  "use the A/V offset control (or 'Estimate') if they are not aligned.")
        return m

    # ---- image helpers
    def gain_map(self) -> np.ndarray:
        """Data-driven receive-coil shading estimate: heavily blurred temporal mean image,
        normalised to mean 1. Dividing frames by it flattens coil sensitivity."""
        if self._gain is None:
            sub = self.frames[:: max(1, self.n // 400)].astype(np.float32)
            mean = sub.mean(axis=0)
            k = int(max(3, round(self.w / 6))) | 1
            g = cv2.GaussianBlur(mean, (0, 0), k / 2.0)
            g = np.maximum(g, 0.15 * g.max())
            g /= g.mean()
            self._gain = g.astype(np.float32)
        return self._gain

    # ---- audio helpers
    def analysis_signal(self, burst_s: float = 0.0, burst_atten: float = 0.1) -> np.ndarray | None:
        """Audio used for display/analysis: optionally attenuates the scanner start-up burst."""
        if self.audio is None:
            return None
        y = self.audio
        if burst_s > 0:
            y = y.copy()
            y[: int(burst_s * self.fs)] *= burst_atten
        return y

    def envelope_minmax(self, y: np.ndarray, t0: float, t1: float, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Min / max of audio in ``n`` equal bins between timeline times t0..t1 (NaN where there is no audio)."""
        edges = np.linspace(t0, t1, n + 1)
        idx = np.round((edges - self.av_offset) * self.fs).astype(np.int64)
        mn = np.full(n, np.nan, np.float32)
        mx = np.full(n, np.nan, np.float32)
        L = len(y)
        for i in range(n):
            a, b = idx[i], max(idx[i + 1], idx[i] + 1)
            if b <= 0 or a >= L:
                continue
            seg = y[max(a, 0):min(b, L)]
            if seg.size:
                mn[i], mx[i] = seg.min(), seg.max()
        return mn, mx

    # ---- A/V sync suggestion
    def estimate_av_offset(self, max_lag: float = 4.0) -> dict:
        """Suggest an A/V offset by cross-correlating image-motion energy with the audio envelope.

        Returns the lag curve so the UI can show how convincing the peak is. Treat as a hint:
        scanner noise and silence can make it ambiguous.
        """
        if self.audio is None:
            return {"ok": False, "reason": "no audio"}
        fps = self.fps
        # motion energy: mean absolute frame difference, in the central part of the image
        f = self.frames.astype(np.float32)
        motion = np.abs(np.diff(f, axis=0)).mean(axis=(1, 2))
        motion = np.r_[motion[0], motion]
        # audio envelope sampled at video frame times
        y = np.abs(self.audio)
        win = max(1, int(self.fs / fps))
        c = np.cumsum(np.r_[0.0, y])
        env_all = (c[win:] - c[:-win]) / win
        env_t = (np.arange(len(env_all)) + win / 2) / self.fs
        def hp(x):
            k = int(2.0 * fps)
            ker = np.ones(k) / k
            return x - np.convolve(x, ker, mode="same")
        def z(x):
            x = x - x.mean()
            s = x.std()
            return x / s if s > 0 else x
        tgrid = np.arange(0, self.audio_dur, 1.0 / fps)
        env = np.interp(tgrid, env_t, env_all)
        m = z(hp(motion))
        e = z(hp(env))
        L = int(max_lag * fps)
        lags = np.arange(-L, L + 1)
        corr = np.zeros(len(lags))
        for k, lag in enumerate(lags):
            # audio sample at tgrid index j corresponds to timeline time (j/fps + lag/fps)
            if lag >= 0:
                a = m[lag:]; b = e[: len(m) - lag] if len(e) >= len(m) - lag else e
            else:
                a = m[: len(m) + lag]; b = e[-lag:]
            nn = min(len(a), len(b))
            if nn < 20:
                continue
            corr[k] = float(np.mean(a[:nn] * b[:nn]))
        best = int(np.argmax(corr))
        srt = np.sort(corr)[::-1]
        # prominence: peak vs best value outside +-0.5 s of the peak
        mask = np.abs(lags - lags[best]) > int(0.5 * fps)
        second = corr[mask].max() if mask.any() else 0.0
        return {"ok": True, "offset": float(lags[best] / fps), "peak": float(corr[best]), "second": float(second),
                "lags": (lags / fps).round(4).tolist(), "corr": corr.round(4).tolist()}
