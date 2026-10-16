"""Flask application: serves the UI and a small JSON / binary API.

Everything runs locally; nothing leaves your machine.
"""
from __future__ import annotations

import csv
import io
import json
import math
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
import webbrowser
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_file, send_from_directory
from scipy.io import wavfile

from . import acoustics as ac
from .media import Media, discover
from .tract import TractAnalyzer, FrameResult

STATIC = Path(__file__).parent / "static"


def clean(o):
    """Make numpy / NaN safe for strict JSON."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def J(o, status=200):
    return Response(json.dumps(clean(o)), status=status, mimetype="application/json")


class Session:
    """The currently open recording and everything derived from it."""

    DEFAULT_SETTINGS = dict(av_offset=0.0, burst_s=0.55, burst_atten=0.1, flip_h=False, fov_mm=280.0, pixel_mm=None, flatten=False)

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.media: Media | None = None
        self.tract: TractAnalyzer | None = None
        self.settings = dict(self.DEFAULT_SETTINGS)
        self.labels: list[dict] = []
        self.sg_ref: dict = {}
        self.batch: dict | None = None
        self.lock = threading.RLock()
        self.jobs: dict[str, dict] = {}
        self.cache_dir = Path(tempfile.mkdtemp(prefix="rtmri_"))
        self.audio_wav: Path | None = None
        self.view: dict = {}

    # ---- helpers
    @property
    def pixel_mm(self) -> float:
        s = self.settings
        if s.get("pixel_mm"):
            return float(s["pixel_mm"])
        return float(s["fov_mm"]) / self.media.w

    def session_path(self) -> Path:
        return self.out_dir / self.media.token / "session.json"

    def save(self):
        if not self.media:
            return
        p = self.session_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {"version": 1, "token": self.media.token, "video": self.media.video_path, "audio": self.media.audio_path,
                "settings": self.settings, "labels": self.labels, "tract": self.tract.state() if self.tract else {}, "view": self.view}
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(clean(data), indent=1))
        tmp.replace(p)

    def open(self, video: str, wav: str | None):
        with self.lock:
            sp = None
            prev = None
            # peek at an existing session to know flip_h before decoding
            guess = self.out_dir / Path(video).stem / "session.json"
            if guess.exists():
                try:
                    prev = json.loads(guess.read_text())
                except Exception:
                    prev = None
            settings = dict(self.DEFAULT_SETTINGS)
            if prev:
                settings.update(prev.get("settings", {}))
            m = Media.load(video, wav, flip_h=bool(settings.get("flip_h")))
            m.av_offset = float(settings.get("av_offset", 0.0))
            self.media, self.settings = m, settings
            self.labels = (prev or {}).get("labels", [])
            self.view = (prev or {}).get("view", {})
            self.batch = None
            self.tract = TractAnalyzer(m.w, m.h, self.pixel_mm)
            if prev and prev.get("tract"):
                try:
                    self.tract.load_state(prev["tract"], ref_img_loader=lambda f: self.tract._prepared(m.frames[f], m.gain_map() if self.settings["flatten"] else None))
                except Exception as e:  # a stale session must never stop a file from opening
                    m.warnings.append(f"Could not restore saved tract analysis ({e}).")
            self.tract.p.pixel_mm = self.pixel_mm
            self._prepare_audio()

    def _prepare_audio(self):
        m = self.media
        self.sg_ref = {}
        self.audio_wav = None
        if m.audio is None:
            return
        y16 = (np.clip(m.audio, -1, 1) * 32767).astype(np.int16)
        p = self.cache_dir / f"{uuid.uuid4().hex}.wav"
        wavfile.write(str(p), m.fs, y16)
        self.audio_wav = p

    def signal(self):
        m = self.media
        return m.analysis_signal(self.settings["burst_s"], self.settings["burst_atten"])

    def sg_top(self, win_ms: float, preemph: bool) -> float:
        key = (round(win_ms, 1), bool(preemph), round(self.settings["burst_s"], 2), round(self.settings["burst_atten"], 3))
        if key not in self.sg_ref:
            self.sg_ref[key] = ac.spectrogram_ref_db(self.signal(), self.media.fs, win_ms, preemph)
        return self.sg_ref[key]

    def gain(self):
        return self.media.gain_map() if self.settings.get("flatten") else None


def clamp_dur(d: float) -> float:
    return float(min(max(d, 0.2), 10.0))


def create_app(root, out_dir) -> Flask:
    root = Path(root) if root else None
    out_dir = Path(out_dir)
    app = Flask(__name__, static_folder=None)
    S = Session(out_dir)
    state = {"root": root or "", "files": discover(root) if root and root.exists() else []}

    # ---------------------------------------------------------------- static
    @app.get("/")
    def index():
        return send_from_directory(STATIC, "index.html")

    @app.get("/static/<path:name>")
    def static_files(name):
        r = send_from_directory(STATIC, name)
        r.headers["Cache-Control"] = "no-store"
        return r

    # ---------------------------------------------------------------- files / session
    @app.get("/api/files")
    def files():
        return J({"root": str(state["root"]), "files": state["files"], "open": S.media.token if S.media else None})

    @app.post("/api/scan")
    def scan():
        p = Path(request.json.get("path", "")).expanduser()
        if not p.exists():
            return J({"error": f"{p} does not exist"}, 400)
        state["root"] = p if p.is_dir() else p.parent
        state["files"] = discover(p)
        return J({"root": str(state["root"]), "files": state["files"]})

    @app.get("/api/browse")
    def browse():
        """List sub-folders of a folder on this computer (for the in-app folder picker)."""
        import os
        from .media import _subdir, VIDEO_EXTS, AUDIO_EXTS
        raw = request.args.get("path") or str(state["root"] or Path.home())
        p = Path(raw).expanduser()
        try:
            p = p.resolve()
        except Exception:
            pass
        while not p.is_dir() and p != p.parent:
            p = p.parent
        dirs, err = [], None
        try:
            for d in sorted(p.iterdir(), key=lambda q: q.name.lower()):
                try:
                    if d.is_dir() and not d.name.startswith("."):
                        dirs.append({"name": d.name, "path": str(d)})
                except OSError:
                    pass
        except OSError as e:
            err = str(e)
        info = {"video": None, "audio": None}
        try:
            vd, ad = _subdir(p, ("mp4", "video", "videos")), _subdir(p, ("wav", "audio"))
            cnt = lambda d, ex: sum(1 for q in d.iterdir() if q.is_file() and q.suffix.lower() in ex and not q.name.startswith("."))
            if vd: info["video"] = [vd.name, cnt(vd, VIDEO_EXTS)]
            if ad: info["audio"] = [ad.name, cnt(ad, AUDIO_EXTS)]
            info["loose_videos"] = cnt(p, VIDEO_EXTS)
        except OSError:
            pass
        drives = [f"{c}:\\" for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if os.name == "nt" and Path(f"{c}:\\").exists()]
        return J({"path": str(p), "parent": str(p.parent) if p.parent != p else None, "dirs": dirs, "home": str(Path.home()),
                  "drives": drives, "info": info, "error": err})

    def meta():
        m = S.media
        return {
            "token": m.token, "video": m.video_path, "audio": m.audio_path, "audio_source": m.audio_source,
            "w": m.w, "h": m.h, "n": m.n, "fps": m.fps, "frame_times": np.round(m.frame_times, 5),
            "time_source": m.time_source, "video_dur": m.video_dur, "audio_dur": m.audio_dur, "t_end": m.t_end,
            "fs": m.fs, "has_audio": m.audio is not None, "warnings": m.warnings, "settings": S.settings,
            "pixel_mm": S.pixel_mm, "labels": S.labels, "view": S.view, "tract": tract_state(),
            "chunk": max(1, int(8_000_000 // (m.w * m.h))),
        }

    @app.post("/api/open")
    def open_():
        d = request.json
        try:
            S.open(d["video"], d.get("wav"))
        except Exception as e:
            return J({"error": str(e)}, 400)
        return J(meta())

    @app.get("/api/meta")
    def meta_():
        if not S.media:
            return J({"open": False})
        return J(meta())

    @app.get("/favicon.ico")
    def favicon():
        return Response(b"", status=204)

    def need():
        if not S.media:
            raise RuntimeError("Open a recording first.")

    @app.errorhandler(Exception)
    def on_error(e):
        import traceback
        traceback.print_exc()
        code = getattr(e, "code", 500)
        return J({"error": str(e)}, code if isinstance(code, int) else 500)

    @app.post("/api/settings")
    def settings():
        need()
        d = request.json
        flip_changed = "flip_h" in d and bool(d["flip_h"]) != bool(S.settings["flip_h"])
        for k in S.DEFAULT_SETTINGS:
            if k in d:
                S.settings[k] = d[k]
        S.media.av_offset = float(S.settings["av_offset"])
        if flip_changed:
            S.media.frames = np.ascontiguousarray(S.media.frames[:, :, ::-1])
            S.media._gain = None
            S.tract = TractAnalyzer(S.media.w, S.media.h, S.pixel_mm)
            S.batch = None
        S.tract.p.pixel_mm = S.pixel_mm
        S.tract.cache.clear()
        S.save()
        return J({"settings": S.settings, "pixel_mm": S.pixel_mm, "t_end": S.media.t_end, "flip_changed": flip_changed, "tract": tract_state()})

    @app.post("/api/view")
    def view():
        need()
        S.view = request.json
        S.save()
        return J({"ok": True})

    @app.get("/api/estimate_sync")
    def est_sync():
        need()
        return J(S.media.estimate_av_offset())

    # ---------------------------------------------------------------- video
    @app.get("/api/frames")
    def frames():
        need()
        a = int(request.args.get("start", 0)); c = int(request.args.get("count", 1))
        a = max(0, min(a, S.media.n)); b = min(S.media.n, a + c)
        return Response(S.media.frames[a:b].tobytes(), mimetype="application/octet-stream", headers={"X-Count": str(b - a)})

    @app.get("/api/gain")
    def gain():
        need()
        return Response(S.media.gain_map().astype(np.float32).tobytes(), mimetype="application/octet-stream")

    # ---------------------------------------------------------------- audio
    @app.get("/api/audio.wav")
    def audio_wav():
        need()
        if S.audio_wav is None:
            return J({"error": "no audio"}, 404)
        return send_file(S.audio_wav, mimetype="audio/wav", conditional=True, max_age=0)

    @app.get("/api/waveform")
    def waveform():
        need()
        if S.media.audio is None:
            return J({"error": "no audio"}, 404)
        t0 = float(request.args["t0"]); t1 = float(request.args["t1"]); n = int(request.args.get("n", 1000))
        mn, mx = S.media.envelope_minmax(S.signal(), t0, t1, n)
        return J({"t0": t0, "t1": t1, "min": np.round(mn, 4), "max": np.round(mx, 4)})

    @app.get("/api/spectrogram")
    def spectrogram():
        need()
        if S.media.audio is None:
            return J({"error": "no audio"}, 404)
        q = request.args
        t0 = float(q["t0"]); t1 = float(q["t1"])
        win = float(q.get("win", 20)); step = float(q.get("step", 5)); fmax = float(q.get("fmax", 5000))
        pre = q.get("preemph", "1") in ("1", "true"); dr = float(q.get("dr", 60)); top = float(q.get("top", 0))
        nfft = int(q["nfft"]) if q.get("nfft") else None
        db, freqs, times = ac.spectrogram_db(S.signal(), S.media.fs, t0, t1, win, step, fmax, pre, nfft, S.media.av_offset)
        ref = S.sg_top(win, pre) + top
        img = ac.db_to_gray(db, ref, dr)[::-1]
        ok, png = cv2.imencode(".png", np.ascontiguousarray(img))
        meta_ = {"t0": float(times[0]), "t1": float(times[-1]), "step": step / 1000.0, "fmax": float(freqs[-1]), "rows": img.shape[0], "cols": img.shape[1]}
        return Response(png.tobytes(), mimetype="image/png", headers={"X-Meta": json.dumps(meta_)})

    def formant_args():
        q = request.args
        return dict(n_formants=int(q.get("nf", 4)), max_freq=float(q.get("fmaxf", 5500)), win_ms=float(q.get("fwin", 25)),
                    step_ms=float(q.get("fstep", 5)), order=int(q["order"]) if q.get("order") not in (None, "", "0") else None,
                    max_bw=float(q.get("maxbw", 600)), min_level_db=float(q.get("minlvl", -40)), speaker_scale=float(q.get("scale", 1.0)))

    @app.get("/api/formants")
    def formants():
        need()
        if S.media.audio is None:
            return J({"error": "no audio"}, 404)
        t0 = float(request.args["t0"]); t1 = float(request.args["t1"])
        r = ac.track_formants(S.signal(), S.media.fs, t0, t1, off=S.media.av_offset, **formant_args())
        return J({"t": np.round(r["t"], 4), "F": np.round(r["F"], 1), "B": np.round(r["B"], 1)})

    @app.get("/api/formants.csv")
    def formants_csv():
        need()
        t0 = float(request.args.get("t0", 0)); t1 = float(request.args.get("t1", S.media.t_end))
        r = ac.track_formants(S.signal(), S.media.fs, t0, t1, off=S.media.av_offset, **formant_args())
        nf = r["F"].shape[1]
        buf = io.StringIO(); w = csv.writer(buf)
        w.writerow(["time_s"] + [f"F{i+1}_Hz" for i in range(nf)] + [f"B{i+1}_Hz" for i in range(nf)])
        for k, t in enumerate(r["t"]):
            w.writerow([f"{t:.4f}"] + ["" if np.isnan(v) else f"{v:.1f}" for v in r["F"][k]] + ["" if np.isnan(v) else f"{v:.1f}" for v in r["B"][k]])
        return Response(buf.getvalue(), mimetype="text/csv", headers={"Content-Disposition": f"attachment; filename={S.media.token}_formants.csv"})

    # ---------------------------------------------------------------- tract
    def tract_state():
        t = S.tract
        if t is None:
            return {}
        return {"params": {k: v for k, v in t.state()["params"].items()}, "landmarks": t.landmarks,
                "grid": t.grid.to_json() if t.grid else None, "ref_frame": t.ref_frame, "pixel_mm": S.pixel_mm,
                "n_edited_frames": len(t.edits)}

    @app.get("/api/tract/state")
    def tract_state_():
        need()
        return J(tract_state())

    @app.post("/api/tract/landmarks")
    def tract_landmarks():
        need()
        with S.lock:
            try:
                S.tract.set_landmarks(request.json["landmarks"])
            except ValueError as e:
                return J({"error": str(e)}, 400)
            S.batch = None
            S.save()
        return J(tract_state())

    @app.post("/api/tract/params")
    def tract_params():
        need()
        with S.lock:
            S.tract.set_params(request.json)
            S.tract.p.pixel_mm = S.pixel_mm
            S.batch = None
            S.save()
        return J(tract_state())

    @app.post("/api/tract/clear")
    def tract_clear():
        need()
        with S.lock:
            S.tract = TractAnalyzer(S.media.w, S.media.h, S.pixel_mm)
            S.batch = None
            S.save()
        return J(tract_state())

    def frame_payload(f, res: FrameResult):
        pred = S.tract.predicted_formants(res)
        return {"frame": f, "time": float(S.media.frame_times[f]), **res.to_json(), "predicted_formants": pred}

    @app.post("/api/tract/frame")
    def tract_frame():
        need()
        f = int(request.json["frame"])
        with S.lock:
            res = S.tract.extract(f, S.media.frames[f], S.gain())
        return J(frame_payload(f, res))

    @app.get("/api/tract/frame.csv")
    def tract_frame_csv():
        need()
        f = int(request.args["frame"])
        with S.lock:
            r = S.tract.extract(f, S.media.frames[f], S.gain())
        buf = io.StringIO(); w = csv.writer(buf)
        w.writerow(["frame", "time_s", "gridline", "distance_from_glottis_mm", "aperture_mm", "area_cm2",
                    "centre_x_px", "centre_y_px", "tongue_side_x_px", "tongue_side_y_px", "wall_side_x_px", "wall_side_y_px"])
        for g in range(r.ngl):
            w.writerow([f + 1, f"{S.media.frame_times[f]:.4f}", g + 1, f"{r.x_mm[g]:.2f}", f"{r.aperture_mm[g]:.2f}", f"{r.area_cm2[g]:.3f}",
                        f"{r.cen[g,0]:.2f}", f"{r.cen[g,1]:.2f}", f"{r.lf[g,0]:.2f}", f"{r.lf[g,1]:.2f}", f"{r.rt[g,0]:.2f}", f"{r.rt[g,1]:.2f}"])
        return Response(buf.getvalue(), mimetype="text/csv",
                        headers={"Content-Disposition": f"attachment; filename={S.media.token}_frame{f+1}_area_function.csv"})

    @app.post("/api/synth")
    def synth():
        """Synthesise a vowel from this frame's area function (Rosenberg source -> tube filter)."""
        need()
        d = request.json
        f = int(d["frame"]); f0 = float(d.get("f0", 120)); dur = float(d.get("dur", 2.0))
        with S.lock:
            r = S.tract.extract(f, S.media.frames[f], S.gain())
        y = ac.synthesize_vowel(r.area_cm2, r.seg_len_cm, 16000, clamp_dur(dur), f0)
        buf = io.BytesIO()
        wavfile.write(buf, 16000, (y * 32767).astype(np.int16))
        return Response(buf.getvalue(), mimetype="audio/wav", headers={"Content-Disposition": f"attachment; filename={S.media.token}_frame{f+1}_synth.wav"})

    @app.post("/api/tract/ref")
    def tract_ref():
        need()
        f = int(request.json["frame"])
        with S.lock:
            if request.json.get("clear"):
                S.tract.ref_rt, S.tract.ref_frame, S.tract.ref_img = {}, None, None
                S.tract.cache.clear()
            else:
                S.tract.set_reference(f, S.media.frames[f], S.gain())
            S.batch = None
            S.save()
        return J(tract_state())

    @app.post("/api/tract/edit")
    def tract_edit():
        need()
        d = request.json
        f = int(d["frame"])
        with S.lock:
            res0 = S.tract.extract(f, S.media.frames[f], S.gain())
            S.tract.edit_point(f, float(d["x"]), float(d["y"]), d["side"], res0.shift)
            res = S.tract.extract(f, S.media.frames[f], S.gain())
            S.batch = None
            S.save()
        return J(frame_payload(f, res))

    @app.post("/api/tract/clear_edits")
    def tract_clear_edits():
        need()
        f = int(request.json["frame"])
        with S.lock:
            S.tract.clear_edits(f)
            S.batch = None
            res = S.tract.extract(f, S.media.frames[f], S.gain())
            S.save()
        return J(frame_payload(f, res))

    @app.post("/api/tract/batch")
    def tract_batch():
        need()
        d = request.json
        a = max(0, int(d["start"])); b = min(S.media.n - 1, int(d["end"])); step = max(1, int(d.get("step", 1)))
        frames_ = list(range(a, b + 1, step))
        jid = uuid.uuid4().hex[:8]
        job = {"id": jid, "progress": 0.0, "done": False, "error": None, "n": len(frames_)}
        S.jobs[jid] = job

        def run():
            try:
                with S.lock:
                    results = {}
                    gain = S.gain()
                    for i, f in enumerate(frames_):
                        results[f] = S.tract.extract(f, S.media.frames[f], gain, use_cache=False)
                        job["progress"] = (i + 1) / len(frames_)
                    S.batch = {"frames": frames_, "results": results}
                job["done"] = True
            except Exception as e:  # noqa
                job["error"] = str(e)
                job["done"] = True

        threading.Thread(target=run, daemon=True).start()
        return J(job)

    @app.get("/api/job/<jid>")
    def job_(jid):
        return J(S.jobs.get(jid, {"error": "unknown job", "done": True}))

    def batch_arrays():
        b = S.batch
        if not b:
            return None
        fr = b["frames"]; R = [b["results"][f] for f in fr]
        ng = max(r.ngl for r in R)
        def pad(get):
            M = np.full((len(R), ng), np.nan)
            for i, r in enumerate(R):
                v = get(r); M[i, : len(v)] = v
            return M
        return {"frames": fr, "times": S.media.frame_times[fr], "x_mm": pad(lambda r: r.x_mm), "aperture_mm": pad(lambda r: r.aperture_mm),
                "area_cm2": pad(lambda r: r.area_cm2), "vt_len_mm": np.array([r.vt_len_mm for r in R]), "ngl": np.array([r.ngl for r in R]),
                "mid_area_mm2": np.array([r.mid_area_mm2 for r in R])}

    @app.get("/api/tract/batch")
    def tract_batch_get():
        need()
        a = batch_arrays()
        if a is None:
            return J({"error": "no batch results"}, 404)
        return J({"frames": a["frames"], "times": np.round(a["times"], 4), "aperture_mm": np.round(a["aperture_mm"], 2),
                  "x_mm": np.round(a["x_mm"], 2), "vt_len_mm": np.round(a["vt_len_mm"], 2), "ngl": a["ngl"], "mid_area_mm2": np.round(a["mid_area_mm2"], 1)})

    @app.get("/api/tract/batch.csv")
    def tract_batch_csv():
        need()
        a = batch_arrays()
        if a is None:
            return J({"error": "no batch results"}, 404)
        kind = request.args.get("kind", "long")
        buf = io.StringIO(); w = csv.writer(buf)
        if kind == "summary":
            w.writerow(["frame", "time_s", "n_gridlines", "tract_length_mm", "midsagittal_area_mm2"])
            for i, f in enumerate(a["frames"]):
                w.writerow([f, f"{a['times'][i]:.4f}", a["ngl"][i], f"{a['vt_len_mm'][i]:.2f}", f"{a['mid_area_mm2'][i]:.1f}"])
        else:
            w.writerow(["frame", "time_s", "gridline", "distance_from_glottis_mm", "aperture_mm", "area_cm2"])
            for i, f in enumerate(a["frames"]):
                for g in range(a["ngl"][i]):
                    w.writerow([f, f"{a['times'][i]:.4f}", g + 1, f"{a['x_mm'][i, g]:.2f}", f"{a['aperture_mm'][i, g]:.2f}", f"{a['area_cm2'][i, g]:.3f}"])
        return Response(buf.getvalue(), mimetype="text/csv",
                        headers={"Content-Disposition": f"attachment; filename={S.media.token}_area_function_{kind}.csv"})

    # ---------------------------------------------------------------- labels
    @app.post("/api/labels")
    def labels():
        need()
        S.labels = request.json["labels"]
        S.save()
        return J({"ok": True})

    @app.get("/api/labels.csv")
    def labels_csv():
        need()
        buf = io.StringIO(); w = csv.writer(buf)
        w.writerow(["label", "start_s", "end_s", "start_frame", "end_frame"])
        for l in sorted(S.labels, key=lambda l: l["t0"]):
            w.writerow([l["text"], f"{l['t0']:.4f}", f"{l['t1']:.4f}", S.media.frame_at(l["t0"]), S.media.frame_at(l["t1"])])
        return Response(buf.getvalue(), mimetype="text/csv", headers={"Content-Disposition": f"attachment; filename={S.media.token}_labels.csv"})

    # ---------------------------------------------------------------- export clip
    @app.get("/api/export/clip")
    def export_clip():
        need()
        m = S.media
        t0 = float(request.args["t0"]); t1 = float(request.args["t1"]); scale = int(request.args.get("scale", 3))
        kind = request.args.get("kind", "mp4")
        i0, i1 = m.frame_at(t0), m.frame_at(t1)
        if kind == "wav":
            if m.audio is None:
                return J({"error": "no audio"}, 404)
            a = int(max(0, (t0 - m.av_offset) * m.fs)); b = int((t1 - m.av_offset) * m.fs)
            buf = io.BytesIO()
            wavfile.write(buf, m.fs, (np.clip(m.audio[a:b], -1, 1) * 32767).astype(np.int16))
            return Response(buf.getvalue(), mimetype="audio/wav", headers={"Content-Disposition": f"attachment; filename={m.token}_{t0:.2f}-{t1:.2f}.wav"})
        ff = shutil.which("ffmpeg")
        fr = m.frames[i0:i1 + 1]
        W, H = m.w * scale, m.h * scale
        out_dir = S.out_dir / m.token
        out_dir.mkdir(parents=True, exist_ok=True)
        name = f"{m.token}_{t0:.2f}-{t1:.2f}"
        if ff:
            out = out_dir / f"{name}.mp4"
            wav_tmp = None
            cmd = [ff, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{W}x{H}", "-r", f"{m.fps:.4f}", "-i", "-"]
            if m.audio is not None:
                a = int(max(0, (m.frame_times[i0] - m.av_offset) * m.fs)); b = int((m.frame_times[i1] + 1 / m.fps - m.av_offset) * m.fs)
                wav_tmp = S.cache_dir / f"{uuid.uuid4().hex}.wav"
                wavfile.write(str(wav_tmp), m.fs, (np.clip(m.audio[a:b], -1, 1) * 32767).astype(np.int16))
                cmd += ["-i", str(wav_tmp), "-c:a", "aac", "-shortest"]
            cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(out)]
            p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            for f in fr:
                p.stdin.write(cv2.resize(f, (W, H), interpolation=cv2.INTER_CUBIC).tobytes())
            p.stdin.close(); p.wait()
            if p.returncode != 0:
                return J({"error": "ffmpeg failed"}, 500)
            return send_file(out, as_attachment=True, download_name=out.name)
        out = out_dir / f"{name}.avi"
        vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"MJPG"), m.fps, (W, H), False)
        for f in fr:
            vw.write(cv2.resize(f, (W, H), interpolation=cv2.INTER_CUBIC))
        vw.release()
        return send_file(out, as_attachment=True, download_name=out.name)

    @app.post("/api/save_session")
    def save_session():
        need()
        S.save()
        return J({"path": str(S.session_path())})

    app.S = S  # for tests
    return app


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="rtmri_web", description="Browser-based rtMRI + audio inspector")
    ap.add_argument("path", nargs="?", default=None, help="optional: root folder containing mp4/ and wav/ sub-folders (if omitted, pick the folder in the browser; otherwise any folder of videos, searched recursively) or a single video file")
    ap.add_argument("--wav", help="audio file to pair with a single video file")
    ap.add_argument("--out", default="rtmri_output", help="where sessions, clips and exports are written")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args(argv)
    path = Path(a.path).expanduser() if a.path else None
    app = create_app(path, Path(a.out).expanduser())
    S = app.S
    first = None
    if path is not None and path.is_file():
        first = {"video": str(path), "wav": a.wav}
        if not a.wav:
            f = discover(path)
            first["wav"] = f[0]["wav"] if f else None
    if first:
        try:
            S.open(first["video"], first["wav"])
        except Exception as e:
            print("Could not open:", e)
    url = f"http://{a.host}:{a.port}/"
    print(f"\n  rtMRI Inspector running at {url}\n  (Ctrl+C to stop)\n")
    if not a.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    app.run(host=a.host, port=a.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
