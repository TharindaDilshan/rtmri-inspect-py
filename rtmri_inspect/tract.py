"""Vocal-tract analysis grid, airway centre-line, tissue boundaries and area function.

A Python port of the method in ``inspect_rtmri.m`` (set_grid / get_grid_profiles /
find_tissue_bnds / extract_AF_frame), with these deliberate differences:

* Everything is done in native image pixels with sub-pixel sampling (no 2.3x pre-scaling).
* Each frame is contrast-normalised before sampling, so the intensity thresholds behave the
  same on bright and dark recordings.
* The centre-line is the cheapest path through all gridlines (dynamic programming) and the
  end node is chosen by total cost, not "the first minimum on the last gridline".
* Boundaries are located by walking outward from the centre-line until the profile crosses the
  threshold, with linear interpolation between samples.
* Optional rigid registration (phase correlation) to follow head motion.

Conventions: image coordinates, x to the right, y down, pixel centres at integer coordinates.
Gridline g runs from its *inner* end (tongue / lower lip side) to its *outer* end
(palate / pharyngeal wall side). Positions along a gridline are parameter t in [0, 1].
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from math import atan2, cos, sin, pi, floor, ceil

import cv2
import numpy as np
from scipy import ndimage
from scipy.fft import dct, idct
from scipy.signal import argrelextrema

from .acoustics import tube_formants

DISPLAY_REF_W = 340.0  # width of the MATLAB display window the original weights were tuned in


# --------------------------------------------------------------------------- parameters
@dataclass
class TractParams:
    wid: float                      # gridline length (px)
    spacing: float                  # distance between gridlines along the tract (px)
    ithrsh: float = 0.55            # boundary threshold: fraction of the way from airway to tissue intensity
    diwtcl: float = 0.15            # centre-line weight: intensity (vs distance) in the path cost
    smooth_sigma: float = 0.8       # spatial smoothing of the frame before sampling (px)
    use_ref: bool = True            # use palate / pharyngeal wall from the reference frame
    use_shift: bool = False         # follow head motion by registering each frame to the reference frame
    recenter: bool = False          # recompute centre-line as midpoint between boundaries
    smooth_tongue: bool = False     # DCT-smooth the tongue surface (gridlines in tng range)
    dct_keep: float = 0.99          # fraction of DCT energy kept when smoothing
    pixel_mm: float = 1.0           # mm per native pixel
    area_alpha: float = 1.5         # area (cm^2) = alpha * d(cm)^beta
    area_beta: float = 1.4
    pha: list[int] | None = None    # [first, last] gridline index of pharyngeal-wall reference contour
    pal: list[int] | None = None    # palate reference contour
    tng: list[int] | None = None    # tongue (for smoothing)

    @staticmethod
    def defaults_for(w: int, pixel_mm: float) -> "TractParams":
        return TractParams(wid=round(60 * w / DISPLAY_REF_W, 1), spacing=round(8 * w / DISPLAY_REF_W, 2), pixel_mm=pixel_mm)

    def update(self, d: dict):
        for k, v in d.items():
            if hasattr(self, k) and v is not None:
                setattr(self, k, type(getattr(self, k))(v) if not isinstance(getattr(self, k), (list, type(None))) else v)
        return self


# --------------------------------------------------------------------------- grid
@dataclass
class Grid:
    ends: np.ndarray                # (N, 2, 2): [gridline, inner/outer, x/y]
    n_pharynx: int                  # number of horizontal pharyngeal lines
    pha: list[int]
    pal: list[int]
    tng: list[int]
    or1: tuple
    or2: tuple
    rad1: float
    rad2: float
    theta: float
    notes: list[str] = field(default_factory=list)

    @property
    def n(self): return int(self.ends.shape[0])

    def lengths(self):
        return np.linalg.norm(self.ends[:, 1] - self.ends[:, 0], axis=1)

    def to_json(self):
        return {"ends": self.ends.round(3).tolist(), "n_pharynx": self.n_pharynx, "pha": self.pha, "pal": self.pal, "tng": self.tng,
                "or1": list(self.or1), "or2": list(self.or2), "rad1": self.rad1, "rad2": self.rad2, "notes": self.notes}


def build_grid(lm: dict, wid: float, spacing: float, w: int, h: int) -> Grid:
    """Semi-polar (Ohman-style) grid from four landmarks.

    lm: glot, mpal (highest palate point), dent (alveolar ridge), mlab (lip limit); each [x, y] in px.
    The face is assumed to point to the LEFT (use the 'flip image' option otherwise).
    """
    notes = []
    gx, gy = lm["glot"]; px, py = lm["mpal"]; ax, ay = lm["dent"]; lx, ly = lm["mlab"]
    rad1 = gx - px
    if rad1 < 4 * spacing:
        raise ValueError("Glottis must be clearly to the right of the highest palate point "
                         "(is the image flipped? see 'Face points right' in settings).")
    cx = float(round(px)); cy = float(round(py + rad1 + wid / 6.0))
    theta = atan2(cy - ay, cx - ax)
    if not (np.deg2rad(5) < theta < np.deg2rad(85)):
        theta = float(np.clip(theta, np.deg2rad(5), np.deg2rad(85)))
        notes.append("Alveolar ridge point is unusual relative to the grid origin; angle was clamped.")
    RR = rad1 + wid / 2 + wid / 4
    ax2 = cx - RR * cos(theta); ay2 = cy - RR * sin(theta)
    rad2 = float(np.hypot(cx - ax2, cy - ay2) - rad1)
    ends = []

    # 1. horizontal lines through the pharynx
    r, R = rad1 - wid / 2, rad1 + wid / 2
    yy = gy - spacing
    last_yy = None
    while yy >= cy:
        ends.append([[cx + r, yy], [cx + R, yy]])
        last_yy = yy
        yy -= spacing
    n_h = len(ends)
    rem = (last_yy - cy) if last_yy is not None else 0.0

    # 2. polar rays about origin 1 (lingual)
    dth = spacing / rad1
    th = (spacing - rem) / rad1 if n_h else dth
    tht = pi - theta
    pal1 = None
    ix_vel = max(n_h - 1, 0)
    while th <= tht:
        x1, y1 = cx + r * cos(th), cy - r * sin(th)
        x2, y2 = cx + R * cos(th), cy - R * sin(th)
        ends.append([[x1, y1], [x2, y2]])
        if th < np.deg2rad(20):
            ix_vel = len(ends) - 1
        if x2 < px and pal1 is None:
            pal1 = max(len(ends) - 2, 0)
        th += dth
    n_p1 = len(ends)

    # 3. rays about origin 2 (alveolar ridge), sweeping from "toward origin 1" to straight down
    tx, ty = ax2, ay2
    r2, R2 = rad2 - wid / 2, rad2 + wid / 2
    dth2 = spacing / max(rad2, 1e-6)
    th = theta + dth2
    while th <= pi / 2 + 1e-9:
        x1, y1 = tx + r2 * cos(th), ty + r2 * sin(th)
        x2, y2 = tx + R2 * cos(th), ty + R2 * sin(th)
        ends.append([[x2, y2], [x1, y1]])
        th += dth2
    n_p2 = len(ends)

    # 4. vertical lines through the front of the oral cavity, out past the lips
    ym_lo, ym_hi = ty + R2, ty + r2
    xmn = max(lx - 4 * spacing, spacing)
    xx = tx - spacing
    while xx >= xmn:
        ends.append([[xx, ym_lo], [xx, ym_hi]])
        xx -= spacing

    E = np.array(ends, float)
    N = len(E)
    if pal1 is None:
        pal1 = n_p1 - 1
    pha = [max(n_h // 2 - 1, 0), ix_vel]
    pal = [min(pal1, N - 1), min(n_p2 + 3, N - 1)]
    tng = [min(n_h // 2, N - 1), min(n_p2 + 3, N - 1)]
    # clip to the image (profiles are sampled with nearest-edge padding, but warn if the grid leaves the frame)
    if (E[..., 0] < -1).any() or (E[..., 0] > w).any() or (E[..., 1] < -1).any() or (E[..., 1] > h).any():
        notes.append("Part of the grid lies outside the image.")
    return Grid(E, n_h, pha, pal, tng, (cx, cy), (float(ax2), float(ay2)), float(rad1), rad2, float(theta), notes)


# --------------------------------------------------------------------------- per-frame processing
def normalise_frame(img: np.ndarray, sigma: float, gain: np.ndarray | None = None) -> np.ndarray:
    """Float image, optionally flattened by ``gain``, percentile-normalised to 0..255, lightly smoothed."""
    f = img.astype(np.float32)
    if gain is not None:
        f = f / gain
    lo, hi = np.percentile(f, [1.0, 99.5])
    f = np.clip((f - lo) / max(hi - lo, 1e-6), 0, 1) * 255.0
    if sigma > 0:
        f = cv2.GaussianBlur(f, (0, 0), sigma)
    return f


def estimate_shift(ref: np.ndarray, img: np.ndarray, max_shift: float = 6.0) -> tuple[float, float]:
    """Sub-pixel translation of ``img`` relative to ``ref`` (phase correlation, Hann-windowed)."""
    win = cv2.createHanningWindow((ref.shape[1], ref.shape[0]), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(ref.astype(np.float32), img.astype(np.float32), win)
    if resp < 0.03 or abs(dx) > max_shift or abs(dy) > max_shift:
        return 0.0, 0.0
    return float(dx), float(dy)


class Profiles:
    """Sampling coordinates for a grid (computed once per grid)."""
    def __init__(self, grid: Grid):
        L = grid.lengths()
        self.npts = int(max(24, ceil(np.max(L) * 2)))
        t = np.linspace(0, 1, self.npts)
        e = grid.ends
        self.xs = e[:, 0, 0][:, None] + (e[:, 1, 0] - e[:, 0, 0])[:, None] * t[None, :]
        self.ys = e[:, 0, 1][:, None] + (e[:, 1, 1] - e[:, 0, 1])[:, None] * t[None, :]
        self.t = t
        self.L = L

    def sample(self, img: np.ndarray, dx=0.0, dy=0.0) -> np.ndarray:
        coords = np.vstack([(self.ys + dy).ravel(), (self.xs + dx).ravel()])
        v = ndimage.map_coordinates(img, coords, order=1, mode="nearest")
        return v.reshape(self.xs.shape)


def _local_minima(p: np.ndarray) -> np.ndarray:
    idx = argrelextrema(p, np.less_equal, order=1)[0]
    idx = idx[(idx > 1) & (idx < len(p) - 2)]
    if len(idx) > 1:  # collapse plateaus
        keep = [idx[0]]
        for i in idx[1:]:
            if i - keep[-1] > 1:
                keep.append(i)
        idx = np.array(keep)
    if len(idx) == 0:
        idx = np.array([int(np.argmin(p[2:-2])) + 2])
    return idx


def _tissue_score(p: np.ndarray) -> float:
    return float(np.percentile(p, 90))


def _walk(p: np.ndarray, c: int, itb: float, direction: int) -> float:
    """Walk from index c outward until the profile reaches ``itb``; return fractional index."""
    n = len(p)
    j = c
    while 0 <= j + direction < n:
        nxt = j + direction
        if p[nxt] >= itb:
            a, b = p[j], p[nxt]
            frac = 0.0 if b == a else float(np.clip((itb - a) / (b - a), 0, 1))
            return j + direction * frac
        j = nxt
    return float(j)


@dataclass
class FrameResult:
    ngl: int
    cen_t: np.ndarray               # (ngl,) centre parameter on each gridline
    lf_t: np.ndarray
    rt_t: np.ndarray
    cen: np.ndarray                 # (ngl, 2) px
    lf: np.ndarray
    rt: np.ndarray
    aperture_mm: np.ndarray
    x_mm: np.ndarray                # distance from glottis along the centre-line (mm)
    area_cm2: np.ndarray
    seg_len_cm: np.ndarray
    shift: tuple
    vt_len_mm: float
    mid_area_mm2: float
    edited: list[int]

    def to_json(self):
        r = lambda a, k=3: np.round(a, k).tolist()
        return {"ngl": self.ngl, "cen": r(self.cen), "lf": r(self.lf), "rt": r(self.rt), "aperture_mm": r(self.aperture_mm, 2),
                "x_mm": r(self.x_mm, 2), "area_cm2": r(self.area_cm2, 3), "shift": list(np.round(self.shift, 2)),
                "vt_len_mm": round(self.vt_len_mm, 2), "mid_area_mm2": round(self.mid_area_mm2, 1), "edited": self.edited}


def find_airway(prof: np.ndarray, grid: Grid, P: Profiles, glot: np.ndarray, p: TractParams, to_disp: float, dx=0.0, dy=0.0):
    """Return (ngl, centre index per gridline) for one frame.

    ``to_disp`` converts native pixels to the 340-px display units in which the original
    intensity-vs-distance weighting (``diwtcl``) was tuned."""
    N = grid.n
    # --- lip limit: last gridline that still contains tissue
    score = np.array([_tissue_score(prof[g]) for g in range(N)])
    bg = float(np.median(score[-4:]))
    st = float(np.median(score[: max(int(0.7 * N), 5)]))
    th = bg + 0.45 * (st - bg)
    above = np.where(score > th)[0]
    ngl = int(above.max()) + 1 if len(above) else N
    ngl = max(ngl, 8)

    # --- centre-line: dynamic programming through intensity minima
    a = p.diwtcl
    prev_xy = (glot + np.array([dx, dy]))[None, :]
    prev_cost = np.zeros(1)
    layers = []
    back = []
    for g in range(ngl):
        pg = prof[g]
        idx = _local_minima(pg)
        xy = np.stack([P.xs[g, idx] + dx, P.ys[g, idx] + dy], axis=1)
        d = np.linalg.norm(prev_xy[:, None, :] - xy[None, :, :], axis=2) * to_disp
        w = a * pg[idx][None, :] + (1 - a) * d
        tot = prev_cost[:, None] + w
        bp = tot.argmin(axis=0)
        cost = tot[bp, np.arange(len(idx))]
        layers.append(idx)
        back.append(bp)
        prev_xy, prev_cost = xy, cost
    k = int(np.argmin(prev_cost))
    path = np.zeros(ngl, int)
    for g in range(ngl - 1, -1, -1):
        path[g] = layers[g][k]
        k = back[g][k]
    return ngl, path


def find_boundaries(prof: np.ndarray, ngl: int, path: np.ndarray, ithrsh: float):
    lf = np.zeros(ngl); rt = np.zeros(ngl)
    for g in range(ngl):
        pg = prof[g]; c = int(path[g])
        v = pg[c]
        rmx = pg[c + 1:].max() if c + 1 < len(pg) else v
        lmx = pg[:c].max() if c > 0 else v
        imx = min(lmx, rmx)
        itb = v + (imx - v) * ithrsh
        rt[g] = _walk(pg, c, itb, +1)
        lf[g] = _walk(pg, c, itb, -1)
    return lf, rt


def smooth_tongue_surface(lf_idx: np.ndarray, rng: list[int], keep: float):
    """DCT low-pass of the tongue-surface position along the tongue gridlines (like smooth_ibnd.m)."""
    a, b = rng
    seg = lf_idx[a:b + 1].astype(float)
    if len(seg) < 6:
        return lf_idx
    c = dct(seg, norm="ortho")
    order = np.argsort(-np.abs(c))
    cum = np.sqrt(np.cumsum(c[order] ** 2) / np.sum(c ** 2))
    k = int(np.searchsorted(cum, keep)) + 1
    mask = np.zeros_like(c, bool)
    mask[order[:k]] = True
    out = lf_idx.copy()
    out[a:b + 1] = idct(np.where(mask, c, 0), norm="ortho")
    return out


class TractAnalyzer:
    """Holds the grid, reference contours and manual edits; extracts the area function from frames."""

    def __init__(self, w: int, h: int, pixel_mm: float):
        self.w, self.h = w, h
        self.p = TractParams.defaults_for(w, pixel_mm)
        self.landmarks: dict = {}
        self.grid: Grid | None = None
        self.P: Profiles | None = None
        self.ref_frame: int | None = None
        self.ref_rt: dict[int, float] = {}       # gridline -> outer boundary param t (palate / pharynx)
        self.ref_img: np.ndarray | None = None
        self.edits: dict[int, dict[int, dict[str, float]]] = {}   # frame -> gl -> {'lf': t, 'rt': t}
        self.cache: dict[int, FrameResult] = {}

    # ---- configuration
    def set_landmarks(self, lm: dict):
        self.landmarks = {k: [float(v[0]), float(v[1])] for k, v in lm.items()}
        self.reset_ranges()
        self.rebuild()

    def rebuild(self):
        self.cache.clear()
        if all(k in self.landmarks for k in ("glot", "mpal", "dent", "mlab")):
            self.grid = build_grid(self.landmarks, self.p.wid, self.p.spacing, self.w, self.h)
            self.P = Profiles(self.grid)
            # default reference ranges unless the user set them
            if not self.p.pha: self.p.pha = list(self.grid.pha)
            if not self.p.pal: self.p.pal = list(self.grid.pal)
            if not self.p.tng: self.p.tng = list(self.grid.tng)
            self.grid.pha, self.grid.pal, self.grid.tng = list(self.p.pha), list(self.p.pal), list(self.p.tng)
            self.ref_rt = {}
            self.ref_frame = None
        else:
            self.grid = None
            self.P = None

    def reset_ranges(self):
        self.p.pha = self.p.pal = self.p.tng = None

    def set_params(self, d: dict):
        old = (self.p.wid, self.p.spacing, tuple(self.p.pha or []), tuple(self.p.pal or []), tuple(self.p.tng or []))
        self.p.update(d)
        new = (self.p.wid, self.p.spacing, tuple(self.p.pha or []), tuple(self.p.pal or []), tuple(self.p.tng or []))
        if old[:2] != new[:2]:
            self.reset_ranges_if_geometry()
            self.rebuild()
        elif old != new and self.grid is not None:
            self.grid.pha, self.grid.pal, self.grid.tng = list(self.p.pha), list(self.p.pal), list(self.p.tng)
            self.cache.clear()
        else:
            self.cache.clear()

    def reset_ranges_if_geometry(self):
        self.p.pha = self.p.pal = self.p.tng = None

    # ---- extraction
    def _prepared(self, img, gain):
        return normalise_frame(img, self.p.smooth_sigma, gain)

    def extract(self, f: int, img: np.ndarray, gain: np.ndarray | None = None, use_cache: bool = True) -> FrameResult:
        if self.grid is None:
            raise ValueError("Place the four landmarks and build the grid first.")
        if use_cache and f in self.cache:
            return self.cache[f]
        p, grid, P = self.p, self.grid, self.P
        im = self._prepared(img, gain)
        dx = dy = 0.0
        if p.use_shift and self.ref_img is not None and f != self.ref_frame:
            dx, dy = estimate_shift(self.ref_img, im)
        prof = P.sample(im, dx, dy)
        glot = np.array(self.landmarks["glot"], float)
        ngl, path = find_airway(prof, grid, P, glot, p, DISPLAY_REF_W / self.w, dx, dy)
        lf_i, rt_i = find_boundaries(prof, ngl, path, p.ithrsh)
        last = P.npts - 1
        if p.smooth_tongue:
            lf_i = smooth_tongue_surface(lf_i, [min(grid.tng[0], ngl - 1), min(grid.tng[1], ngl - 1)], p.dct_keep)
        t_lf = np.clip(lf_i / last, 0, 1); t_rt = np.clip(rt_i / last, 0, 1); t_c = path / last
        # reference palate / pharynx
        if p.use_ref and self.ref_rt:
            for g in range(ngl):
                if (grid.pal[0] <= g <= grid.pal[1] or grid.pha[0] <= g <= grid.pha[1]) and g in self.ref_rt:
                    t_rt[g] = self.ref_rt[g]
        # manual edits win
        edited = []
        for g, e in self.edits.get(f, {}).items():
            if g < ngl:
                if "lf" in e: t_lf[g] = e["lf"]
                if "rt" in e: t_rt[g] = e["rt"]
                edited.append(int(g))
        t_lf = np.minimum(t_lf, t_rt)             # lf cannot pass rt: closure -> zero aperture
        t_c = np.clip(t_c, t_lf, t_rt) if not p.recenter else 0.5 * (t_lf + t_rt)
        res = self._pack(f, ngl, t_c, t_lf, t_rt, (dx, dy), edited)
        self.cache[f] = res
        return res

    def _pack(self, f, ngl, t_c, t_lf, t_rt, shift, edited) -> FrameResult:
        P, p = self.P, self.p
        g = np.arange(ngl)
        e = self.grid.ends[:ngl]
        pt = lambda t: np.stack([e[:, 0, 0] + (e[:, 1, 0] - e[:, 0, 0]) * t + shift[0],
                                 e[:, 0, 1] + (e[:, 1, 1] - e[:, 0, 1]) * t + shift[1]], axis=1)
        cen, lf, rt = pt(t_c), pt(t_lf), pt(t_rt)
        L = P.L[:ngl]
        ap = (t_rt - t_lf) * L * p.pixel_mm
        glot = np.array(self.landmarks["glot"], float) + np.array(shift)
        seg = np.linalg.norm(np.diff(np.vstack([glot, cen]), axis=0), axis=1) * p.pixel_mm
        x = np.cumsum(seg)
        d_cm = ap / 10.0
        area = p.area_alpha * np.maximum(d_cm, 0) ** p.area_beta
        mid = float(np.sum(ap[:-1] * np.diff(x))) if ngl > 1 else 0.0
        return FrameResult(ngl, t_c, t_lf, t_rt, cen, lf, rt, ap, x, area, seg / 10.0, shift, float(x[-1]), mid, edited)

    # ---- reference contours
    def set_reference(self, f: int, img: np.ndarray, gain=None):
        """Use frame ``f`` as the source of the palate and pharyngeal-wall contours (static anatomy)."""
        saved = self.p.use_ref
        self.p.use_ref = False
        self.cache.pop(f, None)
        res = self.extract(f, img, gain, use_cache=False)
        self.p.use_ref = saved
        self.ref_frame = f
        self.ref_img = self._prepared(img, gain)
        self.ref_rt = {int(g): float(res.rt_t[g]) for g in range(res.ngl)
                       if self.grid.pal[0] <= g <= self.grid.pal[1] or self.grid.pha[0] <= g <= self.grid.pha[1]}
        self.cache.clear()

    def set_shift_reference(self, f, img, gain=None):
        self.ref_frame = f
        self.ref_img = self._prepared(img, gain)

    # ---- manual editing
    def edit_point(self, f: int, x: float, y: float, side: str, shift=(0, 0)) -> int:
        """Move the ``side`` ('lf' or 'rt') boundary on the gridline nearest to (x, y) to the projection of (x, y)."""
        e = self.grid.ends
        a = e[:, 0, :] + np.array(shift); b = e[:, 1, :] + np.array(shift)
        ab = b - a
        t = np.clip(np.einsum("ij,ij->i", np.array([x, y]) - a, ab) / np.einsum("ij,ij->i", ab, ab), 0, 1)
        proj = a + ab * t[:, None]
        d = np.linalg.norm(proj - np.array([x, y]), axis=1)
        g = int(np.argmin(d))
        self.edits.setdefault(f, {}).setdefault(g, {})[side] = float(t[g])
        self.cache.pop(f, None)
        return g

    def clear_edits(self, f: int):
        self.edits.pop(f, None)
        self.cache.pop(f, None)

    # ---- acoustic check
    def predicted_formants(self, res: FrameResult, n: int = 4) -> list[float]:
        return tube_formants(res.area_cm2, np.maximum(res.seg_len_cm, 0.02), n).round(0).tolist()

    # ---- (de)serialisation
    def state(self) -> dict:
        p = asdict(self.p)
        return {"params": p, "landmarks": self.landmarks, "ref_frame": self.ref_frame, "ref_rt": {str(k): v for k, v in self.ref_rt.items()},
                "edits": {str(f): {str(g): e for g, e in d.items()} for f, d in self.edits.items()}}

    def load_state(self, s: dict, ref_img_loader=None):
        if not s:
            return
        pp = s.get("params", {})
        self.p.update({k: v for k, v in pp.items() if k not in ("pha", "pal", "tng")})
        self.p.pha, self.p.pal, self.p.tng = pp.get("pha"), pp.get("pal"), pp.get("tng")
        lm = s.get("landmarks", {})
        if lm:
            keep = (self.p.pha, self.p.pal, self.p.tng)
            self.landmarks = lm
            self.grid = build_grid(lm, self.p.wid, self.p.spacing, self.w, self.h)
            self.P = Profiles(self.grid)
            self.p.pha, self.p.pal, self.p.tng = [k or d for k, d in zip(keep, (self.grid.pha, self.grid.pal, self.grid.tng))]
            self.grid.pha, self.grid.pal, self.grid.tng = list(self.p.pha), list(self.p.pal), list(self.p.tng)
        self.ref_frame = s.get("ref_frame")
        self.ref_rt = {int(k): float(v) for k, v in s.get("ref_rt", {}).items()}
        self.edits = {int(f): {int(g): e for g, e in d.items()} for f, d in s.get("edits", {}).items()}
        self.cache.clear()
        if ref_img_loader and self.ref_frame is not None:
            self.ref_img = ref_img_loader(self.ref_frame)
