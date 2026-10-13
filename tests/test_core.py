import numpy as np
from scipy import signal
from rtmri_web import acoustics as ac

def test_tube_uniform():
    f = ac.tube_formants(np.full(35, 5.0), np.full(35, 0.5))
    assert np.allclose(f, [500, 1500, 2500, 3500], atol=5)

def test_formants_synthetic():
    fs = 16000; n = fs * 2
    src = np.zeros(n); src[::145] = 1
    y = src
    for f, b in [(700, 90), (1200, 110), (2600, 160), (3500, 200)]:
        r = np.exp(-np.pi * b / fs); th = 2 * np.pi * f / fs
        y = signal.lfilter([1], [1, -2 * r * np.cos(th), r * r], y)
    y = (y / abs(y).max()).astype(np.float32)
    r = ac.track_formants(y, fs, 0.3, 1.8)
    med = np.nanmedian(r["F"], axis=0)
    assert np.allclose(med, [700, 1200, 2600, 3500], rtol=0.12)
