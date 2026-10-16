# rtMRI Inspect: Quick Guide

A short walkthrough from opening a recording to exporting an area function..

## 1. Start the app

```bash
python -m rtmri_inspect
```

A browser tab opens. Pick the directory with mp4 and wav folders from the file explorer. If a recording has no matching audio, the list says "(no audio)" and the audio tools are hidden. The video still works.

## 2. The screen

| Area | What it is |
|---|---|
| **Video** (left) | Current frame, play controls, frame/time boxes, speed, loop, *Image display* (brightness, contrast, gamma, smoothing, coil-shading flatten) |
| **Audio** (top right) | Overview waveform (whole file), zoomed waveform, spectrogram, all showing the same time window |
| **Tabs** (bottom right) | *Area function*, *Spectrogram & formants*, *Sync & data*, *Labels & export* |

The **red dashed line** is the current time. The video always shows the frame at that time.

## 3. Moving around

- **Click** the waveform or spectrogram: move the cursor; the video follows.
- **Drag** on the waveform or spectrogram: select an interval (shaded). **Double-click** clears it.
- **Mouse wheel** zooms the window around the pointer. **Shift + wheel** pans.
- **Overview**: click or drag the teal box to move the window.
- **Hover** the spectrogram: time and frequency (Hz) appear at the cursor and in the top-right readout.
- Play with **Space**, *▶ window* or *▶ selection*. Step frames with **← / →**.

All shortcuts are listed under the **?** button.

## 4. Check synchronisation first

Open **Sync & data**.

1. Read any warning about audio/video length.
2. Look at a clear event (for example a plosive release or a vowel onset). Does the mouth move when the sound occurs?
3. If not, set **A/V offset** (positive delays the audio) or use the −0.1 / −0.01 / +0.01 / +0.1 buttons.
4. *Estimate from motion vs. audio* gives a hint only. Trust your eyes over it.

Also set the **field of view width (mm)**, or the **pixel size**, to match your acquisition. This scales every length and area. If the face points right, tick **face points right**.

## 5. Spectrogram and formants

- Turn on **formants** in the audio header (or press **f**) to overlay the tracks. Colours are F1 red, F2 amber, F3 green, F4 blue.
- **Spectrogram & formants** tab: window, step, max frequency, dynamic range, level shift, colour map; formant count, ceiling, window, LPC order, bandwidth limit, silence gate, speaker scale.
- Tips: if tracks jump between harmonics, raise the formant window or lower the ceiling (try 5000 for male, 5500–6000 for female). If quiet speech is skipped, lower the silence gate (for example −50 dB). If the start of the recording is loud scanner noise, set *attenuate first (s)* in **Sync & data**.
- Always compare tracks with the spectrogram bands.

## 6. Extract an area function

Open the **Area function** tab.

**Step 1: place landmarks.** Choose a frame with a clearly open vocal tract (for example a low vowel). Click each button and then click the image, in order:

1. **Glottis**: bottom of the airway at the level of the vocal folds.
2. **Highest palate point**: top of the hard palate.
3. **Alveolar ridge**: just behind the upper front teeth.
4. **Lip limit**: mid-lip position, the front end of the tract.

After each click the next landmark arms itself. Click an armed button again (or press **Esc**) to cancel. The grid appears once all four are placed.

**Step 2: look at the result.** On the video you see the grid (red lines), the airway centre-line (yellow), the tongue/lower side boundary (green) and the palate/wall boundary (blue). The plot shows aperture (mm) against distance from the glottis, or area (cm²) if you switch the unit. Hovering the plot highlights the matching gridline on the image. The tract length is shown above the plot; for an adult it should be roughly 160–185 mm. A very different value usually means the pixel size is wrong.

**Step 3: improve it if needed.**

- **Set reference frame**: pick a frame where the palate and pharyngeal wall are clear. Those two (static) boundaries are then reused for every frame, which makes results steadier.
- **Edit boundary**: choose the tongue/lower side or palate/wall side, then click where the boundary should be. The nearest gridline is moved. *Clear edits (this frame)* undoes them.
- **Grid & threshold settings**: gridline spacing (smaller = more gridlines; about 2–2.5 px is the practical limit), boundary threshold, centre-line weight, smoothing, optional head-motion tracking, tongue smoothing, and the area-law constants α and β.
- *Reset grid* starts again with new landmarks.

Landmarks, edits and settings are saved automatically per recording.

**Step 4: export.**

- **This frame**: CSV of the frame currently shown.
- **Export range**: choose selection, window or whole file, a frame step, and either *all gridlines* (one row per frame and gridline) or *one row per frame* (tract length, midsagittal area).

### CSV columns (all gridlines)

| Column | Meaning |
|---|---|
| `frame`, `time_s` | Frame number (1-based) and its time |
| `gridline` | Gridline number, from the glottis (1) towards the lips |
| `distance_from_glottis_mm` | Distance along the centre-line |
| `aperture_mm` | Distance between the two boundaries along the gridline |
| `area_cm2` | α · d^β with d in cm |
| `*_x_px`, `*_y_px` | Centre, tongue-side and wall-side points in image pixels (single-frame export) |

## 7. Tube model and synthesis

Under the plot, the **tube model** line lists the resonances (F1–F4) of the current area function, with the formants measured from the audio at the same time for comparison. This model has no excitation: it only finds where the tube resonates (closed glottis, open lips, no losses).

**Synthesize vowel** plays a source-filter vowel for the current frame: a Rosenberg glottal pulse train at the chosen f0 through the tube model, for the chosen length. **Save wav** downloads it. Differences from the recorded speech are expected (midsagittal-only model, simple losses, nasal tract absent).

## 8. Labels and other exports

In **Labels & export**:

- **Labels**: type a name and click *Add for selection / window*. Click a label in the list to jump to it. CSV export available.
- **Export**: video + audio clip (MP4, needs `ffmpeg`), audio (WAV), spectrogram image (PNG), formant tracks (CSV) for a selection, window or whole file.
- *Frame PNG* (under the video) saves the current frame at 4× size.

Sessions and clips are written to the output folder (`rtmri_output/<recording name>/` by default; change with `--out`).

## 9. Common problems

| Problem | What to do |
|---|---|
| Grid lies off the airway | Landmarks belong to another recording or frame. *Reset grid*, place them again |
| Tract length far from expected | Check field of view / pixel size in **Sync & data** |
| Boundary wrong in a few frames | Use *Edit boundary*; use a reference frame for palate and wall |
| Audio and video out of step | Adjust the **A/V offset** |
| Formant tracks look wrong | Adjust ceiling, window, bandwidth and silence gate; compare with the spectrogram |
| Nothing appears for the area function | A grid needs all four landmarks; the extraction message appears under the plot if it fails |
| Changing gridline spacing cleared my reference | Spacing changes rebuild the grid; set the reference frame again |

## 10. Quick reference

| Key / action | Effect |
|---|---|
| **Space** | Play / pause (to the end of the selection if inside one) |
| **← / →** | Previous / next frame (**Shift** ×10, **Ctrl** ±1 s) |
| **w** / **s** | Play window / selection |
| **f** | Toggle formant overlay |
| **Home / End** | Start / end of the recording |
| **Esc** | Cancel landmark or edit mode, clear selection |
| **?** | Show shortcuts |
