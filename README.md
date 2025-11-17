# split-clips

Split any YouTube video (trailer, music video, movie clip) into **scenes** automatically.

## Features
- Input: **YouTube URL** → downloads with `yt-dlp`.
- Detects scene boundaries with **PySceneDetect** (content detector by default).
- Exports **scene clips** (lossless stream copy) using `ffmpeg` + a **CSV** with timestamps.
- Simple one-file CLI script: `scene_slicer.py`.
- Works locally or in Docker.

> ⚠️ Use only where you have rights/permission to download/process the content. Respect YouTube's Terms of Service and copyright.

---

## Quickstart (local)

### 0) Prereqs
- Python 3.12
- `ffmpeg` on PATH (macOS `brew install ffmpeg`, Ubuntu `sudo apt-get install ffmpeg`, Windows via [ffmpeg.org] downloads)

### 1) Install
```bash
python -m venv .venv && source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Run
```bash
python scene_slicer.py "https://www.youtube.com/watch?v=q7dKevkSFdc" -o outputs --threshold 27 --min-export-len 30
```

This will:
1. Download the video into a temp folder.
2. Detect scenes with PySceneDetect's **ContentDetector** (sensitive to visual changes).
3. Save scene clips to `outputs/<video_id>/scenes/` and a manifest at `outputs/<video_id>/scenes.csv`.

### Common options
```bash
python scene_slicer.py URL -o outputs/run   --threshold 30            # higher = fewer cuts (default 27)
  --min-scene-len 1.0       # seconds; reject ultra-short shots (default 0.6)
  --max-duration 0          # 0 = no limit; else stop after N seconds
  --start 0                 # start offset seconds
  --detector content        # content | adaptive | histogram
  --dry-run                 # just print detected cuts; don't export clips
  --keep-temp               # keep downloaded source file
```

### Output
- `outputs/<video_id>/scenes/scene_0001__00h00m00s000__00h00m04s500.mp4`
- `outputs/<video_id>/scenes.csv` (start, end, duration, file)

---

## Quickstart (Docker)

```bash
docker build -t scene-slicer .
docker run --rm -v "$PWD/outputs:/app/outputs" scene-slicer \
  python scene_slicer.py "YOUTUBE_URL_HERE" -o outputs --threshold 27
```

## Streamlit UI

Launch a simple web UI locally:

```bash
# from the repo root, in your venv
pip install -r requirements.txt
streamlit run streamlit_app.py
```

- Paste a YouTube URL **or** upload a local video.
- Choose detector/threshold options.
- Click **Run Scene Slicing**.
- Download a ZIP of the scene clips when done.

**Tip (YouTube warnings):** If you hit SABR/nsig warnings, try the *Cookies from browser* field (e.g., `chrome` or `chrome:Profile 1`) and make sure `yt-dlp` is up to date.
