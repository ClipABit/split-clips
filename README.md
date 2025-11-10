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
- Python 3.9+
- `ffmpeg` on PATH (macOS `brew install ffmpeg`, Ubuntu `sudo apt-get install ffmpeg`, Windows via [ffmpeg.org] downloads)

### 1) Install
```bash
python -m venv .venv && source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Run
```bash
python scene_slicer.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ" -o outputs/demo   --threshold 27 --min-scene-len 0.8
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

---

## Repo Structure
```
scene-slicer/
  ├─ scene_slicer.py
  ├─ requirements.txt
  ├─ Dockerfile
  ├─ README.md
  ├─ LICENSE
  └─ .gitignore
```

## Notes
- For noisy music videos, try `--threshold 32` or `--detector adaptive`.
- For very fast cuts, **lower** threshold (e.g., `--threshold 22`) and `--min-scene-len 0.3`.
- You can re-run on a **local file** instead of a URL as well.


---

## Troubleshooting

**YouTube SABR / nsig warnings**  
If you see warnings like *"nsig extraction failed"* or *"YouTube is forcing SABR streaming"*:
- Update yt-dlp to the latest: `pip install -U yt-dlp` (or `yt-dlp -U` if installed globally).
- The tool now tries alternate player clients (`android`, `tvhtml5`, then `web`) to avoid SABR-restricted formats.
- Optionally use your browser cookies to access formats available to your account:
  ```bash
  python scene_slicer.py "URL" --cookies-from-browser chrome
  # or a specific profile:
  python scene_slicer.py "URL" --cookies-from-browser "chrome:Profile 1"
  ```

**`ffmpeg` not found**  
Ensure ffmpeg is installed and on PATH. On Windows, open a new terminal after updating PATH.


---

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
