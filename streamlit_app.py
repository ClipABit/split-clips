# streamlit_app.py
import streamlit as st
from pathlib import Path
import tempfile
import shutil
import pandas as pd
from typing import Optional, Tuple

# Import functions from the CLI module
from scene_slicer import (
    ensure_ffmpeg,
    download_youtube,
    detect_scenes,
    export_scenes_ffmpeg,
    write_manifest,
    human_timecode,
)

st.set_page_config(page_title="Scene Slicer", page_icon="🎬", layout="wide")

st.title("🎬 Scene Slicer")
st.caption("Download a YouTube video (or use a local file) and split it into scenes.")

with st.expander("⚠️ Usage & Terms", expanded=False):
    st.write(
        "- Only download/process videos where you have rights/permission.\n"
        "- Respect YouTube and content providers' Terms of Service and copyright.\n"
    )

# --- Sidebar options ---
st.sidebar.header("Options")
input_mode = st.sidebar.radio("Input Source", ["YouTube URL", "Local File"], index=0)
cookies_from_browser = None
url = ""
uploaded_file = None

if input_mode == "YouTube URL":
    url = st.sidebar.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    cookies_help = "Optionally use cookies from your browser (e.g., 'chrome', 'edge', 'firefox' or 'chrome:Profile 1')."
    cookies_from_browser = st.sidebar.text_input("Cookies from browser (optional)", help=cookies_help, placeholder="chrome")
else:
    uploaded_file = st.sidebar.file_uploader("Upload a local video", type=["mp4", "mkv", "webm", "mov"])

detector = st.sidebar.selectbox("Detector", options=["content", "adaptive", "histogram"], index=0)
threshold = st.sidebar.slider("Threshold (higher = fewer cuts)", min_value=5, max_value=60, value=27, step=1)
min_scene_len = st.sidebar.number_input("Minimum scene length (seconds)", min_value=0.1, max_value=10.0, value=0.6, step=0.1)
start = st.sidebar.number_input("Start offset (seconds)", min_value=0.0, value=0.0, step=0.5)
max_duration = st.sidebar.number_input("Max analyze duration (seconds, 0 = full)", min_value=0.0, value=0.0, step=1.0)
do_dry_run = st.sidebar.checkbox("Dry run (no clip export)", value=False)

out_root = Path("outputs")
st.sidebar.write("---")
run_btn = st.sidebar.button("▶️ Run Scene Slicing", type="primary")

# Session state for results
if "last_output_dir" not in st.session_state:
    st.session_state.last_output_dir = None
if "manifest_df" not in st.session_state:
    st.session_state.manifest_df = None
if "video_id" not in st.session_state:
    st.session_state.video_id = None

def process_video():
    ensure_ffmpeg()
    tmpdir = Path(tempfile.mkdtemp(prefix="scene_slicer_ui_"))
    try:
        # Determine source video
        if input_mode == "Local File":
            if not uploaded_file:
                st.error("Please upload a local video file.")
                return
            tmp_video = tmpdir / uploaded_file.name
            with open(tmp_video, "wb") as f:
                f.write(uploaded_file.read())
            video_path = tmp_video
            video_id = "localfile"
        else:
            if not url.strip():
                st.error("Please enter a YouTube URL.")
                return
            st.write("**[1/3] Downloading…**")
            video_path, video_id = download_youtube(url.strip(), tmpdir, cookies_from_browser=cookies_from_browser)
            st.success(f"Downloaded: {video_path.name}")

        st.write("**[2/3] Detecting scenes…**")
        scenes, video = detect_scenes(video_path, detector, threshold, min_scene_len, start, max_duration)
        st.info(f"Found **{len(scenes)}** scenes")

        base_out = out_root / video_id
        scenes_dir = base_out / "scenes"
        base_out.mkdir(parents=True, exist_ok=True)

        st.write("**[3/3] Exporting clips…**")
        outputs = export_scenes_ffmpeg(video_path, scenes, scenes_dir, prefix="scene_", dry_run=do_dry_run)

        # Write manifest after export using real filenames (if not dry-run)
        manifest_csv = write_manifest(scenes, base_out, base_prefix="scene_", file_names=(outputs if not do_dry_run else None))

        # Load manifest to display
        import csv
        rows = []
        with open(manifest_csv, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == 0:
                    headers = next(csv.reader([line]))
                else:
                    rows.append(next(csv.reader([line])))
        df = pd.DataFrame(rows, columns=["scene_number", "start_sec", "end_sec", "duration_sec", "output_file"])
        st.session_state.manifest_df = df
        st.session_state.last_output_dir = base_out
        st.session_state.video_id = video_id

        st.success("Done!")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

if run_btn:
    with st.spinner("Processing… this can take a moment"):
        process_video()

# --- Results section ---
if st.session_state.manifest_df is not None:
    st.subheader("Scenes")
    st.dataframe(st.session_state.manifest_df, use_container_width=True)

    # Download ZIP of scenes
    if st.session_state.last_output_dir is not None and not do_dry_run:
        scenes_path = st.session_state.last_output_dir / "scenes"
        zip_name = f"{st.session_state.video_id}_scenes.zip"
        zip_path = st.session_state.last_output_dir / zip_name

        if scenes_path.exists():
            # Build zip on demand
            import zipfile, io
            mem = io.BytesIO()
            with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as z:
                for f in sorted(scenes_path.iterdir()):
                    if f.is_file() and f.suffix.lower() in [".mp4", ".mkv", ".webm", ".mov"]:
                        z.write(f, arcname=f.name)
            st.download_button("⬇️ Download Scenes ZIP", data=mem.getvalue(), file_name=zip_name, mime="application/zip")

    # Show where files are saved
    st.caption(f"Output folder: `{st.session_state.last_output_dir}`")
