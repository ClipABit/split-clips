# streamlit_app.py
import streamlit as st
from pathlib import Path
import tempfile
import gc
import time
import pandas as pd
from typing import Optional, Tuple

from scene_slicer import SceneSlicer

st.set_page_config(page_title="Scene Slicer", page_icon="🎬", layout="wide")
st.title("🎬 Scene Slicer")
st.caption("Download a YouTube video (or use a local file) and split it into scenes.")

slicer = SceneSlicer()

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
start = st.sidebar.number_input("Start offset (seconds)", min_value=0.0, value=0.0, step=0.5)
max_duration = st.sidebar.number_input("Max analyze duration (seconds, 0 = full)", min_value=0.0, value=0.0, step=1.0)
do_dry_run = st.sidebar.checkbox("Dry run (no clip export)", value=False)
min_export_len = st.sidebar.number_input(
    "Drop clips shorter than (seconds)",
    min_value=0.0,
    value=30.0,
    step=0.5,
    help="Any detected scene shorter than this will not be exported.",
)

out_root = Path("outputs")
st.sidebar.write("---")
run_btn = st.sidebar.button("▶️ Run Scene Slicing", type="primary")

# Session state
if "last_output_dir" not in st.session_state:
    st.session_state.last_output_dir = None
if "manifest_df" not in st.session_state:
    st.session_state.manifest_df = None
if "video_id" not in st.session_state:
    st.session_state.video_id = None
if "clip_paths" not in st.session_state:
    st.session_state.clip_paths = []

def process_video():
    try:
        slicer.ensure_ffmpeg()
    except Exception as e:
        st.error(f"ffmpeg check failed: {e}")
        return

    with tempfile.TemporaryDirectory(prefix="scene_slicer_ui_") as tmp:
        tmpdir = Path(tmp)
        video = None          # ensure defined in outer scope
        try:
            # Source
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
                video_path, video_id = slicer.download_youtube(url.strip(), tmpdir, cookies_from_browser=cookies_from_browser)
                st.success(f"Downloaded: {video_path.name}")

            st.write("**[2/3] Detecting scenes…**")
            scenes, video = slicer.detect_scenes(
                video_path=video_path,
                detector=detector,
                threshold=threshold,
                start=start,
                max_duration=max_duration,
                min_scene_len=min_export_len
            )
            st.info(f"Detected **{len(scenes)}** raw scenes")

            base_out = out_root / video_id
            scenes_dir = base_out / "scenes"
            base_out.mkdir(parents=True, exist_ok=True)

            st.write("**[3/3] Exporting clips…**")
            outputs, kept_scenes = slicer.export_scenes_ffmpeg(
                video_path,
                scenes,
                scenes_dir,
                prefix="scene_",
                dry_run=do_dry_run,
                min_export_len=min_export_len,
            )

            if not do_dry_run:
                manifest_csv = slicer.write_manifest(kept_scenes, base_out, base_prefix="scene_")

                # Load manifest via pandas
                df = pd.read_csv(manifest_csv)  
                st.session_state.manifest_df = df

            st.session_state.clip_paths = outputs if not do_dry_run else []
            st.session_state.last_output_dir = base_out
            st.session_state.video_id = video_id

            st.success(f"Done! Exported {len(outputs)} scenes ≥ {min_export_len:.2f}s.")
        finally:
            # Best-effort: release scenedetect handles, delete reference, run GC and allow a short delay
            try:
                slicer.close_video(video)
            except Exception as e:
                st.error(f"Error closing video: {e}")
                return
            try:
                # remove Python reference and collect so OS handles get closed
                del video
            except Exception as e:
                st.error(f"Error deleting video reference: {e}")
            gc.collect()
            time.sleep(0.25)

if run_btn:
    with st.spinner("Processing…"):
        process_video()

# --- Results
if (
    st.session_state.manifest_df is not None
    and not do_dry_run
    and st.session_state.get("clip_paths")
):
    st.subheader("Download scenes")

    import zipfile, io
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as z:
        for full_path in st.session_state.clip_paths:
            p = Path(full_path)
            if p.is_file():
                z.write(p, arcname=p.name)

    zip_name = f"{st.session_state.video_id}_scenes.zip"
    st.download_button(
        "⬇️ Download Scenes ZIP",
        data=mem.getvalue(),
        file_name=zip_name,
        mime="application/zip",
    )

    st.caption(f"Output folder: `{st.session_state.last_output_dir}`")
