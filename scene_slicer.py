#!/usr/bin/env python3
"""
scene_slicer.py
Download a YouTube video and split it into scenes.

Usage:
  python scene_slicer.py "https://www.youtube.com/watch?v=XXXX" -o outputs/run --threshold 27
"""
import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional

from tqdm import tqdm

# --- PySceneDetect ---
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector, AdaptiveDetector, ThresholdDetector
from scenedetect.video_splitter import split_video_ffmpeg

# --- yt-dlp ---
import yt_dlp

def sanitize(text: str) -> str:
    text = re.sub(r"[^\w\-\.]+", "_", text.strip())
    return text[:100] or "video"

def download_youtube(url: str, tmpdir: Path, cookies_from_browser: Optional[str] = None) -> Tuple[Path, str]:
    """
    Download the best merged file (video+audio) using yt-dlp, with robustness against
    YouTube SABR/nsig changes by trying multiple player clients and optional browser cookies.
    Returns (filepath, video_id).
    """
    # Try a few client profiles that are less likely to hit SABR-restricted formats.
    client_candidates = ["android", "tvhtml5", "web"]
    last_err = None

    # Optional cookies-from-browser (e.g., "chrome", "edge", "firefox" or "chrome:Profile 1")
    cfb = None
    if cookies_from_browser:
        parts = [s.strip() for s in cookies_from_browser.split(":", 1)]
        cfb = tuple(parts) if len(parts) == 2 else (parts[0],)

    for client in client_candidates:
        ydl_opts = {
            "outtmpl": str(tmpdir / "%(id)s.%(ext)s"),
            "format": "bv*+ba/b",   # best video+audio, else best single
            "merge_output_format": "mp4",
            "noprogress": True,
            "quiet": True,
            "restrictfilenames": True,
            "retries": 10,
            "concurrent_fragment_downloads": 5,
            "http_headers": {"User-Agent": "Mozilla/5.0"},
            "extractor_args": {"youtube": {"player_client": [client]}},
        }
        if cfb:
            ydl_opts["cookiesfrombrowser"] = cfb

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                vid_id = info.get("id", "video")
                ext = info.get("ext", "mp4")
                file_path = tmpdir / f"{vid_id}.{ext}"
                if not file_path.exists():
                    # sometimes yt-dlp remuxes to mp4
                    alt = tmpdir / f"{vid_id}.mp4"
                    file_path = alt if alt.exists() else file_path
                return file_path, vid_id
        except Exception as e:
            last_err = e
            continue  # try the next client

    # If all clients failed, raise the last error for visibility
    raise last_err


def human_timecode(seconds: float) -> str:
    ms = int((seconds - int(seconds)) * 1000)
    s = int(seconds) % 60
    m = (int(seconds) // 60) % 60
    h = int(seconds) // 3600
    return f"{h:02d}h{m:02d}m{s:02d}s{ms:03d}"

def detect_scenes(video_path: Path, detector: str, threshold: float, min_scene_len: float, start: float, max_duration: float):
    video = open_video(str(video_path))
    sm = SceneManager()
    # Convert seconds to frames for min_scene_len if needed (PySceneDetect handles either but ensure float seconds ok)
    if detector == "content":
        det = ContentDetector(threshold=threshold, min_scene_len=int(max(2, min_scene_len * video.base_timecode.framerate)))
    elif detector == "adaptive":
        det = AdaptiveDetector(adaptive_threshold=threshold, min_scene_len=int(max(2, min_scene_len * video.base_timecode.framerate)))
    elif detector == "histogram":
        # Histogram via ThresholdDetector (brightness threshold) isn't the same as color histogram diff,
        # but included as an alternate static threshold approach.
        det = ThresholdDetector(threshold=threshold, min_scene_len=int(max(2, min_scene_len * video.base_timecode.framerate)))
    else:
        raise ValueError("Unknown detector: " + detector)

    sm.add_detector(det)

    # Set time range
    if start > 0 or max_duration > 0:
        start_frame = int(start * video.base_timecode.framerate)
        end_frame = None if max_duration <= 0 else int((start + max_duration) * video.base_timecode.framerate)
        sm.detect_scenes(video=video, start_time=video.base_timecode[start_frame], end_time=(video.base_timecode[end_frame] if end_frame else None))
    else:
        sm.detect_scenes(video=video)

    scene_list = sm.get_scene_list()
    return scene_list, video



def export_scenes_ffmpeg(video_path: Path, scenes, out_dir: Path, prefix="scene_", dry_run=False):
    """
    Export scenes by invoking ffmpeg directly. This avoids API/template differences across
    PySceneDetect versions and guarantees filenames match the manifest.
    Returns a list of output file names (basename only).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    if dry_run:
        return outputs

    import subprocess

    def tc_seconds(tc):
        return tc.get_seconds() if hasattr(tc, "get_seconds") else float(tc)

    for idx, (start_tc, end_tc) in enumerate(scenes, start=1):
        start_sec = tc_seconds(start_tc)
        end_sec = tc_seconds(end_tc)
        duration = max(0.001, end_sec - start_sec)
        out_name = f"{prefix}{idx:04d}__{human_timecode(start_sec)}__{human_timecode(end_sec)}.mp4"
        out_file = out_dir / out_name

        # -ss before -i for fast seek; -to is duration when paired this way
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{start_sec:.3f}",
            "-i", str(video_path),
            "-to", f"{duration:.3f}",
            "-c", "copy",
            str(out_file),
        ]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            # Fallback to re-encode if copy fails (e.g., cuts between non-keyframes)
            cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-ss", f"{start_sec:.3f}",
                "-i", str(video_path),
                "-to", f"{duration:.3f}",
                "-c:v", "libx264",
                "-c:a", "aac",
                "-movflags", "+faststart",
                str(out_file),
            ]
            subprocess.run(cmd, check=True)
        outputs.append(out_name)
    return outputs

def write_manifest(scenes, out_dir: Path, base_prefix="scene_", file_names=None):
    csv_path = out_dir / "scenes.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["scene_number", "start_sec", "end_sec", "duration_sec", "output_file"])
        for i, (start_time, end_time) in enumerate(scenes, start=1):
            start_sec = start_time.get_seconds()
            end_sec = end_time.get_seconds()
            duration = end_sec - start_sec
            pattern_name = f"{base_prefix}{i:04d}__{human_timecode(start_sec)}__{human_timecode(end_sec)}.mp4"
            writer.writerow([i, f"{start_sec:.3f}", f"{end_sec:.3f}", f"{duration:.3f}", pattern_name])
    return csv_path

def ensure_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception:
        print("ERROR: ffmpeg not found on PATH. Please install ffmpeg.", file=sys.stderr)
        sys.exit(2)

def main():
    parser = argparse.ArgumentParser(description="Download a YouTube video and split it into scenes.")
    parser.add_argument("input", help="YouTube URL or local video file path")
    parser.add_argument("-o", "--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--threshold", type=float, default=27.0, help="Detector threshold (higher=fewer cuts). Default 27.")
    parser.add_argument("--min-scene-len", type=float, default=0.6, help="Minimum scene length in seconds. Default 0.6s")
    parser.add_argument("--detector", choices=["content", "adaptive", "histogram"], default="content", help="Scene detector type")
    parser.add_argument("--start", type=float, default=0.0, help="Start offset (seconds)")
    parser.add_argument("--max-duration", type=float, default=0.0, help="Max duration to analyze (0 = full video)")
    parser.add_argument("--dry-run", action="store_true", help="Only detect & print; do not export scenes")
    parser.add_argument("--keep-temp", action="store_true", help="Keep downloaded file in tmp dir")
    parser.add_argument("--cookies-from-browser", type=str, default=None, help="Use browser cookies (e.g., chrome | edge | firefox or chrome:Profile 1)")
    args = parser.parse_args()

    ensure_ffmpeg()

    tmpdir = Path(tempfile.mkdtemp(prefix="scene_slicer_"))
    download_path = None
    video_id = "localfile"

    try:
        input_path = Path(args.input)
        if input_path.exists():
            video_path = input_path
        else:
            print("[1/3] Downloading…")
            video_path, video_id = download_youtube(args.input, tmpdir, cookies_from_browser=args.cookies_from_browser)
            print(f"    saved: {video_path.name}")

        print("[2/3] Detecting scenes…")
        scenes, video = detect_scenes(video_path, args.detector, args.threshold, args.min_scene_len, args.start, args.max_duration)
        print(f"    found {len(scenes)} scenes")

        # Prepare output folders
        base_out = Path(args.output_dir) / video_id
        scenes_dir = base_out / "scenes"
        base_out.mkdir(parents=True, exist_ok=True)

        # Write manifest CSV
        # Export
        print("[3/3] Exporting clips…")
        outputs = export_scenes_ffmpeg(video_path, scenes, scenes_dir, prefix="scene_", dry_run=args.dry_run)

        # Write manifest CSV after export, using actual filenames to ensure a match.
        manifest_csv = write_manifest(scenes, base_out, base_prefix="scene_", file_names=outputs if not args.dry_run else None)
        print(f"    wrote: {manifest_csv}")

        if not args.dry_run:
            print(f"    clips: {scenes_dir}")

        print("Done.")
    finally:
        if not args.keep_temp:
            shutil.rmtree(tmpdir, ignore_errors=True)

if __name__ == "__main__":
    main()
