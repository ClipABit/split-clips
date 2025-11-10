#!/usr/bin/env python3
"""
scene_slicer.py
Download a YouTube video and split it into scenes.

Usage:
  python scene_slicer.py "https://www.youtube.com/watch?v=XXXX" -o outputs --threshold 27 --min-export-len 30
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
    last_err = None
    cfb = None
    if cookies_from_browser:
        parts = [s.strip() for s in cookies_from_browser.split(":", 1)]
        cfb = tuple(parts) if len(parts) == 2 else (parts[0],)

    ydl_opts = {
        "outtmpl": str(tmpdir / "%(id)s.%(ext)s"),
        "format": "bv*+ba/b",
        "merge_output_format": "mp4",
        "quiet": True,
        "retries": 10,
        "concurrent_fragment_downloads": 5,
        "http_headers": {"User-Agent": "Mozilla/5.0"},
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
                alt = tmpdir / f"{vid_id}.mp4"
                file_path = alt if alt.exists() else file_path
            return file_path, vid_id
    except Exception as e:
        last_err = e

    raise last_err


def human_timecode(seconds: float) -> str:
    ms = int((seconds - int(seconds)) * 1000)
    s = int(seconds) % 60
    m = (int(seconds) // 60) % 60
    h = int(seconds) // 3600
    return f"{h:02d}h{m:02d}m{s:02d}s{ms:03d}"


def detect_scenes(
    video_path: Path,
    detector: str,
    threshold: float,
    start: float,
    max_duration: float,
    min_scene_len: float,
):
    video = open_video(str(video_path))

    sm = SceneManager()

    # Minimal internal min_scene_len of 2 frames just to avoid pathological 1-frame cuts.
    min_frames = max(2, int(round(video.base_timecode.framerate * min_scene_len)))

    if detector == "content":
        det = ContentDetector(threshold=threshold, min_scene_len=min_frames)
    elif detector == "adaptive":
        det = AdaptiveDetector(adaptive_threshold=threshold, min_scene_len=min_frames)
    elif detector == "histogram":
        det = ThresholdDetector(threshold=threshold, min_scene_len=min_frames)
    else:
        raise ValueError("Unknown detector: " + detector)

    sm.add_detector(det)

    if start > 0 or max_duration > 0:
        start_frame = int(start * video.base_timecode.framerate)
        end_frame = None if max_duration <= 0 else int((start + max_duration) * video.base_timecode.framerate)
        sm.detect_scenes(
            video=video,
            start_time=video.base_timecode[start_frame],
            end_time=(video.base_timecode[end_frame] if end_frame else None),
        )
    else:
        sm.detect_scenes(video=video)

    scene_list = sm.get_scene_list()
    return scene_list, video


def export_scenes_ffmpeg(
    video_path: Path,
    scenes,
    out_dir: Path,
    prefix: str = "scene_",
    dry_run: bool = False,
    force_reencode: bool = False,
    crf: int = 18,
    preset: str = "slow",
    min_export_len: float = 0.0,  # NEW: skip exporting scenes shorter than this
):
    """
    Export scenes to individual clips using ffmpeg.

    - Skips any scene whose duration < min_export_len (seconds).
    - If force_reencode == False:
        Try stream copy (-c copy) first; fallback to re-encode if needed.
    - If force_reencode == True:
        Always re-encode with libx264/aac.
    - Returns: (outputs, kept_scenes) where kept_scenes only includes exported scenes.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    kept_scenes = []

    if dry_run:
        # In dry-run, we still enforce the filter but don't write files.
        for (start_tc, end_tc) in scenes:
            dur = end_tc.get_seconds() - start_tc.get_seconds()
            if dur + 1e-9 >= min_export_len:
                kept_scenes.append((start_tc, end_tc))
        return outputs, kept_scenes

    def tc_seconds(tc):
        return tc.get_seconds() if hasattr(tc, "get_seconds") else float(tc)

    for idx, (start_tc, end_tc) in enumerate(scenes, start=1):
        start_sec = tc_seconds(start_tc)
        end_sec = tc_seconds(end_tc)
        duration = max(0.001, end_sec - start_sec)

        # NEW: skip short scenes right here
        if duration + 1e-9 < min_export_len:
            continue

        kept_scenes.append((start_tc, end_tc))
        out_name = f"{prefix}{len(kept_scenes):04d}__{human_timecode(start_sec)}__{human_timecode(end_sec)}.mp4"
        out_file = out_dir / out_name

        if force_reencode:
            cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-ss", f"{start_sec:.3f}", "-i", str(video_path),
                "-to", f"{duration:.3f}",
                "-c:v", "libx264", "-preset", str(preset), "-crf", str(crf),
                "-c:a", "aac", "-movflags", "+faststart",
                str(out_file),
            ]
            subprocess.run(cmd, check=True)
        else:
            copy_cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-ss", f"{start_sec:.3f}", "-i", str(video_path),
                "-to", f"{duration:.3f}",
                "-c", "copy", "-movflags", "+faststart",
                str(out_file),
            ]
            result = subprocess.run(copy_cmd)
            if result.returncode != 0 or not out_file.exists() or out_file.stat().st_size == 0:
                if out_file.exists():
                    out_file.unlink(missing_ok=True)
                reenc_cmd = [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-ss", f"{start_sec:.3f}", "-i", str(video_path),
                    "-to", f"{duration:.3f}",
                    "-c:v", "libx264", "-preset", str(preset), "-crf", str(crf),
                    "-c:a", "aac", "-movflags", "+faststart",
                    str(out_file),
                ]
                subprocess.run(reenc_cmd, check=True)

        outputs.append(str(out_file))

    return outputs, kept_scenes


def write_manifest(scenes, out_dir: Path, base_prefix="scene_"):
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
    parser.add_argument("--detector", choices=["content", "adaptive", "histogram"], default="content", help="Scene detector type")
    parser.add_argument("--start", type=float, default=0.0, help="Start offset (seconds)")
    parser.add_argument("--max-duration", type=float, default=0.0, help="Max duration to analyze (0 = full video)")
    parser.add_argument("--dry-run", action="store_true", help="Only detect & print; do not export scenes")
    parser.add_argument("--keep-temp", action="store_true", help="Keep downloaded file in tmp dir")
    parser.add_argument("--cookies-from-browser", type=str, default=None, help="Use browser cookies (e.g., chrome | edge | firefox or chrome:Profile 1)")
    parser.add_argument("--min-export-len", type=float, default=0.0, help="Skip exporting scenes shorter than this many seconds (default: 0).")
    parser.add_argument("--force-reencode", action="store_true", help="Always re-encode clips with x264/aac instead of trying stream copy.")
    parser.add_argument("--downscale", type=int, default=2, help="(Advanced) Downscale factor for detection (2,3,4). Speeds up detection. 0=off.")
    args = parser.parse_args()

    ensure_ffmpeg()

    tmpdir = Path(tempfile.mkdtemp(prefix="scene_slicer_"))
    video_id = "localfile"

    try:
        # 1) Resolve input (local vs YouTube)
        input_path = Path(args.input)
        if input_path.exists():
            video_path = input_path
            video_id = sanitize(input_path.stem)
            print(f"[1/3] Using local file: {video_path}")
        else:
            print("[1/3] Downloading…")
            video_path, video_id = download_youtube(args.input, tmpdir, cookies_from_browser=args.cookies_from_browser)
            print(f"    saved: {video_path.name}")

        # 2) Detect scenes
        print("[2/3] Detecting scenes…")
        downscale = args.downscale if args.downscale and args.downscale > 0 else 0
        scenes, video = detect_scenes(
            video_path=video_path,
            detector=args.detector,
            threshold=args.threshold,
            min_scene_len=args.min_scene_len,
            start=args.start,
            max_duration=args.max_duration,
            downscale_factor=(downscale if downscale else 1),
        )
        print(f"    found {len(scenes)} raw scenes")

        # 3) Prepare output dirs
        base_out = Path(args.output_dir) / video_id
        scenes_dir = base_out / "scenes"
        base_out.mkdir(parents=True, exist_ok=True)

        # 4) Export (skip scenes shorter than min-export-len right here)
        print("[3/3] Exporting clips…")
        clips, kept_scenes = export_scenes_ffmpeg(
            video_path,
            scenes,
            scenes_dir,
            prefix="scene_",
            dry_run=args.dry_run,
            force_reencode=args.force_reencode,
            min_export_len=args.min_export_len,
        )
        print(f"    Exported {len(clips)} clips ≥ {args.min_export_len:.2f}s → {scenes_dir}")

        # 5) Manifest for the kept scenes only
        if not args.dry_run:
            manifest_csv = write_manifest(kept_scenes, base_out, base_prefix="scene_")
            print(f"    Wrote manifest: {manifest_csv}")

    finally:
        if not args.keep_temp:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
