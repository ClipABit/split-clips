import csv
import re
import subprocess
from pathlib import Path
from typing import Tuple, Optional

# --- PySceneDetect ---
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector, AdaptiveDetector, ThresholdDetector

# --- yt-dlp ---
import yt_dlp

class SceneSlicer:
    """
    Function library for downloading and slicing Youtube videos into scenes.
    """
    def sanitize(self, text: str) -> str:
        """
        Sanitize text to be safe for filenames.
        """
        text = re.sub(r"[^\w\-\.]+", "_", text.strip())
        return text[:100] or "video"


    def download_youtube(self, url: str, tmpdir: Path, cookies_from_browser: Optional[str] = None) -> Tuple[Path, str]:
        """
        Download the best merged file (video+audio) using yt-dlp, with robustness against
        YouTube SABR/nsig changes by trying multiple player clients and optional browser cookies.
        Returns (filepath, video_id).
        """
        last_err = None     # error from last attempt
        cfb = None          # cookiesfrombrowser tuple
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
            print(f"Download attempt failed: {e}")

        raise last_err


    def human_timecode(self, seconds: float) -> str:
        """
        Convert seconds to more visible timecode
        """
        ms = int((seconds - int(seconds)) * 1000)
        s = int(seconds) % 60
        m = (int(seconds) // 60) % 60
        h = int(seconds) // 3600
        return f"{h:02d}h{m:02d}m{s:02d}s{ms:03d}"


    def detect_scenes(
        self,
        video_path: Path,
        detector: str,
        threshold: float,
        start: float,
        max_duration: float,
        min_scene_len: float,
    ):
        """
        Detect scenes in the video using the specified detector and parameters.
        ContentDetector: detects spot changes using weighted average of pixel changes in HSV colorspace.
        AdaptiveDetector: performs rolling average on differences in HSV colorspace
        ThresholdDetector: detects slow transitions using average pixel intensity differences in RGB colorspace.
        Returns: (scene_list, video_object)
        """
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

    def close_video(self, video):
        """
        Helper function that attempt to release/close any underlying file handles from the scenedetect video object.
        """
        if video is None:
            return
        try:
            if hasattr(video, "release"):
                video.release()
        except Exception as e:
            print(f"Error releasing video: {e}")
            return
        try:
            if hasattr(video, "close"):
                video.close()
        except Exception as e:
            print(f"Error closing video: {e}")
            return
        # Some scenedetect versions wrap OpenCV objects:
        try:
            reader = getattr(video, "reader", None)
            if reader is not None and hasattr(reader, "release"):
                try:
                    reader.release()
                except Exception as e:
                    print(f"Error releasing video reader: {e}")
                    return
        except Exception as e:
            print(f"Error accessing video reader: {e}")
            return

    def export_scenes_ffmpeg(self,
        video_path: Path,
        scenes,
        out_dir: Path,
        prefix: str = "scene_",
        dry_run: bool = False,
        force_reencode: bool = False,
        crf: int = 18,
        preset: str = "slow",
        min_export_len: float = 0.0,
    ):
        """Export scenes to individual clips using ffmpeg, skipping any scene whose duration < min_export_len (seconds)."""
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
            out_name = f"{prefix}{len(kept_scenes):04d}__{self.human_timecode(start_sec)}__{self.human_timecode(end_sec)}.mp4"
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
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL, close_fds=True)
            else:
                copy_cmd = [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-ss", f"{start_sec:.3f}", "-i", str(video_path),
                    "-to", f"{duration:.3f}",
                    "-c", "copy", "-movflags", "+faststart",
                    str(out_file),
                ]
                result = subprocess.run(copy_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL, close_fds=True)
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
                    subprocess.run(reenc_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL, close_fds=True)

            outputs.append(str(out_file))

        return outputs, kept_scenes


    def write_manifest(self, scenes, out_dir: Path, base_prefix="scene_"):
        csv_path = out_dir / "scenes.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["scene_number", "start_sec", "end_sec", "duration_sec", "output_file"])
            for i, (start_time, end_time) in enumerate(scenes, start=1):
                start_sec = start_time.get_seconds()
                end_sec = end_time.get_seconds()
                duration = end_sec - start_sec
                pattern_name = f"{base_prefix}{i:04d}__{self.human_timecode(start_sec)}__{self.human_timecode(end_sec)}.mp4"
                writer.writerow([i, f"{start_sec:.3f}", f"{end_sec:.3f}", f"{duration:.3f}", pattern_name])
        return csv_path


    def ensure_ffmpeg(self):
        try:
            subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check="true", stdin=subprocess.DEVNULL, close_fds=True)
        except Exception:
            raise FileNotFoundError("ffmpeg not found. Please install ffmpeg and ensure it's in your PATH.")
