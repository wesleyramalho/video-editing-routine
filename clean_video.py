#!/usr/bin/env python3
"""
clean_video.py
-----------------------------------------------------------
Takes an MP4 video and generates an OpenShot project (.osp)
and an FCPXML file with filler-word cuts and long-pause cuts
already applied as separate clips on the timeline.

Open the .osp in OpenShot (File > Open Project) or the
.fcpxml in DaVinci Resolve / Final Cut Pro to fine-tune
each cut before exporting the final video.

Dependencies (same as clean_audio.py):
  - openai-whisper
  - ffmpeg + ffprobe
  - python3

Usage:
  python3 clean_video.py video.mp4
  python3 clean_video.py video.mp4 output.osp
  python3 clean_video.py video.mp4 --model small --silence 0.8
"""

import sys
import os
import subprocess
import argparse
import json
import uuid
import shutil
import random

# Reuse detection logic from the audio script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clean_audio import (
    PT_FILLER_WORDS,
    DEFAULT_PAUSE_THRESHOLD,
    DEFAULT_PADDING,
    normalize_word,
    transcribe,
    identify_cuts,
    invert_to_keep,
)

OPENSHOT_QT_VERSION = "3.2.1"
LIBOPENSHOT_VERSION = "0.4.0"
FCPXML_VERSION = "1.9"

# Zoom effect defaults (Ken Burns style)
ZOOM_MIN_DURATION = 3.0      # only zoom clips >= 3 seconds
ZOOM_PROBABILITY = 0.30      # ~30% of eligible clips get zoom
ZOOM_SCALE_START = 1.0
ZOOM_SCALE_END = 1.08        # subtle: 8% zoom-in

# Caption defaults (TikTok/Shorts style)
CAPTION_MAX_WORDS = 5
CAPTION_MAX_DURATION = 2.0   # seconds
CAPTION_MAX_GAP = 0.5        # split caption group if gap between words > 0.5s


def check_dependencies():
    missing = []
    if not shutil.which("ffmpeg"):
        missing.append("ffmpeg (brew install ffmpeg)")
    if not shutil.which("ffprobe"):
        missing.append("ffprobe (bundled with ffmpeg)")
    try:
        import whisper  # noqa: F401
    except ImportError:
        missing.append("whisper (pip3 install openai-whisper --break-system-packages)")
    if missing:
        print("ERROR - Missing dependencies:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)


def ffprobe_metadata(input_file):
    """
    Use ffprobe to extract video metadata:
    width, height, fps, duration, codecs, etc.
    Returns a structured dict.
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        input_file,
    ]
    out = subprocess.check_output(cmd).decode()
    info = json.loads(out)

    video_stream = None
    audio_stream = None
    for s in info.get("streams", []):
        if s.get("codec_type") == "video" and video_stream is None:
            video_stream = s
        elif s.get("codec_type") == "audio" and audio_stream is None:
            audio_stream = s

    if video_stream is None:
        print("ERROR - The file has no video stream")
        sys.exit(1)

    width = int(video_stream.get("width", 1920))
    height = int(video_stream.get("height", 1080))

    # fps comes as "30/1" or "30000/1001"
    fps_str = video_stream.get("r_frame_rate", "30/1")
    fps_parts = fps_str.split("/")
    fps_num = int(fps_parts[0])
    fps_den = int(fps_parts[1]) if len(fps_parts) > 1 else 1

    duration = float(info.get("format", {}).get("duration", 0.0))
    video_length = int(round(duration * fps_num / fps_den))

    vcodec = video_stream.get("codec_name", "h264")
    video_bit_rate = int(video_stream.get("bit_rate", 2000000)) if video_stream.get("bit_rate") else 2000000

    if audio_stream is not None:
        has_audio = 1
        acodec = audio_stream.get("codec_name", "aac")
        sample_rate = int(audio_stream.get("sample_rate", 48000))
        channels = int(audio_stream.get("channels", 2))
        audio_bit_rate = int(audio_stream.get("bit_rate", 192000)) if audio_stream.get("bit_rate") else 192000
    else:
        has_audio = 0
        acodec = ""
        sample_rate = 48000
        channels = 2
        audio_bit_rate = 0

    file_size = int(info.get("format", {}).get("size", 0))

    # channel_layout: 1=mono, 3=stereo (bitmask); default to stereo
    channel_layout = 3 if channels == 2 else (1 if channels == 1 else 3)

    return {
        "width": width,
        "height": height,
        "fps_num": fps_num,
        "fps_den": fps_den,
        "duration": duration,
        "video_length": video_length,
        "vcodec": vcodec,
        "acodec": acodec,
        "sample_rate": sample_rate,
        "channels": channels,
        "channel_layout": channel_layout,
        "video_bit_rate": video_bit_rate,
        "audio_bit_rate": audio_bit_rate,
        "has_audio": has_audio,
        "file_size": file_size,
    }


def build_file_entry(file_id, abs_path, meta):
    """Build a files[] entry for the .osp project."""
    return {
        "id": file_id,
        "path": abs_path,
        "media_type": "video",
        "width": meta["width"],
        "height": meta["height"],
        "fps": {"num": meta["fps_num"], "den": meta["fps_den"]},
        "duration": meta["duration"],
        "video_length": str(meta["video_length"]),
        "sample_rate": meta["sample_rate"],
        "channels": meta["channels"],
        "channel_layout": meta["channel_layout"],
        "has_audio": meta["has_audio"],
        "has_video": 1,
        "has_single_image": 0,
        "acodec": meta["acodec"],
        "vcodec": meta["vcodec"],
        "audio_bit_rate": meta["audio_bit_rate"],
        "video_bit_rate": meta["video_bit_rate"],
        "video_stream_index": 0,
        "audio_stream_index": 0,
        "audio_timebase": {"num": 1, "den": meta["sample_rate"]},
        "video_timebase": {"num": meta["fps_den"], "den": meta["fps_num"]},
        "pixel_ratio": {"num": 1, "den": 1},
        "display_ratio": {"num": 16, "den": 9},
        "pixel_format": -1,
        "file_size": str(meta["file_size"]),
        "type": "FFmpegReader",
        "interlaced_frame": False,
        "top_field_first": True,
    }


def point(y, interpolation=0):
    """Helper for an OpenShot animation Point."""
    return {"Points": [{"co": {"X": 1.0, "Y": y}, "interpolation": interpolation}]}


def zoom_points(clip_frames, start_value, end_value, interpolation=0):
    """Two-keyframe OpenShot animation: start_value at frame 1, end_value at last frame.

    Used for scale_x / scale_y to produce a Ken Burns-style slow zoom.
    """
    end_frame = max(2, clip_frames)
    return {
        "Points": [
            {"co": {"X": 1, "Y": float(start_value)}, "interpolation": interpolation},
            {"co": {"X": end_frame, "Y": float(end_value)}, "interpolation": interpolation},
        ]
    }


def build_clip_entry(clip_id, file_entry, file_id, position, start, end, title,
                     zoom_range=None, fps_num=30, fps_den=1):
    """Build a clips[] entry for the .osp project.

    zoom_range: None for no zoom, or (start_scale, end_scale) tuple for animated zoom.
    """
    if zoom_range:
        clip_duration = end - start
        clip_frames = max(2, int(round(clip_duration * fps_num / fps_den)))
        scale_x = zoom_points(clip_frames, zoom_range[0], zoom_range[1])
        scale_y = zoom_points(clip_frames, zoom_range[0], zoom_range[1])
    else:
        scale_x = point(1.0)
        scale_y = point(1.0)

    return {
        "id": clip_id,
        "layer": 4000000,
        "position": position,
        "start": start,
        "end": end,
        "file_id": file_id,
        "title": title,
        "image": "",
        "reader": file_entry,
        "has_audio": point(1.0, interpolation=2),
        "has_video": point(1.0, interpolation=2),
        "alpha": point(1.0),
        "anchor": 0,
        "channel_filter": point(-1.0),
        "channel_mapping": point(-1.0),
        "display": 0,
        "duration": file_entry["duration"],
        "effects": [],
        "gravity": 4,
        "location_x": point(0.0),
        "location_y": point(0.0),
        "mixing": 0,
        "origin_x": point(0.5),
        "origin_y": point(0.5),
        "parentObjectId": "",
        "perspective_c1_x": point(-1.0),
        "perspective_c1_y": point(-1.0),
        "perspective_c2_x": point(-1.0),
        "perspective_c2_y": point(-1.0),
        "perspective_c3_x": point(-1.0),
        "perspective_c3_y": point(-1.0),
        "perspective_c4_x": point(-1.0),
        "perspective_c4_y": point(-1.0),
        "rotation": point(0.0),
        "scale": 1,
        "scale_x": scale_x,
        "scale_y": scale_y,
        "shear_x": point(0.0),
        "shear_y": point(0.0),
        "time": point(1.0),
        "volume": point(1.0),
        "wave_color": {
            "red": point(0.0),
            "green": point(123.0),
            "blue": point(255.0),
            "alpha": point(255.0),
        },
        "waveform": False,
    }


def select_zoomed_indices(keep, min_duration=ZOOM_MIN_DURATION,
                          probability=ZOOM_PROBABILITY, seed=None):
    """Return a set of clip indices that should get the zoom effect.

    Eligibility: clip duration >= min_duration.
    Selection: `probability` fraction of eligible clips, chosen randomly.
    """
    rng = random.Random(seed)
    eligible = [i for i, (s, e) in enumerate(keep) if (e - s) >= min_duration]
    n = int(round(len(eligible) * probability))
    if n <= 0:
        return set()
    return set(rng.sample(eligible, min(n, len(eligible))))


def build_osp_project(input_video, keep, meta, zoomed_indices=None):
    """Build the full OpenShot project JSON.

    zoomed_indices: optional set of clip indices (into `keep`) that should
    receive a Ken Burns zoom effect.
    """
    abs_path = os.path.abspath(input_video)
    name = os.path.basename(input_video)
    zoomed_indices = zoomed_indices or set()

    file_id = "F0"
    file_entry = build_file_entry(file_id, abs_path, meta)

    clips = []
    position = 0.0
    for i, (start, end) in enumerate(keep):
        clip_duration = end - start
        zoom_range = (ZOOM_SCALE_START, ZOOM_SCALE_END) if i in zoomed_indices else None
        clip = build_clip_entry(
            clip_id=f"C{i}",
            file_entry=file_entry,
            file_id=file_id,
            position=position,
            start=start,
            end=end,
            title=name,
            zoom_range=zoom_range,
            fps_num=meta["fps_num"],
            fps_den=meta["fps_den"],
        )
        clips.append(clip)
        position += clip_duration

    timeline_duration = position

    # Profile tuned to the video dimensions
    if meta["height"] >= 2000:
        profile = "HDTV 2160p 30 fps"
    elif meta["height"] >= 1000:
        profile = "HD 1080p 30 fps"
    elif meta["height"] >= 700:
        profile = "HD 720p 30 fps"
    else:
        profile = "SD 480p 30 fps"

    return {
        "id": uuid.uuid4().hex[:10].upper(),
        "fps": {"num": meta["fps_num"], "den": meta["fps_den"]},
        "display_ratio": {"num": 16, "den": 9},
        "pixel_ratio": {"num": 1, "den": 1},
        "width": meta["width"],
        "height": meta["height"],
        "sample_rate": meta["sample_rate"],
        "channels": meta["channels"],
        "channel_layout": meta["channel_layout"],
        "settings": {},
        "clips": clips,
        "effects": [],
        "files": [file_entry],
        "duration": max(timeline_duration, 30.0),
        "scale": 15.0,
        "tick_pixels": 100,
        "playhead_position": 0,
        "profile": profile,
        "export_settings": None,
        "layers": [
            {"id": "L1", "label": "", "number": 1000000, "y": 0, "lock": False},
            {"id": "L2", "label": "", "number": 2000000, "y": 0, "lock": False},
            {"id": "L3", "label": "", "number": 3000000, "y": 0, "lock": False},
            {"id": "L4", "label": "", "number": 4000000, "y": 0, "lock": False},
            {"id": "L5", "label": "", "number": 5000000, "y": 0, "lock": False},
        ],
        "markers": [],
        "progress": [],
        "history": {"undo": [], "redo": []},
        "version": {
            "openshot-qt": OPENSHOT_QT_VERSION,
            "libopenshot": LIBOPENSHOT_VERSION,
        },
    }


def fcp_time(seconds, fps_num, fps_den):
    """Convert seconds to FCPXML time format, frame-aligned.

    Ex: 11.68s @ 30000/1001 fps -> "350350/30000s"
    (350 frames * 1001 / 30000 ~= 11.678s, frame-aligned)
    """
    if seconds <= 0:
        return "0s"
    frames = round(seconds * fps_num / fps_den)
    if frames == 0:
        return "0s"
    return f"{frames * fps_den}/{fps_num}s"


def build_fcpxml(input_video, keep, meta, zoomed_indices=None):
    """Build an FCPXML 1.9 string (imports in DaVinci Resolve / FCP).

    zoomed_indices: optional set of clip indices that should receive a
    keyframed scale transform (Ken Burns zoom-in).
    """
    from urllib.parse import quote
    from xml.sax.saxutils import escape as xml_escape

    abs_path = os.path.abspath(input_video)
    path_uri = "file://" + quote(abs_path)
    name = os.path.basename(input_video)
    name_safe = xml_escape(name)
    name_noext = xml_escape(os.path.splitext(name)[0])
    zoomed_indices = zoomed_indices or set()

    fps_num = meta["fps_num"]
    fps_den = meta["fps_den"]
    frame_duration = f"{fps_den}/{fps_num}s"

    # Format name follows Final Cut Pro naming convention
    fps_tag = f"{fps_num/fps_den:g}".replace(".", "")
    if meta["height"] >= 2000:
        format_name = f"FFVideoFormat2160p{fps_tag}"
    elif meta["height"] >= 1000:
        format_name = f"FFVideoFormat1080p{fps_tag}"
    elif meta["height"] >= 700:
        format_name = f"FFVideoFormat720p{fps_tag}"
    else:
        format_name = "FFVideoFormatRateUndefined"

    asset_duration = fcp_time(meta["duration"], fps_num, fps_den)

    clips_xml = []
    position = 0.0
    for i, (start, end) in enumerate(keep):
        duration = end - start
        offset_str = fcp_time(position, fps_num, fps_den)
        start_str = fcp_time(start, fps_num, fps_den)
        duration_str = fcp_time(duration, fps_num, fps_den)

        if i in zoomed_indices:
            # Animated scale (Ken Burns) — keyframe times are clip-local,
            # starting at 0s and ending at clip duration
            zoom_xml = (
                f'\n                            <adjust-transform>'
                f'\n                                <param name="scale">'
                f'\n                                    <keyframe time="0s" value="{ZOOM_SCALE_START} {ZOOM_SCALE_START}"/>'
                f'\n                                    <keyframe time="{duration_str}" value="{ZOOM_SCALE_END} {ZOOM_SCALE_END}"/>'
                f'\n                                </param>'
                f'\n                            </adjust-transform>'
                f'\n                        '
            )
            clips_xml.append(
                f'                        <asset-clip ref="r2" '
                f'offset="{offset_str}" start="{start_str}" duration="{duration_str}" '
                f'name="{name_safe}">'
                f'{zoom_xml}</asset-clip>'
            )
        else:
            clips_xml.append(
                f'                        <asset-clip ref="r2" '
                f'offset="{offset_str}" start="{start_str}" duration="{duration_str}" '
                f'name="{name_safe}"/>'
            )
        position += duration

    sequence_duration = fcp_time(position, fps_num, fps_den)
    spine_xml = "\n".join(clips_xml)

    if meta["has_audio"]:
        audio_attr = f' hasAudio="1" audioSources="1" audioChannels="{meta["channels"]}" audioRate="{meta["sample_rate"]}"'
    else:
        audio_attr = ""

    return f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE fcpxml>
<fcpxml version="{FCPXML_VERSION}">
    <resources>
        <format id="r1" name="{format_name}" frameDuration="{frame_duration}" width="{meta["width"]}" height="{meta["height"]}"/>
        <asset id="r2" name="{name_noext}" start="0s" duration="{asset_duration}" hasVideo="1"{audio_attr} format="r1">
            <media-rep kind="original-media" src="{path_uri}"/>
        </asset>
    </resources>
    <library>
        <event name="Automatic Cuts">
            <project name="{name_noext}_cuts">
                <sequence format="r1" duration="{sequence_duration}" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="{meta["sample_rate"]}Hz">
                    <spine>
{spine_xml}
                    </spine>
                </sequence>
            </project>
        </event>
    </library>
</fcpxml>
'''


def srt_time(seconds):
    """Format seconds as HH:MM:SS,mmm for SRT subtitles."""
    if seconds < 0:
        seconds = 0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms >= 1000:
        ms = 0
        s += 1
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def map_to_new_timeline(source_time, keep):
    """Map a source-video timestamp to its timestamp on the cut timeline.

    Returns None if the source_time falls in a cut gap (the word was removed).
    """
    cumulative = 0.0
    for src_start, src_end in keep:
        if src_start <= source_time < src_end:
            return cumulative + (source_time - src_start)
        cumulative += (src_end - src_start)
    # Tolerate the very last boundary
    if keep and abs(source_time - keep[-1][1]) < 1e-3:
        return cumulative
    return None


def build_captions(result, keep,
                   max_words=CAPTION_MAX_WORDS,
                   max_duration=CAPTION_MAX_DURATION,
                   max_gap=CAPTION_MAX_GAP):
    """Walk the Whisper result, remap word timestamps to the cut timeline,
    skip words that fell inside cut gaps, and group the rest into captions.
    """
    remapped = []
    for segment in result.get("segments", []):
        for info in segment.get("words", []):
            text = info.get("word", "").strip()
            if not text:
                continue
            src_start = float(info.get("start", 0))
            src_end = float(info.get("end", 0))
            new_start = map_to_new_timeline(src_start, keep)
            new_end = map_to_new_timeline(src_end, keep)
            if new_start is None or new_end is None:
                continue  # word was cut out
            if new_end <= new_start:
                new_end = new_start + 0.05
            remapped.append((new_start, new_end, text))

    captions = []
    group_words = []
    group_start = None
    group_end = None

    def flush():
        if group_words and group_start is not None:
            captions.append((group_start, group_end, " ".join(group_words)))

    for (ws, we, text) in remapped:
        if not group_words:
            group_words = [text]
            group_start = ws
            group_end = we
            continue

        gap = ws - group_end
        group_duration = we - group_start
        if (gap > max_gap
                or group_duration > max_duration
                or len(group_words) >= max_words):
            flush()
            group_words = [text]
            group_start = ws
            group_end = we
        else:
            group_words.append(text)
            group_end = we

    flush()
    return captions


def write_srt(captions, output_file):
    """Write captions list [(start, end, text), ...] to a UTF-8 SRT file."""
    with open(output_file, "w", encoding="utf-8") as f:
        for i, (start, end, text) in enumerate(captions, start=1):
            f.write(f"{i}\n")
            f.write(f"{srt_time(start)} --> {srt_time(end)}\n")
            f.write(f"{text.strip()}\n\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate OpenShot (.osp) + FCPXML (DaVinci/FCP) project with automatic cuts applied.",
    )
    parser.add_argument("input", help="Input MP4 video")
    parser.add_argument("output", nargs="?", help="Output .osp file (optional)")
    parser.add_argument("--model", default="small",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size (default: small)")
    parser.add_argument("--silence", type=float, default=DEFAULT_PAUSE_THRESHOLD,
                        help=f"Seconds of pause above which to cut (default: {DEFAULT_PAUSE_THRESHOLD})")
    parser.add_argument("--pad", type=float, default=DEFAULT_PADDING,
                        help=f"Safety margin in seconds around each filler (default: {DEFAULT_PADDING})")
    parser.add_argument("--zoom", action="store_true",
                        help="Add a subtle Ken Burns zoom to ~30%% of clips longer than 3s")
    parser.add_argument("--captions", action="store_true",
                        help="Generate an SRT subtitle file aligned to the cut timeline")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible zoom selection (default: non-deterministic)")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR - File not found: {args.input}")
        sys.exit(1)

    output_osp = args.output
    if not output_osp:
        base, _ = os.path.splitext(args.input)
        output_osp = f"{base}_cuts.osp"

    base_out, _ = os.path.splitext(output_osp)
    output_json = f"{base_out}.json"
    output_fcpxml = f"{base_out}.fcpxml"
    output_srt = f"{base_out}.srt"

    check_dependencies()

    print(f"Input:    {args.input}")
    print(f"OpenShot: {output_osp}")
    print(f"FCPXML:   {output_fcpxml}")
    if args.captions:
        print(f"Captions: {output_srt}")
    print(f"Fallback: {output_json}")
    print(f"Model:    {args.model}")
    print(f"Silence:  cut pauses > {args.silence}s")
    if args.zoom:
        print(f"Zoom:     enabled (~{int(ZOOM_PROBABILITY*100)}% of clips >= {ZOOM_MIN_DURATION}s, scale {ZOOM_SCALE_START}->{ZOOM_SCALE_END})")
    print("-" * 60)

    print("[1/4] Extracting video metadata with ffprobe...")
    meta = ffprobe_metadata(args.input)
    print(f"      {meta['width']}x{meta['height']} @ {meta['fps_num']}/{meta['fps_den']} fps")
    print(f"      duration: {meta['duration']:.1f}s | audio: {meta['acodec']} {meta['sample_rate']}Hz {meta['channels']}ch")
    print("-" * 60)

    result = transcribe(args.input, model=args.model)
    cuts = identify_cuts(result, PT_FILLER_WORDS, args.silence, args.pad)
    keep = invert_to_keep(cuts, meta["duration"])

    if not keep:
        print("ERROR - No segments left to keep.")
        sys.exit(1)

    final_time = sum(e - s for s, e in keep)
    savings = meta["duration"] - final_time
    print(f"      {len(keep)} clips will be generated")
    print(f"      original duration: {meta['duration']:.1f}s")
    print(f"      final duration:    {final_time:.1f}s")
    print(f"      savings:           {savings:.1f}s ({savings/meta['duration']*100:.1f}%)")
    print("-" * 60)

    zoomed_indices = set()
    if args.zoom:
        zoomed_indices = select_zoomed_indices(keep, seed=args.seed)
        print(f"      zoom applied to {len(zoomed_indices)} of {len(keep)} clips")

    print(f"[4/4] Writing OpenShot project and FCPXML...")
    project = build_osp_project(args.input, keep, meta, zoomed_indices=zoomed_indices)
    with open(output_osp, "w", encoding="utf-8") as f:
        json.dump(project, f, indent=2, ensure_ascii=False)

    fcpxml = build_fcpxml(args.input, keep, meta, zoomed_indices=zoomed_indices)
    with open(output_fcpxml, "w", encoding="utf-8") as f:
        f.write(fcpxml)

    # Fallback: simple JSON with the keep segments
    fallback = {
        "video": os.path.abspath(args.input),
        "original_duration": meta["duration"],
        "final_duration": final_time,
        "zoomed_clip_indices": sorted(zoomed_indices),
        "keep_segments": [
            {"position": sum(e - s for s, e in keep[:i]), "source_start": start, "source_end": end, "duration": end - start}
            for i, (start, end) in enumerate(keep)
        ],
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(fallback, f, indent=2, ensure_ascii=False)

    if args.captions:
        captions = build_captions(result, keep)
        write_srt(captions, output_srt)
        print(f"      {len(captions)} caption lines written")

    print(f"\nDone!")
    print(f"  OpenShot:            {output_osp}")
    print(f"  DaVinci Resolve/FCP: {output_fcpxml}")
    print(f"  Fallback JSON:       {output_json}")
    if args.captions:
        print(f"  SRT captions:        {output_srt}")


if __name__ == "__main__":
    main()
