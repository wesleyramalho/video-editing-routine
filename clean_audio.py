#!/usr/bin/env python3
"""
clean_audio.py
-----------------------------------------------------------
Automatically removes filler words and long pauses from
Portuguese audio files (mp3, wav, m4a, aac, flac).

How it works:
  1. Uses Whisper (OpenAI) locally to transcribe the audio
     with per-word timestamps
  2. Identifies filler words ("eh", "hmm", "um"...) and long
     silences between words
  3. Uses ffmpeg to cut those segments and produce a clean audio

SETUP (run once):
  brew install ffmpeg
  pip3 install openai-whisper --break-system-packages

USAGE:
  python3 clean_audio.py input.mp3
  python3 clean_audio.py input.m4a output.mp3
  python3 clean_audio.py input.wav --model small
  python3 clean_audio.py input.mp3 --silence 1.2

Optional flags:
  --model     tiny | base | small | medium | large  (default: base)
              larger = more accurate, but slower
  --silence   seconds of silence above which to cut (default: 0.8)
  --pad       safety margin in seconds around speech (default: 0.1)
"""

import sys
import os
import subprocess
import argparse
import shutil

# ----------------------------------------------------------
# CONFIGURATION: filler words (conservative mode)
# ----------------------------------------------------------
# These are the words cut when they appear standalone.
# Conservative mode = only obvious cuts, no risk of cutting
# legitimate speech. Portuguese-specific (Whisper pinned to pt).
PT_FILLER_WORDS = {
    "e", "eh", "ehh", "ehhh", "eeh", "eeeh",
    "ah", "ahh", "ahhh",
    "ahm", "ahn",
    "hm", "hmm", "hmmm", "hmmmm",
    "um", "umm", "uhm", "uhmm",
    "uh", "uhh",
    "oh", "ohh",
    "mmm", "mm",
}

# Minimum pause to distinguish "natural" silence from
# "wasted" silence (seconds)
DEFAULT_PAUSE_THRESHOLD = 0.8

# Margin to avoid cutting mid-speech (seconds)
DEFAULT_PADDING = 0.1

# ----------------------------------------------------------
# HELPERS
# ----------------------------------------------------------

def check_dependencies():
    """Verify ffmpeg and whisper are installed."""
    missing = []
    if not shutil.which("ffmpeg"):
        missing.append("ffmpeg (install with: brew install ffmpeg)")
    try:
        import whisper  # noqa: F401
    except ImportError:
        missing.append("whisper (install with: pip3 install openai-whisper --break-system-packages)")
    if missing:
        print("ERROR - Missing dependencies:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)


def normalize_word(word):
    """Lowercase and strip punctuation for comparison."""
    return word.strip().lower().strip(".,!?;:()\"'-")


def transcribe(input_file, model="base"):
    """Run Whisper and return the result with per-word timestamps."""
    import whisper
    print(f"[1/3] Loading Whisper model '{model}'...")
    w = whisper.load_model(model)

    print(f"[2/3] Transcribing audio (this may take a few minutes)...")
    result = w.transcribe(
        input_file,
        language="pt",
        word_timestamps=True,
        verbose=False,
    )
    return result


def identify_cuts(result, fillers, pause_threshold, padding):
    """
    Walks each transcribed word and identifies (start, end)
    ranges that should be cut.
    """
    cuts = []
    previous_end = 0.0
    total_words = 0
    fillers_found = 0

    for segment in result.get("segments", []):
        for info in segment.get("words", []):
            raw_word = info.get("word", "")
            word = normalize_word(raw_word)
            start = float(info.get("start", 0))
            end = float(info.get("end", 0))
            total_words += 1

            # Long pause before this word?
            silence_gap = start - previous_end
            if silence_gap > pause_threshold:
                # Keep a small natural breath
                cut_start = previous_end + 0.15
                cut_end = start - 0.15
                if cut_end > cut_start:
                    cuts.append((cut_start, cut_end))

            # Is this word a filler?
            if word in fillers:
                cuts.append((start - padding, end + padding))
                fillers_found += 1

            previous_end = end

    print(f"      {total_words} words transcribed")
    print(f"      {fillers_found} fillers detected")
    print(f"      {len([c for c in cuts if (c[1]-c[0]) > 0]) - fillers_found} long silences detected")
    return cuts


def total_duration(input_file):
    """Use ffprobe to get total duration in seconds."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        input_file,
    ]
    out = subprocess.check_output(cmd).decode().strip()
    return float(out)


def invert_to_keep(cuts, duration):
    """
    Given a list of CUT ranges, return the inverted list:
    the ranges that should be KEPT.
    """
    cuts = [(max(0, s), e) for s, e in cuts if e > s]
    cuts.sort()

    # Merge overlapping cuts
    merged = []
    for start, end in cuts:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    keep = []
    current = 0.0
    for start, end in merged:
        if start > current:
            keep.append((current, start))
        current = max(current, end)
    if current < duration:
        keep.append((current, duration))

    # Drop microscopic segments (<0.05s) that become artifacts
    return [(s, e) for s, e in keep if (e - s) > 0.05]


def render(input_file, keep, output_file):
    """
    Build an ffmpeg filter_complex that extracts only the
    'keep' ranges and concatenates them to the output file.
    """
    if not keep:
        print("ERROR - No segments left to keep. Check the input audio.")
        sys.exit(1)

    print(f"[3/3] Rendering clean audio...")

    parts = []
    labels = []
    for i, (start, end) in enumerate(keep):
        parts.append(f"[0:a]atrim=start={start:.3f}:end={end:.3f},asetpts=PTS-STARTPTS[a{i}]")
        labels.append(f"[a{i}]")

    filter_str = ";".join(parts) + ";"
    filter_str += "".join(labels) + f"concat=n={len(keep)}:v=0:a=1[out]"

    cmd = [
        "ffmpeg", "-y",
        "-i", input_file,
        "-filter_complex", filter_str,
        "-map", "[out]",
        output_file,
    ]
    # Suppress verbose ffmpeg output, show only errors
    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        print("ERROR while rendering:")
        print(result.stderr.decode())
        sys.exit(1)


# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Remove filler words and long pauses from Portuguese audio.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", help="Input audio file (mp3, wav, m4a, aac, flac)")
    parser.add_argument("output", nargs="?", help="Output file (optional)")
    parser.add_argument("--model", default="base",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size (default: base)")
    parser.add_argument("--silence", type=float, default=DEFAULT_PAUSE_THRESHOLD,
                        help=f"Seconds of pause above which to cut (default: {DEFAULT_PAUSE_THRESHOLD})")
    parser.add_argument("--pad", type=float, default=DEFAULT_PADDING,
                        help=f"Safety margin in seconds around each filler (default: {DEFAULT_PADDING})")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR - File not found: {args.input}")
        sys.exit(1)

    output = args.output
    if not output:
        base, ext = os.path.splitext(args.input)
        output = f"{base}_clean{ext}"

    check_dependencies()

    print(f"Input:    {args.input}")
    print(f"Output:   {output}")
    print(f"Model:    {args.model}")
    print(f"Silence:  cut pauses > {args.silence}s")
    print("-" * 60)

    result = transcribe(args.input, model=args.model)
    cuts = identify_cuts(result, PT_FILLER_WORDS, args.silence, args.pad)
    duration = total_duration(args.input)
    keep = invert_to_keep(cuts, duration)

    final_time = sum(e - s for s, e in keep)
    savings = duration - final_time
    print(f"      original duration: {duration:.1f}s")
    print(f"      final duration:    {final_time:.1f}s")
    print(f"      savings:           {savings:.1f}s ({savings/duration*100:.1f}%)")
    print("-" * 60)

    render(args.input, keep, output)

    print(f"\nDone! Clean audio saved to: {output}")


if __name__ == "__main__":
    main()
