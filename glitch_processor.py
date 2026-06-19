#!/usr/bin/env python3
"""
glitch_processor.py
===================
Convert a normal video into a self-contained, highly-compressible glitch-art
animation that plays in any modern browser by simply opening an .html file.

Core idea
---------
We DO NOT store the glitch. We store a clean, tiny, recognizable version of the
video (downsampled + palette-quantized + delta-encoded). All the glitch effects
(RGB split, datamosh-style block tears, scanlines, noise, fringing, corruption)
are generated *live in the browser*. That keeps the stored data extremely
gzip-friendly while the output still looks "broken digital" and artistic.

Pipeline
--------
1. Decode + downscale + resample fps with ffmpeg (any format) -> raw RGB frames.
   (Optional OpenCV path if --decoder opencv and cv2 is installed.)
2. Uniformly quantize each frame to `levels` per channel -> 1 byte palette index
   per pixel (levels**3 colors, e.g. 6 -> 216 colors).
3. Delta-encode: emit a full keyframe every --keyint frames (or when a frame
   differs too much), otherwise emit only the changed (position, index) pairs.
4. Serialize to compact JSON -> gzip -> base64 -> embed into a single .html file.
   The browser decompresses with the native DecompressionStream('gzip').

Usage
-----
    python3 glitch_processor.py input.mp4
    python3 glitch_processor.py clip.mov -o out.html --preset subtle-neon
    python3 glitch_processor.py clip.webm --width 128 --height 72 --fps 15
    python3 glitch_processor.py --selftest          # generate synthetic demo
    python3 glitch_processor.py clip.mp4 --play-text # ASCII preview in terminal

Run `python3 glitch_processor.py --help` for all options.
"""

import argparse
import base64
import gzip
import json
import math
import os
import shutil
import subprocess
import sys

import numpy as np

# --------------------------------------------------------------------------- #
# Glitch presets (these only set the *defaults* baked into the HTML UI; every
# value is also tweakable live with the sliders in the browser).
# --------------------------------------------------------------------------- #
# Each preset deliberately leans on a DIFFERENT dominant effect so they read as
# distinct looks (not just "more/less of the same"):
#   heavy-datamosh -> blocky displacement + corruption
#   subtle-neon    -> pure RGB chroma split, almost nothing else
#   vhs            -> scanlines + horizontal tearing + analog noise, no blocks
PRESETS = {
    # scanGap = display px between line starts (separation); scanSize = px thickness of each dark line.
    # static = TV "snow" (grayscale grain blended over the picture; 1.0 = untuned channel).
    "heavy-datamosh": {
        "rgb": 8, "block": 0.7, "tear": 0.35, "noise": 0.04, "static": 0.0,
        "corrupt": 0.07, "blockSize": 20, "bands": 10,
        "scanline": 0.0, "scanGap": 4, "scanSize": 1,
    },
    "subtle-neon": {
        "rgb": 7, "block": 0.0, "tear": 0.06, "noise": 0.0, "static": 0.0,
        "corrupt": 0.0, "blockSize": 30, "bands": 3,
        "scanline": 0.0, "scanGap": 4, "scanSize": 1,
    },
    "vhs": {
        "rgb": 3, "block": 0.0, "tear": 0.5, "noise": 0.09, "static": 0.12,
        "corrupt": 0.0, "blockSize": 40, "bands": 22,
        "scanline": 0.5, "scanGap": 6, "scanSize": 2,
    },
    "clean": {  # almost no glitch -- useful to confirm the source is recognizable
        "rgb": 0, "block": 0.0, "tear": 0.0, "noise": 0.0, "static": 0.0,
        "corrupt": 0.0, "blockSize": 24, "bands": 0,
        "scanline": 0.0, "scanGap": 4, "scanSize": 1,
    },

    # --- Dominant-color mosaic presets -------------------------------------- #
    # mosaic=True turns the base layer into solid dominant-color blocks; box =
    # rendered block px, sample = px region scanned (from the PRISTINE frame) for
    # the dominant color. Glitch fields still apply, overlaid on top of the blocks.
    "clean-mosaic": {
        "rgb": 0, "block": 0.0, "tear": 0.0, "noise": 0.0, "static": 0.0,
        "corrupt": 0.0, "blockSize": 24, "bands": 0,
        "scanline": 0.0, "scanGap": 4, "scanSize": 1,
        "mosaic": True, "box": 16, "sample": 16,
    },
    "glitch-mosaic": {
        "rgb": 5, "block": 0.25, "tear": 0.2, "noise": 0.02, "static": 0.0,
        "corrupt": 0.02, "blockSize": 18, "bands": 8,
        "scanline": 0.1, "scanGap": 5, "scanSize": 1,
        "mosaic": True, "box": 12, "sample": 14,
    },
    "heavy-datamosh-mosaic": {
        "rgb": 9, "block": 0.7, "tear": 0.4, "noise": 0.05, "static": 0.05,
        "corrupt": 0.08, "blockSize": 16, "bands": 12,
        "scanline": 0.0, "scanGap": 4, "scanSize": 1,
        "mosaic": True, "box": 10, "sample": 18,
    },
}

# Every preset gets mosaic fields so the UI toggle/sliders always have values.
for _p in PRESETS.values():
    _p.setdefault("mosaic", False)
    _p.setdefault("box", 8)
    _p.setdefault("sample", 8)

# --------------------------------------------------------------------------- #
# Frame decoding
# --------------------------------------------------------------------------- #
def have_ffmpeg():
    return shutil.which("ffmpeg") is not None


def decode_ffmpeg(path, w, h, fps, max_frames=None):
    """Yield (h, w, 3) uint8 RGB frames using ffmpeg for decode/scale/resample."""
    if not have_ffmpeg():
        raise RuntimeError(
            "ffmpeg not found on PATH. Install it (EndeavourOS: "
            "`sudo pacman -S ffmpeg`) or use --decoder opencv."
        )
    # scale=...:flags=area gives clean averaged downsampling (like cv2 INTER_AREA)
    vf = f"fps={fps},scale={w}:{h}:flags=area"
    cmd = [
        "ffmpeg", "-v", "error", "-i", path,
        "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    frame_bytes = w * h * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    count = 0
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            yield np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
            count += 1
            if max_frames and count >= max_frames:
                break
    finally:
        proc.stdout.close()
        err = proc.stderr.read().decode("utf-8", "ignore")
        proc.stderr.close()
        proc.wait()
        if count == 0 and err.strip():
            raise RuntimeError("ffmpeg failed:\n" + err.strip())


def decode_opencv(path, w, h, fps, max_frames=None):
    """Optional OpenCV decode path (only used with --decoder opencv)."""
    try:
        import cv2
    except ImportError as e:
        raise RuntimeError("OpenCV not installed (`pip install opencv-python`).") from e
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open: {path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    step = max(1, round(src_fps / fps))
    idx = 0
    out = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            small = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            yield small[:, :, ::-1].copy()  # BGR -> RGB
            out += 1
            if max_frames and out >= max_frames:
                break
        idx += 1
    cap.release()


def synthetic_frames(w, h, fps, seconds=5):
    """Procedural demo animation (used for --selftest / when no video given)."""
    n = int(fps * seconds)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    for i in range(n):
        t = i / fps
        # moving plasma
        r = (np.sin(xx / 8 + t * 2) + np.cos(yy / 6 - t * 1.5)) * 0.5 + 0.5
        g = (np.sin((xx + yy) / 10 - t * 2.2)) * 0.5 + 0.5
        b = (np.cos(xx / 7 + yy / 9 + t)) * 0.5 + 0.5
        frame = np.stack([r, g, b], axis=-1)
        # a bright bouncing block so motion is obvious
        bx = int((math.sin(t * 1.7) * 0.4 + 0.5) * (w - w // 4))
        by = int((math.cos(t * 1.3) * 0.4 + 0.5) * (h - h // 4))
        frame[by:by + h // 4, bx:bx + w // 4] = [1.0, 0.2, 0.6]
        yield (np.clip(frame, 0, 1) * 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Quantization + delta encoding
# --------------------------------------------------------------------------- #
def quantize(frame, levels):
    """RGB uint8 (h,w,3) -> flat int array of palette indices in [0, levels**3)."""
    q = (frame.astype(np.int32) * levels) // 256          # 0..levels-1 per channel
    idx = (q[..., 0] * levels + q[..., 1]) * levels + q[..., 2]
    return idx.reshape(-1).astype(np.int32)


def encode(frames_iter, w, h, levels, keyint, delta_thresh):
    """
    Build the compact frame list.
      keyframe -> {"t": 0, "p": [idx, idx, ...]}        (full grid)
      delta    -> {"t": 1, "d": [pos, idx, pos, idx]}   (flat changed pairs)
    A delta is promoted to a keyframe if more than `delta_thresh` of the pixels
    changed (cheaper and avoids drift).
    """
    frames = []
    prev = None
    npix = w * h
    nframes = 0
    for frame in frames_iter:
        cur = quantize(frame, levels)
        if prev is None:
            frames.append({"t": 0, "p": cur.tolist()})
        else:
            changed = np.nonzero(cur != prev)[0]
            if nframes % keyint == 0 or len(changed) > delta_thresh * npix:
                frames.append({"t": 0, "p": cur.tolist()})
            else:
                pairs = np.empty(len(changed) * 2, dtype=np.int32)
                pairs[0::2] = changed
                pairs[1::2] = cur[changed]
                frames.append({"t": 1, "d": pairs.tolist()})
        prev = cur
        nframes += 1
    return frames, nframes


def build_payload(frames, w, h, levels, fps):
    return {
        "w": w, "h": h, "levels": levels, "fps": fps,
        "n": len(frames), "frames": frames,
    }


def pack(payload):
    """JSON -> gzip -> base64 string (decompressed natively in the browser)."""
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    gz = gzip.compress(raw, 9)
    return base64.b64encode(gz).decode("ascii"), len(raw), len(gz)


def extract_audio(path, bitrate="24k"):
    """
    Extract the source soundtrack with ffmpeg, downmixed to mono and compressed,
    so it can be embedded (base64) and played synced in the browser.
    Returns {"mime": ..., "b64": ...} or None (no audio stream / ffmpeg missing).
    Tries Opus-in-Ogg first (tiny + universally supported in modern browsers),
    then falls back to MP3.
    """
    if not have_ffmpeg():
        return None
    attempts = [
        ("audio/ogg", ["-c:a", "libopus", "-b:a", bitrate, "-f", "ogg"]),
        ("audio/mpeg", ["-c:a", "libmp3lame", "-b:a", bitrate, "-f", "mp3"]),
    ]
    for mime, enc in attempts:
        cmd = ["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1"] + enc + ["pipe:1"]
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode == 0 and len(proc.stdout) > 256:
            return {"mime": mime, "b64": base64.b64encode(proc.stdout).decode("ascii")}
    return None


# --------------------------------------------------------------------------- #
# HTML generation
# --------------------------------------------------------------------------- #
def build_html(data_b64, ui_config, audio=None):
    """Inject data + default UI config (+ optional embedded audio) into the template."""
    html = HTML_TEMPLATE
    html = html.replace("/*__DATA__*/null", json.dumps(data_b64) if data_b64 else "null")
    html = html.replace("/*__CONFIG__*/({})", "(" + json.dumps(ui_config) + ")")
    html = html.replace("/*__AUDIO__*/null", json.dumps(audio) if audio else "null")
    return html


# --------------------------------------------------------------------------- #
# Terminal ASCII fallback
# --------------------------------------------------------------------------- #
def play_text(payload, cols=80, frames_limit=None):
    """Crude ASCII playback in the terminal (the 'text/JSON fallback' viewer)."""
    import time
    w, h, levels = payload["w"], payload["h"], payload["fps"],
    w = payload["w"]; h = payload["h"]; levels = payload["levels"]; fps = payload["fps"]
    ramp = " .:-=+*#%@"
    # reconstruct full index buffers from deltas
    buf = np.zeros(w * h, dtype=np.int32)
    aspect = 0.5  # terminal chars are ~2x tall
    out_w = min(cols, w)
    out_h = max(1, int(h * (out_w / w) * aspect))
    try:
        for fi, f in enumerate(payload["frames"]):
            if f["t"] == 0:
                buf = np.array(f["p"], dtype=np.int32)
            else:
                d = f["d"]
                if d:
                    arr = np.array(d, dtype=np.int32)
                    buf[arr[0::2]] = arr[1::2]
            # index -> luminance
            r = (buf // (levels * levels)) % levels
            g = (buf // levels) % levels
            b = buf % levels
            lum = (0.3 * r + 0.59 * g + 0.11 * b) / (levels - 1)
            img = lum.reshape(h, w)
            sys.stdout.write("\033[H\033[2J")  # home + clear
            for oy in range(out_h):
                sy = int(oy / out_h * h)
                row = []
                for ox in range(out_w):
                    sx = int(ox / out_w * w)
                    v = img[sy, sx]
                    row.append(ramp[min(len(ramp) - 1, int(v * (len(ramp) - 1)))])
                sys.stdout.write("".join(row) + "\n")
            sys.stdout.flush()
            time.sleep(1.0 / fps)
            if frames_limit and fi + 1 >= frames_limit:
                break
    except KeyboardInterrupt:
        pass


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Convert a video into a self-contained glitch-art HTML animation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("input", nargs="?", help="Input video (any ffmpeg format).")
    ap.add_argument("-o", "--output", default="glitch_animation.html",
                    help="Output HTML file.")
    ap.add_argument("--width", type=int, default=96, help="Grid width (pixels).")
    ap.add_argument("--height", type=int, default=54, help="Grid height (pixels).")
    ap.add_argument("--fps", type=int, default=12, help="Playback / sampling fps.")
    ap.add_argument("--levels", type=int, default=6,
                    help="Quantization levels per channel (levels**3 colors).")
    ap.add_argument("--keyint", type=int, default=24,
                    help="Force a keyframe every N frames.")
    ap.add_argument("--delta-thresh", type=float, default=0.45,
                    help="Promote delta->keyframe if more than this fraction changed.")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="Cap number of frames (useful for long clips).")
    ap.add_argument("--preset", choices=list(PRESETS), default="heavy-datamosh",
                    help="Default glitch preset baked into the player.")
    ap.add_argument("--scale", type=int, default=8,
                    help="Display upscale factor (canvas = grid * scale).")
    ap.add_argument("--mosaic", action="store_true",
                    help="Start with dominant-color mosaic mode ON (overrides preset).")
    ap.add_argument("--box-size", type=int, default=None,
                    help="Mosaic block render size in display px (e.g. 4, 8, 16, 32).")
    ap.add_argument("--sample-size", type=int, default=None,
                    help="Mosaic dominant-color sampling area in display px "
                         "(independent of box size; defaults to the box size).")
    ap.add_argument("--audio", dest="audio", action="store_true", default=None,
                    help="Embed the source soundtrack (synced playback). Default: "
                         "embed automatically when the source has audio.")
    ap.add_argument("--no-audio", dest="audio", action="store_false",
                    help="Do not embed the source soundtrack (keeps the file smaller).")
    ap.add_argument("--audio-bitrate", default="24k",
                    help="Embedded audio bitrate (mono), e.g. 16k/24k/32k.")
    ap.add_argument("--decoder", choices=["ffmpeg", "opencv"], default="ffmpeg",
                    help="Frame decoder backend.")
    ap.add_argument("--json", dest="json_out", default=None,
                    help="Also write raw (uncompressed) JSON to this path.")
    ap.add_argument("--play-text", action="store_true",
                    help="Play an ASCII preview in the terminal (no HTML written).")
    ap.add_argument("--selftest", action="store_true",
                    help="Use a synthetic demo animation (no input video needed).")
    ap.add_argument("--build-web", nargs="?", const="index.html", default=None,
                    metavar="PATH",
                    help="Write the in-browser converter page (default index.html) "
                         "for GitHub Pages, then exit. No input video needed.")
    args = ap.parse_args()

    # --- web converter page (client-side; for GitHub Pages) ----------------- #
    if args.build_web is not None:
        with open(args.build_web, "w") as f:
            f.write(build_web_html())
        print(f"[✓] Wrote {args.build_web} — host it (e.g. GitHub Pages) and "
              f"anyone can convert videos in-browser.", file=sys.stderr)
        return

    w, h, levels, fps = args.width, args.height, args.levels, args.fps
    if not (2 <= levels <= 8):
        ap.error("--levels must be between 2 and 8 (levels**3 must fit in a byte; "
                 "8**3=512 still fits Uint16 in JS but 6 is the sweet spot).")

    # --- choose frame source ------------------------------------------------ #
    if args.selftest or not args.input:
        if not args.input and not args.selftest:
            print("[i] No input video given -> generating synthetic demo "
                  "(use a path argument for a real video).", file=sys.stderr)
        src = synthetic_frames(w, h, fps)
    else:
        if not os.path.isfile(args.input):
            ap.error(f"Input file not found: {args.input}")
        if args.decoder == "opencv":
            src = decode_opencv(args.input, w, h, fps, args.max_frames)
        else:
            src = decode_ffmpeg(args.input, w, h, fps, args.max_frames)

    # --- encode ------------------------------------------------------------- #
    print(f"[i] Encoding {w}x{h} @ {fps}fps, {levels} levels/channel ...",
          file=sys.stderr)
    frames, nframes = encode(src, w, h, levels, args.keyint, args.delta_thresh)
    if nframes == 0:
        ap.error("No frames decoded. Is the input a valid video?")
    payload = build_payload(frames, w, h, levels, fps)
    print(f"[i] {nframes} frames encoded.", file=sys.stderr)

    # --- terminal preview --------------------------------------------------- #
    if args.play_text:
        print(f"[i] Playing ASCII preview ({nframes} frames). Ctrl-C to stop.",
              file=sys.stderr)
        play_text(payload)
        return

    # --- optional raw JSON sidecar ----------------------------------------- #
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(payload, f, separators=(",", ":"))
        print(f"[i] Wrote raw JSON -> {args.json_out}", file=sys.stderr)

    # --- optional embedded soundtrack -------------------------------------- #
    # Default: embed when a real video is given and has audio (args.audio is None).
    # --audio forces it; --no-audio (args.audio is False) skips it entirely.
    audio = None
    want_audio = args.audio is not False and args.input and os.path.isfile(args.input)
    if want_audio:
        audio = extract_audio(args.input, args.audio_bitrate)
        if audio:
            print(f"[i] Embedded soundtrack: {len(audio['b64'])*3//4/1024:.1f} KB "
                  f"({audio['mime']}, mono {args.audio_bitrate}).", file=sys.stderr)
        elif args.audio:
            print("[!] --audio requested but no audio track was extracted.", file=sys.stderr)

    # --- pack + write HTML -------------------------------------------------- #
    data_b64, raw_len, gz_len = pack(payload)
    ui_config = {
        "scale": args.scale,
        "preset": args.preset,
        "presets": PRESETS,
        # mosaic overrides (None -> let the preset decide; values win over preset)
        "mosaic": True if args.mosaic else None,
        "boxSize": args.box_size,
        "sampleSize": args.sample_size,
    }
    html = build_html(data_b64, ui_config, audio)
    with open(args.output, "w") as f:
        f.write(html)

    html_size = len(html.encode("utf-8"))
    print(f"[i] JSON {raw_len/1024:.1f} KB -> gzip {gz_len/1024:.1f} KB "
          f"-> HTML {html_size/1024:.1f} KB", file=sys.stderr)
    if args.input and os.path.isfile(args.input):
        src_size = os.path.getsize(args.input)
        print(f"[i] Source video {src_size/1024:.1f} KB -> "
              f"HTML is {100*html_size/src_size:.1f}% of original.", file=sys.stderr)
    print(f"[✓] Wrote {args.output} — open it in any browser.", file=sys.stderr)


# --------------------------------------------------------------------------- #
# HTML PLAYER TEMPLATE
# (kept in sync with glitch_animation.html — the standalone file is this same
#  template with DATA=null so it plays a synthetic demo with zero setup.)
# --------------------------------------------------------------------------- #
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Glitch Animation</title>
<style>
  :root { --bg:#07080c; --fg:#d8f5ff; --acc:#ff2bd6; --acc2:#23f0ff; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:14px/1.4 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
         display:flex; flex-direction:column; align-items:center; min-height:100vh; }
  h1 { font-size:15px; letter-spacing:.25em; text-transform:uppercase; margin:18px 0 4px;
       color:var(--acc2); text-shadow:0 0 8px var(--acc2); }
  .sub { opacity:.5; margin-bottom:14px; font-size:11px; }
  .stage { position:relative; background:#000; border:1px solid #1a2430;
           box-shadow:0 0 40px rgba(35,240,255,.12); image-rendering:pixelated; }
  canvas { display:block; image-rendering:pixelated; }
  /* ---- control panel : sectioned / collapsible, compact ---- */
  .panel { width:min(820px,94vw); margin:16px 0 48px;
           border:1px solid #1a2735; border-radius:8px; background:#0a0d13;
           overflow:hidden; box-shadow:0 10px 40px rgba(0,0,0,.4); }
  .head { display:flex; gap:10px; align-items:center; padding:12px 16px;
          background:linear-gradient(180deg,#0f141d,#0b0f17); border-bottom:1px solid #1a2735; }
  .head .grow { flex:1; }
  .head .tag { font-size:10px; opacity:.45; letter-spacing:.08em; }

  .sec { border-bottom:1px solid #131c26; }
  .sec:last-child { border-bottom:0; }
  .sec > summary { list-style:none; cursor:pointer; user-select:none;
                   padding:11px 16px; display:flex; align-items:center; gap:9px;
                   font-size:10.5px; text-transform:uppercase; letter-spacing:.16em;
                   color:var(--acc2); transition:background .12s; }
  .sec > summary:hover { background:rgba(35,240,255,.05); }
  .sec > summary::-webkit-details-marker { display:none; }
  .sec > summary::before { content:"▸"; font-size:9px; opacity:.7; transition:transform .15s; }
  .sec[open] > summary::before { transform:rotate(90deg); }
  .sec > summary .muted { font-size:8.5px; letter-spacing:.1em; color:var(--fg);
                          opacity:.38; text-transform:none; }
  .sec > summary .spacer { flex:1; }
  .sec .body { padding:2px 16px 16px; }

  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(155px,1fr)); gap:13px 18px; }
  .row { display:flex; flex-direction:column; gap:5px; }
  .row label { font-size:9.5px; text-transform:uppercase; letter-spacing:.1em; opacity:.62;
               display:flex; justify-content:space-between; gap:8px; }
  .row .val { color:var(--acc); opacity:1; }
  input[type=range] { width:100%; accent-color:var(--acc); height:3px; }

  select, button { background:#11161f; color:var(--fg); border:1px solid #243140;
                   border-radius:5px; padding:7px 11px; font:inherit; font-size:12px; cursor:pointer; }
  button:hover, select:hover { border-color:var(--acc2); }
  button.on { border-color:var(--acc); color:var(--acc); }

  .toolbar { display:flex; gap:8px; flex-wrap:wrap; align-items:center; }
  .toolbar .grow { flex:1; }
  .toolbar .sep { width:1px; align-self:stretch; background:#1a2735; margin:0 2px; }
  .bar { display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
  .bar .grow { flex:1; }
  .chk { font-size:10px; opacity:.7; display:flex; align-items:center; gap:6px;
         text-transform:uppercase; letter-spacing:.08em; }
  button.rec { border-color:#ff3b3b; color:#ff3b3b; animation:pulse 1s infinite; }
  @keyframes pulse { 50% { opacity:.55; } }
  .msg { font-size:11px; color:var(--acc); min-height:1.3em; padding:0 16px 12px; }
  .tag { font-size:10px; opacity:.4; }
  a { color:var(--acc2); }
</style>
</head>
<body>
  <h1>// Glitch Animation</h1>
  <div class="sub" id="meta">loading…</div>
  <div class="stage"><canvas id="screen"></canvas></div>

  <div class="panel">

    <!-- transport header -->
    <div class="head">
      <button id="playBtn">⏸ Pause</button>
      <button id="loopBtn">🔁 Loop: on</button>
      <select id="preset" title="Preset / favorite" style="max-width:200px;"></select>
      <span class="grow"></span>
      <span class="tag" id="frameTag">frame 0/0</span>
    </div>

    <!-- PLAYBACK -->
    <details class="sec" open>
      <summary>Playback &amp; master</summary>
      <div class="body grid">
        <div class="row">
          <label>Speed <span class="val" id="speedV">1.0×</span></label>
          <input type="range" id="speed" min="0.1" max="3" step="0.1" value="1">
        </div>
        <div class="row">
          <label>Master ×ALL effects <span class="val" id="intV">1.0×</span></label>
          <input type="range" id="intensity" min="0" max="2.5" step="0.05" value="1">
        </div>
      </div>
    </details>

    <!-- GLITCH -->
    <details class="sec" open>
      <summary>Glitch effects <span class="muted">scaled by Master</span></summary>
      <div class="body grid">
        <div class="row">
          <label>RGB split <span class="val" id="rgbV"></span></label>
          <input type="range" id="rgb" min="0" max="24" step="1">
        </div>
        <div class="row">
          <label>Block tears <span class="val" id="blockV"></span></label>
          <input type="range" id="block" min="0" max="1" step="0.02">
        </div>
        <div class="row">
          <label>Scanline tears <span class="val" id="tearV"></span></label>
          <input type="range" id="tear" min="0" max="1" step="0.02">
        </div>
        <div class="row">
          <label>Noise (b/w sparkle) <span class="val" id="noiseV"></span></label>
          <input type="range" id="noise" min="0" max="0.25" step="0.005">
        </div>
        <div class="row">
          <label>TV static (snow) <span class="val" id="staticV"></span></label>
          <input type="range" id="static" min="0" max="1" step="0.02">
        </div>
        <div class="row">
          <label>Corruption <span class="val" id="corruptV"></span></label>
          <input type="range" id="corrupt" min="0" max="0.25" step="0.005">
        </div>
      </div>
    </details>

    <!-- SCANLINES / CRT -->
    <details class="sec">
      <summary>Scanlines / CRT <span class="muted">independent of Master</span></summary>
      <div class="body grid">
        <div class="row">
          <label>Darkness <span class="val" id="scanV"></span></label>
          <input type="range" id="scanline" min="0" max="1" step="0.02">
        </div>
        <div class="row">
          <label>Line gap <span class="val" id="scanGapV"></span></label>
          <input type="range" id="scanGap" min="2" max="40" step="1">
        </div>
        <div class="row">
          <label>Line size <span class="val" id="scanSizeV"></span></label>
          <input type="range" id="scanSize" min="1" max="20" step="1">
        </div>
      </div>
    </details>

    <!-- MOSAIC -->
    <details class="sec">
      <summary>Dominant-color mosaic
        <span class="muted">base layer · sampled from pristine frame</span>
        <span class="spacer"></span>
        <button id="mosaicToggle" style="padding:3px 10px;font-size:10px;">off</button>
      </summary>
      <div class="body grid">
        <div class="row">
          <label>Box size (render) <span class="val" id="boxV"></span></label>
          <input type="range" id="box" min="2" max="64" step="1">
        </div>
        <div class="row">
          <label>Sampling area <span class="val" id="sampleV"></span></label>
          <input type="range" id="sample" min="2" max="96" step="1">
        </div>
      </div>
    </details>

    <!-- SOUND -->
    <details class="sec">
      <summary>Sound</summary>
      <div class="body toolbar">
        <button id="soundBtn" title="Enable audio (click to allow sound)">🔇 Sound: off</button>
        <button id="sfxBtn" title="Procedural glitch sound effects">✧ SFX: on</button>
        <label class="chk">Vol
          <input type="range" id="volume" min="0" max="1" step="0.02" value="0.6" style="width:120px;"></label>
      </div>
    </details>

    <!-- EXPORT -->
    <details class="sec" open>
      <summary>Export &amp; share</summary>
      <div class="body toolbar">
        <button id="dlBtn" title="Render &amp; download the whole animation offline (WebCodecs — no replay, no dropped frames)">⬇ Download video</button>
        <label class="chk">format
          <select id="dlFormat" style="padding:5px 8px;">
            <option value="webm">WebM</option><option value="mp4">MP4</option>
          </select></label>
        <label class="chk"><input type="checkbox" id="dlAudio" checked> with audio</label>
        <label class="chk">loops
          <select id="dlLoops" style="padding:5px 8px;">
            <option value="1">1</option><option value="2">2</option>
            <option value="3">3</option><option value="5">5</option>
          </select></label>
        <span class="muted" id="dlEst" title="Rough estimate from resolution, fps, length and effect intensity"></span>
        <span class="sep"></span>
        <button id="recBtn" title="Manually start/stop a live recording">🎞 Record</button>
        <button id="pngBtn" title="Download the current frame as PNG">📷 PNG</button>
        <span class="grow"></span>
        <button id="linkBtn" title="Copy a link that reproduces this exact look">🔗 Copy look</button>
        <button id="randBtn" title="Randomize all effect sliders">🎲 Random</button>
      </div>
      <div class="body toolbar" style="padding-top:0;">
        <button id="favBtn" title="Save current settings as a reusable favorite">★ Save favorite</button>
        <button id="favDelBtn" title="Delete the selected favorite">🗑 Delete favorite</button>
      </div>
    </details>

    <div class="msg" id="exportMsg"></div>
  </div>

<script>
// ===========================================================================
//  EMBEDDED DATA + CONFIG  (replaced by glitch_processor.py)
// ===========================================================================
const DATA_B64 = /*__DATA__*/null;          // base64(gzip(json)) or null -> synthetic demo
const CONFIG   = /*__CONFIG__*/({});         // { scale, preset, presets:{...} }
const AUDIO    = /*__AUDIO__*/null;          // { mime, b64 } embedded soundtrack, or null

// Default config when running the standalone file with no Python step:
const DEFAULTS = {
  scale: 8,
  preset: "heavy-datamosh",
  presets: {
    "heavy-datamosh": {rgb:8, block:0.7, tear:0.35, noise:0.04, static:0.0,  corrupt:0.07, blockSize:20, bands:10, scanline:0.0, scanGap:4, scanSize:1, mosaic:false, box:8, sample:8},
    "subtle-neon":    {rgb:7, block:0.0, tear:0.06, noise:0.0,  static:0.0,  corrupt:0.0,  blockSize:30, bands:3,  scanline:0.0, scanGap:4, scanSize:1, mosaic:false, box:8, sample:8},
    "vhs":            {rgb:3, block:0.0, tear:0.5,  noise:0.09, static:0.12, corrupt:0.0,  blockSize:40, bands:22, scanline:0.5, scanGap:6, scanSize:2, mosaic:false, box:8, sample:8},
    "clean":          {rgb:0, block:0.0, tear:0.0,  noise:0.0,  static:0.0,  corrupt:0.0,  blockSize:24, bands:0,  scanline:0.0, scanGap:4, scanSize:1, mosaic:false, box:8, sample:8},
    "clean-mosaic":          {rgb:0, block:0.0,  tear:0.0, noise:0.0,  static:0.0,  corrupt:0.0,  blockSize:24, bands:0,  scanline:0.0, scanGap:4, scanSize:1, mosaic:true, box:16, sample:16},
    "glitch-mosaic":         {rgb:5, block:0.25, tear:0.2, noise:0.02, static:0.0,  corrupt:0.02, blockSize:18, bands:8,  scanline:0.1, scanGap:5, scanSize:1, mosaic:true, box:12, sample:14},
    "heavy-datamosh-mosaic": {rgb:9, block:0.7,  tear:0.4, noise:0.05, static:0.05, corrupt:0.08, blockSize:16, bands:12, scanline:0.0, scanGap:4, scanSize:1, mosaic:true, box:10, sample:18},
  }
};
const CFG = Object.assign({}, DEFAULTS, CONFIG || {});
CFG.presets = Object.assign({}, DEFAULTS.presets, (CONFIG && CONFIG.presets) || {});

// ===========================================================================
//  DATA LOADING / DECODING
//  Produces: { W, H, LEVELS, FPS, FRAMES:[Uint8Array(W*H) of palette indices] }
// ===========================================================================
async function ungzipToJSON(b64) {
  const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
  const buf = await new Response(stream).arrayBuffer();
  return JSON.parse(new TextDecoder().decode(buf));
}

function reconstructFrames(payload) {
  // Expand keyframes + deltas into full per-frame index buffers.
  const { w, h, frames } = payload;
  const npix = w * h;
  const out = [];
  let cur = new Uint16Array(npix);
  for (const f of frames) {
    if (f.t === 0) {
      cur = Uint16Array.from(f.p);
    } else {
      cur = cur.slice();                 // copy previous, apply delta pairs
      const d = f.d;
      for (let i = 0; i < d.length; i += 2) cur[d[i]] = d[i + 1];
    }
    out.push(cur);
  }
  return out;
}

function syntheticAnimation() {
  // Self-contained demo so the bare HTML file plays with zero setup.
  const W = 96, H = 54, LEVELS = 6, FPS = 12, N = 60;
  const FRAMES = [];
  for (let i = 0; i < N; i++) {
    const t = i / FPS;
    const buf = new Uint16Array(W * H);
    const bx = Math.floor((Math.sin(t * 1.7) * 0.4 + 0.5) * (W - W / 4));
    const by = Math.floor((Math.cos(t * 1.3) * 0.4 + 0.5) * (H - H / 4));
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      let r = (Math.sin(x / 8 + t * 2) + Math.cos(y / 6 - t * 1.5)) * 0.5 + 0.5;
      let g = Math.sin((x + y) / 10 - t * 2.2) * 0.5 + 0.5;
      let b = Math.cos(x / 7 + y / 9 + t) * 0.5 + 0.5;
      if (x >= bx && x < bx + W / 4 && y >= by && y < by + H / 4) { r = 1; g = 0.2; b = 0.6; }
      const qr = Math.min(LEVELS - 1, r * LEVELS | 0);
      const qg = Math.min(LEVELS - 1, g * LEVELS | 0);
      const qb = Math.min(LEVELS - 1, b * LEVELS | 0);
      buf[y * W + x] = (qr * LEVELS + qg) * LEVELS + qb;
    }
    FRAMES.push(buf);
  }
  return { W, H, LEVELS, FPS, FRAMES };
}

async function loadAnimation() {
  if (!DATA_B64) return syntheticAnimation();
  const payload = await ungzipToJSON(DATA_B64);
  return {
    W: payload.w, H: payload.h, LEVELS: payload.levels, FPS: payload.fps,
    FRAMES: reconstructFrames(payload),
  };
}

// ===========================================================================
//  PALETTE (index -> RGB) lookup built from the uniform-quantization scheme.
// ===========================================================================
function buildPalette(levels) {
  const n = levels * levels * levels;
  const pal = new Uint8Array(n * 3);
  const scale = 255 / (levels - 1);
  for (let i = 0; i < n; i++) {
    const r = (i / (levels * levels) | 0) % levels;
    const g = (i / levels | 0) % levels;
    const b = i % levels;
    pal[i * 3]     = Math.round(r * scale);
    pal[i * 3 + 1] = Math.round(g * scale);
    pal[i * 3 + 2] = Math.round(b * scale);
  }
  return pal;
}

// ===========================================================================
//  GLITCH RENDERER
//  Clean low-res frame -> upscaled canvas -> live procedural glitch.
// ===========================================================================
const screen = document.getElementById("screen");
const ctx = screen.getContext("2d", { willReadFrequently: true });
const small = document.createElement("canvas");
const sctx = small.getContext("2d", { willReadFrequently: true });

let ANIM, PAL, dispW, dispH, baseImg;

function setupCanvas(anim) {
  small.width = anim.W; small.height = anim.H;
  const scale = Math.max(2, CFG.scale | 0);
  dispW = anim.W * scale; dispH = anim.H * scale;
  screen.width = dispW; screen.height = dispH;
  ctx.imageSmoothingEnabled = false;
}

// Draw the clean frame to the small canvas, then upscale (nearest) to the big
// canvas, and grab the upscaled pixels as our glitch source buffer.
function rasterizeBase(buf) {
  const W = ANIM.W, H = ANIM.H;
  const img = sctx.createImageData(W, H);
  const d = img.data;
  for (let i = 0; i < W * H; i++) {
    const idx = buf[i] * 3;
    d[i * 4]     = PAL[idx];
    d[i * 4 + 1] = PAL[idx + 1];
    d[i * 4 + 2] = PAL[idx + 2];
    d[i * 4 + 3] = 255;
  }
  sctx.putImageData(img, 0, 0);
  ctx.drawImage(small, 0, 0, dispW, dispH);
  baseImg = ctx.getImageData(0, 0, dispW, dispH);
}

// ===========================================================================
//  DOMINANT-COLOR MOSAIC (base layer)
//  IMPORTANT: the dominant color is sampled from `grid` — the PRISTINE decoded
//  frame (palette indices) — NEVER from baseImg after glitch. We overwrite
//  baseImg with solid blocks here, then glitchFrame() overlays on top.
// ===========================================================================
let mosaicHist = null;   // histogram over palette indices (allocated once we know LEVELS)

// Most-frequent palette index (histogram peak) within a grid rectangle.
function dominantIndex(grid, gx0, gy0, gx1, gy1) {
  const W = ANIM.W;
  const touched = [];
  let best = grid[gy0 * W + gx0], bestCount = 0;
  for (let gy = gy0; gy < gy1; gy++) {
    const row = gy * W;
    for (let gx = gx0; gx < gx1; gx++) {
      const idx = grid[row + gx];
      if (mosaicHist[idx] === 0) touched.push(idx);
      const c = ++mosaicHist[idx];
      if (c > bestCount) { bestCount = c; best = idx; }
    }
  }
  for (let i = 0; i < touched.length; i++) mosaicHist[touched[i]] = 0;  // reset for next box
  return best;
}

// Rewrite baseImg as solid dominant-color blocks. `boxPx` = rendered block size,
// `samplePx` = size of the region scanned for the dominant color (independent).
function applyMosaic(grid, boxPx, samplePx) {
  const d = baseImg.data;
  const W = ANIM.W, H = ANIM.H;
  const sc = dispW / W;                 // display px per grid cell (= CFG.scale)
  const box = Math.max(1, boxPx | 0);
  const half = Math.max(sc, samplePx) / 2;
  for (let by = 0; by < dispH; by += box) {
    for (let bx = 0; bx < dispW; bx += box) {
      // sample region (in grid space) centered on this block, from PRISTINE grid
      const cx = bx + box / 2, cy = by + box / 2;
      let gx0 = Math.floor((cx - half) / sc), gx1 = Math.ceil((cx + half) / sc);
      let gy0 = Math.floor((cy - half) / sc), gy1 = Math.ceil((cy + half) / sc);
      if (gx0 < 0) gx0 = 0; if (gy0 < 0) gy0 = 0;
      if (gx1 > W) gx1 = W; if (gy1 > H) gy1 = H;
      if (gx1 <= gx0) gx1 = gx0 + 1; if (gy1 <= gy0) gy1 = gy0 + 1;
      const idx = dominantIndex(grid, gx0, gy0, gx1, gy1) * 3;
      const r = PAL[idx], g = PAL[idx + 1], b = PAL[idx + 2];
      // fill the rendered block
      const x1 = Math.min(dispW, bx + box), y1 = Math.min(dispH, by + box);
      for (let y = by; y < y1; y++) {
        let di = (y * dispW + bx) << 2;
        for (let x = bx; x < x1; x++) {
          d[di] = r; d[di + 1] = g; d[di + 2] = b; d[di + 3] = 255; di += 4;
        }
      }
    }
  }
}

function sampleClamp(x, y, off) {
  // clamp coords, return source byte offset for channel sampling
  if (x < 0) x = 0; else if (x >= dispW) x = dispW - 1;
  if (y < 0) y = 0; else if (y >= dispH) y = dispH - 1;
  return ((y * dispW + x) << 2) + off;
}

// Small seeded PRNG (mulberry32). A *seeded* RNG is the whole point of the fix:
// the glitch layout is deterministic per `seed`, so it HOLDS for a displayed
// frame instead of strobing at 60fps. That makes each slider's effect legible
// and presets read as distinct looks. The seed advances once per source frame.
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function glitchFrame(p, seed) {
  const rnd = mulberry32(seed >>> 0);
  const s = baseImg.data;
  const out = ctx.createImageData(dispW, dispH);
  const o = out.data;

  // --- horizontal scanline tears: shift random horizontal bands sideways --- //
  // tear strength scales the *number* of torn rows too, so 0 = perfectly clean
  // and high values rip large chunks of the frame sideways.
  const tear = new Int16Array(dispH);
  const bands = (p.bands * (0.3 + p.tear)) | 0;
  for (let i = 0; i < bands; i++) {
    const y0 = rnd() * dispH | 0;
    const bh = 2 + (rnd() * dispH * 0.18 | 0);
    const off = ((rnd() * 2 - 1) * dispW * 0.5 * p.tear) | 0;
    for (let y = y0; y < Math.min(dispH, y0 + bh); y++) tear[y] = off;
  }

  // --- block displacement (datamosh-style) -------------------------------- //
  const bs = Math.max(4, p.blockSize | 0);
  const bcols = Math.ceil(dispW / bs), brows = Math.ceil(dispH / bs);
  const bx = new Int16Array(bcols * brows), by = new Int16Array(bcols * brows);
  for (let i = 0; i < bcols * brows; i++) {
    if (rnd() < p.block) {
      bx[i] = ((rnd() * 2 - 1) * bs * 2.2) | 0;
      by[i] = ((rnd() * 2 - 1) * bs * 1.2) | 0;
    }
  }

  const chroma = p.rgb | 0;
  for (let y = 0; y < dispH; y++) {
    const tx = tear[y];
    const brow = (y / bs | 0) * bcols;
    for (let x = 0; x < dispW; x++) {
      const bi = brow + (x / bs | 0);
      const sx = x + tx + bx[bi];
      const sy = y + by[bi];
      const di = (y * dispW + x) << 2;
      // RGB channel separation / color fringing
      o[di]     = s[sampleClamp(sx + chroma, sy, 0)];
      o[di + 1] = s[sampleClamp(sx,          sy, 1)];
      o[di + 2] = s[sampleClamp(sx - chroma, sy, 2)];
      o[di + 3] = 255;
    }
  }

  // --- region corruption: tear-and-quantize horizontal slabs -------------- //
  // Number of corrupt slabs scales with the slider, so it's continuous (not a
  // rare all-or-nothing flash) and clearly responds to the control.
  const slabs = Math.round(p.corrupt * 14);
  for (let n = 0; n < slabs; n++) {
    const cy = rnd() * dispH | 0;
    const ch = (dispH * (0.04 + rnd() * 0.22)) | 0;
    const shift = (rnd() * dispW) | 0;
    const qbits = 2 + (rnd() * 4 | 0);
    const mask = ~((1 << qbits) - 1) & 0xff;
    const tint = rnd() < 0.5;             // alternate which channel gets blown out
    for (let y = cy; y < Math.min(dispH, cy + ch); y++) {
      for (let x = 0; x < dispW; x++) {
        const di = (y * dispW + x) << 2;
        const si = sampleClamp((x + shift) % dispW, y, 0);
        o[di]     = (s[si]     & mask) | (tint ? (~mask & 0xff) : 0);
        o[di + 1] = (s[si + 1] & mask);
        o[di + 2] = (s[si + 2] & mask) | (tint ? 0 : (~mask & 0xff));
      }
    }
  }

  // --- noise sparkle (single-pixel b/w dots) ------------------------------ //
  const noiseCount = (p.noise * dispW * dispH) | 0;
  for (let i = 0; i < noiseCount; i++) {
    const di = ((rnd() * dispW * dispH) | 0) << 2;
    const v = rnd() < 0.5 ? 255 : 0;
    o[di] = v; o[di + 1] = v; o[di + 2] = v;
  }

  // --- TV static "snow": grayscale grain blended over the whole picture --- //
  // At p.static = 1 the picture is fully replaced by random gray grain (an
  // untuned channel); lower values mix snow over the image like weak reception.
  if (p.static > 0) {
    const amt = p.static;
    const grain = Math.max(1, (CFG.scale / 4) | 0);   // chunky snow, scales with display
    for (let y = 0; y < dispH; y += grain) {
      for (let x = 0; x < dispW; x += grain) {
        const g = (rnd() * 255) | 0;                  // one gray value per grain cell
        const x1 = Math.min(dispW, x + grain), y1 = Math.min(dispH, y + grain);
        for (let yy = y; yy < y1; yy++) {
          for (let xx = x; xx < x1; xx++) {
            const di = (yy * dispW + xx) << 2;
            o[di]     += (g - o[di])     * amt;
            o[di + 1] += (g - o[di + 1]) * amt;
            o[di + 2] += (g - o[di + 2]) * amt;
          }
        }
      }
    }
  }

  ctx.putImageData(out, 0, 0);

  // --- scanline / CRT overlay (drawn over the pixels) --------------------- //
  if (p.scanline > 0) {
    const gap = Math.max(1, p.scanGap | 0);             // separation between line starts
    const size = Math.max(1, Math.min(gap, p.scanSize | 0)); // thickness of each dark line
    ctx.globalAlpha = p.scanline;
    ctx.fillStyle = "#000";
    for (let y = 0; y < dispH; y += gap) ctx.fillRect(0, y, dispW, size);
    ctx.globalAlpha = 1;
  }
}

// ===========================================================================
//  PLAYBACK LOOP + UI
// ===========================================================================
const ui = {
  playBtn: byId("playBtn"), loopBtn: byId("loopBtn"), frameTag: byId("frameTag"),
  meta: byId("meta"), presetSel: byId("preset"),
  speed: byId("speed"), intensity: byId("intensity"),
  rgb: byId("rgb"), block: byId("block"), tear: byId("tear"),
  noise: byId("noise"), static: byId("static"), corrupt: byId("corrupt"),
  scanline: byId("scanline"), scanGap: byId("scanGap"), scanSize: byId("scanSize"),
  box: byId("box"), sample: byId("sample"), mosaicToggle: byId("mosaicToggle"),
  pngBtn: byId("pngBtn"), recBtn: byId("recBtn"), linkBtn: byId("linkBtn"),
  randBtn: byId("randBtn"), exportMsg: byId("exportMsg"),
  soundBtn: byId("soundBtn"), sfxBtn: byId("sfxBtn"), volume: byId("volume"),
  favBtn: byId("favBtn"), favDelBtn: byId("favDelBtn"),
  dlBtn: byId("dlBtn"), dlAudio: byId("dlAudio"), dlLoops: byId("dlLoops"),
  dlFormat: byId("dlFormat"), dlEst: byId("dlEst"),
};
function byId(id){ return document.getElementById(id); }

let state = { playing: true, loop: true, speed: 1, intensity: 1, frame: 0, acc: 0, last: 0, seed: 1 };
let params = {};       // current base preset values (pre-intensity)
let mosaicOn = false;  // mosaic toggle (set by preset / toggle button)

function applyPresetToUI(name) {
  const p = CFG.presets[name] || CFG.presets["heavy-datamosh"];
  params = Object.assign({}, p);
  ui.rgb.value = p.rgb; ui.block.value = p.block; ui.tear.value = p.tear;
  ui.noise.value = p.noise; ui.corrupt.value = p.corrupt; ui.scanline.value = p.scanline;
  ui.static.value = p.static != null ? p.static : 0;
  ui.scanGap.value = p.scanGap != null ? p.scanGap : 4;
  ui.scanSize.value = p.scanSize != null ? p.scanSize : 1;
  ui.box.value = p.box != null ? p.box : 8;
  ui.sample.value = p.sample != null ? p.sample : 8;
  mosaicOn = !!p.mosaic;
  updateMosaicBtn();
  readUIParams();   // rebuild params uniformly from the UI (keeps mosaic in sync)
}

function updateMosaicBtn() {
  ui.mosaicToggle.textContent = mosaicOn ? "ON" : "off";
  ui.mosaicToggle.style.borderColor = mosaicOn ? "var(--acc)" : "#243140";
  ui.mosaicToggle.style.color = mosaicOn ? "var(--acc)" : "var(--fg)";
}

function readUIParams() {
  params = {
    rgb: +ui.rgb.value, block: +ui.block.value, tear: +ui.tear.value,
    noise: +ui.noise.value, static: +ui.static.value,
    corrupt: +ui.corrupt.value, scanline: +ui.scanline.value,
    scanGap: +ui.scanGap.value, scanSize: +ui.scanSize.value,
    mosaic: mosaicOn, box: +ui.box.value, sample: +ui.sample.value,
    blockSize: (CFG.presets[ui.presetSel.value] || {}).blockSize || 22,
    bands: (CFG.presets[ui.presetSel.value] || {}).bands || 12,
  };
  syncReadouts();
}

function effectiveParams() {
  const k = state.intensity;
  return {
    rgb: params.rgb * k,
    block: Math.min(1, params.block * k),
    tear: Math.min(1, params.tear * k),
    noise: Math.min(0.5, params.noise * k),
    static: Math.min(1, params.static * k),
    corrupt: Math.min(0.5, params.corrupt * k),
    scanline: params.scanline,  // independent of master (CRT overlay, not glitch intensity)
    scanGap: params.scanGap,    // geometry: not scaled by master
    scanSize: params.scanSize,
    blockSize: params.blockSize,
    bands: Math.round(params.bands * Math.min(2, k)),
    mosaic: params.mosaic, box: params.box, sample: params.sample,  // base layer, not master-scaled
  };
}

function syncReadouts() {
  byId("rgbV").textContent     = (+ui.rgb.value).toFixed(0) + "px";
  byId("blockV").textContent   = (+ui.block.value).toFixed(2);
  byId("tearV").textContent    = (+ui.tear.value).toFixed(2);
  byId("noiseV").textContent   = (+ui.noise.value).toFixed(3);
  byId("staticV").textContent  = (+ui.static.value).toFixed(2);
  byId("corruptV").textContent = (+ui.corrupt.value).toFixed(3);
  byId("scanV").textContent    = (+ui.scanline.value).toFixed(2);
  byId("scanGapV").textContent  = (+ui.scanGap.value).toFixed(0) + "px";
  byId("scanSizeV").textContent = (+ui.scanSize.value).toFixed(0) + "px";
  byId("boxV").textContent     = (+ui.box.value).toFixed(0) + "px";
  byId("sampleV").textContent  = (+ui.sample.value).toFixed(0) + "px";
  byId("speedV").textContent   = state.speed.toFixed(1) + "×";
  byId("intV").textContent     = state.intensity === 0
      ? "0× — ALL OFF" : state.intensity.toFixed(2) + "×";
}

// Render whatever frame we're on with the CURRENT params + CURRENT seed.
// Used for instant feedback when paused / when a slider or preset changes — the
// seed is held so only the dimension you're dragging changes, making each
// slider's contribution obvious.
// Draw one fully-composited glitch frame to the canvas (deterministic for a
// given frameIndex + seed). Shared by live playback and the offline exporter.
function drawGlitchFrame(frameIndex, seed) {
  const grid = ANIM.FRAMES[frameIndex];
  const ep = effectiveParams();
  rasterizeBase(grid);                              // pristine upscaled frame
  if (ep.mosaic) applyMosaic(grid, ep.box, ep.sample);  // base layer = mosaic (sampled from pristine grid)
  glitchFrame(ep, seed);                            // glitch overlays on top
  return ep;
}
function renderCurrent() {
  const ep = drawGlitchFrame(state.frame, state.seed);
  audioReact(ep);                                   // drive procedural SFX from the same params
  ui.frameTag.textContent = `frame ${state.frame + 1}/${ANIM.FRAMES.length}`;
  updateEstimate();                                 // refresh the est. download size for these effects
}

// Advance to a new displayed frame: bump the seed so the glitch layout changes
// once per frame (coherent, legible) instead of strobing every rAF tick.
function showFrame(i) {
  state.frame = i;
  state.seed = (state.seed * 1664525 + 1013904223) >>> 0;
  renderCurrent();
}

function tick(ts) {
  requestAnimationFrame(tick);
  if (!state.last) state.last = ts;
  const dt = (ts - state.last) / 1000; state.last = ts;
  if (!state.playing) return;
  state.acc += dt * state.speed;
  const spf = 1 / ANIM.FPS;
  let next = state.frame, advanced = false;
  while (state.acc >= spf) {
    state.acc -= spf;
    next++;
    advanced = true;
    if (next >= ANIM.FRAMES.length) {
      if (state.loop) { next = 0; if (audioEl && soundOn) audioEl.currentTime = 0; }
      else { next = ANIM.FRAMES.length - 1; state.playing = false; updatePlayBtn(); break; }
    }
  }
  if (advanced) showFrame(next);
  syncAudioTransport();
}

function updatePlayBtn() {
  ui.playBtn.textContent = state.playing ? "⏸ Pause" : "▶ Play";
}

// ===========================================================================
//  EXPORT + REPRODUCIBILITY
// ===========================================================================
const LOOK_IDS = ["rgb","block","tear","noise","static","corrupt",
                  "scanline","scanGap","scanSize","box","sample"];

function flash(msg) {
  ui.exportMsg.textContent = msg;
  clearTimeout(flash._t);
  flash._t = setTimeout(() => { ui.exportMsg.textContent = ""; }, 2200);
}

function downloadBlob(blob, name) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = name;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// --- snapshot current frame as PNG --------------------------------------- //
function downloadPNG() {
  screen.toBlob(b => { downloadBlob(b, `glitch-frame-${state.frame + 1}.png`);
                       flash("Saved PNG ✓"); }, "image/png");
}

// --- record the live canvas to a .webm video ----------------------------- //
let recorder = null, recChunks = [];
function toggleRecord() {
  if (recorder && recorder.state !== "inactive") { recorder.stop(); return; }
  if (!screen.captureStream || typeof MediaRecorder === "undefined") {
    flash("Recording not supported in this browser"); return;
  }
  if (!state.playing) { state.playing = true; updatePlayBtn(); }  // need motion to capture
  // Force audio ON so there's a live graph to capture (this click is the gesture).
  if (!soundOn) setSoundOn(true);

  const stream = screen.captureStream(ANIM.FPS);
  // Merge the audio tap into the recorded stream (captureStream is video-only).
  let hasAudio = false;
  if (streamDest && streamDest.stream) {
    for (const tr of streamDest.stream.getAudioTracks()) { stream.addTrack(tr); hasAudio = true; }
  }
  const types = hasAudio
    ? ["video/webm;codecs=vp9,opus", "video/webm;codecs=vp8,opus", "video/webm"]
    : ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"];
  const mime = types.find(t => MediaRecorder.isTypeSupported(t)) || "";
  recorder = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
  recChunks = [];
  recorder.ondataavailable = e => { if (e.data.size) recChunks.push(e.data); };
  recorder.onstop = () => {
    downloadBlob(new Blob(recChunks, { type: "video/webm" }), "glitch.webm");
    ui.recBtn.textContent = "🎞 Record"; ui.recBtn.classList.remove("rec");
    flash("Saved glitch.webm ✓");
  };
  recorder.start();
  ui.recBtn.textContent = "⏹ Stop"; ui.recBtn.classList.add("rec");
  flash(hasAudio ? "Recording with audio… click Stop" : "Recording (video only)… click Stop");
}

// --- one-click "download the whole video" (auto length) ------------------ //
// Plays the animation from frame 0 for exactly `loops` full passes while
// recording, then stops and downloads. Audio is optional via the checkbox.
let exporting = false;
function exportVideo() {
  if (exporting || (recorder && recorder.state !== "inactive")) { flash("Already recording"); return; }
  if (!screen.captureStream || typeof MediaRecorder === "undefined") {
    flash("Video export not supported in this browser"); return;
  }
  const withAudio = ui.dlAudio.checked;
  const loops = Math.max(1, +ui.dlLoops.value || 1);
  if (withAudio && !soundOn) setSoundOn(true);

  // restart from the top so the export captures a clean, complete pass
  state.frame = 0; state.acc = 0; state.playing = true; updatePlayBtn();
  if (audioEl && withAudio) { try { audioEl.currentTime = 0; } catch (e) {} }
  renderCurrent();

  const stream = screen.captureStream(ANIM.FPS);
  let hasAudio = false;
  if (withAudio && streamDest && streamDest.stream) {
    for (const tr of streamDest.stream.getAudioTracks()) { stream.addTrack(tr); hasAudio = true; }
  }
  const types = hasAudio
    ? ["video/webm;codecs=vp9,opus", "video/webm;codecs=vp8,opus", "video/webm"]
    : ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"];
  const mime = types.find(t => MediaRecorder.isTypeSupported(t)) || "";
  const rec = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
  const chunks = [];
  rec.ondataavailable = e => { if (e.data.size) chunks.push(e.data); };
  rec.onstop = () => {
    downloadBlob(new Blob(chunks, { type: "video/webm" }), "glitch-video.webm");
    exporting = false;
    ui.dlBtn.textContent = "⬇ Download video"; ui.dlBtn.classList.remove("rec");
    flash("Saved glitch-video.webm ✓");
  };
  exporting = true;
  ui.dlBtn.textContent = "● Exporting…"; ui.dlBtn.classList.add("rec");
  rec.start();

  // real-time capture: duration = loops × (frames / fps), adjusted for speed
  const durMs = loops * (ANIM.FRAMES.length / ANIM.FPS) * 1000 / Math.max(0.1, state.speed);
  flash(`Exporting ${loops} loop${loops > 1 ? "s" : ""}${hasAudio ? " + audio" : ""}…`);
  setTimeout(() => { if (rec.state !== "inactive") rec.stop(); }, durMs + 200);
}

// ===========================================================================
//  OFFLINE EXPORT (WebCodecs) + minimal WebM muxer
//  Deterministic frame-by-frame encode — no real-time playback, no dropped
//  frames, exact fps, audio muxed at correct timestamps. Falls back to the
//  real-time recorder (exportVideo) when WebCodecs isn't available.
// ===========================================================================
// --- tiny EBML / WebM writer -------------------------------------------- //
function _idBytes(id) {
  const len = id > 0xFFFFFF ? 4 : id > 0xFFFF ? 3 : id > 0xFF ? 2 : 1;
  const b = new Uint8Array(len);
  for (let i = len - 1; i >= 0; i--) { b[i] = id & 0xff; id = Math.floor(id / 256); }
  return b;
}
function _vint(n) {                       // size as Matroska variable-length int
  let L = 1; while (n >= Math.pow(2, 7 * L) - 1) L++;
  const b = new Uint8Array(L); let v = n;
  for (let i = L - 1; i >= 0; i--) { b[i] = v & 0xff; v = Math.floor(v / 256); }
  b[0] |= 1 << (8 - L);
  return b;
}
function _cat(arrs) {
  let n = 0; for (const a of arrs) n += a.length;
  const out = new Uint8Array(n); let o = 0;
  for (const a of arrs) { out.set(a, o); o += a.length; }
  return out;
}
function _el(id, payload) { return _cat([_idBytes(id), _vint(payload.length), payload]); }
function _uint(n) {                       // minimal big-endian unsigned
  if (n === 0) return new Uint8Array([0]);
  const b = []; let v = n; while (v > 0) { b.unshift(v & 0xff); v = Math.floor(v / 256); }
  return new Uint8Array(b);
}
function _f64(f) { const b = new Uint8Array(8); new DataView(b.buffer).setFloat64(0, f, false); return b; }
function _str(s) { const b = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) b[i] = s.charCodeAt(i); return b; }
function _i16(v) { const b = new Uint8Array(2); new DataView(b.buffer).setInt16(0, v, false); return b; }

function muxWebM(o) {
  // o: {width,height,vCodecId, video:[{data,timeMs,key}], durationMs,
  //     audio?:{sampleRate,channels,codecPrivate,chunks:[{data,timeMs}]}}
  const ebml = _el(0x1A45DFA3, _cat([
    _el(0x4286, _uint(1)), _el(0x42F7, _uint(1)), _el(0x42F2, _uint(4)), _el(0x42F3, _uint(8)),
    _el(0x4282, _str("webm")), _el(0x4287, _uint(2)), _el(0x4285, _uint(2)),
  ]));
  const info = _el(0x1549A966, _cat([
    _el(0x2AD7B1, _uint(1000000)),                 // TimestampScale = 1 ms
    _el(0x4D80, _str("glitch")), _el(0x5741, _str("glitch")),
    _el(0x4489, _f64(o.durationMs)),               // Duration (ms units)
  ]));
  const trackEls = [_el(0xAE, _cat([
    _el(0xD7, _uint(1)), _el(0x73C5, _uint(1)), _el(0x83, _uint(1)), _el(0x9C, _uint(0)),
    _el(0x86, _str(o.vCodecId)),
    _el(0xE0, _cat([_el(0xB0, _uint(o.width)), _el(0xBA, _uint(o.height))])),
  ]))];
  if (o.audio) {
    const aParts = [
      _el(0xD7, _uint(2)), _el(0x73C5, _uint(2)), _el(0x83, _uint(2)), _el(0x9C, _uint(0)),
      _el(0x86, _str("A_OPUS")),
    ];
    if (o.audio.codecPrivate) aParts.push(_el(0x63A2, o.audio.codecPrivate));
    aParts.push(_el(0xE1, _cat([_el(0xB5, _f64(o.audio.sampleRate)), _el(0x9F, _uint(o.audio.channels))])));
    trackEls.push(_el(0xAE, _cat(aParts)));
  }
  const tracks = _el(0x1654AE6B, _cat(trackEls));

  // merge blocks (video=track1, audio=track2), ordered by time
  const blocks = [];
  for (const v of o.video) blocks.push({ t: v.timeMs, track: 1, key: v.key, data: v.data });
  if (o.audio) for (const a of o.audio.chunks) blocks.push({ t: a.timeMs, track: 2, key: true, data: a.data });
  blocks.sort((a, b) => a.t - b.t || a.track - b.track);

  // group into clusters (new cluster on each video keyframe, or every ~30s)
  const clusters = []; let cur = null, clusterT = 0;
  for (const b of blocks) {
    const startNew = !cur || (b.track === 1 && b.key && (b.t - clusterT) > 0) || (b.t - clusterT) > 30000;
    if (startNew) { cur = []; clusterT = b.t; clusters.push({ t: clusterT, blocks: cur }); }
    const sb = _el(0xA3, _cat([_vint(b.track), _i16(b.t - clusterT), new Uint8Array([b.key ? 0x80 : 0]), b.data]));
    cur.push(sb);
  }
  const clusterEls = clusters.map(c => _el(0x1F43B675, _cat([_el(0xE7, _uint(c.t)), ..._chunk(c.blocks)])));
  const segment = _el(0x18538067, _cat([info, tracks, ..._chunk(clusterEls)]));
  return new Blob([ebml, segment], { type: "video/webm" });
}
function _chunk(a) { return a; }   // identity helper (keeps spread readable)

// --- audio: decode embedded track -> Opus chunks via AudioEncoder -------- //
function _opusHead(channels, sampleRate) {
  const b = new Uint8Array(19); const dv = new DataView(b.buffer);
  _str("OpusHead").forEach((c, i) => b[i] = c);
  b[8] = 1; b[9] = channels; dv.setUint16(10, 0, true); dv.setUint32(12, sampleRate, true);
  dv.setUint16(16, 0, true); b[18] = 0;
  return b;
}
async function encodeAudioOpus(durationSec) {
  if (typeof AudioEncoder === "undefined" || !AUDIO) return null;
  const AC = window.AudioContext || window.webkitAudioContext; if (!AC) return null;
  const ctx = new AC();
  let buf;
  try {
    const bytes = Uint8Array.from(atob(AUDIO.b64), c => c.charCodeAt(0));
    buf = await ctx.decodeAudioData(bytes.buffer);
  } catch (e) { ctx.close(); return null; }
  ctx.close();
  const sr = buf.sampleRate, ch = Math.min(2, buf.numberOfChannels);
  const planes = []; for (let c = 0; c < ch; c++) planes.push(buf.getChannelData(c));
  const srcLen = buf.length, total = Math.floor(durationSec * sr);
  const chunks = []; let desc = null;
  const aenc = new AudioEncoder({
    output: (chunk, meta) => {
      if (!desc && meta && meta.decoderConfig && meta.decoderConfig.description)
        desc = new Uint8Array(meta.decoderConfig.description);
      const d = new Uint8Array(chunk.byteLength); chunk.copyTo(d);
      chunks.push({ data: d, timeMs: Math.round(chunk.timestamp / 1000) });
    },
    error: e => console.error(e),
  });
  aenc.configure({ codec: "opus", sampleRate: sr, numberOfChannels: ch, bitrate: 96000 });
  const block = Math.floor(sr * 0.02);   // 20 ms frames
  for (let off = 0; off < total; off += block) {
    const n = Math.min(block, total - off);
    const data = new Float32Array(n * ch);
    for (let c = 0; c < ch; c++) { const pl = planes[c]; for (let i = 0; i < n; i++) data[c * n + i] = pl[(off + i) % srcLen]; }
    const ad = new AudioData({ format: "f32-planar", sampleRate: sr, numberOfFrames: n, numberOfChannels: ch, timestamp: Math.round(off / sr * 1e6), data });
    aenc.encode(ad); ad.close();
    if (aenc.encodeQueueSize > 24) await new Promise(r => setTimeout(r));
  }
  await aenc.flush(); aenc.close();
  if (!chunks.length) return null;
  return { sampleRate: sr, channels: ch, codecPrivate: desc || _opusHead(ch, sr), chunks };
}

// --- bitrate model + live size estimate ---------------------------------- //
// Bitrate scales with effect "entropy": noise / static / corruption are
// high-frequency and need many more bits to look good, so both the encoder
// quality AND the size estimate move with the selected effects.
function estimateVideoBitrate() {
  const p = effectiveParams();
  const score = 0.18
    + p.noise * 1.6 + p.static * 0.9 + p.corrupt * 2.2
    + p.block * 0.35 + p.tear * 0.25 + (p.rgb / 24) * 0.2
    + (p.mosaic ? -0.06 : 0);                 // flat blocks compress better
  const bpp = Math.max(0.05, 0.12 + score);   // bits per pixel
  const bps = bpp * dispW * dispH * ANIM.FPS;
  return Math.round(Math.min(28e6, Math.max(1e6, bps)));
}
function fmtSize(bytes) {
  return bytes >= 1048576 ? (bytes / 1048576).toFixed(1) + " MB"
       : Math.max(1, Math.round(bytes / 1024)) + " KB";
}
function updateEstimate() {
  if (!ANIM || !ui.dlEst) return;
  const loops = Math.max(1, +ui.dlLoops.value || 1);
  const dur = loops * ANIM.FRAMES.length / ANIM.FPS;
  const vb = estimateVideoBitrate();
  const ab = (ui.dlAudio.checked && AUDIO) ? (ui.dlFormat.value === "mp4" ? 128000 : 96000) : 0;
  const bytes = (vb + ab) * dur / 8 * 1.02;   // +container overhead
  ui.dlEst.textContent = "≈ " + fmtSize(bytes) + " · " + Math.round(vb / 1e6 * 10) / 10 + " Mb/s";
}

async function exportOffline() {
  if (exporting || (recorder && recorder.state !== "inactive")) { flash("Already exporting"); return; }
  if (typeof VideoEncoder === "undefined" || typeof VideoFrame === "undefined") {
    flash("No WebCodecs — using the live recorder instead"); return exportVideo();
  }
  const withAudio = ui.dlAudio.checked;
  const loops = Math.max(1, +ui.dlLoops.value || 1);
  const mp4 = ui.dlFormat.value === "mp4";
  const fps = ANIM.FPS, N = ANIM.FRAMES.length, W = dispW, H = dispH;
  const bitrate = estimateVideoBitrate();

  // pick a supported codec for the chosen container
  const candidates = mp4
    ? [["avc1.640028", "avc"], ["avc1.4d0028", "avc"], ["avc1.42001f", "avc"]]
    : [["vp09.00.10.08", "V_VP9"], ["vp8", "V_VP8"]];
  let vCodec = null, vTag = null;
  for (const [c, tag] of candidates) {
    try {
      const cfg = { codec: c, width: W, height: H, bitrate, framerate: fps };
      if (mp4) cfg.avc = { format: "avc" };            // length-prefixed NALs for MP4
      const s = await VideoEncoder.isConfigSupported(cfg);
      if (s.supported) { vCodec = c; vTag = tag; break; }
    } catch (e) {}
  }
  if (!vCodec) {
    if (mp4) { flash("MP4/H.264 not supported here — falling back to WebM"); ui.dlFormat.value = "webm"; return exportOffline(); }
    flash("WebCodecs video unsupported — using live recorder"); return exportVideo();
  }

  exporting = true;
  ui.dlBtn.textContent = "● Encoding…"; ui.dlBtn.classList.add("rec");
  const wasPlaying = state.playing; state.playing = false; updatePlayBtn();
  try {
    const video = []; let avcC = null;
    const venc = new VideoEncoder({
      output: (chunk, meta) => {
        if (mp4 && !avcC && meta && meta.decoderConfig && meta.decoderConfig.description)
          avcC = new Uint8Array(meta.decoderConfig.description);
        const d = new Uint8Array(chunk.byteLength); chunk.copyTo(d);
        video.push({ data: d, timeMs: Math.round(chunk.timestamp / 1000), key: chunk.type === "key" });
      },
      error: e => { throw e; },
    });
    const vcfg = { codec: vCodec, width: W, height: H, bitrate, framerate: fps };
    if (mp4) vcfg.avc = { format: "avc" };
    venc.configure(vcfg);
    const gop = Math.max(1, fps), durUs = Math.round(1e6 / fps), totalFrames = loops * N;
    let seed = 1;
    for (let f = 0; f < totalFrames; f++) {
      seed = (seed * 1664525 + 1013904223) >>> 0;
      drawGlitchFrame(f % N, seed);
      const vf = new VideoFrame(screen, { timestamp: Math.round(f * 1e6 / fps), duration: durUs });
      venc.encode(vf, { keyFrame: f % gop === 0 }); vf.close();
      if (venc.encodeQueueSize > 12) await new Promise(r => setTimeout(r));
      if (f % 8 === 0) flash(`Encoding video… ${Math.round(100 * f / totalFrames)}%`);
    }
    await venc.flush(); venc.close();

    const durationMs = Math.round(totalFrames / fps * 1000);
    let blob, fname;
    if (mp4) {
      let audio = null;
      if (withAudio && AUDIO) { flash("Encoding audio (AAC)…"); audio = await encodeAudioAac(loops * N / fps); }
      flash("Muxing MP4…");
      blob = muxMP4({ width: W, height: H, fps, video, avcC, durationMs, audio });
      fname = "glitch-video.mp4";
    } else {
      let audio = null;
      if (withAudio && AUDIO) { flash("Encoding audio (Opus)…"); audio = await encodeAudioOpus(loops * N / fps); }
      flash("Muxing WebM…");
      blob = muxWebM({ width: W, height: H, vCodecId: vTag, video, durationMs, audio });
      fname = "glitch-video.webm";
    }
    downloadBlob(blob, fname);
    flash(`Saved ${fname} ✓ (${(blob.size / 1048576).toFixed(1)} MB)`);
  } catch (err) {
    console.error(err); flash("Offline export failed (" + err.message + ") — try WebM or the 🎞 Record button");
  } finally {
    exporting = false; ui.dlBtn.textContent = "⬇ Download video"; ui.dlBtn.classList.remove("rec");
    state.playing = wasPlaying; updatePlayBtn();
  }
}

// --- MP4 (ISO-BMFF) muxer : H.264 video + optional AAC audio ------------- //
function _u32be(n) { const b = new Uint8Array(4); new DataView(b.buffer).setUint32(0, n >>> 0); return b; }
function _u16be(n) { const b = new Uint8Array(2); new DataView(b.buffer).setUint16(0, n); return b; }
function _strz(s) { return _cat([_str(s), new Uint8Array([0])]); }
function _mb(type, ...parts) {
  const body = _cat(parts), h = new Uint8Array(8);
  new DataView(h.buffer).setUint32(0, 8 + body.length);
  for (let i = 0; i < 4; i++) h[4 + i] = type.charCodeAt(i);
  return _cat([h, body]);
}
function _mfb(type, ver, flags, ...parts) {
  return _mb(type, new Uint8Array([ver, (flags >>> 16) & 255, (flags >>> 8) & 255, flags & 255]), ...parts);
}
const _MATRIX = _cat([_u32be(0x00010000), _u32be(0), _u32be(0), _u32be(0), _u32be(0x00010000),
                      _u32be(0), _u32be(0), _u32be(0), _u32be(0x40000000)]);
function _aacAsc(sr, ch) {
  const rates = [96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000, 7350];
  let idx = rates.indexOf(sr); if (idx < 0) idx = 4;
  const v = (2 << 11) | (idx << 7) | (ch << 3);
  return new Uint8Array([(v >> 8) & 0xff, v & 0xff]);
}
function muxMP4(o) {
  const { width, height, fps, video, avcC, durationMs, audio } = o;
  const MS = 1000;
  // mdat: video samples then audio frames
  const parts = []; let pos = 0;
  const vOff = [], vSizes = [], vSync = [];
  video.forEach((s, i) => { vOff.push(pos); vSizes.push(s.data.length); if (s.key) vSync.push(i + 1); parts.push(s.data); pos += s.data.length; });
  const aOff = [], aSizes = [];
  if (audio) audio.frames.forEach(f => { aOff.push(pos); aSizes.push(f.length); parts.push(f); pos += f.length; });
  const mdatData = _cat(parts);

  const descr = (tag, p) => _cat([new Uint8Array([tag, p.length]), p]);
  const esdsBox = (asc) => _mfb("esds", 0, 0, descr(0x03, _cat([
    _u16be(0), new Uint8Array([0]),
    descr(0x04, _cat([new Uint8Array([0x40, 0x15, 0, 0, 0]), _u32be(0), _u32be(0), descr(0x05, asc)])),
    descr(0x06, new Uint8Array([0x02])),
  ])));
  const dinf = () => _mb("dinf", _mfb("dref", 0, 0, _u32be(1), _mfb("url ", 0, 1)));
  const avc1 = () => _mb("avc1", _cat([
    new Uint8Array(6), _u16be(1), _u16be(0), _u16be(0), _u32be(0), _u32be(0), _u32be(0),
    _u16be(width), _u16be(height), _u32be(0x00480000), _u32be(0x00480000), _u32be(0),
    _u16be(1), new Uint8Array(32), _u16be(0x0018), _u16be(0xFFFF),
  ]), _mb("avcC", avcC));
  const mp4a = (sr, ch, asc) => _mb("mp4a", _cat([
    new Uint8Array(6), _u16be(1), _u32be(0), _u32be(0),
    _u16be(ch), _u16be(16), _u16be(0), _u16be(0), _u32be(sr * 65536),
  ]), esdsBox(asc));

  function buildMoov(base) {
    const vN = video.length;
    const stblV = _mb("stbl",
      _mfb("stsd", 0, 0, _u32be(1), avc1()),
      _mfb("stts", 0, 0, _u32be(1), _u32be(vN), _u32be(1)),
      _mfb("stsc", 0, 0, _u32be(1), _u32be(1), _u32be(1), _u32be(1)),
      _mfb("stsz", 0, 0, _u32be(0), _u32be(vN), _cat(vSizes.map(_u32be))),
      _mfb("stco", 0, 0, _u32be(vN), _cat(vOff.map(x => _u32be(base + x)))),
      _mfb("stss", 0, 0, _u32be(vSync.length), _cat(vSync.map(_u32be))));
    const trakV = _mb("trak",
      _mfb("tkhd", 0, 7, _u32be(0), _u32be(0), _u32be(1), _u32be(0), _u32be(durationMs),
           _u32be(0), _u32be(0), _u16be(0), _u16be(0), _u16be(0), _u16be(0), _MATRIX,
           _u32be(width * 65536), _u32be(height * 65536)),
      _mb("mdia",
        _mfb("mdhd", 0, 0, _u32be(0), _u32be(0), _u32be(fps), _u32be(vN), _u16be(0x55C4), _u16be(0)),
        _mb("hdlr", _u32be(0), _str("vide"), _u32be(0), _u32be(0), _u32be(0), _strz("VideoHandler")),
        _mb("minf", _mfb("vmhd", 0, 1, _u16be(0), _u16be(0), _u16be(0), _u16be(0)), dinf(), stblV)));
    let trakA = new Uint8Array(0);
    if (audio) {
      const aN = audio.frames.length, sr = audio.sampleRate, spf = 1024;
      const aDur = Math.round(aN * spf / sr * MS);
      const stblA = _mb("stbl",
        _mfb("stsd", 0, 0, _u32be(1), mp4a(sr, audio.channels, audio.asc)),
        _mfb("stts", 0, 0, _u32be(1), _u32be(aN), _u32be(spf)),
        _mfb("stsc", 0, 0, _u32be(1), _u32be(1), _u32be(1), _u32be(1)),
        _mfb("stsz", 0, 0, _u32be(0), _u32be(aN), _cat(aSizes.map(_u32be))),
        _mfb("stco", 0, 0, _u32be(aN), _cat(aOff.map(x => _u32be(base + x)))));
      trakA = _mb("trak",
        _mfb("tkhd", 0, 7, _u32be(0), _u32be(0), _u32be(2), _u32be(0), _u32be(aDur),
             _u32be(0), _u32be(0), _u16be(0), _u16be(0), _u16be(0x0100), _u16be(0), _MATRIX,
             _u32be(0), _u32be(0)),
        _mb("mdia",
          _mfb("mdhd", 0, 0, _u32be(0), _u32be(0), _u32be(sr), _u32be(aN * spf), _u16be(0x55C4), _u16be(0)),
          _mb("hdlr", _u32be(0), _str("soun"), _u32be(0), _u32be(0), _u32be(0), _strz("SoundHandler")),
          _mb("minf", _mfb("smhd", 0, 0, _u16be(0), _u16be(0)), dinf(), stblA)));
    }
    const mvhd = _mfb("mvhd", 0, 0, _u32be(0), _u32be(0), _u32be(MS), _u32be(durationMs),
      _u32be(0x00010000), _u16be(0x0100), _u16be(0), _u32be(0), _u32be(0), _MATRIX,
      _u32be(0), _u32be(0), _u32be(0), _u32be(0), _u32be(0), _u32be(0), _u32be(audio ? 3 : 2));
    return _mb("moov", mvhd, trakV, trakA);
  }
  const ftyp = _mb("ftyp", _str("isom"), _u32be(0x200), _str("isom"), _str("iso2"), _str("avc1"), _str("mp41"));
  const base = ftyp.length + buildMoov(0).length + 8;   // +8 = mdat box header
  return new Blob([ftyp, buildMoov(base), _mb("mdat", mdatData)], { type: "video/mp4" });
}

// --- audio: decode embedded track -> AAC frames via AudioEncoder --------- //
async function encodeAudioAac(durationSec) {
  if (typeof AudioEncoder === "undefined" || !AUDIO) return null;
  const AC = window.AudioContext || window.webkitAudioContext; if (!AC) return null;
  const ctx = new AC(); let buf;
  try { buf = await ctx.decodeAudioData(Uint8Array.from(atob(AUDIO.b64), c => c.charCodeAt(0)).buffer); }
  catch (e) { ctx.close(); return null; }
  ctx.close();
  const sr = buf.sampleRate, ch = Math.min(2, buf.numberOfChannels);
  const planes = []; for (let c = 0; c < ch; c++) planes.push(buf.getChannelData(c));
  const srcLen = buf.length, total = Math.floor(durationSec * sr);
  const frames = []; let asc = null;
  const aenc = new AudioEncoder({
    output: (chunk, meta) => {
      if (!asc && meta && meta.decoderConfig && meta.decoderConfig.description) asc = new Uint8Array(meta.decoderConfig.description);
      const d = new Uint8Array(chunk.byteLength); chunk.copyTo(d); frames.push(d);
    }, error: e => console.error(e),
  });
  aenc.configure({ codec: "mp4a.40.2", sampleRate: sr, numberOfChannels: ch, bitrate: 128000 });
  for (let off = 0; off < total; off += 1024) {
    const n = Math.min(1024, total - off);
    const data = new Float32Array(n * ch);
    for (let c = 0; c < ch; c++) { const pl = planes[c]; for (let i = 0; i < n; i++) data[c * n + i] = pl[(off + i) % srcLen]; }
    const ad = new AudioData({ format: "f32-planar", sampleRate: sr, numberOfFrames: n, numberOfChannels: ch, timestamp: Math.round(off / sr * 1e6), data });
    aenc.encode(ad); ad.close();
    if (aenc.encodeQueueSize > 24) await new Promise(r => setTimeout(r));
  }
  await aenc.flush(); aenc.close();
  if (!frames.length) return null;
  return { sampleRate: sr, channels: ch, asc: asc || _aacAsc(sr, ch), frames };
}

// --- shareable "look" link (every setting encoded in the URL hash) ------- //
function snapshotLook() {
  const s = { preset: ui.presetSel.value, mosaic: mosaicOn,
              speed: state.speed, intensity: state.intensity };
  for (const id of LOOK_IDS) s[id] = +ui[id].value;
  return s;
}
function applyLook(s) {
  if (s.preset && CFG.presets[s.preset]) { ui.presetSel.value = s.preset; applyPresetToUI(s.preset); }
  for (const id of LOOK_IDS) if (s[id] != null) ui[id].value = s[id];
  if (s.mosaic != null) { mosaicOn = !!s.mosaic; updateMosaicBtn(); }
  if (s.speed != null) { state.speed = s.speed; ui.speed.value = s.speed; }
  if (s.intensity != null) { state.intensity = s.intensity; ui.intensity.value = s.intensity; }
  readUIParams(); renderCurrent();
}
function copyLook() {
  const hash = "#look=" + encodeURIComponent(JSON.stringify(snapshotLook()));
  history.replaceState(null, "", location.pathname + location.search + hash);
  if (navigator.clipboard) {
    navigator.clipboard.writeText(location.href)
      .then(() => flash("Look link copied to clipboard ✓"))
      .catch(() => flash("Look saved to URL (copy from address bar)"));
  } else {
    flash("Look saved to URL (copy from address bar)");
  }
}
function restoreLookFromURL() {
  if (!location.hash.startsWith("#look=")) return false;
  try { applyLook(JSON.parse(decodeURIComponent(location.hash.slice(6)))); return true; }
  catch (e) { console.warn("bad look hash", e); return false; }
}

// --- randomize all effect sliders ---------------------------------------- //
function randomize() {
  const r = (a, b) => a + Math.random() * (b - a);
  ui.rgb.value = Math.round(r(0, 16));
  ui.block.value = r(0, 0.8).toFixed(2);
  ui.tear.value = r(0, 0.6).toFixed(2);
  ui.noise.value = r(0, 0.12).toFixed(3);
  ui.static.value = r(0, 0.3).toFixed(2);
  ui.corrupt.value = r(0, 0.12).toFixed(3);
  ui.scanline.value = r(0, 0.7).toFixed(2);
  ui.scanGap.value = Math.round(r(2, 14));
  ui.scanSize.value = Math.round(r(1, 5));
  ui.box.value = Math.round(r(4, 40));
  ui.sample.value = Math.round(r(4, 48));
  mosaicOn = Math.random() < 0.5; updateMosaicBtn();
  readUIParams(); renderCurrent();
  flash("Randomized — 🔗 Copy look to keep it");
}

// ===========================================================================
//  AUDIO  (1) embedded soundtrack synced to playback, and/or
//         (2) procedural WebAudio glitch SFX that react to the effects.
//  Both need a user gesture to start (browser autoplay policy) -> 🔇 button.
// ===========================================================================
let soundOn = false, sfxOn = !AUDIO;   // SFX default on only when there's no real track
let actx = null, master = null, sfx = {}, audioEl = null, streamDest = null;

function initAudio() {
  if (actx) return;
  const AC = window.AudioContext || window.webkitAudioContext;
  if (AC) {
    actx = new AC();
    master = actx.createGain();
    master.gain.value = +ui.volume.value;
    master.connect(actx.destination);
    // tap for recording: everything through master is ALSO sent here so the
    // MediaRecorder can pick up an audio track (canvas.captureStream is video-only)
    streamDest = actx.createMediaStreamDestination();
    master.connect(streamDest);
    // looping white-noise bed -> bandpass -> gain (static / sparkle / corruption)
    const buf = actx.createBuffer(1, actx.sampleRate * 2, actx.sampleRate);
    const dch = buf.getChannelData(0);
    for (let i = 0; i < dch.length; i++) dch[i] = Math.random() * 2 - 1;
    const noise = actx.createBufferSource(); noise.buffer = buf; noise.loop = true;
    const nFilter = actx.createBiquadFilter(); nFilter.type = "bandpass";
    nFilter.frequency.value = 1200; nFilter.Q.value = 0.6;
    const nGain = actx.createGain(); nGain.gain.value = 0;
    noise.connect(nFilter).connect(nGain).connect(master); noise.start();
    // low sawtooth hum (scanlines)
    const hum = actx.createOscillator(); hum.type = "sawtooth"; hum.frequency.value = 60;
    const humGain = actx.createGain(); humGain.gain.value = 0;
    hum.connect(humGain).connect(master); hum.start();
    // square carrier (rgb split / block / corruption chirps)
    const car = actx.createOscillator(); car.type = "square"; car.frequency.value = 180;
    const carGain = actx.createGain(); carGain.gain.value = 0;
    car.connect(carGain).connect(master); car.start();
    sfx = { nFilter, nGain, humGain, car, carGain };
  }
  if (AUDIO) {
    audioEl = new Audio("data:" + AUDIO.mime + ";base64," + AUDIO.b64);
    audioEl.loop = true;
    // route through the graph (so master volume applies AND it can be recorded);
    // fall back to direct output if MediaElementSource isn't available.
    if (actx) {
      try { actx.createMediaElementSource(audioEl).connect(master); audioEl.volume = 1; }
      catch (e) { audioEl.volume = +ui.volume.value; }
    } else {
      audioEl.volume = +ui.volume.value;
    }
  }
}

// Map the current effect params to the SFX graph (called each rendered frame).
function audioReact(p) {
  if (!actx || !soundOn || !sfxOn || !sfx.nGain) {
    if (sfx.nGain) {
      const t = actx ? actx.currentTime : 0;
      sfx.nGain.gain.setTargetAtTime(0, t, 0.05);
      sfx.humGain.gain.setTargetAtTime(0, t, 0.05);
      sfx.carGain.gain.setTargetAtTime(0, t, 0.05);
    }
    return;
  }
  const t = actx.currentTime;
  const noiseLvl = Math.min(0.6, p.static * 0.5 + p.noise * 1.4 + p.corrupt * 1.6);
  sfx.nGain.gain.setTargetAtTime(noiseLvl, t, 0.04);
  sfx.nFilter.frequency.setTargetAtTime(700 + p.rgb * 220 + p.static * 3200, t, 0.05);
  sfx.humGain.gain.setTargetAtTime(Math.min(0.14, p.scanline * 0.14), t, 0.05);
  sfx.car.frequency.setTargetAtTime(110 + p.rgb * 70 + (p.mosaic ? 160 : 0), t, 0.02);
  sfx.carGain.gain.setTargetAtTime((p.block > 0.3 || p.corrupt > 0.05) ? 0.05 : 0, t, 0.03);
}

function setSoundOn(on) {
  soundOn = on;
  ui.soundBtn.textContent = on ? "🔊 Sound: on" : "🔇 Sound: off";
  if (on) {
    initAudio();
    if (actx && actx.state === "suspended") actx.resume();
    if (audioEl && state.playing) { audioEl.currentTime = state.frame / ANIM.FPS; audioEl.play().catch(() => {}); }
    flash(AUDIO ? "Soundtrack on" : "Glitch SFX on");
  } else {
    if (audioEl) audioEl.pause();
    if (actx) actx.suspend();
  }
}
function setVolume(v) {
  if (master) master.gain.value = v;
  if (audioEl) audioEl.volume = v;
}
// keep the embedded soundtrack roughly in step with the animation clock
function syncAudioTransport() {
  if (!audioEl || !soundOn) return;
  audioEl.playbackRate = Math.min(4, Math.max(0.25, state.speed));
  if (state.playing && audioEl.paused) audioEl.play().catch(() => {});
  if (!state.playing && !audioEl.paused) audioEl.pause();
}

// ===========================================================================
//  FAVORITES  (persisted in localStorage; appear in the preset dropdown)
// ===========================================================================
const FAV_KEY = "glitchFavorites";
function loadFavs() {
  try { return JSON.parse(localStorage.getItem(FAV_KEY) || "{}"); }
  catch (e) { return {}; }
}
function saveFavs(f) {
  try { localStorage.setItem(FAV_KEY, JSON.stringify(f)); } catch (e) {}
}
function rebuildPresetOptions(selected) {
  const favs = loadFavs();
  ui.presetSel.innerHTML = "";
  for (const name of Object.keys(CFG.presets)) {
    const o = document.createElement("option"); o.value = name; o.textContent = name;
    ui.presetSel.appendChild(o);
  }
  const favNames = Object.keys(favs);
  if (favNames.length) {
    const grp = document.createElement("optgroup"); grp.label = "★ favorites";
    for (const name of favNames) {
      const o = document.createElement("option");
      o.value = "fav:" + name; o.textContent = "★ " + name;
      grp.appendChild(o);
    }
    ui.presetSel.appendChild(grp);
  }
  if (selected) ui.presetSel.value = selected;
}
function saveFavorite() {
  const name = (prompt("Name this favorite:", "my-look") || "").trim();
  if (!name) return;
  const favs = loadFavs();
  favs[name] = snapshotLook();
  saveFavs(favs);
  rebuildPresetOptions("fav:" + name);
  flash("Saved favorite ★ " + name);
}
function deleteFavorite() {
  const v = ui.presetSel.value;
  if (!v.startsWith("fav:")) { flash("Select a ★ favorite to delete"); return; }
  const name = v.slice(4);
  const favs = loadFavs();
  delete favs[name]; saveFavs(favs);
  rebuildPresetOptions(CFG.preset);
  applyPresetToUI(CFG.preset); renderCurrent();
  flash("Deleted favorite " + name);
}
// applyPreset OR favorite, depending on the dropdown value
function selectPreset(value) {
  if (value.startsWith("fav:")) {
    const fav = loadFavs()[value.slice(4)];
    if (fav) applyLook(fav);
  } else {
    applyPresetToUI(value);
  }
  renderCurrent();
}

function wireUI() {
  rebuildPresetOptions(CFG.preset);
  applyPresetToUI(CFG.preset);
  // CLI overrides (--mosaic / --box-size / --sample-size) win over the preset:
  if (CFG.mosaic != null) { mosaicOn = !!CFG.mosaic; updateMosaicBtn(); }
  if (CFG.boxSize != null) ui.box.value = CFG.boxSize;
  if (CFG.sampleSize != null) ui.sample.value = CFG.sampleSize;
  readUIParams();
  ui.sfxBtn.textContent = "✧ SFX: " + (sfxOn ? "on" : "off");

  // re-render on every change so tweaks are visible instantly (even when paused)
  ui.presetSel.addEventListener("change", () => selectPreset(ui.presetSel.value));
  for (const id of ["rgb","block","tear","noise","static","corrupt","scanline","scanGap","scanSize","box","sample"])
    ui[id].addEventListener("input", () => { readUIParams(); renderCurrent(); });
  ui.mosaicToggle.addEventListener("click", () => {
    mosaicOn = !mosaicOn; updateMosaicBtn(); readUIParams(); renderCurrent();
  });
  ui.pngBtn.addEventListener("click", downloadPNG);
  ui.recBtn.addEventListener("click", toggleRecord);
  ui.dlBtn.addEventListener("click", exportOffline);
  for (const id of ["dlFormat", "dlAudio", "dlLoops"])
    ui[id].addEventListener("change", updateEstimate);
  ui.linkBtn.addEventListener("click", copyLook);
  ui.randBtn.addEventListener("click", randomize);
  ui.soundBtn.addEventListener("click", () => setSoundOn(!soundOn));
  ui.sfxBtn.addEventListener("click", () => {
    sfxOn = !sfxOn; ui.sfxBtn.textContent = "✧ SFX: " + (sfxOn ? "on" : "off"); renderCurrent();
  });
  ui.volume.addEventListener("input", () => setVolume(+ui.volume.value));
  ui.favBtn.addEventListener("click", saveFavorite);
  ui.favDelBtn.addEventListener("click", deleteFavorite);
  ui.speed.addEventListener("input", () => { state.speed = +ui.speed.value; syncReadouts(); syncAudioTransport(); });
  ui.intensity.addEventListener("input", () => { state.intensity = +ui.intensity.value; syncReadouts(); renderCurrent(); });
  ui.playBtn.addEventListener("click", () => { state.playing = !state.playing; updatePlayBtn(); syncAudioTransport(); });
  ui.loopBtn.addEventListener("click", () => {
    state.loop = !state.loop;
    ui.loopBtn.textContent = "🔁 Loop: " + (state.loop ? "on" : "off");
  });
  window.addEventListener("keydown", (e) => {
    if (e.code === "Space") { e.preventDefault(); state.playing = !state.playing; updatePlayBtn(); syncAudioTransport(); }
  });
}

(async function init() {
  try {
    ANIM = await loadAnimation();
  } catch (err) {
    byId("meta").textContent = "Failed to load data: " + err.message;
    console.error(err);
    return;
  }
  PAL = buildPalette(ANIM.LEVELS);
  mosaicHist = new Uint32Array(ANIM.LEVELS ** 3);   // histogram over palette indices
  setupCanvas(ANIM);
  wireUI();
  restoreLookFromURL();   // reproduce a shared look if the URL carries one
  ui.meta.textContent =
    `${ANIM.W}×${ANIM.H} grid · ${ANIM.FRAMES.length} frames · ${ANIM.FPS} fps · `
    + (DATA_B64 ? "embedded video" : "synthetic demo (run glitch_processor.py on a real video)");
  showFrame(0);
  requestAnimationFrame(tick);
})();
</script>
</body>
</html>
"""

# --------------------------------------------------------------------------- #
# WEB CONVERTER (for GitHub Pages)
# A 100% client-side page: upload a video -> decode frames in-browser (canvas) ->
# quantize + delta-encode -> gzip (native CompressionStream) -> rebuild the same
# standalone player and offer it as a download. No server, no dependencies.
# The player template is embedded (base64) so the two never drift.
# --------------------------------------------------------------------------- #
def build_web_html():
    """Build index.html: the in-browser converter, with the player embedded."""
    tpl_b64 = base64.b64encode(HTML_TEMPLATE.encode("utf-8")).decode("ascii")
    html = WEB_TEMPLATE
    html = html.replace('/*__PLAYER_B64__*/""', json.dumps(tpl_b64))
    html = html.replace("/*__PRESETS__*/({})", "(" + json.dumps(PRESETS) + ")")
    return html


WEB_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Glitch Video Converter</title>
<style>
  :root { --bg:#07080c; --fg:#d8f5ff; --acc:#ff2bd6; --acc2:#23f0ff; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg); min-height:100vh;
         font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
         display:flex; flex-direction:column; align-items:center; padding:24px 14px 60px; }
  h1 { font-size:17px; letter-spacing:.22em; text-transform:uppercase; margin:4px 0;
       color:var(--acc2); text-shadow:0 0 10px var(--acc2); }
  .sub { opacity:.5; font-size:11px; margin-bottom:20px; text-align:center; }
  .wrap { width:min(900px,96vw); display:grid; gap:16px; }
  .card { border:1px solid #1a2735; border-radius:8px; background:#0a0d13; padding:16px 18px;
          box-shadow:0 8px 30px rgba(0,0,0,.35); }
  .card h2 { margin:0 0 12px; font-size:10.5px; letter-spacing:.16em; text-transform:uppercase;
             color:var(--acc2); }
  #drop { border:2px dashed #2a4a55; border-radius:8px; padding:34px 18px; text-align:center;
          cursor:pointer; transition:.15s; color:#9fd; }
  #drop.hot { border-color:var(--acc); background:rgba(255,43,214,.06); }
  #drop b { color:var(--acc2); }
  .fname { font-size:12px; color:var(--acc); margin-top:8px; min-height:1.2em; }
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:13px 18px; }
  .row { display:flex; flex-direction:column; gap:5px; }
  .row label { font-size:9.5px; text-transform:uppercase; letter-spacing:.1em; opacity:.62;
               display:flex; justify-content:space-between; gap:8px; }
  .row .val { color:var(--acc); }
  input[type=range]{ width:100%; accent-color:var(--acc); height:3px; }
  input[type=number], select { background:#11161f; color:var(--fg); border:1px solid #243140;
       border-radius:5px; padding:7px 9px; font:inherit; font-size:12px; }
  .chk { font-size:11px; opacity:.85; display:flex; align-items:center; gap:7px; }
  button { background:#11161f; color:var(--fg); border:1px solid #243140; border-radius:6px;
           padding:11px 18px; font:inherit; font-size:13px; cursor:pointer; }
  button:hover:not(:disabled){ border-color:var(--acc2); }
  button:disabled{ opacity:.45; cursor:not-allowed; }
  button.primary{ border-color:var(--acc); color:var(--acc); }
  .barwrap { height:6px; background:#10161e; border-radius:4px; overflow:hidden; margin-top:12px; }
  .bar { height:100%; width:0; background:linear-gradient(90deg,var(--acc),var(--acc2)); transition:width .15s; }
  .status { font-size:11.5px; margin-top:9px; min-height:1.3em; color:var(--acc2); }
  iframe { width:100%; height:640px; border:1px solid #1a2735; border-radius:8px; background:#000; }
  .muted { opacity:.5; font-size:10.5px; }
  .actions { display:flex; gap:10px; flex-wrap:wrap; align-items:center; margin-top:6px; }
  a.ghost { color:var(--acc2); font-size:11px; }
</style>
</head>
<body>
  <h1>// Glitch Video Converter</h1>
  <div class="sub">Upload a video, tune the look, and download a single self-contained
    HTML glitch animation. 100% in your browser — nothing is uploaded anywhere.</div>

  <div class="wrap">
    <div class="card">
      <h2>1 · Choose a video</h2>
      <div id="drop">Drop a video here, or <b>click to browse</b><br>
        <span class="muted">MP4 / WebM / MOV (whatever your browser can play)</span>
        <div class="fname" id="fname"></div>
      </div>
      <input type="file" id="file" accept="video/*" hidden>
    </div>

    <div class="card">
      <h2>2 · Settings</h2>
      <div class="grid">
        <div class="row"><label>Grid width <span class="val" id="gwV"></span></label>
          <input type="range" id="gw" min="48" max="240" step="4" value="96"></div>
        <div class="row"><label>Grid height <span class="val" id="ghV"></span></label>
          <input type="range" id="gh" min="27" max="135" step="3" value="54"></div>
        <div class="row"><label>FPS <span class="val" id="fpsV"></span></label>
          <input type="range" id="fps" min="6" max="24" step="1" value="12"></div>
        <div class="row"><label>Colour levels <span class="val" id="levV"></span></label>
          <input type="range" id="levels" min="3" max="8" step="1" value="6"></div>
        <div class="row"><label>Display scale <span class="val" id="scaleV"></span></label>
          <input type="range" id="scale" min="4" max="14" step="1" value="8"></div>
        <div class="row"><label>Max frames (0 = all) <span class="val" id="mfV"></span></label>
          <input type="range" id="maxFrames" min="0" max="900" step="30" value="0"></div>
        <div class="row"><label>Preset</label>
          <select id="preset"></select></div>
        <div class="row"><label>&nbsp;</label>
          <label class="chk"><input type="checkbox" id="mosaic"> Mosaic base layer</label></div>
        <div class="row"><label>&nbsp;</label>
          <label class="chk"><input type="checkbox" id="audio"> Embed audio (WAV)</label></div>
        <div class="row"><label>Audio rate <span class="val" id="arV"></span></label>
          <select id="audioRate">
            <option value="8000">8 kHz (tiny)</option>
            <option value="11025" selected>11 kHz</option>
            <option value="22050">22 kHz</option></select></div>
      </div>
      <div class="actions">
        <button class="primary" id="convert" disabled>⚙ Convert</button>
        <span class="muted">Larger grid / fps / more frames = bigger file &amp; slower.</span>
      </div>
      <div class="barwrap"><div class="bar" id="prog"></div></div>
      <div class="status" id="status">Waiting for a video…</div>
    </div>

    <div class="card" id="resultCard" style="display:none;">
      <h2>3 · Result</h2>
      <div class="actions">
        <button class="primary" id="download">⬇ Download standalone HTML</button>
        <span class="muted" id="sizeInfo"></span>
      </div>
      <p class="muted" style="margin:10px 0 12px;">Live preview (this IS the file you'll
        download — play, tweak its sliders, export video/PNG inside it):</p>
      <iframe id="preview" sandbox="allow-scripts allow-same-origin allow-downloads allow-modals"></iframe>
    </div>
  </div>

<script>
const PLAYER_B64 = /*__PLAYER_B64__*/"";
const PRESETS    = /*__PRESETS__*/({});
const PLAYER_TEMPLATE = new TextDecoder().decode(
  Uint8Array.from(atob(PLAYER_B64), c => c.charCodeAt(0)));

const $ = id => document.getElementById(id);
let currentFile = null, lastHtml = null;

// ---- settings readouts -------------------------------------------------- //
const SLIDERS = { gw:"gwV", gh:"ghV", fps:"fpsV", levels:"levV", scale:"scaleV", maxFrames:"mfV" };
function syncVals() {
  for (const [id, vid] of Object.entries(SLIDERS)) $(vid).textContent = $(id).value;
}
for (const id of Object.keys(SLIDERS)) $(id).addEventListener("input", syncVals);

// populate presets
for (const name of Object.keys(PRESETS)) {
  const o = document.createElement("option"); o.value = name; o.textContent = name;
  $("preset").appendChild(o);
}
$("preset").value = "heavy-datamosh";
syncVals();

// ---- file handling ------------------------------------------------------ //
const drop = $("drop"), fileInput = $("file");
drop.addEventListener("click", () => fileInput.click());
drop.addEventListener("dragover", e => { e.preventDefault(); drop.classList.add("hot"); });
drop.addEventListener("dragleave", () => drop.classList.remove("hot"));
drop.addEventListener("drop", e => {
  e.preventDefault(); drop.classList.remove("hot");
  if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", () => { if (fileInput.files[0]) setFile(fileInput.files[0]); });
function setFile(f) {
  currentFile = f;
  $("fname").textContent = f.name + "  (" + (f.size / 1048576).toFixed(1) + " MB)";
  $("convert").disabled = false;
  $("status").textContent = "Ready to convert.";
}

// ---- helpers ------------------------------------------------------------ //
function progress(f) { $("prog").style.width = Math.round(f * 100) + "%"; }
function status(s) { $("status").textContent = s; }
function once(el, ev) { return new Promise(res => el.addEventListener(ev, res, { once: true })); }
function bytesToB64(bytes) {
  let bin = ""; const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk)
    bin += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
  return btoa(bin);
}

// ---- decode frames from the uploaded video (canvas seeking) ------------- //
async function decodeFrames(file, w, h, fps, maxFrames, onProg) {
  const url = URL.createObjectURL(file);
  const video = document.createElement("video");
  video.src = url; video.muted = true; video.playsInline = true; video.preload = "auto";
  await once(video, "loadedmetadata");
  let duration = video.duration;
  if (!isFinite(duration) || duration <= 0) duration = (maxFrames || 120) / fps;
  const cnv = document.createElement("canvas"); cnv.width = w; cnv.height = h;
  const c = cnv.getContext("2d", { willReadFrequently: true });
  const frames = [];
  const dt = 1 / fps;
  const total = maxFrames > 0 ? maxFrames : Math.floor(duration * fps);
  for (let i = 0; i < total; i++) {
    const t = i * dt;
    if (t > duration) break;
    // small epsilon so the first seek differs from the initial 0 (else "seeked" may never fire)
    video.currentTime = Math.min(Math.max(t, 0.001), Math.max(0.001, duration - 0.001));
    await once(video, "seeked");
    c.drawImage(video, 0, 0, w, h);
    frames.push(c.getImageData(0, 0, w, h).data.slice());
    if (onProg) onProg((i + 1) / total);
  }
  URL.revokeObjectURL(url);
  return frames;
}

// ---- quantize + delta-encode (mirrors glitch_processor.py) -------------- //
function encode(frames, w, h, levels, keyint, deltaThresh, fps) {
  const npix = w * h, out = [];
  let prev = null;
  for (let n = 0; n < frames.length; n++) {
    const rgba = frames[n];
    const cur = new Uint16Array(npix);
    for (let i = 0; i < npix; i++) {
      const qr = (rgba[i * 4] * levels) >> 8;
      const qg = (rgba[i * 4 + 1] * levels) >> 8;
      const qb = (rgba[i * 4 + 2] * levels) >> 8;
      cur[i] = (qr * levels + qg) * levels + qb;
    }
    if (!prev) {
      out.push({ t: 0, p: Array.from(cur) });
    } else {
      const changed = [];
      for (let i = 0; i < npix; i++) if (cur[i] !== prev[i]) changed.push(i);
      if (n % keyint === 0 || changed.length > deltaThresh * npix) {
        out.push({ t: 0, p: Array.from(cur) });
      } else {
        const d = new Array(changed.length * 2);
        for (let k = 0; k < changed.length; k++) { d[k * 2] = changed[k]; d[k * 2 + 1] = cur[changed[k]]; }
        out.push({ t: 1, d });
      }
    }
    prev = cur;
  }
  return { w, h, levels, fps, n: out.length, frames: out };
}

// ---- gzip (native) + base64 --------------------------------------------- //
async function gzipBase64(obj) {
  const json = JSON.stringify(obj);
  const stream = new Blob([json]).stream().pipeThrough(new CompressionStream("gzip"));
  const buf = await new Response(stream).arrayBuffer();
  return bytesToB64(new Uint8Array(buf));
}

// ---- optional audio: decode -> mono/resample -> WAV --------------------- //
function encodeWav(samples, rate) {
  const n = samples.length, buf = new ArrayBuffer(44 + n * 2), v = new DataView(buf);
  const wstr = (off, s) => { for (let i = 0; i < s.length; i++) v.setUint8(off + i, s.charCodeAt(i)); };
  wstr(0, "RIFF"); v.setUint32(4, 36 + n * 2, true); wstr(8, "WAVE"); wstr(12, "fmt ");
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, rate, true); v.setUint32(28, rate * 2, true);
  v.setUint16(32, 2, true); v.setUint16(34, 16, true); wstr(36, "data"); v.setUint32(40, n * 2, true);
  let o = 44;
  for (let i = 0; i < n; i++) { let s = Math.max(-1, Math.min(1, samples[i])); v.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7fff, true); o += 2; }
  return buf;
}
async function extractAudioWav(file, rate) {
  const AC = window.AudioContext || window.webkitAudioContext;
  if (!AC || !window.OfflineAudioContext) return null;
  const ctx = new AC();
  let decoded;
  try { decoded = await ctx.decodeAudioData(await file.arrayBuffer()); }
  catch (e) { ctx.close(); return null; }
  ctx.close();
  const len = Math.ceil(decoded.duration * rate);
  if (len <= 0) return null;
  const off = new OfflineAudioContext(1, len, rate);
  const src = off.createBufferSource(); src.buffer = decoded; src.connect(off.destination); src.start();
  const rendered = await off.startRendering();
  return { mime: "audio/wav", b64: bytesToB64(new Uint8Array(encodeWav(rendered.getChannelData(0), rate))) };
}

// ---- assemble the standalone player (mirrors Python build_html) --------- //
function buildStandalone(dataB64, cfg, audio) {
  let html = PLAYER_TEMPLATE;
  html = html.replace("/*__DATA__*/null", JSON.stringify(dataB64));
  html = html.replace("/*__CONFIG__*/({})", "(" + JSON.stringify(cfg) + ")");
  html = html.replace("/*__AUDIO__*/null", audio ? JSON.stringify(audio) : "null");
  return html;
}

// ---- main convert flow -------------------------------------------------- //
$("convert").addEventListener("click", async () => {
  if (!currentFile) return;
  if (typeof CompressionStream === "undefined") {
    status("Your browser lacks CompressionStream — please update it."); return;
  }
  $("convert").disabled = true; progress(0);
  try {
    const w = +$("gw").value, h = +$("gh").value, fps = +$("fps").value, levels = +$("levels").value;
    status("Decoding frames… (seeking through the video)");
    const frames = await decodeFrames(currentFile, w, h, fps, +$("maxFrames").value, f => progress(f * 0.6));
    if (!frames.length) throw new Error("No frames decoded — is the file a playable video?");
    status("Encoding + compressing…"); progress(0.65);
    const payload = encode(frames, w, h, levels, 24, 0.45, fps);
    const dataB64 = await gzipBase64(payload); progress(0.8);
    let audio = null;
    if ($("audio").checked) { status("Embedding audio…"); audio = await extractAudioWav(currentFile, +$("audioRate").value); }
    progress(0.9);
    const cfg = { scale: +$("scale").value, preset: $("preset").value, presets: PRESETS,
                  mosaic: $("mosaic").checked ? true : null, boxSize: null, sampleSize: null };
    lastHtml = buildStandalone(dataB64, cfg, audio);
    $("preview").srcdoc = lastHtml;
    $("resultCard").style.display = "";
    const kb = new Blob([lastHtml]).size / 1024;
    const srcKb = currentFile.size / 1024;
    $("sizeInfo").textContent = `${frames.length} frames · ${kb.toFixed(0)} KB`
      + (audio ? " (incl. audio)" : "") + ` · ${(100 * kb / srcKb).toFixed(1)}% of source`;
    status("Done ✓  Download below, or tweak inside the preview.");
    progress(1);
  } catch (err) {
    status("Error: " + err.message); console.error(err); progress(0);
  } finally {
    $("convert").disabled = false;
  }
});

$("download").addEventListener("click", () => {
  if (!lastHtml) return;
  const base = (currentFile && currentFile.name.replace(/\.[^.]+$/, "")) || "glitch";
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([lastHtml], { type: "text/html" }));
  a.download = base + ".glitch.html";
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
});
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
