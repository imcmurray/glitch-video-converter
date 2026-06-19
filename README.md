# GlitchVideo

[![Deploy converter to GitHub Pages](https://github.com/imcmurray/glitch-video-converter/actions/workflows/pages.yml/badge.svg)](https://github.com/imcmurray/glitch-video-converter/actions/workflows/pages.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-23f0ff.svg)](LICENSE)

### ▶ Try it in your browser: **https://imcmurray.github.io/glitch-video-converter/**

Convert a normal video into a **stylized digital glitch-art animation** that plays
in any browser by opening a single self-contained `.html` file — no server, no
external dependencies, no internet.

> The link above is the **in-browser converter** (auto-deployed from this repo via
> GitHub Pages). Drop in a video, tune the look, and download a standalone glitch
> animation — nothing is uploaded anywhere; it all runs on your machine.

The output looks "broken digital" and artistic — strong **RGB channel splits**,
**datamosh-style block tears**, **horizontal scanline glitches**, **color
fringing**, **noise**, and occasional **full-frame corruption** — while still
keeping the recognizable motion of the source.

Two ways to use it: the **Python CLI** (best quality, any format via ffmpeg) or a
**100% in-browser converter** you can host on GitHub Pages so anyone can drop in a
video and download a glitch animation — no install, no upload to any server.

---

## Web converter (GitHub Pages — no install)

Build the converter page and host it as a static site:

```bash
python3 glitch_processor.py --build-web        # writes index.html
```

Then host `index.html` anywhere static. **This repo deploys itself** via
GitHub Actions ([`.github/workflows/pages.yml`](.github/workflows/pages.yml)):
on every push to `main` it reruns `--build-web` and publishes the fresh page to
Pages, so the hosted app never drifts from `glitch_processor.py`. To replicate on
your own fork: **Settings → Pages → Source: GitHub Actions**, then push.

Visitors **drop in a video**, tune grid size / fps / colours / preset / mosaic /
audio, hit **Convert**, preview live, and **download a standalone `.glitch.html`**.
Everything runs client-side:

- **Decode** — the file is loaded into a hidden `<video>`, seeked frame-by-frame,
  and drawn to a tiny canvas (which does the downscaling).
- **Encode** — quantize + delta-encode in JS, gzip via the native
  `CompressionStream`, base64 — the *same* format the CLI produces.
- **Package** — the in-browser code rebuilds the exact same player and hands it to
  you as a download. The player template is embedded in `index.html`, so the
  converter and the CLI never drift.

Notes & limits:
- Works with whatever the browser can natively play (**MP4/H.264, WebM**, usually
  MOV). Exotic codecs would need `ffmpeg.wasm` (not bundled — keeps the page tiny).
- **Audio** is optional (off by default; procedural SFX already gives sound for
  free). When ticked, the source audio is decoded and re-embedded as mono WAV at
  the chosen sample rate — reliable and dependency-free, just larger than the
  CLI's Opus. For the smallest files with real audio, use the CLI + `--audio`.
- Nothing is uploaded — the video never leaves the visitor's machine.

---

## How it works

Two front-ends (CLI + browser) share **one** player and **one** data format. The
core idea: store a clean, tiny, recognizable video; generate the glitch *live*.

```
            ┌──────────────────────── CLI (Python) ───────────────────────┐
 video ───▶ │  ffmpeg decode → downscale 96×54 → palette-quantize (1 B/px) │
            │  → delta-encode (keyframe + changed pixels) → gzip → base64  │
            └──────────────────────────────────┬──────────────────────────┘
                                                │   (identical format)
            ┌──────────────────── Browser (index.html) ───────────────────┐
 video ───▶ │  <video>+canvas decode → same quantize/delta → CompressionStream gzip
            └──────────────────────────────────┬──────────────────────────┘
                                                ▼
                         data = base64( gzip( JSON ) )   ◀── a few % of the source
                                                │
                                                ▼  injected into the shared template
            ┌─────────────── Self-contained player .html ─────────────────┐
 open ───▶  │  DecompressionStream → frames →  PER FRAME, LIVE:            │
            │    pristine grid → [dominant-colour mosaic] → glitch overlay │
            │    (RGB split · datamosh blocks · tears · noise · TV static  │
            │     · corruption · scanlines) → canvas                       │
            │  + sound (procedural SFX / embedded track)                   │
            │  + offline WebCodecs export → WebM (VP9/Opus) | MP4 (H.264/AAC)
            └─────────────────────────────────────────────────────────────┘
```

**Why store the signal, not the glitch?** Glitch artifacts are high-entropy noise
that compresses terribly. By storing only the clean low-res frames and rendering
every effect procedurally at playback, the data stays tiny *and* the glitch
re-rolls organically each frame. The dominant-colour mosaic always samples the
**pristine** decoded frame — never the glitched output — so effects layer on top
without feeding back into the sampling.

> _Want a preview image here?_ Drop a screenshot/GIF of the player into the repo
> (e.g. `docs/preview.gif`) and reference it — I kept this text-only so the README
> has no binary assets.

---

## The trick (why it stays tiny)

We **don't store the glitch.** We store a clean, tiny, recognizable version of the
video and generate every glitch effect *live in the browser*:

| Stage | What happens | Why it compresses |
|-------|--------------|-------------------|
| Downsample | Each frame → small grid (default **96×54**) | ~99% fewer pixels |
| Quantize | Each pixel → 1-byte palette index (`levels³` colors) | 3 bytes → 1 byte |
| Delta-encode | Keyframe + only changed pixels between frames | Static areas cost ~0 |
| Pack | JSON → **gzip** → base64, embedded in the HTML | Index streams gzip extremely well |
| Decompress | Browser uses the native `DecompressionStream('gzip')` | **Zero JS dependencies** |

Because the *glitch is procedural*, it costs **zero bytes** and flickers
organically every frame.

> **Size expectation:** on real footage (with mostly-static backgrounds) the HTML
> is typically a few percent of the source. On pathological inputs where *every*
> pixel changes every frame (e.g. full-screen plasma / `testsrc2`), delta encoding
> can't help and you'll see 20–40%. Lower `--width/--height`, `--fps`, or
> `--levels` to trade detail for size.

---

## Files

| File | Purpose |
|------|---------|
| `glitch_processor.py` | Processes a video → `glitch_animation.html` + data |
| `glitch_animation.html` | Standalone player; plays a **synthetic demo** if opened directly (no Python needed) |
| `README.md` | This file |

---

## Requirements (EndeavourOS / Arch)

- **Python 3.8+** and **NumPy**
- **ffmpeg** (default decoder — handles every common format)

```bash
sudo pacman -S ffmpeg python-numpy
# or in a venv:  pip install numpy
```

OpenCV is **optional** (only if you pass `--decoder opencv`):

```bash
pip install opencv-python
```

A modern browser (Chrome/Edge/Firefox/Safari) — needs `DecompressionStream`,
available in all current browsers.

---

## Quick start

```bash
# 1. Just see it work (no video required) — open the standalone demo:
xdg-open glitch_animation.html

# 2. Glitch a real video:
python3 glitch_processor.py input.mp4
xdg-open glitch_animation.html        # the processor overwrites this file

# 3. Pick a different output name + preset:
python3 glitch_processor.py clip.mov -o my_glitch.html --preset subtle-neon
```

The resulting `.html` is fully self-contained — email it, host it anywhere, or
open it offline.

---

## Presets

Pick with `--preset`; every value is still tweakable live with sliders in the player.

| Preset | Vibe |
|--------|------|
| `heavy-datamosh` | Aggressive block tears, big RGB splits, frequent corruption *(default)* |
| `subtle-neon` | Gentle chroma fringing + light tears, mostly clean |
| `vhs` | Strong scanlines, wide soft tears, analog noise |
| `clean` | Almost no glitch — useful to confirm the source is recognizable |
| `clean-mosaic` | Dominant-color block mosaic, no glitch |
| `glitch-mosaic` | Mosaic base + moderate glitch overlay |
| `heavy-datamosh-mosaic` | Mosaic base + heavy datamosh overlay |

See **Dominant-Color Mosaic mode** below for the mosaic-specific controls.

---

## All options

```text
python3 glitch_processor.py [input] [options]

  input                 Input video (any ffmpeg-supported format). Omit for a
                        synthetic demo.

  -o, --output FILE     Output HTML file (default: glitch_animation.html)
  --width N             Grid width  (default 96)
  --height N            Grid height (default 54)
  --fps N               Playback / sampling fps (default 12)
  --levels N            Quantization levels per channel, 2-8 (default 6 → 216 colors)
  --keyint N            Force a keyframe every N frames (default 24)
  --delta-thresh F      Promote delta→keyframe if > this fraction changed (default 0.45)
  --max-frames N        Cap number of frames (useful for long clips)
  --preset NAME         heavy-datamosh | subtle-neon | vhs | clean
                        | clean-mosaic | glitch-mosaic | heavy-datamosh-mosaic
  --scale N             Display upscale factor; canvas = grid × scale (default 8)
  --audio               Embed the source soundtrack (synced playback).
                        Default: auto-embed when the source has audio.
  --no-audio            Do not embed the soundtrack (smaller file)
  --audio-bitrate B     Embedded audio bitrate, mono (default 24k)
  --mosaic              Start with dominant-color mosaic mode ON (overrides preset)
  --box-size N          Mosaic block render size in display px (e.g. 4, 8, 16, 32)
  --sample-size N       Mosaic dominant-color sampling area in display px
                        (independent of box size; defaults to box size)
  --decoder {ffmpeg,opencv}   Frame decoder backend (default ffmpeg)
  --json FILE           Also write raw uncompressed JSON sidecar
  --play-text           Play an ASCII preview in the terminal (writes no HTML)
  --selftest            Use a built-in synthetic animation (no input video)
```

### Examples

```bash
# Higher detail (bigger file), 15 fps:
python3 glitch_processor.py clip.mp4 --width 128 --height 72 --fps 15

# Tiniest possible: tiny grid, few colors, low fps:
python3 glitch_processor.py clip.mp4 --width 64 --height 36 --fps 8 --levels 4

# Long video, just the first 5 seconds at 12 fps:
python3 glitch_processor.py movie.mkv --fps 12 --max-frames 60

# Terminal-only ASCII playback (the text fallback):
python3 glitch_processor.py clip.mp4 --play-text

# Keep a raw JSON copy for tooling / inspection:
python3 glitch_processor.py clip.mp4 --json frames.json

# Dominant-color mosaic, 16px blocks, sampled over a wider 24px area:
python3 glitch_processor.py clip.mp4 --mosaic --box-size 16 --sample-size 24

# Mosaic preset straight from the dropdown default:
python3 glitch_processor.py clip.mp4 --preset glitch-mosaic
```

---

## Dominant-Color Mosaic mode

A second **base layer** that replaces the picture with a grid of solid blocks,
each filled with the single **dominant color** of that region — then the usual
glitch effects overlay *on top* of the blocks.

- **Box size** — the rendered block size (display px). Smaller = more detail.
- **Sampling area** — the size of the region scanned to pick each block's
  dominant color, **independent of the box**. Equal to box size = classic mosaic;
  larger = smoother/blended blocks; smaller = punchier local detail.
- Dominant color is found by **histogram peak** over the palette.

> **Pristine-sampling guarantee:** the dominant color is always sampled from the
> *clean decoded frame* (the pre-glitch palette grid) — **never** from the
> already-glitched output. In the player, `applyMosaic()` rewrites the base layer
> from the pristine grid buffer *before* `glitchFrame()` runs, so RGB splits,
> tears, noise, etc. are layered on afterward and never feed back into sampling.

Toggle it live with the **▦ Mosaic** button, or bake it in with `--mosaic` /
`--box-size` / `--sample-size`. Three ready-made presets:

| Preset | Look |
|--------|------|
| `clean-mosaic` | Solid dominant-color blocks, no glitch |
| `glitch-mosaic` | Mosaic blocks + moderate RGB/tear/block glitch |
| `heavy-datamosh-mosaic` | Small blocks + wide sampling + heavy datamosh on top |

---

## In-browser controls

- **⏸ Pause / ▶ Play** (or press **Space**)
- **🔁 Loop** on/off
- **Speed** 0.1×–3×
- **Master ×ALL** — multiplies every *glitch* effect (0–2.5×). Does **not** touch
  the Scanline/CRT group or the Mosaic base layer.
- Per-effect sliders: **RGB split**, **block tears**, **scanline tears**,
  **noise (b/w sparkle)**, **TV static (snow)**, **corruption**
- **Scanlines / CRT** group (independent of master): **darkness**, **gap**, **size**
- **▦ Dominant-Color Mosaic** group: **on/off** toggle, **box size**, **sampling area**
- **Preset** dropdown — reload any baked-in look

Adjusting sliders is live; nothing is re-encoded.

### Export & sharing

The controls are organized into collapsible sections (Playback, Glitch,
Scanlines/CRT, Mosaic, Sound, Export). The **Export & share** section gives you:

- **⬇ Download video** — **offline, frame-accurate** export. Renders every frame
  deterministically with [WebCodecs](https://developer.mozilla.org/docs/Web/API/WebCodecs_API)
  (`VideoEncoder` + `AudioEncoder`) and muxes with a tiny built-in muxer —
  **no real-time playback, no dropped frames, exact fps**, audio at correct
  timestamps. Faster than real-time. Options beside it:
  - **format** — **WebM** (VP9 + Opus) or **MP4** (H.264 + AAC, best for phones /
    social apps). Each falls back gracefully if the codec isn't available.
  - **with audio** — encode the embedded soundtrack and mux it in.
  - **loops** — how many full passes to render (1–5).
  - **≈ size** — a live estimate of the output size and bitrate. It tracks your
    **selected effects**: the encoder bitrate scales with effect entropy
    (noise / static / corruption need far more bits to look good), so heavier
    glitch shows a bigger estimate *and* actually gets more bitrate.
  On browsers without WebCodecs it automatically falls back to the live recorder.
- **🎞 Record** — the legacy *real-time* recorder (records the screen as it plays).
  Kept for capturing a specific window or for browsers without WebCodecs; note it
  can drop frames if the tab is backgrounded.
- **📷 PNG** — download the current glitched frame.
- **🔗 Copy look** — encodes **every** setting (preset, all sliders, mosaic,
  speed, master) into the page URL and copies the link. **Open that link and the
  exact look is restored** — the simplest way to reproduce or share a result.
- **🎲 Random** — randomizes all effect sliders; hit **🔗 Copy look** to keep one.

> Both video exports mix audio via a WebAudio `MediaStreamDestination` tap —
> `canvas.captureStream()` is video-only, so the audio is routed separately and
> added to the recorded stream.

> Tip: to turn a discovered look into a permanent preset, **🔗 Copy look**, read
> the values out of the URL hash, and add them as a new entry in `PRESETS`
> (in `glitch_processor.py`) — they'll appear in the dropdown on the next build.

### Favorites (save looks you reuse)

- **★ Save favorite** — names the current settings and stores them in the
  browser's `localStorage`. Favorites show up in the **preset dropdown** under an
  *★ favorites* group and survive page reloads.
- Selecting a favorite restores it exactly; **🗑** deletes the selected one.

Favorites are per-browser (local). To share a look across machines use **🔗 Copy
look** (URL), or bake it into `PRESETS` for a permanent dropdown entry.

### Sound

The animation has **two** audio paths:

- **Procedural glitch SFX** (✧ SFX) — generated live with WebAudio, **zero bytes**.
  It reacts to the effects in real time: TV-static/noise → broadband hiss,
  scanlines → mains hum, RGB/block/corruption → digital chirps. Works on any file,
  including the synthetic demo.
- **Embedded soundtrack** — with `--audio` (auto when the source has audio) the
  original track is compressed to mono Opus, embedded, and played **synced** to
  the animation (speed slider scales playback rate; loop re-syncs).

Click **🔇 Sound: off** once to enable audio — browsers block autoplay until a
user gesture. **VOL** controls the master volume; **✧ SFX** toggles the procedural
layer (default on only when there's no embedded soundtrack).

---

## How the glitch is rendered (browser side)

Each frame: the clean low-res buffer is upscaled (nearest-neighbor) to the display
canvas, then a single pass applies, per pixel:

1. **Scanline tears** — random horizontal bands shifted sideways.
2. **Block displacement** — grid blocks randomly offset (datamosh feel).
3. **RGB channel separation** — R/B sampled at ±`rgb` px offset (split + fringing).
4. **Region corruption** — occasional band quantized/bit-masked + shifted.
5. **Noise** — sparkle pixels.
6. **Scanline overlay** — CRT darkening every other row.

All effects use fresh randomness each frame, so the glitch shimmers even on a
paused or slow source.

---

## Customizing further

- **New preset:** add an entry to `PRESETS` in `glitch_processor.py` (it gets
  baked into the player's dropdown automatically).
- **Different palette feel:** change `--levels` (fewer = chunkier, posterized;
  more = smoother but larger).
- **Sharper vs. softer scaling:** change `--scale` (display only; doesn't affect
  file size).

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ffmpeg not found` | `sudo pacman -S ffmpeg`, or use `--decoder opencv` |
| `OpenCV not installed` | `pip install opencv-python`, or drop `--decoder opencv` |
| Blank player / "Failed to load data" | Browser too old for `DecompressionStream` — update it |
| File too big | Lower `--width/--height`, `--fps`, or `--levels`; use `--max-frames` |
| Looks too clean / too wild | Switch `--preset`, or use the **Intensity** slider live |

---

## License

Do whatever you want with it.
