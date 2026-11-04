"""Build the demo video from the running app.

    uv run uvicorn app.main:app --port 8765    # in another terminal
    uv run python video/build.py

Narration uses ElevenLabs when ELEVENLABS_API_KEY is set in .env, and the
macOS voice "Rishi" otherwise. Footage is a real recording of the console
driven by Playwright, so everything on screen is the actual product.
"""

from __future__ import annotations

import json
import os
import subprocess
import textwrap
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "video" / "out"
APP = os.environ.get("SAATH_URL", "http://localhost:8765")
W, H = 1280, 720

SCENES = [
    {
        "id": "intro",
        "say": "On a busy night, a crisis text line answers people in the order they arrive. "
        "Someone who has just taken pills can end up waiting behind someone asking for clinic timings, "
        "and some people give up before anyone replies.",
    },
    {
        "id": "queue",
        "say": "Saath reorders that queue by suicide risk. It reads each waiting message, estimates where it sits "
        "on the Columbia suicide severity scale, and moves the most dangerous conversations to the top. "
        "Waiting time counts too, so nobody is stuck at the bottom.",
    },
    {
        "id": "detail",
        "say": "Counsellors can see the reasoning. The lines that drove the score are highlighted, and the card "
        "shows a range of plausible levels instead of a single confident guess.",
    },
    {
        "id": "language",
        "say": "At most one in twenty high-risk messages may fall outside the urgent band, and in testing, "
        "two percent did. Messages in romanised Hindi or Telugu, which the model still reads "
        "poorly, are never put in the standard queue.",
    },
    {
        "id": "evidence",
        "say": "Here is one typical night, replayed twice with the same arrivals. Over three hundred such nights, "
        "twenty-two percent of high-risk texters gave up waiting in first-come order, and fifteen percent "
        "with Saath. The gain held when we changed staffing, patience and how common high-risk messages are. "
        "It runs on a laptop CPU, so no message ever leaves the helpline.",
    },
    {
        "id": "outro",
        "say": "If you or someone you know is struggling, in India you can call Tele-MANAS on 1 4 4 1 6.",
    },
]


def env(name):
    path = ROOT / ".env"
    if path.exists():
        for line in path.read_text().splitlines():
            if line.startswith(name + "="):
                return line.split("=", 1)[1].strip() or None
    return os.environ.get(name)


def run(*cmd):
    subprocess.run(cmd, check=True, capture_output=True)


def duration(path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(out.stdout.strip())


# ---------- narration ----------
def narrate(scene) -> Path:
    wav = OUT / f"{scene['id']}.wav"
    key = env("ELEVENLABS_API_KEY")
    if key:
        voice = env("ELEVENLABS_VOICE_ID") or "JBFqnCBsd6RMkjVDRZzb"
        req = urllib.request.Request(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice}?output_format=mp3_44100_128",
            data=json.dumps({"text": scene["say"], "model_id": "eleven_multilingual_v2"}).encode(),
            headers={"xi-api-key": key, "content-type": "application/json"},
        )
        mp3 = OUT / f"{scene['id']}.mp3"
        mp3.write_bytes(urllib.request.urlopen(req, timeout=120).read())
        run("ffmpeg", "-y", "-i", str(mp3), "-ar", "44100", "-ac", "2", str(wav))
    else:
        aiff = OUT / f"{scene['id']}.aiff"
        run("say", "-v", "Rishi", "-r", "175", "-o", str(aiff), scene["say"])
        run("ffmpeg", "-y", "-i", str(aiff), "-ar", "44100", "-ac", "2", str(wav))
    return wav


# ---------- pexels intro still ----------
def pexels_still() -> Path:
    path = OUT / "intro.jpg"
    if path.exists():
        return path
    key = env("PEXELS_API_KEY")
    q = urllib.parse.quote("city at night window rain")
    req = urllib.request.Request(
        f"https://api.pexels.com/v1/search?query={q}&orientation=landscape&per_page=5",
        headers={"Authorization": key, "User-Agent": "saath-demo-video"},
    )
    photos = json.load(urllib.request.urlopen(req, timeout=60))["photos"]
    photo = photos[0]
    (OUT / "intro_credit.txt").write_text(f"Photo: {photo['photographer']} on Pexels, {photo['url']}\n")
    img = urllib.request.Request(photo["src"]["large2x"], headers={"User-Agent": "saath-demo-video"})
    path.write_bytes(urllib.request.urlopen(img, timeout=120).read())
    return path


# ---------- captions ----------
def _font(size, bold=False):
    for name in ["Inter-SemiBold.ttf", "Helvetica.ttc"] if bold else ["Inter-Regular.ttf", "Helvetica.ttc"]:
        for base in ["/System/Library/Fonts", "/Library/Fonts", str(Path.home() / "Library/Fonts")]:
            p = Path(base) / name
            if p.exists():
                return ImageFont.truetype(str(p), size)
    return ImageFont.load_default(size)


def caption_png(text, path):
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    font = _font(30)
    lines = textwrap.wrap(text, 62)
    lh = 40
    box_h = lh * len(lines) + 28
    y0 = H - box_h - 36
    d.rounded_rectangle([(W - 1040) // 2, y0, (W + 1040) // 2, y0 + box_h], 14, fill=(10, 10, 10, 190))
    for i, line in enumerate(lines):
        tw = d.textlength(line, font=font)
        d.text(((W - tw) / 2, y0 + 14 + i * lh), line, font=font, fill=(245, 244, 240, 255))
    img.save(path)


def captions_for(scene, dur):
    """Split narration into sentences and time each by its share of characters."""
    import re

    parts = [p.strip() for p in re.split(r"(?<=[.,])\s+(?=[A-Z])|(?<=\.)\s+", scene["say"]) if p.strip()]
    total = sum(len(p) for p in parts)
    t, out = 0.0, []
    for i, p in enumerate(parts):
        span = dur * len(p) / total
        png = OUT / f"cap_{scene['id']}_{i}.png"
        caption_png(p, png)
        out.append((png, t, t + span))
        t += span
    return out


def title_png(path, big, small):
    img = Image.new("RGB", (W, H), (21, 21, 20))
    d = ImageDraw.Draw(img)
    f1, f2 = _font(64, bold=True), _font(28)
    tw = d.textlength(big, font=f1)
    d.text(((W - tw) / 2, H / 2 - 70), big, font=f1, fill=(236, 235, 231))
    for i, line in enumerate(small.split("\n")):
        tw = d.textlength(line, font=f2)
        d.text(((W - tw) / 2, H / 2 + 20 + i * 40), line, font=f2, fill=(156, 152, 144))
    img.save(path)


# ---------- footage ----------
async def record(scene_id, seconds) -> Path:
    from playwright.async_api import async_playwright

    vid_dir = OUT / f"rec_{scene_id}"
    vid_dir.mkdir(exist_ok=True)
    for old in vid_dir.glob("*.webm"):
        old.unlink()
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        ctx = await browser.new_context(
            viewport={"width": W, "height": H},
            record_video_dir=str(vid_dir),
            record_video_size={"width": W, "height": H},
            color_scheme="dark",
            device_scale_factor=1,
        )
        page = await ctx.new_page()
        await page.goto(APP)
        await page.wait_for_selector("#queue li", timeout=120_000)
        t = page.wait_for_timeout

        if scene_id == "queue":
            await page.click("[data-order=fifo]")
            await page.click("#shift")
            await t(int(seconds * 1000 * 0.4))
            await page.click("[data-order=risk]")
            await t(int(seconds * 1000 * 0.6))
        elif scene_id == "detail":
            await page.click("#shift")
            await t(7000)
            await page.click("#shift")  # pause so the card stays put
            await page.click(".card.critical, .card.urgent")
            await page.wait_for_selector("#detail .msg", timeout=60_000)
            await t(800)
            await page.screenshot(path=str(ROOT / "docs" / "queue.png"))
            await t(int(seconds * 1000) - 6800)
        elif scene_id == "language":
            await page.click("[data-tab=try]")
            await page.fill("#try-text", "")
            await page.type("#try-text", "kaash main hota hi nahi. kisi ko fark nahi padta", delay=45)
            await page.click("#try-form button")
            await page.wait_for_selector("#try-result .msg", timeout=60_000)
            await t(int(seconds * 1000) - 4000)
        elif scene_id == "evidence":
            await page.click("[data-tab=evidence]")
            await page.wait_for_selector("#replay", timeout=60_000)
            await t(1500)
            await page.click("#replay")  # restart so the race is on camera
            await t(int(seconds * 1000 * 0.6))
            await page.screenshot(path=str(ROOT / "docs" / "evidence.png"))
            await page.mouse.wheel(0, 420)
            await t(int(seconds * 1000 * 0.4))
        await ctx.close()
        await browser.close()
    return next(vid_dir.glob("*.webm"))


def clip_from_footage(webm, dur, caps, out):
    """Keep the last `dur` seconds of the recording (skips page load) and burn captions."""
    total = duration(webm)
    start = max(0.0, total - dur)
    inputs = ["-ss", f"{start:.2f}", "-i", str(webm)]
    for png, *_ in caps:
        inputs += ["-i", str(png)]
    chain, last = [], "[0:v]"
    for i, (_, a, b) in enumerate(caps, start=1):
        chain.append(f"{last}[{i}:v]overlay=0:0:enable='between(t,{a:.2f},{b:.2f})'[v{i}]")
        last = f"[v{i}]"
    run(
        "ffmpeg",
        "-y",
        *inputs,
        "-filter_complex",
        ";".join(chain) or "null",
        "-map",
        last,
        "-t",
        f"{dur:.2f}",
        "-r",
        "30",
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
        "-crf",
        "20",
        str(out),
    )


def clip_from_still(img, dur, caps, out, zoom=True):
    inputs = ["-loop", "1", "-i", str(img)]
    for png, *_ in caps:
        inputs += ["-i", str(png)]
    frames = int(dur * 30)
    base = (
        f"[0:v]scale={W * 2}:-1,crop={W * 2}:{H * 2},zoompan=z='min(zoom+0.0006,1.12)':d={frames}:s={W}x{H}:fps=30,"
        "eq=brightness=-0.14:saturation=0.35[b]"
        if zoom
        else f"[0:v]scale={W}:{H}[b]"
    )
    chain, last = [base], "[b]"
    for i, (_, a, b) in enumerate(caps, start=1):
        chain.append(f"{last}[{i}:v]overlay=0:0:enable='between(t,{a:.2f},{b:.2f})'[v{i}]")
        last = f"[v{i}]"
    run(
        "ffmpeg",
        "-y",
        *inputs,
        "-filter_complex",
        ";".join(chain),
        "-map",
        last,
        "-t",
        f"{dur:.2f}",
        "-r",
        "30",
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
        "-crf",
        "20",
        str(out),
    )


def main():
    import asyncio

    OUT.mkdir(parents=True, exist_ok=True)
    pad = 0.6
    clips = []

    title = OUT / "title.png"
    title_png(
        title, "Saath", "Risk-ordered queue triage for crisis text lines\nNeural Storm 2026 · HI-05 · Yasaswi Paruchuri"
    )
    clip_from_still(title, 3.0, [], OUT / "c_title.mp4", zoom=False)
    run("ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-t", "3.0", str(OUT / "a_title.wav"))
    clips.append((OUT / "c_title.mp4", OUT / "a_title.wav"))

    for scene in SCENES:
        wav = narrate(scene)
        dur = duration(wav) + pad
        caps = captions_for(scene, dur - pad)
        clip = OUT / f"c_{scene['id']}.mp4"
        if scene["id"] == "intro":
            clip_from_still(pexels_still(), dur, caps, clip)
        elif scene["id"] == "outro":
            card = OUT / "outro.png"
            title_png(card, "Tele-MANAS 14416", "Free, 24×7, 20 languages\nEmergency: 112")
            clip_from_still(card, dur + 1.5, caps, clip, zoom=False)
            dur += 1.5
        else:
            webm = asyncio.run(record(scene["id"], dur + 1))
            clip_from_footage(webm, dur, caps, clip)
        padded = OUT / f"a_{scene['id']}.wav"
        run("ffmpeg", "-y", "-i", str(wav), "-af", f"apad=whole_dur={dur:.2f}", str(padded))
        clips.append((clip, padded))
        print("scene", scene["id"], f"{dur:.1f}s", flush=True)

    vlist = OUT / "v.txt"
    alist = OUT / "a.txt"
    vlist.write_text("".join(f"file '{c}'\n" for c, _ in clips))
    alist.write_text("".join(f"file '{a}'\n" for _, a in clips))
    run("ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(vlist), "-c", "copy", str(OUT / "video_only.mp4"))
    run("ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(alist), str(OUT / "narration.wav"))
    final = ROOT / "video" / "saath_demo.mp4"
    run(
        "ffmpeg",
        "-y",
        "-i",
        str(OUT / "video_only.mp4"),
        "-i",
        str(OUT / "narration.wav"),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        "-shortest",
        str(final),
    )
    print("wrote", final, f"{duration(final):.1f}s")


if __name__ == "__main__":
    main()
