"""
AI Video Generation Pipeline
=============================
Fully automated: one topic in → one YouTube-ready MP4 out.

Usage:
    python main.py "Your Topic Here"

Requires:
    - GEMINI_API_KEY or GROQ_API_KEY in .env (at least one)
    - PEXELS_API_KEY in .env
    - FFmpeg installed and on PATH
"""

import sys
import os
import json
import re
import asyncio
import textwrap
import time
from pathlib import Path
from io import BytesIO

import requests
from PIL import Image, ImageDraw, ImageFont
import edge_tts
from moviepy import (
    ImageClip,
    AudioFileClip,
    CompositeVideoClip,
    concatenate_videoclips,
    TextClip,
    vfx,
)
import google.generativeai as genai
from dotenv import load_dotenv

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")

VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
FPS = 24
VOICE = "en-US-ChristopherNeural"  # Natural male voice

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
TEMP_DIR = OUTPUT_DIR / "temp"


def setup_directories():
    """Create output and temp directories."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    TEMP_DIR.mkdir(exist_ok=True)
    print("[✓] Directories ready.")


# ──────────────────────────────────────────────
# Stage 1: Script Generation (Gemini / Groq)
# ──────────────────────────────────────────────
SCRIPT_PROMPT_TEMPLATE = """\
You are a professional YouTube script writer.

Create a short video script about: "{topic}"

Requirements:
- The video should be 60-90 seconds long when narrated.
- Split the script into 5-7 scenes.
- Each scene should have about 2-3 sentences of narration.
- Provide a visual search query for each scene (used to find stock images).
- Generate an SEO-optimized YouTube title and description.

Return your response as valid JSON with this exact structure:
{{
    "title": "SEO-optimized YouTube title",
    "description": "SEO-optimized YouTube description (2-3 sentences)",
    "scenes": [
        {{
            "scene_number": 1,
            "narration": "The narration text for this scene.",
            "visual_query": "a concise search query for a relevant stock photo"
        }}
    ]
}}

IMPORTANT:
- Return ONLY the JSON, no markdown fences, no extra text.
- Keep narration natural and engaging.
- Visual queries should be specific and descriptive for stock photos.
- Aim for 5-7 scenes total.
"""


def _generate_with_gemini(prompt: str) -> str:
    """Generate script using Google Gemini."""
    print("  [→] Trying Gemini...")
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text.strip()


def _generate_with_groq(prompt: str) -> str:
    """Generate script using Groq (free tier, llama model)."""
    print("  [→] Trying Groq...")
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 2048,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def generate_script(topic: str) -> dict:
    """
    Generate a structured video script using Gemini or Groq.
    Tries Gemini first; falls back to Groq if Gemini fails.
    Returns dict with 'title', 'description', 'scenes' list.
    """
    print(f"\n[Stage 1] Generating script for: {topic}")

    prompt = SCRIPT_PROMPT_TEMPLATE.format(topic=topic)
    raw_text = None

    # Try Gemini first
    if GEMINI_API_KEY:
        try:
            raw_text = _generate_with_gemini(prompt)
            print("  [✓] Gemini responded.")
        except Exception as e:
            print(f"  [!] Gemini failed: {e}")
            print("  [→] Falling back to Groq...")

    # Fallback to Groq
    if raw_text is None and GROQ_API_KEY:
        try:
            raw_text = _generate_with_groq(prompt)
            print("  [✓] Groq responded.")
        except Exception as e:
            print(f"  [!] Groq also failed: {e}")

    if raw_text is None:
        print("[ERROR] All script generation methods failed.")
        print("  Make sure you have a valid GEMINI_API_KEY or GROQ_API_KEY in .env")
        sys.exit(1)

    # Clean markdown fences if present
    raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
    raw_text = re.sub(r"\s*```$", "", raw_text)

    script_data = json.loads(raw_text)

    scene_count = len(script_data.get("scenes", []))
    print(f"[✓] Script generated: \"{script_data.get('title', '')}\" — {scene_count} scenes")
    return script_data


# ──────────────────────────────────────────────
# Stage 2: Voiceover Generation (Edge TTS)
# ──────────────────────────────────────────────
async def _generate_single_voiceover(text: str, output_path: str, voice: str):
    """Generate a single audio file from text using Edge TTS."""
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)


def generate_voiceovers(scenes: list) -> list:
    """
    Generate voiceover MP3 for each scene.
    Returns list of audio file paths.
    """
    print("\n[Stage 2] Generating voiceovers with Edge TTS...")
    audio_paths = []

    for i, scene in enumerate(scenes):
        narration = scene["narration"]
        audio_path = str(TEMP_DIR / f"scene_{i + 1}.mp3")

        asyncio.run(_generate_single_voiceover(narration, audio_path, VOICE))
        audio_paths.append(audio_path)
        print(f"  [✓] Scene {i + 1} audio saved ({len(narration)} chars)")

    print(f"[✓] All {len(audio_paths)} voiceovers generated.")
    return audio_paths


# ──────────────────────────────────────────────
# Stage 3: Visual Fetching (Pexels API)
# ──────────────────────────────────────────────
def fetch_pexels_image(query: str) -> Image.Image | None:
    """Fetch a landscape image from Pexels for the given query."""
    url = "https://api.pexels.com/v1/search"
    headers = {"Authorization": PEXELS_API_KEY}
    params = {"query": query, "per_page": 5, "orientation": "landscape"}

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        if data.get("photos"):
            # Pick the first landscape photo, use the 'large2x' size
            photo_url = data["photos"][0]["src"]["large2x"]
            img_resp = requests.get(photo_url, timeout=30)
            img_resp.raise_for_status()
            return Image.open(BytesIO(img_resp.content))
    except Exception as e:
        print(f"  [!] Pexels fetch failed for '{query}': {e}")

    return None


def create_fallback_image(query: str) -> Image.Image:
    """Create a gradient image with text as a fallback."""
    img = Image.new("RGB", (VIDEO_WIDTH, VIDEO_HEIGHT))
    draw = ImageDraw.Draw(img)

    # Dark gradient background
    for y in range(VIDEO_HEIGHT):
        r = int(20 + (y / VIDEO_HEIGHT) * 30)
        g = int(20 + (y / VIDEO_HEIGHT) * 50)
        b = int(40 + (y / VIDEO_HEIGHT) * 80)
        draw.line([(0, y), (VIDEO_WIDTH, y)], fill=(r, g, b))

    # Add topic text in center
    try:
        font = ImageFont.truetype("arial.ttf", 48)
    except OSError:
        font = ImageFont.load_default()

    text = query[:60]  # Truncate long queries
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (VIDEO_WIDTH - tw) // 2
    y = (VIDEO_HEIGHT - th) // 2
    draw.text((x, y), text, fill=(200, 200, 220), font=font)

    return img


def resize_and_crop(img: Image.Image) -> Image.Image:
    """Resize and center-crop image to exactly 1920×1080."""
    target_ratio = VIDEO_WIDTH / VIDEO_HEIGHT
    img_ratio = img.width / img.height

    if img_ratio > target_ratio:
        # Image is wider → resize by height, crop width
        new_height = VIDEO_HEIGHT
        new_width = int(img.width * (VIDEO_HEIGHT / img.height))
    else:
        # Image is taller → resize by width, crop height
        new_width = VIDEO_WIDTH
        new_height = int(img.height * (VIDEO_WIDTH / img.width))

    img = img.resize((new_width, new_height), Image.LANCZOS)

    # Center crop
    left = (new_width - VIDEO_WIDTH) // 2
    top = (new_height - VIDEO_HEIGHT) // 2
    img = img.crop((left, top, left + VIDEO_WIDTH, top + VIDEO_HEIGHT))

    return img


def fetch_visuals(scenes: list) -> list:
    """
    Fetch or generate an image for each scene.
    Returns list of image file paths.
    """
    print("\n[Stage 3] Fetching visuals from Pexels...")
    image_paths = []

    for i, scene in enumerate(scenes):
        query = scene["visual_query"]
        img_path = str(TEMP_DIR / f"scene_{i + 1}.jpg")

        img = fetch_pexels_image(query)
        if img is None:
            print(f"  [!] Fallback for scene {i + 1}: '{query}'")
            img = create_fallback_image(query)
        else:
            print(f"  [✓] Scene {i + 1} image fetched: '{query}'")

        img = resize_and_crop(img.convert("RGB"))
        img.save(img_path, "JPEG", quality=95)
        image_paths.append(img_path)

    print(f"[✓] All {len(image_paths)} visuals ready.")
    return image_paths


# ──────────────────────────────────────────────
# Stage 4: Video Assembly (MoviePy + FFmpeg)
# ──────────────────────────────────────────────
def create_ken_burns_clip(image_path: str, duration: float) -> CompositeVideoClip:
    """
    Create a clip with a subtle Ken Burns (zoom-in) effect
    to make the still image feel dynamic.
    """
    # Start slightly zoomed out, end at full size
    zoom_start = 1.0
    zoom_end = 1.15

    img_clip = ImageClip(image_path).with_duration(duration)

    def zoom_effect(get_frame, t):
        """Apply smooth zoom over time."""
        import numpy as np
        from PIL import Image as PILImage

        progress = t / duration
        current_zoom = zoom_start + (zoom_end - zoom_start) * progress

        frame = get_frame(t)
        h, w = frame.shape[:2]

        # Calculate crop region for zoom
        new_w = int(w / current_zoom)
        new_h = int(h / current_zoom)
        x_offset = (w - new_w) // 2
        y_offset = (h - new_h) // 2

        cropped = frame[y_offset:y_offset + new_h, x_offset:x_offset + new_w]

        # Resize back to original dimensions
        pil_img = PILImage.fromarray(cropped)
        pil_img = pil_img.resize((w, h), PILImage.LANCZOS)
        return np.array(pil_img)

    zoomed_clip = img_clip.transform(zoom_effect)
    return zoomed_clip


def assemble_video(
    image_paths: list,
    audio_paths: list,
    scenes: list,
    title: str,
) -> str:
    """
    Assemble the final video from images + audio.
    Returns path to the final MP4.
    """
    print("\n[Stage 4] Assembling video with MoviePy...")
    clips = []
    scene_durations = []

    for i, (img_path, aud_path) in enumerate(zip(image_paths, audio_paths)):
        # Load audio to get duration
        audio_clip = AudioFileClip(aud_path)
        duration = audio_clip.duration + 0.5  # Add 0.5s padding between scenes
        scene_durations.append(audio_clip.duration)

        # Create image clip with Ken Burns effect
        video_clip = create_ken_burns_clip(img_path, duration)
        video_clip = video_clip.with_audio(audio_clip)

        # Add fade in/out transitions
        if i == 0:
            video_clip = video_clip.with_effects([vfx.FadeIn(0.8)])
        if i == len(image_paths) - 1:
            video_clip = video_clip.with_effects([vfx.FadeOut(0.8)])

        clips.append(video_clip)
        print(f"  [✓] Scene {i + 1}: {duration:.1f}s")

    # Concatenate all scene clips
    final_clip = concatenate_videoclips(clips, method="compose")

    # Export
    output_path = str(OUTPUT_DIR / "final_video.mp4")
    final_clip.write_videofile(
        output_path,
        fps=FPS,
        codec="libx264",
        audio_codec="aac",
        bitrate="5000k",
        preset="medium",
        threads=4,
        logger="bar",
    )

    # Cleanup
    final_clip.close()
    for clip in clips:
        clip.close()

    total_duration = sum(scene_durations)
    print(f"\n[✓] Video exported: {output_path}")
    print(f"    Duration: {total_duration:.1f}s | Resolution: {VIDEO_WIDTH}x{VIDEO_HEIGHT} | FPS: {FPS}")

    return output_path, scene_durations


# ──────────────────────────────────────────────
# Bonus: Thumbnail Generation
# ──────────────────────────────────────────────
def generate_thumbnail(first_image_path: str, title: str) -> str:
    """Generate a thumbnail from the first scene image with title overlay."""
    print("\n[Bonus] Generating thumbnail...")

    img = Image.open(first_image_path).copy()
    draw = ImageDraw.Draw(img)

    # Semi-transparent dark overlay at the bottom
    overlay = Image.new("RGBA", (VIDEO_WIDTH, 300), (0, 0, 0, 180))
    img = img.convert("RGBA")
    img.paste(overlay, (0, VIDEO_HEIGHT - 300), overlay)
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)

    # Title text
    try:
        font = ImageFont.truetype("arial.ttf", 64)
    except OSError:
        font = ImageFont.load_default()

    # Wrap title text
    wrapped = textwrap.fill(title, width=35)
    bbox = draw.textbbox((0, 0), wrapped, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (VIDEO_WIDTH - tw) // 2
    y = VIDEO_HEIGHT - 280

    # Text shadow
    draw.text((x + 3, y + 3), wrapped, fill=(0, 0, 0), font=font)
    # Main text
    draw.text((x, y), wrapped, fill=(255, 255, 255), font=font)

    thumb_path = str(OUTPUT_DIR / "thumbnail.jpg")
    img.save(thumb_path, "JPEG", quality=95)
    print(f"[✓] Thumbnail saved: {thumb_path}")
    return thumb_path


# ──────────────────────────────────────────────
# Bonus: SRT Subtitles
# ──────────────────────────────────────────────
def generate_subtitles(scenes: list, durations: list) -> str:
    """Generate an SRT subtitle file from scene narrations."""
    print("\n[Bonus] Generating subtitles...")

    srt_path = str(OUTPUT_DIR / "subtitles.srt")
    current_time = 0.0

    with open(srt_path, "w", encoding="utf-8") as f:
        for i, (scene, dur) in enumerate(zip(scenes, durations)):
            start_time = current_time
            end_time = current_time + dur

            # Format SRT timestamps
            def format_ts(seconds):
                h = int(seconds // 3600)
                m = int((seconds % 3600) // 60)
                s = int(seconds % 60)
                ms = int((seconds % 1) * 1000)
                return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

            f.write(f"{i + 1}\n")
            f.write(f"{format_ts(start_time)} --> {format_ts(end_time)}\n")

            # Word-wrap narration for subtitle readability
            narration = scene["narration"]
            wrapped = textwrap.fill(narration, width=50)
            f.write(f"{wrapped}\n\n")

            current_time = end_time + 0.5  # 0.5s gap between scenes

    print(f"[✓] Subtitles saved: {srt_path}")
    return srt_path


# ──────────────────────────────────────────────
# Main Pipeline
# ──────────────────────────────────────────────
def main():
    """Run the full video generation pipeline."""
    if len(sys.argv) < 2:
        print("Usage: python main.py \"Your Topic Here\"")
        sys.exit(1)

    topic = sys.argv[1]

    # Validation
    if not GEMINI_API_KEY and not GROQ_API_KEY:
        print("[ERROR] No AI API key set. Add GEMINI_API_KEY or GROQ_API_KEY to .env")
        sys.exit(1)
    if not PEXELS_API_KEY:
        print("[ERROR] PEXELS_API_KEY not set. Add it to your .env file.")
        sys.exit(1)

    print("=" * 60)
    print(f"  AI VIDEO GENERATION PIPELINE")
    print(f"  Topic: {topic}")
    print("=" * 60)

    start_time = time.time()

    # Setup
    setup_directories()

    # Stage 1: Generate Script
    script_data = generate_script(topic)
    scenes = script_data["scenes"]
    title = script_data.get("title", topic)
    description = script_data.get("description", "")

    # Stage 2: Generate Voiceovers
    audio_paths = generate_voiceovers(scenes)

    # Stage 3: Fetch Visuals
    image_paths = fetch_visuals(scenes)

    # Stage 4: Assemble Video
    video_path, scene_durations = assemble_video(image_paths, audio_paths, scenes, title)

    # Bonus: Thumbnail
    thumbnail_path = generate_thumbnail(image_paths[0], title)

    # Bonus: Subtitles
    srt_path = generate_subtitles(scenes, scene_durations)

    # Bonus: Print SEO info
    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\n  Video:      {video_path}")
    print(f"  Thumbnail:  {thumbnail_path}")
    print(f"  Subtitles:  {srt_path}")
    print(f"\n  SEO Title:       {title}")
    print(f"  SEO Description: {description}")
    print(f"\n  Total time: {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
