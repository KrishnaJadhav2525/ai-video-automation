"""
AI Video Generation Pipeline
=============================
Fully automated: one topic in → one YouTube-ready MP4 out.

Usage:
    python main.py "Your Topic Here"

Requires:
    - GEMINI_API_KEY or GROQ_API_KEY in .env (at least one)
    - At least one image API key: PEXELS_API_KEY, UNSPLASH_ACCESS_KEY, or PIXABAY_API_KEY
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
from concurrent.futures import ThreadPoolExecutor, as_completed
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
import google.generativeai as genai
import argparse
from dotenv import load_dotenv
from pipeline_utils import setup_logging, retry_api_call, retry_async_api_call, tracker, logger

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")

VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
FPS = 24
VOICE = "en-US-ChristopherNeural"  # Natural male voice

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
TEMP_DIR = OUTPUT_DIR / "temp"
STATE_FILE = TEMP_DIR / "pipeline_state.json"


def setup_directories(clean=False):
    """Create output and temp directories."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    TEMP_DIR.mkdir(exist_ok=True)
    if clean and STATE_FILE.exists():
        STATE_FILE.unlink()
    logger.info("[+] Directories ready.")


def save_state(data: dict):
    """Save pipeline state to JSON."""
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_state() -> dict:
    """Load pipeline state from JSON."""
    if not STATE_FILE.exists():
        logger.error("[!] No state file found. Run 'script' stage first.")
        sys.exit(1)
    with open(STATE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


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


@retry_api_call("Gemini")
def _generate_with_gemini(prompt: str) -> str:
    """Generate script using Google Gemini."""
    logger.info("  [->] Trying Gemini...")
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text.strip()


@retry_api_call("Groq")
def _generate_with_groq(prompt: str) -> str:
    """Generate script using Groq (free tier, llama model)."""
    logger.info("  [->] Trying Groq...")
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
    logger.info(f"\n[Stage 1] Generating script for: {topic}")

    prompt = SCRIPT_PROMPT_TEMPLATE.format(topic=topic)
    raw_text = None

    # Try Gemini first
    if GEMINI_API_KEY:
        try:
            raw_text = _generate_with_gemini(prompt)
            logger.info("  [+] Gemini responded.")
        except Exception as e:
            logger.warning(f"  [!] Gemini failed: {e}")
            logger.info("  [->] Falling back to Groq...")

    # Fallback to Groq
    if raw_text is None and GROQ_API_KEY:
        try:
            raw_text = _generate_with_groq(prompt)
            logger.info("  [+] Groq responded.")
        except Exception as e:
            logger.error(f"  [!] Groq also failed: {e}")

    if raw_text is None:
        print("[ERROR] All script generation methods failed.")
        print("  Make sure you have a valid GEMINI_API_KEY or GROQ_API_KEY in .env")
        sys.exit(1)

    # Clean markdown fences if present
    raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
    raw_text = re.sub(r"\s*```$", "", raw_text)

    script_data = json.loads(raw_text)

    scene_count = len(script_data.get("scenes", []))
    logger.info(f"[+] Script generated: \"{script_data.get('title', '')}\" — {scene_count} scenes")
    return script_data


# ──────────────────────────────────────────────
# Stage 2: Voiceover Generation (Edge TTS)
# ──────────────────────────────────────────────
@retry_async_api_call("EdgeTTS")
async def _generate_single_voiceover(text: str, output_path: str, voice: str):
    """Generate a single audio file from text using Edge TTS."""
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)


def generate_voiceovers(scenes: list) -> list:
    """
    Generate voiceover MP3 for each scene.
    Returns list of audio file paths.
    """
    logger.info("\n[Stage 2] Generating voiceovers with Edge TTS...")
    audio_paths = []

    for i, scene in enumerate(scenes):
        narration = scene["narration"]
        audio_path = str(TEMP_DIR / f"scene_{i + 1}.mp3")

        asyncio.run(_generate_single_voiceover(narration, audio_path, VOICE))
        audio_paths.append(audio_path)
        logger.info(f"  [+] Scene {i + 1} audio saved ({len(narration)} chars)")

    logger.info(f"[+] All {len(audio_paths)} voiceovers generated.")
    return audio_paths


# ──────────────────────────────────────────────
# Stage 3: Multi-Source Visual Fetching
#   Sources: Pexels, Unsplash, Pixabay
# ──────────────────────────────────────────────
import random

UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY", "")
PIXABAY_API_KEY = os.getenv("PIXABAY_API_KEY", "")


def _simplify_query(query: str) -> str:
    """Reduce a long visual query to its 2-3 core keywords for broader results."""
    words = query.split()
    # Remove common filler words
    stop_words = {
        "a", "an", "the", "of", "in", "on", "at", "for", "to", "with",
        "and", "or", "is", "are", "was", "were", "be", "been", "being",
        "that", "this", "it", "its", "from", "by", "as", "into", "about",
        "showing", "depicting", "featuring", "photo", "image", "picture",
    }
    keywords = [w for w in words if w.lower() not in stop_words]
    # Return 2-3 most important words
    return " ".join(keywords[:3])


def _download_image(url: str) -> Image.Image | None:
    """Download an image from a URL and return a PIL Image."""
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content))
    except Exception:
        return None


@retry_api_call("Pexels")
def fetch_pexels_image(query: str) -> Image.Image | None:
    """Fetch a random landscape image from Pexels for the given query."""
    if not PEXELS_API_KEY:
        return None
    url = "https://api.pexels.com/v1/search"
    headers = {"Authorization": PEXELS_API_KEY}
    params = {"query": query, "per_page": 15, "orientation": "landscape"}

    # Requests automatically retried by decorator if exception raised
    resp = requests.get(url, headers=headers, params=params, timeout=15)
    resp.raise_for_status()
    photos = resp.json().get("photos", [])
    if photos:
        photo = random.choice(photos)
        return _download_image(photo["src"]["large2x"])
    return None


@retry_api_call("Unsplash")
def fetch_unsplash_image(query: str) -> Image.Image | None:
    """Fetch a random landscape image from Unsplash for the given query."""
    if not UNSPLASH_ACCESS_KEY:
        return None
    url = "https://api.unsplash.com/search/photos"
    headers = {"Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"}
    params = {
        "query": query,
        "per_page": 15,
        "orientation": "landscape",
        "content_filter": "high",
    }

    resp = requests.get(url, headers=headers, params=params, timeout=15)
    resp.raise_for_status()
    results = resp.json().get("results", [])
    if results:
        photo = random.choice(results)
        # Use 'regular' size (~1080px width) for good quality
        return _download_image(photo["urls"]["regular"])
    return None


@retry_api_call("Pixabay")
def fetch_pixabay_image(query: str) -> Image.Image | None:
    """Fetch a random landscape image from Pixabay for the given query."""
    if not PIXABAY_API_KEY:
        return None
    url = "https://pixabay.com/api/"
    params = {
        "key": PIXABAY_API_KEY,
        "q": query,
        "per_page": 15,
        "orientation": "horizontal",
        "image_type": "photo",
        "safesearch": "true",
        "min_width": 1280,
    }

    resp = requests.get(url, headers={}, params=params, timeout=15)
    resp.raise_for_status()
    hits = resp.json().get("hits", [])
    if hits:
        photo = random.choice(hits)
        return _download_image(photo["largeImageURL"])
    return None


# List of all fetchers — tried in random order for variety
_IMAGE_FETCHERS = [
    ("Pexels", fetch_pexels_image),
    ("Unsplash", fetch_unsplash_image),
    ("Pixabay", fetch_pixabay_image),
]


def fetch_image_multi_source(query: str) -> Image.Image | None:
    """
    Try multiple image sources in random order.
    If the original query fails everywhere, retry with simplified keywords.
    """
    # Shuffle the order so different scenes use different sources
    fetchers = list(_IMAGE_FETCHERS)
    random.shuffle(fetchers)

    fallback_chain = " -> ".join([name for name, _ in fetchers])
    logger.info(f"    [Logic] Attempting visuals for '{query}': {fallback_chain}")

    # Attempt 1: original query across all sources
    for name, fetcher in fetchers:
        try:
            img = fetcher(query)
            if img is not None:
                logger.info(f"    [+] Found via {name}: '{query}'")
                return img
        except Exception:
            # Exceptions inside fetcher are handled/retried by decorator, 
            # but if they bubble up (after retries exhausted), we catch here to try next source
            logger.warning(f"    [!] {name} exhausted retries for '{query}'")
            continue

    # Attempt 2: simplified query across all sources
    simple_query = _simplify_query(query)
    if simple_query != query:
        logger.info(f"    [RETRY] Retrying with simplified query: '{simple_query}'")
        random.shuffle(fetchers)
        for name, fetcher in fetchers:
            try:
                img = fetcher(simple_query)
                if img is not None:
                    logger.info(f"    [+] Found via {name}: '{simple_query}'")
                    return img
            except Exception:
                continue

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
    Fetch or generate an image for each scene in PARALLEL.
    Tries Pexels, Unsplash, and Pixabay in random order.
    Returns list of image file paths.
    """
    sources = []
    if PEXELS_API_KEY:
        sources.append("Pexels")
    if UNSPLASH_ACCESS_KEY:
        sources.append("Unsplash")
    if PIXABAY_API_KEY:
        sources.append("Pixabay")

    print(f"\n[Stage 3] Fetching visuals from {', '.join(sources) or 'fallback only'}...")
    
    # Pre-calculate paths to keep order
    results = [None] * len(scenes)
    
    def process_scene(index, scene):
        query = scene["visual_query"]
        img_path = str(TEMP_DIR / f"scene_{index + 1}.jpg")
        
        logger.info(f"  [->] Fetching scene {index + 1}: '{query}'...")
        img = fetch_image_multi_source(query)
        if img is None:
            logger.warning(f"  [!] All sources failed — using fallback for scene {index + 1}: '{query}'")
            img = create_fallback_image(query)
            tracker.record_attempt("Fallback", True, latency=0.1) # Track fallback usage
        else:
            logger.info(f"  [+] Scene {index + 1} image found.")

        # Resize/crop immediately
        try:
             img = resize_and_crop(img.convert("RGB"))
             img.save(img_path, "JPEG", quality=95)
             return index, img_path
        except Exception as e:
            logger.error(f"  [!] Error processing image for scene {index + 1}: {e}")
            tracker.record_attempt("ImageProcessing", False, error_msg=str(e))
            return index, None

    # Run in parallel
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_scene = {executor.submit(process_scene, i, scene): i for i, scene in enumerate(scenes)}
        for future in as_completed(future_to_scene):
            i, path = future.result()
            results[i] = path

    # Check for any failures (shouldn't happen with fallback, but good to be safe)
    final_paths = [p for p in results if p]
    
    if len(final_paths) != len(scenes):
        logger.warning("[!] Warning: Some scenes failed to generate images.")
    
    logger.info(f"[+] All {len(final_paths)} visuals ready.")
    return final_paths


def create_ken_burns_clip(image_path: str, duration: float) -> CompositeVideoClip:
    """
    Create a static clip (Ken Burns disabled for speed).
    """
    # FASTEST: Static image, no effects
    clip = ImageClip(image_path).with_duration(duration)
    
    # Just ensure it fits 720p (already resized in fetch_visuals)
    # The resize_and_crop function handles the aspect ratio.
    
    return clip


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
    logger.info("\n[Stage 4] Assembling video with MoviePy...")
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
        logger.info(f"  [+] Scene {i + 1}: {duration:.1f}s")

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
        preset="ultrafast",  # <--- CRITICAL SPEEDUP
        threads=8,           # Use more threads
        logger="bar",
    )

    # Cleanup
    final_clip.close()
    for clip in clips:
        clip.close()

    total_duration = sum(scene_durations)
    logger.info(f"\n[+] Video exported: {output_path}")
    logger.info(f"    Duration: {total_duration:.1f}s | Resolution: {VIDEO_WIDTH}x{VIDEO_HEIGHT} | FPS: {FPS}")

    return output_path, scene_durations


# ──────────────────────────────────────────────
# Bonus: Thumbnail Generation
# ──────────────────────────────────────────────
def generate_thumbnail(first_image_path: str, title: str) -> str:
    """Generate a thumbnail from the first scene image with title overlay."""
    logger.info("\n[Bonus] Generating thumbnail...")

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
    logger.info(f"[+] Thumbnail saved: {thumb_path}")
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

    logger.info(f"[+] Subtitles saved: {srt_path}")
    return srt_path


# ──────────────────────────────────────────────
# Main Pipeline
# ──────────────────────────────────────────────
def main():
    """Run the video generation pipeline (all-in-one or staged)."""
    parser = argparse.ArgumentParser(description="AI Video Generation Pipeline")
    parser.add_argument("topic", nargs="?", help="Video topic (required for 'script' stage or full run)")
    parser.add_argument("--stage", choices=["script", "audio", "visuals", "assemble"], help="Run a specific stage")
    args = parser.parse_args()

    # Validation
    if not GEMINI_API_KEY and not GROQ_API_KEY:
        print("[ERROR] No AI API key set. Add GEMINI_API_KEY or GROQ_API_KEY to .env")
        sys.exit(1)
    if not PEXELS_API_KEY and not UNSPLASH_ACCESS_KEY and not PIXABAY_API_KEY:
        print("[ERROR] No image API key set. Add PEXELS_API_KEY, UNSPLASH_ACCESS_KEY, or PIXABAY_API_KEY to .env")
        sys.exit(1)

    # Sanity check for n8n unparsed expressions
    if args.topic and ("{{$json" in args.topic or "}}" in args.topic):
        print(f"[ERROR] Invalid topic detected: '{args.topic}'")
        print("  This looks like an unparsed n8n expression.")
        print("  Fix: In n8n, change the Command parameter to 'Expression' mode.")
        sys.exit(1)

    # ──────────────────────────────────────────────
    # Mode 1: Full Pipeline (Legacy / default)
    # ──────────────────────────────────────────────
    if not args.stage:
        if not args.topic:
            print("Usage: python main.py \"Your Topic Here\"")
            sys.exit(1)
        
        print("=" * 60)
        print(f"  AI VIDEO GENERATION PIPELINE (FULL)")
        print(f"  Topic: {args.topic}")
        print("=" * 60)
        setup_directories(clean=True)
        
        # Run all stages
        script_data = generate_script(args.topic)
        audio_paths = generate_voiceovers(script_data["scenes"])
        image_paths = fetch_visuals(script_data["scenes"])
        video_path, _ = assemble_video(image_paths, audio_paths, script_data["scenes"], script_data.get("title", args.topic))
        generate_thumbnail(image_paths[0], script_data.get("title", args.topic))
        # Subs skipped in full run for brevity, or add back if needed
        
        # Generate Error Report
        tracker.generate_report(OUTPUT_DIR / "error_report.json")
        return

    # ──────────────────────────────────────────────
    # Mode 2: Staged Execution (for n8n)
    # ──────────────────────────────────────────────
    print(f"\n=== Running Stage: {args.stage.upper()} ===")
    
    if args.stage == "script":
        if not args.topic:
            print("[ERROR] Topic is required for script stage.")
            sys.exit(1)
        setup_directories(clean=True)
        script_data = generate_script(args.topic)
        # Save state
        state = {
            "topic": args.topic,
            "script": script_data,
            "scenes": script_data["scenes"]
        }
        save_state(state)
        print("[+] State saved. Ready for 'audio'.")

    elif args.stage == "audio":
        state = load_state()
        scenes = state["scenes"]
        audio_paths = generate_voiceovers(scenes)
        state["audio_paths"] = audio_paths
        save_state(state)
        print("[+] State saved. Ready for 'visuals'.")

    elif args.stage == "visuals":
        state = load_state()
        scenes = state["scenes"]
        image_paths = fetch_visuals(scenes)
        state["image_paths"] = image_paths
        save_state(state)
        print("[+] State saved. Ready for 'assemble'.")

    elif args.stage == "assemble":
        state = load_state()
        if "audio_paths" not in state or "image_paths" not in state:
            print("[ERROR] Missing audio or content. Run previous stages first.")
            sys.exit(1)
            
        video_path, durations = assemble_video(
            state["image_paths"], 
            state["audio_paths"], 
            state["scenes"], 
            state["script"].get("title", state["topic"])
        )
        
        # Bonus items
        thumb_path = generate_thumbnail(state["image_paths"][0], state["script"].get("title", state["topic"]))
        srt_path = generate_subtitles(state["scenes"], durations)
        
        logger.info("\n" + "=" * 60)
        logger.info("  PIPELINE COMPLETE")
        logger.info("=" * 60)
        logger.info(f"  Video: {video_path}")
        logger.info(f"  Thumb: {thumb_path}")
        logger.info(f"  Subs:  {srt_path}")
        
    # Generate Error Report
    tracker.generate_report(OUTPUT_DIR / "error_report.json")


if __name__ == "__main__":
    main()
