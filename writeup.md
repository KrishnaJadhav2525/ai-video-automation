# Technical Write-Up: AI Video Generation Pipeline

## Tool Selection Rationale

Every tool in this pipeline was chosen per assignment requirements, prioritizing **zero-cost, local-first execution**. **n8n** (self-hosted) serves as the orchestration layer because it provides a visual workflow builder with a Manual Trigger node that cleanly passes the topic into **Python**, our core programming language. **Google Gemini** (free tier: 15 RPM, 1 million tokens/minute) generates structured JSON scripts — chosen over Groq/Grok for its generous free quota and reliable JSON output. **Edge TTS** provides Microsoft-quality neural voices with zero API keys and no rate limits, making it the most robust free voiceover option. **Pexels API** (200 req/hour free) supplies royalty-free HD stock images. Finally, **MoviePy** wraps **FFmpeg** in a Pythonic API, enabling Ken Burns effects, fade transitions, and precise audio synchronization without shell scripting.

## End-to-End Pipeline Flow

The user clicks n8n's Manual Trigger, which passes a topic string to an Execute Command node. This invokes `main.py "<topic>"`, which runs four sequential stages: (1) Gemini generates a 5–7 scene script with narration and visual queries; (2) Edge TTS converts each narration into MP3; (3) Pexels fetches matching landscape images per scene; (4) MoviePy composites image clips with Ken Burns zoom, overlays audio, applies fade transitions, and exports a 1920×1080 MP4 at 24fps. Bonus outputs include a thumbnail, SRT subtitles, and SEO metadata.

## Biggest Technical Challenge

The hardest problem was making still images feel cinematic. A simple slideshow looks unprofessional, so I implemented a **Ken Burns (zoom-in) effect** using MoviePy's `transform()` function — each frame is dynamically cropped and resized to simulate camera movement. Combined with cross-scene fade transitions and precise audio-to-image synchronization, the result feels like a produced video rather than a slide deck.

## Scaling Improvements

If scaled, the pipeline would benefit from: (1) **parallel scene processing** using `asyncio` for simultaneous TTS + image fetching; (2) **AI-generated images** (Stable Diffusion) instead of stock photos for unique visuals; (3) **subtitle burn-in** with word-level timestamps from Whisper; (4) **queue-based orchestration** replacing n8n's Execute Command with a message broker for concurrent video generation; and (5) **cloud storage integration** to upload finished videos directly to YouTube via its API.
