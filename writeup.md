# AI Video Generation Pipeline - Submission Write-up

## Tools Used
- **Orchestration:** n8n (Workflow automation) & Python (Core logic).
- **Script Generation:** Google Gemini 2.0 Flash (Primary) with Groq/Llama-3 (Fallback).
- **Voiceover:** Edge TTS (`en-US-ChristopherNeural`) for high-quality, free neural speech.
- **Visuals:** Pexels, Unsplash, and Pixabay APIs (Multi-source aggregation for better relevance).
- **Assembly:** MoviePy (FFmpeg wrapper) for video editing; Pillow for thumbnail generation.

## Biggest Challenge
**Execution Speed vs. Quality.** 
Rendering 1080p video with Python is CPU-intensive. Initial builds took over 10 minutes per video, which was unacceptable for automation. Optimizing `moviepy` with `preset='ultrafast'`, implementing threaded writing, and removing complex per-frame effects (like Ken Burns) were crucial steps. This reduced generation time to under 60 seconds while maintaining acceptable visual quality. Additionally, finding relevant free stock images for abstract topics was difficult, requiring a keyword simplification algorithm (`_simplify_query`) and a multi-source fallback system.

## Improvements
1. **Dynamic Visuals:** Replace static images with AI-generated video clips (e.g., Kling/Luma) or stock video footage for better engagement.
2. **Native n8n Integration:** Decompose the monolithic Python script into granular n8n nodes for better error handling, visibility, and scalability.
3. **Advanced Captions:** Implement burn-in subtitles with word-level highlight animations (Karaoke style) instead of static SRT files.
