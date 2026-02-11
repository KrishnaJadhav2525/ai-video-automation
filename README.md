# AI Video Generation Pipeline

Fully automated AI-powered video generation: **one topic in → one YouTube-ready MP4 out**.

Orchestrated by **n8n**, powered by **Gemini**, **Edge TTS**, **Pexels**, and **MoviePy/FFmpeg** — all free-tier.

---

## 📁 Folder Structure

```
ai-video-automation/
├── main.py                 # Complete pipeline script
├── requirements.txt        # Python dependencies
├── n8n_workflow.json       # Importable n8n workflow
├── .env.example            # API key template
├── .env                    # Your actual API keys (create this)
├── writeup.md              # Technical write-up
├── README.md               # This file
└── output/                 # Generated after run
    ├── final_video.mp4     # YouTube-ready video
    ├── thumbnail.jpg       # Auto-generated thumbnail
    ├── subtitles.srt       # SRT subtitle file
    └── temp/               # Intermediate files
```

---

## 🔧 Prerequisites

1. **Python 3.10+** — [python.org](https://www.python.org/downloads/)
2. **FFmpeg** — Must be installed and on your system PATH
   - Windows: `winget install ffmpeg` or download from [ffmpeg.org](https://ffmpeg.org/download.html)
   - Verify: `ffmpeg -version`
3. **n8n Desktop** — [n8n.io/get-started](https://n8n.io/get-started/) (free self-hosted)

---

## 🔑 API Keys (Free)

| Service | Get Key | Free Tier |
|---------|---------|-----------|
| **Google Gemini** | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) | 15 RPM, 1M tokens/min |
| **Pexels** | [pexels.com/api](https://www.pexels.com/api/) | 200 requests/hour |

### Setup API Keys

1. Copy the example file:
   ```bash
   cp .env.example .env
   ```
2. Edit `.env` and add your keys:
   ```
   GEMINI_API_KEY=your_actual_gemini_key
   PEXELS_API_KEY=your_actual_pexels_key
   ```

---

## 🚀 Quick Start (Command Line)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up API keys (see above)

# 3. Run the pipeline
python main.py "The History of Chess"

# 4. Find your video
#    → output/final_video.mp4
#    → output/thumbnail.jpg
#    → output/subtitles.srt
```

---

## ⚡ Quick Start (n8n — Full Automation)

1. **Open n8n Desktop** (or access your self-hosted instance)
2. Go to **Settings → Import Workflow** (or Ctrl+O)
3. Select `n8n_workflow.json` from this folder
4. **Edit the topic**: Click the "Set Topic" node and change the topic string
5. Click **"Test Workflow"** or **"Execute Workflow"**
6. Wait ~2–3 minutes → your video appears in `output/final_video.mp4`

### n8n Workflow Overview

```
Manual Trigger → Set Topic (Code Node) → Execute Command (python main.py)
```

- **Manual Trigger**: Starts the workflow
- **Set Topic**: Contains the video topic (edit it here)
- **Execute Command**: Runs `python main.py "<topic>"` with a 10-minute timeout

---

## 🎬 Pipeline Stages

| Stage | Tool | What it does |
|-------|------|--------------|
| 1. Script Generation | Google Gemini | Generates structured JSON script with scenes, narration, and visual queries |
| 2. Voiceover | Edge TTS | Converts narration to natural-sounding MP3 audio |
| 3. Visuals | Pexels API | Fetches HD landscape images matching each scene |
| 4. Video Assembly | MoviePy + FFmpeg | Combines images + audio with Ken Burns effect, exports 1080p MP4 |

### Bonus Features

- 🖼️ **Thumbnail** — Auto-generated from first scene with title overlay
- 📝 **Subtitles** — SRT file generated from scene narrations
- 🔍 **SEO** — YouTube-optimized title and description printed to console

---

## 📋 Output Specifications

| Property | Value |
|----------|-------|
| Format | MP4 (H.264 + AAC) |
| Resolution | 1920×1080 (1080p) |
| Aspect Ratio | 16:9 |
| FPS | 24 |
| Duration | ~60–90 seconds |
| Bitrate | 5000 kbps |

---

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| `GEMINI_API_KEY not set` | Create `.env` file with your key (see API Keys section) |
| `FFmpeg not found` | Install FFmpeg and ensure it's on your PATH |
| `Pexels fetch failed` | Check your Pexels API key; the pipeline will use fallback images |
| n8n timeout | Increase timeout in Execute Command node settings |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |

---

## 📄 License

This project is for educational/assignment purposes. All APIs used are free-tier.
