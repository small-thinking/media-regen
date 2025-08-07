# Media Generation Pipeline

AI-powered image and video generation/understanding framework.

## Setup

1. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API keys:**
   ```bash
   cp env.example .env
   # Add your OPENAI_API_KEY to .env
   # Add your REPLICATE_API_TOKEN to .env
   ```

## Usage

### Image Regeneration Pipeline

Regenerate images with AI analysis and user preferences:

```bash
python media-gen/image_regen_pipeline.py --image-path photo.jpg --user-interests "basketball, kapybara"
```

**Options:**
- `--image-path`: Input image path (required)
- `--user-interests`: Regeneration preferences (required)
- `--output-folder`: Output directory (default: `~/Downloads`)
- `--aspect-ratio`: Image ratio (default: `1:1`)
- `--output-format`: Format (default: `png`)
- `--debug`: Show detailed prompts

**Examples:**
```bash
# Basic regeneration
python media-gen/image_regen_pipeline.py --image-path landscape.jpg --user-interests "vintage style, steam punk"

# Custom output and aspect ratio
python media-gen/image_regen_pipeline.py \
  --image-path ~/Pictures/photo.jpg \
  --user-interests "make it modern and professional" \
  --output-folder ~/Desktop \
  --aspect-ratio 16:9
```

### Video Regeneration Pipeline

Process videos with AI understanding and generation:

```bash
python media-gen/video_regen_pipeline.py --video-path video.mp4 --user-interests "basketball, kapybara"
```

**Options:**
- `--video-path`: Input video path (required)
- `--user-interests`: Processing preferences (required)
- `--output-folder`: Output directory (default: `~/Downloads`)
- `--screenshot-interval`: Seconds between screenshots (default: `2.0`)

## Extensibility

The framework is designed for easy extension. You can create custom tools for:

- **Arbitrary image generation models** (Stable Diffusion, DALL-E, etc.)
- **Video generation models** (Runway, Pika Labs, etc.)
- **Custom analysis tools** (scene detection, content filtering, etc.)

Tools follow the base class in `media-gen/tools/media_gen_tool_base.py` for consistent integration.
