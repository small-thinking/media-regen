# Media Generation Pipeline

AI-powered image and video generation/understanding framework.

## Quick Start

1. **Setup environment:**
   ```bash
   cp env.example .env
   # Add your REPLICATE_API_TOKEN to .env
   pip install -r requirements.txt
   ```

2. **Generate images:**
   ```bash
   python media-gen/image_regen_pipeline.py --image-path photo.jpg --user-interests "make it more vibrant"
   ```

## Features

- **Image Understanding**: Analyze images with AI vision
- **Image Generation**: Create new images via Replicate
- **Video Understanding**: Extract and analyze video scenes
- **Video Generation**: Generate videos from images and prompts

## Usage

### Image Regeneration
```bash
python media-gen/image_regen_pipeline.py \
  --image-path ~/Pictures/photo.jpg \
  --user-interests "convert to watercolor style" \
  --output-folder ~/Desktop \
  --aspect-ratio 16:9
```

**Options:**
- `--image-path`: Input image (required)
- `--user-interests`: Regeneration preferences (required)
- `--output-folder`: Output directory (default: `~/Downloads`)
- `--aspect-ratio`: Image ratio (default: `1:1`)
- `--output-format`: Format (default: `png`)
- `--debug`: Show detailed prompts

### Video Understanding
```python
from tools.video_understanding_tool import VideoUnderstandingTool

video_tool = VideoUnderstandingTool()
result = video_tool.run({
    "video_path": "video.mp4",
    "user_preference": "cinematic lighting",
    "screenshot_interval": 2.0
})
```

### Video Generation
```python
from tools.replicate_video_gen import ReplicateVideoGen

video_gen = ReplicateVideoGen()
result = video_gen.run({
    "image": "image.jpg",
    "prompt": "serene landscape with gentle movement"
})
```

## Testing

```bash
# Test image generation
python media-gen/image_regen_pipeline.py --image-path media-gen/test_scripts/test_image.png --user-interests "enhance visual appeal"

# Test video tools
python media-gen/test_scripts/test_video_understanding_tool.py
python media-gen/test_scripts/test_replicate_video_gen.py
```

## Architecture

- **`pipeline.py`**: Core pipeline infrastructure
- **`image_regen_pipeline.py`**: Image regeneration CLI
- **`video_regen_pipeline.py`**: Video processing pipeline
- **`tools/`**: Media generation and understanding tools
- **`utils/`**: Utility functions for video processing