# Media Generation Pipeline

AI-powered image and video generation/understanding framework.

## Setup

## Setup

1. **Install uv:**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   # or with pip: pip install uv
   ```

2. **Setup with uv:**
   ```bash
   python setup_uv.py
   ```

3. **Or manually with uv:**
   ```bash
   uv venv
   uv sync
   uv sync --extra dev  # For development dependencies
   cp env.example .env
   ```

4. **Configure API keys:**
   ```bash
   cp env.example .env
   # Add your OPENAI_API_KEY to .env
   # Add your REPLICATE_API_TOKEN to .env
   ```

## Usage

### Running with uv

Use `uv run` to execute commands in the virtual environment:

```bash
# Run any Python script
uv run python media_gen/image_regen_pipeline.py --image-path photo.jpg --user-interests "basketball, kapybara"

# Run tests
uv run pytest

# Run with specific Python version
uv run --python 3.11 python your_script.py
```

### Image Regeneration Pipeline

Regenerate images with AI analysis and user preferences:

```bash
uv run python media_gen/image_regen_pipeline.py --image-path photo.jpg --user-interests "basketball, kapybara"
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
uv run python media_gen/image_regen_pipeline.py --image-path landscape.jpg --user-interests "vintage style, steam punk"

# Custom output and aspect ratio
uv run python media_gen/image_regen_pipeline.py \
  --image-path ~/Pictures/photo.jpg \
  --user-interests "make it modern and professional" \
  --output-folder ~/Desktop \
  --aspect-ratio 16:9
```

### Video Regeneration Pipeline

Process videos with AI understanding and generation:

```bash
uv run python media_gen/video_regen_pipeline.py --video-path video.mp4 --user-interests "basketball, kapybara"
```

**Options:**
- `--video-path`: Input video path (required)
- `--user-interests`: Processing preferences (required)
- `--output-folder`: Output directory (default: `~/Downloads`)
- `--screenshot-interval`: Seconds between screenshots (default: `2.0`)

## Development

### Adding Dependencies

```bash
# Add a new dependency
uv add package-name

# Add a development dependency
uv add --dev package-name

# Update dependencies
uv sync
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=media_gen

# Run specific test file
uv run pytest media_gen/tests/test_image_understanding.py
```

### Code Quality

```bash
# Format code
uv run black .

# Sort imports
uv run isort .

# Type checking
uv run mypy media_gen/

# Linting
uv run flake8 media_gen/
```

## Extensibility

The framework is designed for easy extension. You can create custom tools for:

- **Arbitrary image generation models** (Stable Diffusion, DALL-E, etc.)
- **Video generation models** (Runway, Pika Labs, etc.)
- **Custom analysis tools** (scene detection, content filtering, etc.)

Tools follow the base class in `media_gen/tools/media_gen_tool_base.py` for consistent integration.
