# Test Scripts

This folder contains test scripts that demonstrate and validate the media generation tools with real API calls and external resources.



## Replicate Image Generation Integration Test

### Prerequisites
- Replicate API token in `.env` file
- Internet connection
- Replicate API access

### Running the Test

```bash
# From the media-gen directory
python test_scripts/test_replicate_image_gen.py
```

### What it does
- Generates a single image with a cinematic cat prompt
- Tests the Replicate image generation tool with real API calls
- Saves the generated image to `~/Downloads/polymind_generated_images/`
- Provides helpful error messages for common issues

### Expected Output
The test will show:
- ✅ Confirmation that API token is found
- 🎨 Image generation progress
- 📁 File path and size of generated image
- 🎯 Generation metadata

### Generated Image
- Dynamically named image with timestamp (e.g., `replicate_generated_image_20241201_143022.jpeg`)

### Notes
- This test is not run automatically with unit tests
- It requires a valid Replicate API token
- It makes real API calls and may incur costs
- Generated images are saved in `~/Downloads/polymind_generated_images/` with unique names
- Uses the WAN 2.2 model by default

## Image Understanding Integration Test

### Prerequisites
- OpenAI API key in `.env` file
- Internet connection
- Test image file (`test_image.png`)

### Running the Test

```bash
# From the media-gen directory
python test_scripts/test_image_understanding.py
```

### What it does
- Loads the test image (`test_image.png`)
- Generates an image generation prompt that could be used to recreate the image
- Calls OpenAI's GPT-4o-mini API
- Displays the generated prompt and metadata

### Expected Output
The test will show:
- ✅ Confirmation that test image and API key are found
- 📝 The prompt being used
- 📋 The generated image generation prompt
- 📊 Token usage metadata

### Notes
- This test is not run automatically with unit tests
- It requires a valid OpenAI API key
- It makes real API calls and may incur costs
- The test image should be placed in this folder

## Video Concatenation Test

### Prerequisites
- OpenCV (cv2) installed
- Video files for testing
- Sufficient disk space for output videos

### Running the Test

```bash
# From the media-regen directory
uv run python media_gen/test_scripts/test_video_concatenation.py
```

### What it does
- Tests the new video concatenation functionality in `video_utils.py`
- Demonstrates both folder-based and list-based concatenation
- Shows how to concatenate videos in alphabetical order
- Handles different video formats and resolutions

### Features Tested
- **Folder-based concatenation**: Automatically finds and concatenates all videos in a folder
- **Alphabetical sorting**: Concatenates videos in alphabetical order by filename
- **Format support**: Handles .mp4, .avi, .mov, .mkv, .webm files
- **Resolution conversion**: Can resize videos to target resolution
- **FPS conversion**: Can adjust frame rate of output video
- **Error handling**: Graceful handling of missing files and processing errors

### Configuration
Update the test script to point to your video files:
- `folder_path`: Directory containing video files to concatenate
- `video_paths`: List of specific video file paths
- `output_path`: Where to save the concatenated video

### Expected Output
The test will show:
- 📁 List of video files found
- 🔄 Processing progress for each video
- ✅ Success/failure status
- 📊 Output video properties (duration, frames, resolution, FPS)

### Notes
- This test requires actual video files to be present
- Update the file paths in the script before running
- Output videos are saved in MP4 format using H.264 codec
- The test demonstrates both automatic folder scanning and manual file list approaches 