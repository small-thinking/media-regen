#!/usr/bin/env python3
"""
Video concatenation integration test.

This script demonstrates the video concatenation functionality by processing
videos from a specific folder and concatenating them in alphabetical order.

Requirements:
- OpenCV (cv2) installed
- Video files in the target folder
- Sufficient disk space for output videos
"""

import sys
from pathlib import Path

# Add the parent directory to the path to import media_gen modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import cv2

    from media_gen.utils.video_utils import concatenate_videos_from_folder
except ImportError as e:
    if "cv2" in str(e):
        print("❌ OpenCV package not installed. Please install it with:")
        print("   pip install opencv-python")
        sys.exit(1)
    else:
        raise


def main():
    """Concatenate videos from the specified folder."""
    print("🎬 Video Concatenation Integration Test")
    print("=" * 60)

    # Target folder path
    folder_path = "~/Downloads/video_regen_1754563125/generated_videos"
    output_path = "~/Downloads/concatenated_videos.mp4"

    # Expand user path
    folder = Path(folder_path).expanduser()
    output_file = Path(output_path).expanduser()

    print(f"📁 Source folder: {folder.absolute()}")
    print(f"📁 Output file: {output_file.absolute()}")
    print()

    # Check if folder exists
    if not folder.exists():
        print(f"❌ Error: Folder not found: {folder.absolute()}")
        print("Please ensure the folder exists and contains video files.")
        return

    # List video files in folder
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    video_files = []
    for ext in video_extensions:
        video_files.extend(folder.glob(f"*{ext}"))
        video_files.extend(folder.glob(f"*{ext.upper()}"))

    if not video_files:
        print(f"❌ Error: No video files found in {folder.absolute()}")
        print("Supported formats: .mp4, .avi, .mov, .mkv, .webm")
        return

    # Sort files alphabetically
    video_files = sorted(video_files, key=lambda x: x.name.lower())

    print(f"✅ Found {len(video_files)} video files:")
    for i, video_file in enumerate(video_files, 1):
        file_size = video_file.stat().st_size
        print(f"  {i:2d}. {video_file.name} ({file_size:,} bytes)")
    print()

    # Create output directory if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Get the resolution of the first video to maintain original size
    first_video = video_files[0]
    first_cap = cv2.VideoCapture(str(first_video))
    original_width = int(first_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(first_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = first_cap.get(cv2.CAP_PROP_FPS)
    first_cap.release()

    print("🔄 Starting video concatenation...")
    print("   - Sorting: Alphabetical by filename")
    print(f"   - Target FPS: {original_fps:.1f} (from first video)")
    print(f"   - Target resolution: {original_width}x{original_height} (from first video)")
    print()

    try:
        # Perform concatenation
        result = concatenate_videos_from_folder(
            folder_path=str(folder),
            output_path=str(output_file),
            sort_by_name=True,  # Sort alphabetically
            target_fps=original_fps,  # Use first video's FPS
            target_resolution=(original_width, original_height)  # Use first video's resolution
        )

        # Display results
        if result.success:
            print("✅ Concatenation completed successfully!")
            print()
            print("📊 Output Video Properties:")
            print(f"   📁 File: {result.output_path}")
            print(f"   ⏱️  Duration: {result.total_duration:.2f} seconds")
            print(f"   🎞️  Frames: {result.frame_count:,}")
            print(f"   📐 Resolution: {result.width}x{result.height}")
            print(f"   🎬 FPS: {result.fps}")
            print()

            # Check if output file exists and show file size
            if Path(result.output_path).exists():
                file_size = Path(result.output_path).stat().st_size
                print(f"📏 Output file size: {file_size:,} bytes ({file_size / 1024 / 1024:.1f} MB)")
            else:
                print("⚠️  Warning: Output file not found after concatenation")

        else:
            print("❌ Concatenation failed!")
            print(f"Error: {result.error_message}")
            return

    except Exception as e:
        print(f"❌ Error during concatenation: {e}")
        return

    print("\n✅ Integration test completed successfully!")
    print(f"🎬 Your concatenated video is ready: {output_file.absolute()}")


if __name__ == "__main__":
    main()
