"""
Simple video screenshot utility.

Extract screenshots from video files at regular time intervals or key frames.
"""

import logging
import os
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _expand_path(path: str) -> Path:
    """
    Expand a path, handling tilde (~) for home directory.

    Args:
        path: Path string that may contain tilde

    Returns:
        Expanded Path object
    """
    return Path(os.path.expanduser(path))


class ExtractionMode(Enum):
    """Extraction modes for video frames."""

    INTERVAL = "interval"
    KEYFRAME = "keyframe"


@dataclass
class ScreenshotInfo:
    """Information about an extracted screenshot."""

    frame_number: int
    timestamp: float  # seconds
    timestamp_str: str  # formatted as HH:MM:SS
    file_path: str


@dataclass
class KeyFrameInfo:
    """Information about an extracted key frame."""

    frame_number: int
    timestamp: float  # seconds
    timestamp_str: str  # formatted as HH:MM:SS
    file_path: str
    confidence: float  # key frame detection confidence (0-1)


class VideoExtractor:
    """
    Unified video frame extractor supporting both interval and keyframe modes.

    A single interface for extracting frames from videos using different strategies.
    """

    def __init__(self, video_path: str, mode: Union[ExtractionMode, str] = ExtractionMode.INTERVAL, **kwargs):
        """
        Initialize the video extractor.

        Args:
            video_path: Path to the video file
            mode: Extraction mode ('interval' or 'keyframe')
            **kwargs: Mode-specific parameters
        """
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Convert string to enum if needed
        if isinstance(mode, str):
            mode = ExtractionMode(mode.lower())
        self.mode = mode

        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0

        # Set mode-specific parameters
        self._set_mode_parameters(**kwargs)

        logger.info(
            f"Video loaded: {self.fps:.2f} FPS, "
            f"{self.frame_count} frames, "
            f"{self.duration:.2f}s duration, "
            f"mode: {self.mode.value}"
        )

    def _set_mode_parameters(self, **kwargs) -> None:
        """Set parameters based on extraction mode."""
        if self.mode == ExtractionMode.INTERVAL:
            self.interval_seconds = kwargs.get("interval_seconds", 2.0)
            self.start_time = kwargs.get("start_time", 0.0)
        elif self.mode == ExtractionMode.KEYFRAME:
            self.threshold = kwargs.get("threshold", 30.0)
            self.min_interval_frames = kwargs.get("min_interval_frames", 30)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - release video capture."""
        self.release()

    def release(self) -> None:
        """Release the video capture object."""
        if self.cap:
            self.cap.release()

    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds into HH:MM:SS string."""
        return str(timedelta(seconds=int(seconds)))

    def _calculate_frame_difference(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Calculate the difference between two frames.

        Args:
            frame1: First frame as numpy array
            frame2: Second frame as numpy array

        Returns:
            Difference score (higher = more different)
        """
        # Convert to grayscale for comparison
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Calculate absolute difference
        diff = cv2.absdiff(gray1, gray2)

        # Return mean difference
        return np.mean(diff)

    def extract(
        self, output_dir: Optional[str] = "~/Downloads", filename_prefix: Optional[str] = None
    ) -> List[Union[ScreenshotInfo, KeyFrameInfo]]:
        """
        Extract frames based on the configured mode.

        Args:
            output_dir: Directory to save frames (default: ~/Downloads)
            filename_prefix: Prefix for saved files (auto-generated if None)

        Returns:
            List of ScreenshotInfo or KeyFrameInfo objects
        """
        if filename_prefix is None:
            filename_prefix = self.mode.value

        if self.mode == ExtractionMode.INTERVAL:
            return self._extract_interval(output_dir, filename_prefix)
        elif self.mode == ExtractionMode.KEYFRAME:
            return self._extract_keyframes(output_dir, filename_prefix)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def _extract_interval(self, output_dir: str, filename_prefix: str) -> List[ScreenshotInfo]:
        """Extract frames at regular intervals."""
        if self.interval_seconds <= 0:
            raise ValueError("Interval must be positive")
        if self.start_time < 0:
            raise ValueError("Start time must be non-negative")

        screenshots = []
        frame_interval = int(self.fps * self.interval_seconds)
        start_frame = int(self.start_time * self.fps)

        # Handle output directory with tilde expansion
        output_path = _expand_path(output_dir) if output_dir else _expand_path("~/Downloads")
        output_path.mkdir(parents=True, exist_ok=True)

        for frame_num in range(start_frame, self.frame_count, frame_interval):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = self.cap.read()

            if not ret:
                break

            timestamp = frame_num / self.fps
            timestamp_str = self._format_timestamp(timestamp)

            screenshot_info = ScreenshotInfo(
                frame_number=frame_num, timestamp=timestamp, timestamp_str=timestamp_str, file_path=""
            )

            # Save screenshot
            timestamp_seconds = int(timestamp)
            filename = f"{filename_prefix}_{timestamp_seconds:06d}s_" f"{timestamp_str.replace(':', '-')}.jpg"
            file_path = output_path / filename
            cv2.imwrite(str(file_path), frame)
            screenshot_info.file_path = str(file_path)

            screenshots.append(screenshot_info)
            logger.debug(f"Extracted screenshot at {timestamp_str} " f"(frame {frame_num})")

        logger.info(f"Extracted {len(screenshots)} screenshots " f"at {self.interval_seconds}s intervals")
        return screenshots

    def _extract_keyframes(self, output_dir: str, filename_prefix: str) -> List[KeyFrameInfo]:
        """Extract key frames using scene change detection."""
        key_frames = []
        prev_frame = None
        last_key_frame = -self.min_interval_frames

        # Handle output directory with tilde expansion
        output_path = _expand_path(output_dir) if output_dir else _expand_path("~/Downloads")
        output_path.mkdir(parents=True, exist_ok=True)

        for frame_num in range(self.frame_count):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = self.cap.read()

            if not ret:
                break

            # Skip if too close to last key frame
            if frame_num - last_key_frame < self.min_interval_frames:
                prev_frame = frame
                continue

            if prev_frame is not None:
                # Calculate frame difference
                diff_score = self._calculate_frame_difference(prev_frame, frame)

                # Check if this is a key frame
                if diff_score > self.threshold:
                    timestamp = frame_num / self.fps
                    timestamp_str = self._format_timestamp(timestamp)

                    # Calculate confidence based on difference score
                    confidence = min(1.0, diff_score / 100.0)

                    key_frame_info = KeyFrameInfo(
                        frame_number=frame_num,
                        timestamp=timestamp,
                        timestamp_str=timestamp_str,
                        file_path="",
                        confidence=confidence,
                    )

                    # Save key frame
                    timestamp_seconds = int(timestamp)
                    filename = f"{filename_prefix}_{timestamp_seconds:06d}s_" f"{timestamp_str.replace(':', '-')}.jpg"
                    file_path = output_path / filename
                    cv2.imwrite(str(file_path), frame)
                    key_frame_info.file_path = str(file_path)

                    key_frames.append(key_frame_info)
                    last_key_frame = frame_num

                    logger.debug(
                        f"Extracted key frame at {timestamp_str} " f"(frame {frame_num}, diff: {diff_score:.2f})"
                    )

            prev_frame = frame

        logger.info(f"Extracted {len(key_frames)} key frames " f"with threshold {self.threshold}")
        return key_frames


class KeyFrameExtractor:
    """
    Extract key frames from video files using scene change detection.

    Uses frame difference analysis to identify significant scene changes.
    """

    def __init__(self, video_path: str, threshold: float = 30.0):
        """
        Initialize the key frame extractor.

        Args:
            video_path: Path to the video file
            threshold: Threshold for scene change detection (default: 30.0)
        """
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        self.threshold = threshold

        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0

        logger.info(
            f"Video loaded: {self.fps:.2f} FPS, " f"{self.frame_count} frames, " f"{self.duration:.2f}s duration"
        )

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - release video capture."""
        self.release()

    def release(self) -> None:
        """Release the video capture object."""
        if self.cap:
            self.cap.release()

    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds into HH:MM:SS string."""
        return str(timedelta(seconds=int(seconds)))

    def _calculate_frame_difference(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Calculate the difference between two frames.

        Args:
            frame1: First frame as numpy array
            frame2: Second frame as numpy array

        Returns:
            Difference score (higher = more different)
        """
        # Convert to grayscale for comparison
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Calculate absolute difference
        diff = cv2.absdiff(gray1, gray2)

        # Return mean difference
        return np.mean(diff)

    def extract_key_frames(
        self,
        output_dir: Optional[str] = "~/Downloads",
        filename_prefix: str = "keyframe",
        min_interval_frames: int = 30,
    ) -> List[KeyFrameInfo]:
        """
        Extract key frames using scene change detection.

        Args:
            output_dir: Directory to save key frames (default: ~/Downloads)
            filename_prefix: Prefix for saved files
            min_interval_frames: Minimum frames between key frames

        Returns:
            List of KeyFrameInfo objects
        """
        key_frames = []
        prev_frame = None
        last_key_frame = -min_interval_frames

        # Handle output directory with tilde expansion
        output_path = _expand_path(output_dir) if output_dir else _expand_path("~/Downloads")
        output_path.mkdir(parents=True, exist_ok=True)

        for frame_num in range(self.frame_count):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = self.cap.read()

            if not ret:
                break

            # Skip if too close to last key frame
            if frame_num - last_key_frame < min_interval_frames:
                prev_frame = frame
                continue

            if prev_frame is not None:
                # Calculate frame difference
                diff_score = self._calculate_frame_difference(prev_frame, frame)

                # Check if this is a key frame
                if diff_score > self.threshold:
                    timestamp = frame_num / self.fps
                    timestamp_str = self._format_timestamp(timestamp)

                    # Calculate confidence based on difference score
                    confidence = min(1.0, diff_score / 100.0)

                    key_frame_info = KeyFrameInfo(
                        frame_number=frame_num,
                        timestamp=timestamp,
                        timestamp_str=timestamp_str,
                        file_path="",
                        confidence=confidence,
                    )

                    # Save key frame
                    timestamp_seconds = int(timestamp)
                    filename = f"{filename_prefix}_{timestamp_seconds:06d}s_" f"{timestamp_str.replace(':', '-')}.jpg"
                    file_path = output_path / filename
                    cv2.imwrite(str(file_path), frame)
                    key_frame_info.file_path = str(file_path)

                    key_frames.append(key_frame_info)
                    last_key_frame = frame_num

                    logger.debug(
                        f"Extracted key frame at {timestamp_str} " f"(frame {frame_num}, diff: {diff_score:.2f})"
                    )

            prev_frame = frame

        logger.info(f"Extracted {len(key_frames)} key frames " f"with threshold {self.threshold}")
        return key_frames


class VideoScreenshotExtractor:
    """
    Extract screenshots from video files at regular intervals.

    Simple utility to take screenshots every X seconds from a video file.
    """

    def __init__(self, video_path: str):
        """
        Initialize the screenshot extractor.

        Args:
            video_path: Path to the video file
        """
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0

        logger.info(
            f"Video loaded: {self.fps:.2f} FPS, " f"{self.frame_count} frames, " f"{self.duration:.2f}s duration"
        )

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - release video capture."""
        self.release()

    def release(self) -> None:
        """Release the video capture object."""
        if self.cap:
            self.cap.release()

    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds into HH:MM:SS string."""
        return str(timedelta(seconds=int(seconds)))

    def extract_screenshots(
        self,
        interval_seconds: float = 2.0,
        start_time: float = 0.0,
        output_dir: Optional[str] = "~/Downloads",
        filename_prefix: str = "screenshot",
    ) -> List[ScreenshotInfo]:
        """
        Extract screenshots at regular time intervals.

        Args:
            interval_seconds: Time interval between screenshots (default: 2.0)
            start_time: Start time in seconds (default: 0.0)
            output_dir: Directory to save screenshots (default: ~/Downloads)
            filename_prefix: Prefix for saved files

        Returns:
            List of ScreenshotInfo objects
        """
        if interval_seconds <= 0:
            raise ValueError("Interval must be positive")
        if start_time < 0:
            raise ValueError("Start time must be non-negative")

        screenshots = []
        frame_interval = int(self.fps * interval_seconds)
        start_frame = int(start_time * self.fps)

        # Handle output directory with tilde expansion
        output_path = _expand_path(output_dir) if output_dir else _expand_path("~/Downloads")
        output_path.mkdir(parents=True, exist_ok=True)

        for frame_num in range(start_frame, self.frame_count, frame_interval):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = self.cap.read()

            if not ret:
                break

            timestamp = frame_num / self.fps
            timestamp_str = self._format_timestamp(timestamp)

            screenshot_info = ScreenshotInfo(
                frame_number=frame_num, timestamp=timestamp, timestamp_str=timestamp_str, file_path=""
            )

            # Always save screenshots since we have a default output directory
            # Use timestamp-based filename for chronological sorting
            timestamp_seconds = int(timestamp)
            filename = f"{filename_prefix}_{timestamp_seconds:06d}s_" f"{timestamp_str.replace(':', '-')}.jpg"
            file_path = output_path / filename
            cv2.imwrite(str(file_path), frame)
            screenshot_info.file_path = str(file_path)

            screenshots.append(screenshot_info)
            logger.debug(f"Extracted screenshot at {timestamp_str} " f"(frame {frame_num})")

        logger.info(f"Extracted {len(screenshots)} screenshots " f"at {interval_seconds}s intervals")
        return screenshots


def extract_screenshots(
    video_path: str,
    interval_seconds: float = 2.0,
    start_time: float = 0.0,
    output_dir: Optional[str] = "~/Downloads",
    filename_prefix: str = "screenshot",
) -> List[ScreenshotInfo]:
    """
    Convenience function to extract screenshots from a video file.

    Args:
        video_path: Path to the video file
        interval_seconds: Time interval between screenshots (default: 2.0)
        start_time: Start time in seconds (default: 0.0)
        output_dir: Directory to save screenshots (default: ~/Downloads)
        filename_prefix: Prefix for saved files

    Returns:
        List of ScreenshotInfo objects
    """
    with VideoScreenshotExtractor(video_path) as extractor:
        return extractor.extract_screenshots(
            interval_seconds=interval_seconds,
            start_time=start_time,
            output_dir=output_dir,
            filename_prefix=filename_prefix,
        )


def extract_key_frames(
    video_path: str,
    threshold: float = 30.0,
    output_dir: Optional[str] = "~/Downloads",
    filename_prefix: str = "keyframe",
    min_interval_frames: int = 30,
) -> List[KeyFrameInfo]:
    """
    Convenience function to extract key frames from a video file.

    Args:
        video_path: Path to the video file
        threshold: Threshold for scene change detection (default: 30.0)
        output_dir: Directory to save key frames (default: ~/Downloads)
        filename_prefix: Prefix for saved files
        min_interval_frames: Minimum frames between key frames

    Returns:
        List of KeyFrameInfo objects
    """
    with KeyFrameExtractor(video_path, threshold) as extractor:
        return extractor.extract_key_frames(
            output_dir=output_dir, filename_prefix=filename_prefix, min_interval_frames=min_interval_frames
        )


def extract_frames(
    video_path: str,
    mode: Union[ExtractionMode, str] = ExtractionMode.INTERVAL,
    output_dir: Optional[str] = "~/Downloads",
    filename_prefix: Optional[str] = None,
    **kwargs,
) -> List[Union[ScreenshotInfo, KeyFrameInfo]]:
    """
    Unified convenience function to extract frames from a video file.

    Args:
        video_path: Path to the video file
        mode: Extraction mode ('interval' or 'keyframe')
        output_dir: Directory to save frames (default: ~/Downloads)
        filename_prefix: Prefix for saved files (auto-generated if None)
        **kwargs: Mode-specific parameters

    Returns:
        List of ScreenshotInfo or KeyFrameInfo objects
    """
    with VideoExtractor(video_path, mode, **kwargs) as extractor:
        return extractor.extract(output_dir, filename_prefix)


@dataclass
class ConcatenationInfo:
    """Information about video concatenation process."""

    input_videos: List[str]
    output_path: str
    total_duration: float
    frame_count: int
    width: int
    height: int
    fps: float
    success: bool
    error_message: Optional[str] = None


def concatenate_videos_from_folder(
    folder_path: str,
    output_path: str,
    video_extensions: Optional[List[str]] = None,
    sort_by_name: bool = True,
    target_fps: Optional[float] = None,
    target_resolution: Optional[tuple[int, int]] = None,
) -> ConcatenationInfo:
    """
    Concatenate videos from a folder in alphabetical order.

    Args:
        folder_path: Path to folder containing video files
        output_path: Path for the concatenated output video
        video_extensions: List of video file extensions to include (default: ['.mp4', '.avi', '.mov', '.mkv'])
        sort_by_name: Whether to sort files alphabetically by name (default: True)
        target_fps: Target FPS for output video (uses first video's FPS if None)
        target_resolution: Target resolution (width, height) for output video (uses first video's resolution if None)

    Returns:
        ConcatenationInfo object with process details
    """
    if video_extensions is None:
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']

    folder = Path(folder_path)
    if not folder.exists():
        return ConcatenationInfo(
            input_videos=[],
            output_path=output_path,
            total_duration=0.0,
            frame_count=0,
            width=0,
            height=0,
            fps=0.0,
            success=False,
            error_message=f"Folder not found: {folder_path}"
        )

    # Find all video files
    video_files = []
    for ext in video_extensions:
        video_files.extend(folder.glob(f"*{ext}"))
        video_files.extend(folder.glob(f"*{ext.upper()}"))

    if not video_files:
        return ConcatenationInfo(
            input_videos=[],
            output_path=output_path,
            total_duration=0.0,
            frame_count=0,
            width=0,
            height=0,
            fps=0.0,
            success=False,
            error_message=f"No video files found in {folder_path}"
        )

    # Sort files alphabetically if requested
    if sort_by_name:
        video_files = sorted(video_files, key=lambda x: x.name.lower())

    video_paths = [str(f) for f in video_files]
    logger.info(f"Found {len(video_paths)} video files to concatenate")

    try:
        # Get properties from first video
        first_cap = cv2.VideoCapture(video_paths[0])
        if not first_cap.isOpened():
            raise ValueError(f"Could not open first video: {video_paths[0]}")

        # Use first video's properties as defaults
        default_fps = first_cap.get(cv2.CAP_PROP_FPS)
        default_width = int(first_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        default_height = int(first_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        first_cap.release()

        # Use provided targets or defaults
        fps = target_fps if target_fps is not None else default_fps
        width = target_resolution[0] if target_resolution else default_width
        height = target_resolution[1] if target_resolution else default_height

        # Create output directory if needed
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not out.isOpened():
            raise ValueError(f"Could not create output video writer: {output_path}")

        total_frames = 0
        total_duration = 0.0

        # Process each video
        for video_path in video_paths:
            logger.info(f"Processing video: {Path(video_path).name}")

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.warning(f"Could not open video: {video_path}")
                continue

            video_fps = cap.get(cv2.CAP_PROP_FPS)
            video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_duration = video_frame_count / video_fps if video_fps > 0 else 0

            logger.info(f"  Duration: {video_duration:.2f}s, FPS: {video_fps:.2f}, Frames: {video_frame_count}")

            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Resize frame if target resolution is different
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height))

                # Handle frame rate differences
                if video_fps != fps:
                    # Simple frame dropping/duplication for FPS conversion
                    if video_fps > fps:
                        # Drop frames
                        if frame_idx % int(video_fps / fps) == 0:
                            out.write(frame)
                    else:
                        # Duplicate frames
                        repeat_count = int(fps / video_fps)
                        for _ in range(repeat_count):
                            out.write(frame)
                else:
                    out.write(frame)

                frame_idx += 1
                total_frames += 1

            cap.release()
            total_duration += video_duration

        out.release()

        # Calculate final properties
        final_duration = total_frames / fps if fps > 0 else 0

        logger.info(f"Concatenation completed: {len(video_paths)} videos, "
                    f"{total_frames} frames, {final_duration:.2f}s duration")

        return ConcatenationInfo(
            input_videos=video_paths,
            output_path=output_path,
            total_duration=final_duration,
            frame_count=total_frames,
            width=width,
            height=height,
            fps=fps,
            success=True
        )

    except Exception as e:
        logger.error(f"Error during video concatenation: {str(e)}")
        return ConcatenationInfo(
            input_videos=video_paths,
            output_path=output_path,
            total_duration=0.0,
            frame_count=0,
            width=0,
            height=0,
            fps=0.0,
            success=False,
            error_message=str(e)
        )


def concatenate_videos(
    video_paths: List[str],
    output_path: str,
    target_fps: Optional[float] = None,
    target_resolution: Optional[tuple[int, int]] = None,
) -> ConcatenationInfo:
    """
    Concatenate a list of video files.

    Args:
        video_paths: List of video file paths to concatenate
        output_path: Path for the concatenated output video
        target_fps: Target FPS for output video (uses first video's FPS if None)
        target_resolution: Target resolution (width, height) for output video (uses first video's resolution if None)

    Returns:
        ConcatenationInfo object with process details
    """
    if not video_paths:
        return ConcatenationInfo(
            input_videos=[],
            output_path=output_path,
            total_duration=0.0,
            frame_count=0,
            width=0,
            height=0,
            fps=0.0,
            success=False,
            error_message="No video paths provided"
        )

    # Create a temporary folder structure to use the folder-based function
    import shutil
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        # Copy videos to temp directory with numbered names to preserve order
        temp_video_paths = []
        for i, video_path in enumerate(video_paths):
            video_file = Path(video_path)
            if not video_file.exists():
                logger.warning(f"Video file not found: {video_path}")
                continue

            # Create numbered filename to preserve order
            temp_name = f"{i:04d}_{video_file.name}"
            temp_path = Path(temp_dir) / temp_name
            shutil.copy2(video_path, temp_path)
            temp_video_paths.append(str(temp_path))

        if not temp_video_paths:
            return ConcatenationInfo(
                input_videos=video_paths,
                output_path=output_path,
                total_duration=0.0,
                frame_count=0,
                width=0,
                height=0,
                fps=0.0,
                success=False,
                error_message="No valid video files found"
            )

        # Use the folder-based concatenation
        result = concatenate_videos_from_folder(
            folder_path=temp_dir,
            output_path=output_path,
            sort_by_name=True,  # This will sort by the numbered filenames
            target_fps=target_fps,
            target_resolution=target_resolution
        )

        # Update the input_videos to show original paths
        result.input_videos = video_paths
        return result
