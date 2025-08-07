"""
Utility functions for media generation and processing.

This package contains helper functions for video processing, image manipulation,
and other media-related utilities.
"""

from .video_utils import VideoScreenshotExtractor, extract_screenshots

__all__ = ['extract_screenshots', 'VideoScreenshotExtractor']
