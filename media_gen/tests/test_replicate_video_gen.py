#!/usr/bin/env python3
"""
Test script for Replicate video generation tool using WAN 2.2 i2v fast model.

This script demonstrates how to use the ReplicateVideoGen tool to generate
videos from images and text prompts using the WAN 2.2 i2v fast model.

Usage:
    python tests/test_replicate_video_gen.py

Requirements:
- Replicate API token set in environment variables
- Test image file: tests/test_image.png
"""

import base64
import os
import sys
from pathlib import Path

import pytest

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from media_gen.tools.replicate_video_gen import ReplicateVideoGen


def test_replicate_video_generation():
    """Test the Replicate video generation functionality."""
    # Check for Replicate API token
    if not os.getenv("REPLICATE_API_TOKEN"):
        pytest.skip("REPLICATE_API_TOKEN not found in environment variables")

    # Path to test image
    test_image_path = Path(__file__).parent / "test_image.png"

    if not test_image_path.exists():
        pytest.skip("Test image not found at: {test_image_path}")

    # Initialize the video generation tool
    video_gen = ReplicateVideoGen()

    # Test parameters
    test_prompt = (
        "Close-up shot of an elderly sailor wearing a yellow raincoat, "
        "seated on the deck of a catamaran, slowly puffing on a pipe. "
        "His cat lies quietly beside him with eyes closed, enjoying the "
        "calm. The warm glow of the setting sun bathes the scene, with "
        "gentle waves lapping against the hull and a few seabirds "
        "circling slowly above. The camera slowly pushes in, capturing "
        "this peaceful and harmonious moment."
    )

    # Generate video
    result = video_gen.run(
        {
            "image": str(test_image_path),
            "prompt": test_prompt,
            "output_folder": "~/Downloads/polymind_video_generation",
            "output_format": "mp4",
        }
    )

    # Assertions
    assert result["video_path"], "Video generation should return a path"
    assert result["generation_info"], "Video generation should return info"

    # Check if file exists
    video_path = Path(result["video_path"])
    assert video_path.exists(), f"Video file should exist at {video_path}"


def test_with_data_uri():
    """Test video generation using data URI for image input."""
    # Path to test image
    test_image_path = Path(__file__).parent / "test_image.png"

    if not test_image_path.exists():
        pytest.skip("Test image not found for data URI test")

    # Convert image to data URI
    with open(test_image_path, "rb") as file:
        data = base64.b64encode(file.read()).decode("utf-8")
        data_uri = f"data:application/octet-stream;base64,{data}"

    # Initialize the video generation tool
    video_gen = ReplicateVideoGen()

    # Test parameters
    test_prompt = "A serene landscape with gentle movement and natural lighting"

    result = video_gen.run(
        {
            "image": data_uri,
            "prompt": test_prompt,
            "output_folder": "~/Downloads/polymind_video_generation",
            "output_format": "mp4",
        }
    )

    # Assertions
    assert result["video_path"], "Video generation should return a path"
    assert result["generation_info"], "Video generation should return info"



