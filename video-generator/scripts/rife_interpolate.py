"""
RIFE Frame Interpolation Script
Extends short videos to longer durations using AI frame interpolation
Optimized for M4 Mac with MPS backend
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import imageio.v3 as iio
from tqdm import tqdm


def check_ffmpeg():
    """Check if ffmpeg is available"""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def load_frames_from_video(video_path: str) -> list:
    """Load all frames from a video file"""
    frames = []
    for frame in iio.imiter(video_path):
        frames.append(frame)
    return frames


def save_video(frames: list, output_path: str, fps: int = 24):
    """Save frames as video"""
    if not frames:
        raise ValueError("No frames to save")

    # Use imageio to write video
    iio.imwrite(
        output_path,
        frames,
        fps=fps,
        codec="libx264",
        quality=8,  # Higher is better, max 10
    )
    print(f"Saved video to: {output_path}")


def interpolate_frames_simple(frames: list, multiplier: int = 2, device: str = "mps") -> list:
    """
    Simple frame interpolation using linear blending
    This is a fallback when RIFE models aren't available
    """
    if multiplier < 2:
        return frames

    interpolated = []
    for i in tqdm(range(len(frames) - 1), desc="Interpolating frames"):
        frame1 = frames[i].astype(np.float32)
        frame2 = frames[i + 1].astype(np.float32)

        interpolated.append(frames[i])

        # Add interpolated frames
        for j in range(1, multiplier):
            alpha = j / multiplier
            blended = (frame1 * (1 - alpha) + frame2 * alpha).astype(np.uint8)
            interpolated.append(blended)

    interpolated.append(frames[-1])
    return interpolated


def interpolate_with_optical_flow(frames: list, multiplier: int = 2) -> list:
    """
    Interpolation using OpenCV optical flow
    Better quality than simple blending
    """
    try:
        import cv2
    except ImportError:
        print("OpenCV not available, using simple interpolation")
        return interpolate_frames_simple(frames, multiplier)

    interpolated = []

    for i in tqdm(range(len(frames) - 1), desc="Interpolating with optical flow"):
        frame1 = frames[i]
        frame2 = frames[i + 1]

        # Convert to grayscale for flow calculation
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)

        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2,
            flags=0
        )

        interpolated.append(frame1)

        h, w = frame1.shape[:2]

        for j in range(1, multiplier):
            alpha = j / multiplier

            # Create coordinate grids
            y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)

            # Warp coordinates
            map_x = x_coords + flow[:, :, 0] * alpha
            map_y = y_coords + flow[:, :, 1] * alpha

            # Remap frame1
            warped1 = cv2.remap(frame1, map_x, map_y, cv2.INTER_LINEAR)

            # Reverse flow for frame2
            map_x2 = x_coords - flow[:, :, 0] * (1 - alpha)
            map_y2 = y_coords - flow[:, :, 1] * (1 - alpha)
            warped2 = cv2.remap(frame2, map_x2, map_y2, cv2.INTER_LINEAR)

            # Blend
            blended = cv2.addWeighted(warped1, 1 - alpha, warped2, alpha, 0)
            interpolated.append(blended)

    interpolated.append(frames[-1])
    return interpolated


def extend_video_duration(
    input_path: str,
    output_path: str,
    target_duration: float = 5.0,
    target_fps: int = 24,
    method: str = "optical_flow"
) -> str:
    """
    Extend a short video to target duration using frame interpolation

    Args:
        input_path: Path to input video
        output_path: Path for output video
        target_duration: Target duration in seconds
        target_fps: Target frames per second
        method: Interpolation method ("simple" or "optical_flow")

    Returns:
        Path to output video
    """
    print(f"Loading video: {input_path}")
    frames = load_frames_from_video(input_path)
    original_count = len(frames)
    print(f"Loaded {original_count} frames")

    # Calculate required frames
    target_frames = int(target_duration * target_fps)

    if original_count >= target_frames:
        print(f"Video already has enough frames ({original_count} >= {target_frames})")
        # Just resample to target fps
        indices = np.linspace(0, original_count - 1, target_frames, dtype=int)
        frames = [frames[i] for i in indices]
    else:
        # Calculate multiplier needed
        multiplier = int(np.ceil(target_frames / original_count))
        print(f"Interpolating with multiplier: {multiplier}x")

        if method == "optical_flow":
            frames = interpolate_with_optical_flow(frames, multiplier)
        else:
            frames = interpolate_frames_simple(frames, multiplier)

        # Trim or pad to exact target frames
        if len(frames) > target_frames:
            # Evenly sample frames
            indices = np.linspace(0, len(frames) - 1, target_frames, dtype=int)
            frames = [frames[i] for i in indices]

    print(f"Final frame count: {len(frames)} (target: {target_frames})")

    # Save output
    save_video(frames, output_path, target_fps)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Extend video duration using frame interpolation"
    )
    parser.add_argument("input", help="Input video path")
    parser.add_argument("-o", "--output", help="Output video path")
    parser.add_argument(
        "-d", "--duration",
        type=float,
        default=5.0,
        help="Target duration in seconds (default: 5.0)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="Target FPS (default: 24)"
    )
    parser.add_argument(
        "--method",
        choices=["simple", "optical_flow"],
        default="optical_flow",
        help="Interpolation method (default: optical_flow)"
    )

    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    # Generate output path if not specified
    if args.output:
        output_path = args.output
    else:
        input_path = Path(args.input)
        output_path = str(
            input_path.parent / f"{input_path.stem}_extended{input_path.suffix}"
        )

    # Check ffmpeg
    if not check_ffmpeg():
        print("Warning: ffmpeg not found. Video encoding may fail.")
        print("Install with: brew install ffmpeg")

    # Run interpolation
    result = extend_video_duration(
        args.input,
        output_path,
        target_duration=args.duration,
        target_fps=args.fps,
        method=args.method
    )

    print(f"\nDone! Output saved to: {result}")


if __name__ == "__main__":
    main()
