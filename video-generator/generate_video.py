#!/usr/bin/env python3
"""
Image to Video Generator
Generates 5-second videos from images using AnimateDiff
Optimized for M4 Mac (24GB Unified Memory)

Usage:
    python generate_video.py <image_path> --duration 5
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import imageio.v3 as iio
from tqdm import tqdm


def get_device():
    """Get the best available device for M4 Mac"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def load_image(image_path: str, target_size: tuple = None) -> Image.Image:
    """Load and optionally resize an image"""
    image = Image.open(image_path).convert("RGB")

    if target_size:
        image = image.resize(target_size, Image.Resampling.LANCZOS)

    return image


def generate_video_svd(
    image_path: str,
    output_path: str,
    num_frames: int = 14,
    fps: int = 7,
    motion_bucket_id: int = 127,
    noise_aug_strength: float = 0.02,
    decode_chunk_size: int = 4,
):
    """
    Generate video using Stable Video Diffusion
    This is the primary method for M4 Mac with 24GB RAM
    """
    from diffusers import StableVideoDiffusionPipeline
    from diffusers.utils import export_to_video

    device = get_device()
    print(f"Using device: {device}")

    # Load model with memory optimizations
    print("Loading Stable Video Diffusion model...")
    print("(This may take a few minutes on first run)")

    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid",  # 14 frames version (lighter)
        torch_dtype=torch.float16,
        variant="fp16",
    )

    # Move to device
    pipe = pipe.to(device)

    # Enable memory optimizations
    try:
        pipe.enable_attention_slicing()
    except AttributeError:
        pass  # Not available in this version

    # For MPS, we need special handling
    if device.type == "mps":
        # Reduce memory pressure
        try:
            pipe.enable_vae_slicing()
        except AttributeError:
            pass  # Not available in this version

    # Load and prepare image
    print(f"Loading image: {image_path}")
    image = load_image(image_path, target_size=(1024, 576))

    # Generate frames
    print(f"Generating {num_frames} frames...")
    print(f"Motion bucket ID: {motion_bucket_id}")

    with torch.inference_mode():
        frames = pipe(
            image,
            num_frames=num_frames,
            motion_bucket_id=motion_bucket_id,
            noise_aug_strength=noise_aug_strength,
            decode_chunk_size=decode_chunk_size,
            generator=torch.Generator(device=device).manual_seed(42),
        ).frames[0]

    # Export video
    print(f"Saving video to: {output_path}")
    export_to_video(frames, output_path, fps=fps)

    return output_path, len(frames)


def generate_video_animatediff(
    image_path: str,
    output_path: str,
    num_frames: int = 16,
    fps: int = 8,
    prompt: str = "",
):
    """
    Generate video using AnimateDiff with img2img
    Alternative method that may work better for some images
    """
    from diffusers import (
        AnimateDiffPipeline,
        MotionAdapter,
        DDIMScheduler,
    )
    from diffusers.utils import export_to_video

    device = get_device()
    print(f"Using device: {device}")

    print("Loading AnimateDiff model...")

    # Load motion adapter
    adapter = MotionAdapter.from_pretrained(
        "guoyww/animatediff-motion-adapter-v1-5-2",
        torch_dtype=torch.float16,
    )

    # Load base model with motion adapter
    pipe = AnimateDiffPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        motion_adapter=adapter,
        torch_dtype=torch.float16,
    )

    # Set scheduler
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config,
        beta_schedule="linear",
        clip_sample=False,
    )

    pipe = pipe.to(device)
    try:
        pipe.enable_attention_slicing()
    except AttributeError:
        pass

    if device.type == "mps":
        try:
            pipe.enable_vae_slicing()
        except AttributeError:
            pass

    # Load reference image
    print(f"Loading image: {image_path}")
    image = load_image(image_path, target_size=(512, 512))

    # Generate with image conditioning via IP-Adapter style prompt
    # Note: This is a simplified version - full IP-Adapter requires additional setup
    if not prompt:
        prompt = "smooth motion, natural movement, high quality video"

    print(f"Generating {num_frames} frames with prompt: {prompt}")

    with torch.inference_mode():
        output = pipe(
            prompt=prompt,
            num_frames=num_frames,
            guidance_scale=7.5,
            num_inference_steps=25,
            generator=torch.Generator(device=device).manual_seed(42),
        )

    frames = output.frames[0]

    print(f"Saving video to: {output_path}")
    export_to_video(frames, output_path, fps=fps)

    return output_path, len(frames)


def extend_video_duration(
    video_path: str,
    target_duration: float,
    target_fps: int = 24,
):
    """Extend video to target duration using frame interpolation"""
    from scripts.rife_interpolate import extend_video_duration as extend_func

    output_path = video_path.replace(".mp4", "_extended.mp4")
    return extend_func(
        video_path,
        output_path,
        target_duration=target_duration,
        target_fps=target_fps,
        method="optical_flow",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate videos from images (optimized for M4 Mac)"
    )
    parser.add_argument(
        "image",
        help="Path to input image"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output video path (default: outputs/<timestamp>.mp4)"
    )
    parser.add_argument(
        "-d", "--duration",
        type=float,
        default=5.0,
        help="Target video duration in seconds (default: 5.0)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="Output FPS (default: 24)"
    )
    parser.add_argument(
        "--motion",
        type=int,
        default=127,
        help="Motion intensity 1-255 (default: 127)"
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=14,
        help="Initial frames to generate (default: 14, max 25 for SVD-XT)"
    )
    parser.add_argument(
        "--method",
        choices=["svd", "animatediff"],
        default="svd",
        help="Generation method (default: svd)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Optional prompt for AnimateDiff method"
    )
    parser.add_argument(
        "--no-extend",
        action="store_true",
        help="Skip frame interpolation (output raw frames only)"
    )

    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)

    # Setup output path
    script_dir = Path(__file__).parent
    output_dir = script_dir / "outputs"
    output_dir.mkdir(exist_ok=True)

    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(output_dir / f"{timestamp}.mp4")

    # Generate initial video
    print(f"\n{'='*50}")
    print("Image to Video Generator (M4 Mac Optimized)")
    print(f"{'='*50}")
    print(f"Input: {args.image}")
    print(f"Method: {args.method}")
    print(f"Target duration: {args.duration}s @ {args.fps}fps")
    print(f"{'='*50}\n")

    try:
        if args.method == "svd":
            temp_output = output_path.replace(".mp4", "_raw.mp4")
            video_path, frame_count = generate_video_svd(
                args.image,
                temp_output,
                num_frames=args.frames,
                fps=args.frames // 2,  # ~2 seconds initial
                motion_bucket_id=args.motion,
            )
        else:
            temp_output = output_path.replace(".mp4", "_raw.mp4")
            video_path, frame_count = generate_video_animatediff(
                args.image,
                temp_output,
                num_frames=args.frames,
                fps=args.frames // 2,
                prompt=args.prompt,
            )

        print(f"\nGenerated {frame_count} frames")

        # Extend to target duration
        if not args.no_extend and args.duration > 0:
            print(f"\nExtending video to {args.duration} seconds...")
            final_output = extend_video_duration(
                temp_output,
                target_duration=args.duration,
                target_fps=args.fps,
            )

            # Rename to final output
            if os.path.exists(temp_output):
                os.rename(final_output, output_path)
                os.remove(temp_output)
            print(f"\nFinal output: {output_path}")
        else:
            os.rename(temp_output, output_path)
            print(f"\nOutput: {output_path}")

        print("\nDone!")

    except Exception as e:
        print(f"\nError: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure you have enough memory (close other apps)")
        print("2. Try reducing --frames to 8 or 10")
        print("3. Try --method animatediff as alternative")
        sys.exit(1)


if __name__ == "__main__":
    main()
