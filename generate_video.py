#!/usr/bin/env python3
"""
Stable Video Diffusion を使用した動画生成CLI
M4 Mac (MPS) 向け省メモリ設定
業務と並行利用を想定（メモリ使用量: 約8-10GB）
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
from PIL import Image


def get_device():
    """利用可能なデバイスを取得"""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_pipeline(device: str, low_memory: bool = True):
    """パイプラインを読み込む（省メモリモード対応）"""
    model_id = "stabilityai/stable-video-diffusion-img2vid"

    print(f"モデルを読み込み中: {model_id}")
    print(f"デバイス: {device}")
    print(f"省メモリモード: {'有効' if low_memory else '無効'}")

    # MPS では float32 を使用
    if device == "mps":
        dtype = torch.float32
    elif device == "cuda":
        dtype = torch.float16
    else:
        dtype = torch.float32

    pipe = StableVideoDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        variant="fp16" if device == "cuda" else None,
    )

    if low_memory:
        # 省メモリ設定（業務と並行利用向け）
        print("省メモリ最適化を適用中...")

        if device == "mps":
            # MPSの場合: sequential_cpu_offloadは非対応なので手動で管理
            # まずCPUに配置してからMPSに移動
            pipe = pipe.to("cpu")
            # Attention Slicingのみ有効化
            pipe.enable_attention_slicing("max")
            # 実行時にMPSに移動
            pipe = pipe.to(device)
        else:
            # CUDA/CPUの場合: Sequential CPU Offload
            pipe.enable_sequential_cpu_offload()
            pipe.enable_attention_slicing("max")

        print("省メモリ最適化完了")
    else:
        pipe = pipe.to(device)
        if device == "mps":
            pipe.enable_attention_slicing("max")

    print("モデル読み込み完了")
    return pipe


def prepare_image(image_path: str, width: int, height: int) -> Image.Image:
    """入力画像を準備（リサイズ・正規化）"""
    print(f"画像を読み込み中: {image_path}")

    image = load_image(image_path)

    # SVDは特定のサイズを要求
    # 8の倍数に調整
    width = (width // 8) * 8
    height = (height // 8) * 8

    # リサイズ
    image = image.resize((width, height), Image.Resampling.LANCZOS)

    print(f"  リサイズ: {width}x{height}")
    return image


def generate_video(
    pipe,
    image: Image.Image,
    num_frames: int,
    fps: int,
    motion_bucket_id: int,
    noise_aug_strength: float,
    num_inference_steps: int,
    seed: int | None,
):
    """動画を生成"""
    # シード設定
    if seed is None:
        seed = torch.randint(0, 2**32, (1,)).item()

    generator = torch.Generator(device="cpu").manual_seed(seed)

    print(f"\n動画生成中...")
    print(f"  フレーム数: {num_frames}")
    print(f"  FPS: {fps}")
    print(f"  動きの強さ: {motion_bucket_id}")
    print(f"  ステップ数: {num_inference_steps}")
    print(f"  シード: {seed}")
    print(f"\n※ 省メモリモードのため時間がかかります。しばらくお待ちください...")

    frames = pipe(
        image,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        motion_bucket_id=motion_bucket_id,
        noise_aug_strength=noise_aug_strength,
        generator=generator,
        decode_chunk_size=2,  # メモリ節約: 一度にデコードするフレーム数を制限
    ).frames[0]

    return frames, seed


def save_video(frames, output_dir: Path, seed: int, fps: int):
    """動画を保存"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{seed}.mp4"
    filepath = output_dir / filename

    export_to_video(frames, str(filepath), fps=fps)
    print(f"\n保存完了: {filepath}")
    return filepath


def main():
    parser = argparse.ArgumentParser(
        description="Stable Video Diffusion 動画生成CLI（省メモリ版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 画像から動画を生成
  python generate_video.py outputs/20241222_123456_12345.png

  # フレーム数と動きの強さを指定
  python generate_video.py image.png --frames 14 --motion 100

  # 高品質モード（メモリに余裕がある場合）
  python generate_video.py image.png --no-low-memory --steps 25
        """,
    )

    parser.add_argument(
        "image",
        type=str,
        help="入力画像のパス（PNG/JPG）",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="出力動画の幅 (default: 512)",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=320,
        help="出力動画の高さ (default: 320)",
    )

    parser.add_argument(
        "--frames",
        type=int,
        default=14,
        help="生成するフレーム数。14-25推奨 (default: 14)",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=6,
        help="出力動画のFPS (default: 6)",
    )

    parser.add_argument(
        "--motion",
        type=int,
        default=127,
        help="動きの強さ (1-255)。大きいほど動きが激しい (default: 127)",
    )

    parser.add_argument(
        "--noise",
        type=float,
        default=0.02,
        help="ノイズ強度 (default: 0.02)",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="推論ステップ数 (default: 20)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="シード値。指定しない場合はランダム",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="outputs",
        help="出力ディレクトリ (default: outputs)",
    )

    parser.add_argument(
        "--no-low-memory",
        action="store_true",
        help="省メモリモードを無効化（メモリに余裕がある場合）",
    )

    args = parser.parse_args()

    # 入力画像の確認
    if not Path(args.image).exists():
        print(f"エラー: 画像が見つかりません: {args.image}")
        return

    # 出力ディレクトリ確認
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # デバイス取得
    device = get_device()

    # パイプライン読み込み
    low_memory = not args.no_low_memory
    pipe = load_pipeline(device, low_memory=low_memory)

    # 画像準備
    image = prepare_image(args.image, args.width, args.height)

    # 動画生成
    frames, seed = generate_video(
        pipe=pipe,
        image=image,
        num_frames=args.frames,
        fps=args.fps,
        motion_bucket_id=args.motion,
        noise_aug_strength=args.noise,
        num_inference_steps=args.steps,
        seed=args.seed,
    )

    # 保存
    save_video(frames, output_dir, seed, args.fps)

    # 動画の長さを表示
    duration = args.frames / args.fps
    print(f"動画の長さ: {duration:.1f}秒")


if __name__ == "__main__":
    main()
