#!/usr/bin/env python3
"""
Realistic Vision v5.1 を使用した画像生成CLI
M4 Mac (MPS) 向けに最適化
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler


def get_device():
    """利用可能なデバイスを取得"""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_pipeline(device: str):
    """パイプラインを読み込む"""
    model_id = "SG161222/Realistic_Vision_V5.1_noVAE"

    print(f"モデルを読み込み中: {model_id}")
    print(f"デバイス: {device}")

    # MPS では float32 を使用（float16 は黒画像になる問題あり）
    if device == "mps":
        dtype = torch.float32
    elif device == "cuda":
        dtype = torch.float16
    else:
        dtype = torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )

    # DPM++ 2M Karras スケジューラーを使用（推奨設定）
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        use_karras_sigmas=True,
        algorithm_type="dpmsolver++",
        final_sigmas_type="sigma_min",
    )

    pipe = pipe.to(device)

    # メモリ最適化
    if device == "mps":
        # より小さなスライスサイズで分割（高解像度対応）
        pipe.enable_attention_slicing("max")
        # VAEも分割処理
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()

    print("モデル読み込み完了")
    return pipe


def generate_image(
    pipe,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    cfg_scale: float,
    seed: int | None,
    device: str,
):
    """画像を生成"""
    # シード設定
    if seed is None:
        seed = torch.randint(0, 2**32, (1,)).item()

    generator = torch.Generator(device="cpu").manual_seed(seed)

    print(f"\n生成中...")
    print(f"  プロンプト: {prompt[:50]}..." if len(prompt) > 50 else f"  プロンプト: {prompt}")
    print(f"  サイズ: {width}x{height}")
    print(f"  ステップ: {steps}")
    print(f"  CFG Scale: {cfg_scale}")
    print(f"  シード: {seed}")

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=cfg_scale,
        generator=generator,
    ).images[0]

    return image, seed


def save_image(image, output_dir: Path, seed: int):
    """画像を保存"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{seed}.png"
    filepath = output_dir / filename

    image.save(filepath)
    print(f"\n保存完了: {filepath}")
    return filepath


def main():
    parser = argparse.ArgumentParser(
        description="Realistic Vision v5.1 画像生成CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python generate.py "beautiful sunset over mountains"
  python generate.py "portrait of a woman" --size 768
  python generate.py "coffee on table" --steps 30 --cfg 5.0
        """,
    )

    parser.add_argument(
        "prompt",
        type=str,
        help="生成する画像のプロンプト（英語推奨）",
    )

    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="画像サイズ（正方形）。512 または 768 推奨 (default: 512)",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="画像の幅（--sizeより優先）",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="画像の高さ（--sizeより優先）",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=25,
        help="推論ステップ数。多いほど高品質だが遅い (default: 25)",
    )

    parser.add_argument(
        "--cfg",
        type=float,
        default=5.0,
        help="CFG Scale。プロンプトへの忠実度 (default: 5.0, 推奨: 3.5-7.0)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="シード値。指定しない場合はランダム",
    )

    parser.add_argument(
        "--negative",
        type=str,
        default="(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck",
        help="ネガティブプロンプト",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="outputs",
        help="出力ディレクトリ (default: outputs)",
    )

    args = parser.parse_args()

    # サイズ決定
    width = args.width if args.width else args.size
    height = args.height if args.height else args.size

    # 8の倍数に調整
    width = (width // 8) * 8
    height = (height // 8) * 8

    # 出力ディレクトリ確認
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # デバイス取得
    device = get_device()

    # リアル系プロンプト強化
    enhanced_prompt = f"RAW photo, {args.prompt}, (high detailed skin:1.2), 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3"

    # パイプライン読み込み
    pipe = load_pipeline(device)

    # 画像生成
    image, seed = generate_image(
        pipe=pipe,
        prompt=enhanced_prompt,
        negative_prompt=args.negative,
        width=width,
        height=height,
        steps=args.steps,
        cfg_scale=args.cfg,
        seed=args.seed,
        device=device,
    )

    # 保存
    save_image(image, output_dir, seed)


if __name__ == "__main__":
    main()
