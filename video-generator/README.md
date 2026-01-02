# Video Generator for M4 Mac

画像から5秒程度の動画を生成するツール。
M4 Mac (24GB Unified Memory) に最適化されています。

## 仕組み

1. **Stable Video Diffusion (SVD)** で14フレームの短い動画を生成
2. **光学フロー補間** でフレームを補間し、5秒の動画に拡張

## セットアップ

```bash
cd video-generator
./setup.sh
```

## 使い方

### 基本的な使い方

```bash
source venv/bin/activate

# 画像から5秒動画を生成
python generate_video.py ../outputs/your_image.png

# 出力先を指定
python generate_video.py ../outputs/your_image.png -o my_video.mp4
```

### オプション

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `-d, --duration` | 5.0 | 目標動画長（秒） |
| `--fps` | 24 | 出力FPS |
| `--motion` | 127 | 動きの強さ (1-255) |
| `--frames` | 14 | 初期生成フレーム数 |
| `--method` | svd | 生成方法 (svd/animatediff) |
| `--no-extend` | - | フレーム補間をスキップ |

### 例

```bash
# 穏やかな動き（風景向け）
python generate_video.py image.png --motion 80

# 激しい動き
python generate_video.py image.png --motion 180

# 3秒の動画
python generate_video.py image.png --duration 3

# フレーム補間なし（生のSVD出力）
python generate_video.py image.png --no-extend
```

## フレーム補間のみ

既存の短い動画を5秒に伸ばす場合：

```bash
python scripts/rife_interpolate.py input_video.mp4 -d 5.0
```

## メモリ使用量

| 設定 | 推定メモリ使用量 |
|------|-----------------|
| SVD (14フレーム) | 12-16GB |
| SVD (25フレーム) | 18-22GB |
| AnimateDiff | 10-14GB |

メモリ不足の場合：
1. 他のアプリを閉じる
2. `--frames 10` で初期フレーム数を減らす
3. AnimateDiff を試す: `--method animatediff`

## トラブルシューティング

### MPS (Metal) エラー
```bash
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
python generate_video.py image.png
```

### ffmpeg がない
```bash
brew install ffmpeg
```

### モデルダウンロードが遅い
初回実行時に Hugging Face からモデルをダウンロードします。
約4-5GBのダウンロードが必要です。

## ファイル構成

```
video-generator/
├── generate_video.py      # メインスクリプト
├── scripts/
│   └── rife_interpolate.py  # フレーム補間
├── outputs/               # 生成された動画
├── models/                # ダウンロードしたモデル
├── requirements.txt
├── setup.sh
└── README.md
```
