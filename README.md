# Realistic Vision v5.1を使用した画像生成ツール

## セットアップ

```bash
# 依存関係のインストール
pip install -r requirements.txt
```

## 基本的な使い方

```bash
python generate.py "プロンプト"
```

### 例

```bash
# シンプルな風景
python generate.py "beautiful sunset over mountains"

# ポートレート
python generate.py "portrait of a woman, soft lighting"

# 正方形以外のサイズ
python generate.py "coffee on table" --width 768 --height 512
```

## コマンドオプション

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--size` | 512 | 正方形の画像サイズ |
| `--width` | なし | 画像の幅（--sizeより優先） |
| `--height` | なし | 画像の高さ（--sizeより優先） |
| `--steps` | 25 | 推論ステップ数（多いほど高品質） |
| `--cfg` | 5.0 | CFG Scale（プロンプトへの忠実度） |
| `--seed` | ランダム | シード値（再現性のため） |
| `--negative` | (長いデフォルト値) | ネガティブプロンプト |
| `--output` | outputs | 出力ディレクトリ |

## 推奨設定

### サイズ

| 用途 | サイズ | コマンド |
|------|--------|---------|
| 標準 | 512x512 | `--size 512` |
| 高解像度 | 768x768 | `--size 768` |
| 横長 | 768x512 | `--width 768 --height 512` |
| 縦長 | 512x768 | `--width 512 --height 768` |
| ワイドスクリーン | 1024x576 | `--width 1024 --height 576` |

### ステップ数

| 値 | 品質 | 速度 |
|----|------|------|
| 15 | 低（テスト用） | 速い |
| 25 | 標準 | 普通 |
| 30-40 | 高 | 遅い |

### CFG Scale

| 値 | 効果 |
|----|------|
| 3.0-4.0 | 創造的、プロンプトから離れやすい |
| 5.0-6.0 | バランス（推奨） |
| 7.0-8.0 | プロンプトに忠実、硬い印象になりやすい |

## プロンプトのコツ

### 基本構造

```
[被写体], [スタイル/照明], [品質タグ]
```

### 重み付け

特定の要素を強調または弱める:

```
(強調したい要素:1.3)    # 1.0より大きいと強調
(弱めたい要素:0.8)      # 1.0より小さいと抑制
```

### 良いプロンプトの例

```bash
# ポートレート
python generate.py "portrait of a woman, natural makeup, soft smile, looking at camera, soft studio lighting"

# 風景
python generate.py "japanese garden in autumn, red maple leaves, pond reflection, golden hour lighting"

# 静物
python generate.py "cup of coffee on wooden table, steam rising, morning light from window, bokeh background"
```

## 出力

- 出力先: `outputs/` ディレクトリ
- ファイル名形式: `YYYYMMDD_HHMMSS_シード値.png`
- 例: `20241222_143052_1234567890.png`

## トラブルシューティング

### メモリ不足

- サイズを小さくする（512x512推奨）
- 他のアプリを閉じる

### 黒い画像が生成される

- M4 Macでは自動的にfloat32が使用されるため通常は問題なし
- 問題が続く場合はステップ数を増やす

### 生成が遅い

- M4 Macでは512x512で約30秒〜1分程度
- 高解像度ほど時間がかかる

## 同じ画像を再生成

シード値を指定すると同じ画像を再生成できます:

```bash
# 出力ファイル名からシード値を取得
# 例: 20241222_143052_1234567890.png → シード値は 1234567890

python generate.py "同じプロンプト" --seed 1234567890
```
