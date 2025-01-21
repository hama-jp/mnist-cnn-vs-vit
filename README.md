# MNIST: CNN vs Vision Transformer

MNISTデータセットを使用して、CNNとVision Transformer (ViT) の性能を比較した実装です。
動作確認用のPyGameを使ったサンプルアプリも同梱しました。

## 実装内容

### モデル構造

#### CNN
- 4層のConvolutional層
- BatchNormalization
- Dropout (0.5)
- MaxPooling (2x2)
- 全結合層 (128ユニット)

#### Vision Transformer (ViT)
- パッチサイズ: 4x4
- エンコーダー層数: 6
- Attention heads: 8
- 埋め込みサイズ: 256
- MLPサイズ: 512
- Dropout: 0.1

### データ拡張
- RandomAffine (回転、平行移動、拡大縮小、せん断)
- RandomErasing
- CutMix

### データセット
- MNIST
- EMNIST-digits
- EMNIST-mnist
を組み合わせて使用（合計360,000枚の学習データ）

## 結果

| モデル | Best Accuracy |
|--------|--------------|
| CNN    | 99.08%      |
| ViT    | 99.53%      |

Vision Transformerが従来のCNNモデルを0.46%上回る精度を達成しました。

## ディレクトリ構造

```
.
├── models/
│   ├── __init__.py
│   ├── cnn.py
│   ├── layers.py
│   └── vit.py
├── utils/
│   ├── __init__.py
│   ├── data.py
│   └── training.py
└── train.py
```

## 要件

- Python 3.8+
- PyTorch
- torchvision
- einops
- matplotlib
- numpy

## 使用方法

1. 環境構築:
```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision einops matplotlib numpy
```

2. 学習の実行:
```bash
python train.py
```

## 手書き数字認識アプリ

学習済みViTモデルを使用して、リアルタイムで手書き数字を認識するアプリケーションを提供しています。

### 追加の要件

- pygame

### 使用方法

1. PyGameのインストール:
```bash
source .venv/bin/activate  # 仮想環境が有効でない場合
pip install pygame
```

2. アプリの実行:
```bash
python pygame_app.py
```

### 操作方法

- **左クリック**: 数字を描画
- **スペースキー**: 描画をクリア
- **ESCキー**: アプリを終了

数字を描くと、リアルタイムで上位3つの予測結果が表示されます。
