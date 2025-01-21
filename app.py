import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from models.vit import ViT
import gradio as gr

def load_model(model_path='vit_model_best.pth'):
    """学習済みモデルを読み込む"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # モデルの初期化
    model = ViT(
        image_size=28,
        patch_size=4,
        num_classes=10,
        dim=256,
        depth=6,
        heads=8,
        mlp_dim=512,
        channels=1,
        dim_head=32,
        dropout=0.1,
        emb_dropout=0.1
    )
    
    # チェックポイントの読み込み
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, device

def preprocess_image(image):
    """手書き画像の前処理"""
    # グレースケールに変換
    if image.shape[2] == 3:
        image = np.mean(image, axis=2)
    
    # PIL Imageに変換
    image = Image.fromarray((image * 255).astype(np.uint8))
    
    # モデル用に前処理
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    return transform(image).unsqueeze(0)

def predict(image, model, device):
    """画像を予測"""
    # 画像の前処理
    x = preprocess_image(image)
    x = x.to(device)
    
    # 予測
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        
    # 確率を取得
    probs = probs[0].cpu().numpy()
    return {str(i): float(p) for i, p in enumerate(probs)}

# モデルの読み込み
model, device = load_model()

# Gradioインターフェースの作成
interface = gr.Interface(
    fn=lambda x: predict(x, model, device),
    inputs=gr.Sketchpad(shape=(280, 280), tool="pencil"),
    outputs=gr.Label(num_top_classes=3),
    live=True,
    title="MNISTリアルタイム数字認識",
    description="手書きの数字を書いてください。モデルがリアルタイムで認識を行います。",
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    interface.launch()
