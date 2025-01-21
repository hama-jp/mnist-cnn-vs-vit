import pygame
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from models.vit import ViT

class DigitRecognizer:
    def __init__(self, model_path='vit_model_best.pth'):
        # PyGameの初期化
        pygame.init()
        
        # 画面の設定
        self.width = 800
        self.height = 600
        self.drawing_size = 280
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Handwritten Digit Recognition")
        
        # 描画エリアの設定
        self.drawing_surface = pygame.Surface((self.drawing_size, self.drawing_size))
        self.drawing_surface.fill((255, 255, 255))
        self.drawing = False
        self.last_pos = None
        
        # フォントの設定
        self.font = pygame.font.Font(None, 36)
        
        # モデルの初期化と読み込み
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ViT(
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
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # 画像の前処理用の変換
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    def preprocess_image(self):
        """描画された画像を前処理"""
        # PyGameのサーフェスをnumpy配列に変換
        arr = pygame.surfarray.array3d(self.drawing_surface)
        # 白黒を反転（MNISTは黒背景に白文字なので）
        arr = 255 - arr
        # グレースケールに変換
        arr = np.mean(arr, axis=2).astype(np.uint8)
        # PIL Imageに変換
        img = Image.fromarray(arr)
        # モデル用に前処理
        x = self.transform(img).unsqueeze(0)
        return x
    
    def predict(self):
        """画像を予測"""
        x = self.preprocess_image()
        x = x.to(self.device)
        
        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1)
        
        # 上位3つの予測結果を取得
        values, indices = torch.topk(probs, 3)
        return [(idx.item(), val.item()) for idx, val in zip(indices[0], values[0])]
    
    def draw(self):
        """画面の描画"""
        self.screen.fill((200, 200, 200))
        
        # 描画エリアの表示
        drawing_pos = ((self.width - self.drawing_size) // 2, 50)
        self.screen.blit(self.drawing_surface, drawing_pos)
        pygame.draw.rect(self.screen, (0, 0, 0), (*drawing_pos, self.drawing_size, self.drawing_size), 2)
        
        # 予測結果の表示
        results = self.predict()
        y = self.drawing_size + 100
        for i, (digit, prob) in enumerate(results):
            text = f"{digit}: {prob:.3f}"
            text_surface = self.font.render(text, True, (0, 0, 0))
            text_rect = text_surface.get_rect(center=(self.width // 2, y + i * 40))
            self.screen.blit(text_surface, text_rect)
        
        # 操作説明の表示
        instructions = [
            "Left click to draw",
            "Space to clear",
            "ESC to exit"
        ]
        for i, text in enumerate(instructions):
            text_surface = self.font.render(text, True, (50, 50, 50))
            self.screen.blit(text_surface, (20, self.height - 100 + i * 30))
        
        pygame.display.flip()
    
    def run(self):
        """メインループ"""
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # 左クリック
                        self.drawing = True
                        self.last_pos = None
                
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:  # 左クリック
                        self.drawing = False
                
                elif event.type == pygame.MOUSEMOTION and self.drawing:
                    current_pos = (
                        event.pos[0] - (self.width - self.drawing_size) // 2,
                        event.pos[1] - 50
                    )
                    
                    # 描画エリア内かチェック
                    if (0 <= current_pos[0] < self.drawing_size and 
                        0 <= current_pos[1] < self.drawing_size):
                        if self.last_pos:
                            pygame.draw.line(
                                self.drawing_surface,
                                (0, 0, 0),
                                self.last_pos,
                                current_pos,
                                8
                            )
                        self.last_pos = current_pos
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self.drawing_surface.fill((255, 255, 255))
            
            self.draw()
        
        pygame.quit()

if __name__ == "__main__":
    app = DigitRecognizer()
    app.run()
