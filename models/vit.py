import torch
import torch.nn as nn
from einops import rearrange, repeat
from .layers import Transformer

class ViT(nn.Module):
    def __init__(
        self, 
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
    ):
        super().__init__()
        
        assert image_size % patch_size == 0, 'Image size must be divisible by patch size'
        
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size * patch_size
        
        self.patch_size = patch_size
        
        # パッチの埋め込み
        self.to_patch_embedding = nn.Sequential(
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        
        # Position embeddings + CLS token
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        
        # Transformer
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        
        # MLP head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        
    def forward(self, img):
        # (batch_size, channels, height, width) -> (batch_size, num_patches, patch_dim)
        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                     p1=self.patch_size, p2=self.patch_size)
        
        # パッチの埋め込み
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape
        
        # CLSトークンの追加
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Position embeddingの追加
        x = x + self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        
        # Transformerで特徴抽出
        x = self.transformer(x)
        
        # CLSトークンの出力を分類に使用
        return self.mlp_head(x[:, 0])
