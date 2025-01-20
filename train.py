import torch
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR

from models.cnn import Net
from models.vit import ViT
from utils.data import get_dataloaders
from utils.training import train, test, save_checkpoint, plot_training_progress

def train_model(model_name, model, train_loader, test_loader, epochs=100, lr=3e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.05,
        anneal_strategy='cos'
    )
    
    # 学習の進捗を記録
    accuracies = []
    losses = {'train': [], 'test': []}
    best_accuracy = 0
    
    for epoch in range(1, epochs + 1):
        # 訓練
        train(model, device, train_loader, optimizer, epoch)
        
        # 評価
        accuracy = test(model, device, test_loader)
        accuracies.append(accuracy)
        
        # 最良モデルの保存
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            save_checkpoint(
                epoch,
                model,
                optimizer,
                accuracy,
                losses,
                f'{model_name}_model_best.pth'
            )
        
        # 学習率の表示
        print(f"\nCurrent learning rate: {scheduler.get_last_lr()[0]:.6f}\n")
        
        # スケジューラの更新
        scheduler.step()
        
        # 学習の早期終了（99.5%以上の精度を達成した場合）
        if best_accuracy >= 99.5:
            print(f"Target accuracy achieved. Best accuracy: {best_accuracy:.2f}%")
            break
    
    # 学習曲線のプロット
    plot_training_progress(
        accuracies,
        losses,
        f'{model_name}_training_progress.png'
    )
    
    return best_accuracy

def main():
    # データローダーの取得
    train_loader, test_loader = get_dataloaders(batch_size=128, num_workers=4)
    
    # CNNモデルの学習
    cnn_model = Net()
    cnn_accuracy = train_model(
        'cnn',
        cnn_model,
        train_loader,
        test_loader,
        epochs=20,
        lr=1e-3
    )
    
    # ViTモデルの学習
    vit_model = ViT(
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
    vit_accuracy = train_model(
        'vit',
        vit_model,
        train_loader,
        test_loader,
        epochs=100,
        lr=1e-3
    )
    
    # 結果の比較
    print("\nFinal Results:")
    print(f"CNN Best Accuracy: {cnn_accuracy:.2f}%")
    print(f"ViT Best Accuracy: {vit_accuracy:.2f}%")
    print(f"Improvement: {vit_accuracy - cnn_accuracy:.2f}%")

if __name__ == '__main__':
    main()
