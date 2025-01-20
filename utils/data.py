import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import datasets, transforms
from torchvision.transforms import RandomAffine, RandomErasing

class CutMix:
    """CutMix augmentation"""
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, batch):
        images, labels = batch
        batch_size = len(images)
        
        # ラムダ値の生成
        lam = torch.distributions.beta.Beta(self.alpha, self.alpha).sample()
        
        # ランダムなインデックスを生成
        rand_index = torch.randperm(batch_size)
        
        # バウンディングボックスの生成
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(images.size(), lam)
        
        # 画像の混合
        images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
        
        # ラベルの混合比率の計算
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
        
        return images, labels, labels[rand_index], lam

    def _rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = torch.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = torch.randint(W, (1,))[0]
        cy = torch.randint(H, (1,))[0]

        bbx1 = torch.clamp(cx - cut_w // 2, 0, W)
        bby1 = torch.clamp(cy - cut_h // 2, 0, H)
        bbx2 = torch.clamp(cx + cut_w // 2, 0, W)
        bby2 = torch.clamp(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

def get_dataloaders(batch_size=64, num_workers=4, use_cutmix=True):
    """データローダーの取得"""
    # 基本の変換
    transform = transforms.Compose([
        transforms.ToTensor(),
        RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=15),
        RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 訓練データセット
    mnist_train = datasets.MNIST('data', train=True, download=True, transform=transform)
    emnist_digits_train = datasets.EMNIST('data', split='digits', train=True, download=True, transform=transform)
    emnist_mnist_train = datasets.EMNIST('data', split='mnist', train=True, download=True, transform=transform)
    
    # テストデータセット
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    mnist_test = datasets.MNIST('data', train=False, transform=test_transform)
    emnist_digits_test = datasets.EMNIST('data', split='digits', train=False, transform=test_transform)
    emnist_mnist_test = datasets.EMNIST('data', split='mnist', train=False, transform=test_transform)

    # データセットの結合
    train_dataset = ConcatDataset([mnist_train, emnist_digits_train, emnist_mnist_train])
    test_dataset = ConcatDataset([mnist_test, emnist_digits_test, emnist_mnist_test])

    # データローダーの作成
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
    )

    print(f"Combined training dataset size: {len(train_dataset)}")
    print(f"Combined test dataset size: {len(test_dataset)}")

    return train_loader, test_loader
