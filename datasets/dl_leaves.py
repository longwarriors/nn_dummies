import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
import os
from PIL import Image
from typing import List


class LeavesDataset(Dataset):
    def __init__(self, df: pd.DataFrame, root_dir: str, transform=None):
        self.image_paths: List[str] = [os.path.join(root_dir, relative_path) for relative_path in df['image']]
        self.transform = transform
        if 'label' in df.columns:  # 如果 'label' 列不存在则为测试集
            labels, self.unique_labels = pd.factorize(df['label'])
            self.labels = torch.tensor(labels, dtype=torch.long)
            self.num_classes = len(self.unique_labels)
        else:
            self.labels = None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Returns: (image, label) 如果是训练集
                 image 如果是测试集
        """
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)

        if self.labels is None:  # 测试集
            return image
        else:
            return image, self.labels[idx]


def make_leaves_dl(base_path: str = r'E:\AIdata\kaggle-classify-leaves',
                   batch_size: int = 32,
                   train_spilt: float = 0.9):
    train_path = os.path.join(base_path, 'train.csv')
    test_path = os.path.join(base_path, 'test.csv')
    sample_submission_path = os.path.join(base_path, 'sample_submission.csv')
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    sample_submission_df = pd.read_csv(sample_submission_path)

    # 定义变换
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomRotation(degrees=15),  # 随机旋转
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = val_transform  # 测试集通常与验证集变换一致

    tsf = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomRotation(degrees=20),  # 随机旋转
        transforms.ToTensor()
    ])

    # 创建训练集数据集
    train_set = LeavesDataset(train_df, base_path, transform=train_transform)
    test_set = LeavesDataset(test_df, base_path, transform=test_transform)

    # 分训练集为 9/10 的训练集和 1/10 的验证集
    train_size = int(train_spilt * len(train_set))
    valid_size = len(train_set) - train_size

    # 随机划分数据集
    train_subset, valid_subset = random_split(train_set, [train_size, valid_size])

    # num_workers 指定了用于加载数据的子进程数量。每个子进程负责从数据集中加载一部分数据并将其返回给主进程。
    # pin_memory 是一个布尔值（True 或 False），用于指定是否使用固定内存（pinned memory）来加速数据传输。
    train_loader = DataLoader(train_subset, batch_size, shuffle=True, num_workers=0, pin_memory=True)
    valid_loader = DataLoader(valid_subset, batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, valid_loader, test_loader


if __name__ == '__main__':
    print(f"训练数据预览：\n{train_df.head(10)}")
    print(f"测试数据预览：\n{test_df.head(10)}")
    print(f"提交样例预览：\n{sample_submission_df.head(10)}")
    print(f"训练集大小={train_size}, 验证集大小={valid_size}")
