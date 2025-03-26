import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import os
import pandas as pd


class ResNet(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x  # 残差
        return self.relu(y)


def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(ResNet(input_channels, num_channels, use_1x1conv=True, strides=2))
        else:
            blk.append(ResNet(num_channels, num_channels))
    return blk


class ClassifyLeaves(nn.Module):
    def __init__(self, num_classes, set_device='cpu'):
        super().__init__()
        b1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                           nn.BatchNorm2d(64),
                           nn.ReLU(),
                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
        b3 = nn.Sequential(*resnet_block(64, 128, 2))
        b4 = nn.Sequential(*resnet_block(128, 256, 2))
        b5 = nn.Sequential(*resnet_block(256, 512, 2))
        self.net = nn.Sequential(b1, b2, b3, b4, b5,
                                 nn.AdaptiveAvgPool2d((1, 1)),
                                 nn.Flatten(),
                                 nn.Linear(512, num_classes))
        self.device = torch.device(set_device if torch.cuda.is_available() else 'cpu')
        self.to(self.device)  # 把神经网络实例移到设备上
        print(f"Model initialized on {self.device}")

    def forward(self, x):
        return self.net(x)

    def evaluate(self, dataloader: DataLoader, criterion=nn.CrossEntropyLoss()):
        self.eval()  # 关闭 dropout 和 batch norm 的训练行为
        loss_sum: float = 0.0
        correct_sum: float = 0.0
        num: int = 0
        with torch.no_grad():
            for X, y in dataloader:
                batch_size = X.size(0)  # 或 y.size(0)
                X, y = X.to(self.device), y.to(self.device)
                logits = self(X)
                loss_sum += criterion(logits, y).item() * batch_size  # 按样本数加权
                y_hat = logits.argmax(dim=1)
                correct_sum += (y_hat == y).sum().item()
                num += batch_size
        epoch_loss = loss_sum / num
        epoch_accuracy = correct_sum / num * 100  # 百分比形式
        return epoch_loss, epoch_accuracy

    def train_model(self, train_dl: DataLoader, val_dl: DataLoader,
                    num_epochs: int,
                    learning_rate: float = 0.001,
                    weight_decay=0.01,
                    checkpoint_path: str = '../outputs/checkpoints/kaggle_leaves.pth.tar',
                    resume=False):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        best_val_accuracy: float = 0.0
        start_epoch: int = 0
        if resume and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            self.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_accuracy = checkpoint['best_val_accuracy']
            print(f"Model loaded from epoch {start_epoch - 1} with val_accuracy: {best_val_accuracy:.2f}%")

        for epoch in range(start_epoch, num_epochs):
            self.train()
            loss_sum: float = 0.0
            correct_sum: float = 0.0
            num: int = 0
            for X, y in train_dl:
                batch_size = X.size(0)  # 或 y.size(0)
                X, y = X.to(self.device), y.to(self.device)
                logits = self(X)
                loss = criterion(logits, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()  # 更新参数

                loss_sum += loss.item() * batch_size  # 按样本数加权
                y_hat = logits.argmax(dim=1)
                correct_sum += (y_hat == y).sum().item()
                num += batch_size

            # 计算当前 epoch 训练集的平均损失和准确率
            train_epoch_loss = loss_sum / num
            train_epoch_accuracy = correct_sum / num * 100

            # 评估训练过的模型在验证集的表现
            val_loss, val_accuracy = self.evaluate(val_dl, criterion)

            print('=' * 60)
            print(f'Epoch [{epoch + 1}/{num_epochs}]')
            print(f'Train Loss: {train_epoch_loss:.4f}, Train Accuracy: {train_epoch_accuracy:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

            # 保存检查点
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                checkpoint = {'model_state_dict': self.state_dict(),
                              'optimizer_state_dict': optimizer.state_dict(),
                              'epoch': epoch,
                              'best_val_accuracy': best_val_accuracy}
                torch.save(checkpoint, checkpoint_path, _use_new_zipfile_serialization=True)
                print(f"Checkpoint saved at epoch {epoch + 1} with Val Accuracy: {val_accuracy:.2f}%")

    def create_submission(self, test_dl: DataLoader,
                          unique_labels: list,
                          checkpoint_path: str = '../outputs/checkpoints/kaggle_leaves.pth.tar',
                          submission_path: str = 'submission.csv'):
        """生成 Kaggle 提交文件"""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            self.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model from {checkpoint_path}")
        else:
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        self.eval()
        predictions = []
        with torch.no_grad():
            for X in test_dl:
                X = X.to(self.device)
                logits = self(X)
                y_hat = logits.argmax(dim=1)  # 获取预测类别索引
                predictions.extend(y_hat.cpu().numpy())
        # 索引转换为类别名
        predicted_labels = [unique_labels[idx] for idx in predictions]

        # 获取图片路径
        image_paths = test_dl.dataset.image_paths
        relative_paths = [f"images/{os.path.basename(path)}" for path in image_paths]  # 只取文件名

        # 创建提交 DataFrame
        submission_df = pd.DataFrame({'image': relative_paths, 'label': predicted_labels})
        submission_df.to_csv(submission_path, index=False)  # 保存提交文件
        print(f"Submission file saved to {submission_path}")


if __name__ == '__main__':
    from datasets.dl_leaves import make_leaves_dl

    # 获取数据加载器
    DATA_DIR = r'E:\AIdata\kaggle-classify-leaves'
    train_loader, valid_loader, test_loader = make_leaves_dl(base_path=DATA_DIR, batch_size=16, train_spilt=0.85)
    NUM_CLASSES = train_loader.dataset.dataset.num_classes
    TAR_LABELS = train_loader.dataset.dataset.unique_labels
    task_model = ClassifyLeaves(NUM_CLASSES, set_device='cuda')
    task_model.train_model(train_loader, valid_loader, num_epochs=80, resume=True)
    task_model.create_submission(test_loader, TAR_LABELS, submission_path='../outputs/kaggle_leaves_submission.csv')
