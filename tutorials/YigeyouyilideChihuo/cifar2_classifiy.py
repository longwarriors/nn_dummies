"""https://www.heywhale.com/mw/project/5f2b44a3af3980002cb1ce5f"""
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from datetime import datetime
from torchkeras.metrics import Accuracy
from matplotlib import pyplot as plt
from copy import deepcopy


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.dropout = nn.Dropout2d(0.1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()

        # out_features = num_classes
        # 下面这种输出形式不是常用的
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def display_samples(imgset):
    plt.figure(figsize=(10, 10))
    for i in range(9):
        img, label = imgset[i]
        img = img.permute(1, 2, 0)
        ax = plt.subplot(3, 3, i + 1)
        ax.imshow(img.numpy())
        ax.set_title("label = %d" % label)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


def printlog(info):
    nowtime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n" + "==========" * 8 + f"{nowtime}")
    print(str(info) + "\n")


class StepRunner:
    def __init__(self, net, loss_fn, stage="train", optimizer=None, metrics_dict=None):
        self.net = net
        self.loss_fn = loss_fn
        self.stage = stage
        self.metrics_dict = metrics_dict
        self.optimizer = optimizer

    def step(self, features, labels):
        pred = self.net(features).squeeze()
        loss = self.loss_fn(pred, labels.float())
        if self.optimizer is not None and self.stage == "train":
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        step_metrics = {self.stage + "_" + name: metric_fn(pred, labels).item() for name, metric_fn in
                        self.metrics_dict.items()}
        return loss.item(), step_metrics

    def train_step(self, features, labels):
        self.net.train()  # 训练模式下dropout层才发生作用
        return self.step(features, labels)

    @torch.no_grad()
    def eval_step(self, features, labels):
        self.net.eval()  # 预测模式下dropout层不发生作用
        return self.step(features, labels)

    def __call__(self, features, labels):
        if self.stage == "train":
            return self.train_step(features, labels)
        else:
            return self.eval_step(features, labels)


class EpochRunner:
    def __init__(self, step_runner: StepRunner):
        self.step_runner = step_runner
        self.stage = step_runner.stage

    def __call__(self, dataloader: DataLoader):
        total_loss, step = 0, 0
        loop = tqdm(enumerate(dataloader), total=len(dataloader), file=sys.stdout)
        # epoch_log = {}  # 初始化 epoch_log 变量
        for idx, batch in loop:
            loss, step_metrics = self.step_runner(*batch)
            step_log = dict({self.stage + "_loss": loss}, **step_metrics)
            total_loss += loss
            step += 1
            if idx != len(dataloader) - 1:
                loop.set_postfix(**step_log)
            else:
                epoch_loss = total_loss / step
                epoch_metrics = {self.stage + "_" + name: metric_fn.compute().item() for name, metric_fn in
                                 self.step_runner.metrics_dict.items()}
                epoch_log = dict({self.stage + "_loss": epoch_loss}, **epoch_metrics)
                # epoch_log = {self.stage + "_loss": epoch_loss, **epoch_metrics}  # 优化后的合并字典方法
                loop.set_postfix(**epoch_log)

                for name, metric_fn in self.step_runner.metrics_dict.items():
                    metric_fn.reset()
        return epoch_log


def train_model(net, loss_fn, optimizer, metrics_dict, train_loader, val_loader=None, epochs=10,
                ckpt_path='checkpoint.pt', patience=5, monitor='val_loss', mode='min'):
    history = {}
    for epoch in range(1, epochs + 1):
        printlog(f"Epoch {epoch}/{epochs}")

        # 1，train -------------------------------------------------
        train_step_runner = StepRunner(net, loss_fn, "train", optimizer, deepcopy(metrics_dict))
        train_epoch_runner = EpochRunner(train_step_runner)
        train_metrics = train_epoch_runner(train_loader)

        # for name, metric_fn in train_metrics.items():
        #     history[name].append(metric_fn.compute().item())

        for name, metric in train_metrics.items():
            if name not in history:
                history[name] = []  # Initialize the list for this metric
            history[name].append(metric)  # Append the float value directly

        # 2，validate -------------------------------------------------
        if val_loader:
            val_step_runner = StepRunner(net, loss_fn, "val", None, deepcopy(metrics_dict))
            val_epoch_runner = EpochRunner(val_step_runner)

            with torch.no_grad():
                val_metrics = val_epoch_runner(val_loader)

            val_metrics["epoch"] = epoch

            for name, metric in val_metrics.items():
                history[name] = history.get(name, []) + [metric]

        # 3，early-stopping -------------------------------------------------
        arr_scores = history[monitor]
        best_score_idx = np.argmax(arr_scores) if mode == "max" else np.argmin(arr_scores)
        if best_score_idx == len(arr_scores) - 1:
            torch.save(net.state_dict(), ckpt_path)
            print("<<<<<< reach best {0} : {1} >>>>>>".format(monitor, arr_scores[best_score_idx]), file=sys.stderr)
        if len(arr_scores) - best_score_idx > patience:
            print("<<<<<< {} without improvement in {} epoch, early stopping >>>>>>".format(monitor, patience),
                  file=sys.stderr)
            break
        net.load_state_dict(torch.load(ckpt_path))
    return pd.DataFrame(history)


trans_img = transforms.Compose([
    transforms.ToTensor()
])
train_set = datasets.ImageFolder(root='../../data/CIFAR/cifar2/train/', transform=trans_img)
val_set = datasets.ImageFolder(root='../../data/CIFAR/cifar2/test/', transform=trans_img)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
model = Net()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
metrics_dict = {"acc": Accuracy()}
df_history = train_model(model, criterion, optimizer, metrics_dict, train_loader, val_loader, epochs=10,
                         patience=3, monitor='val_acc', mode='max')
