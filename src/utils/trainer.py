# kaggle优秀训练函数
# https://www.zhihu.com/question/523869554/answer/2560312612

import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def train(network,
          device,
          loss_fn,
          train_loader,
          valid_loader,
          num_epochs,
          batch_size,
          lr,
          lr_min,
          init=True,
          optimizer_type='sgd',
          scheduler_type='Cosine'):
    def init_weights(m):
        '''type(m) == nn.Conv2d or nn.Linear'''
        classname = m.__class__.__name__
        # # for every Linear layer in a model.
        if classname.find('Linear') != -1:
            # Xavier初始化，保持每层输入和输出的方差一致
            nn.init.xavier_normal_(m.weight)

    if init:
        network.apply(init_weights)

    print(f"training on: {device}")
    network.to(device)

    if optimizer_type == 'sgd':
        optimizer = optim.SGD((param for param in network.parameters() if param.requires_grad),
                              lr=lr,
                              weight_decay=0)
    elif optimizer_type == 'adam':
        optimizer = optim.Adam((param for param in network.parameters() if param.requires_grad),
                               lr=lr,
                               weight_decay=0)
    elif optimizer_type == 'adamW':
        optimizer = optim.AdamW((param for param in network.parameters() if param.requires_grad),
                                lr=lr,
                                weight_decay=0)

    if scheduler_type == 'Cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr_min)

    train_losses = []
    train_accuracies = []
    eval_accuracies = []
    best_accuracy = 0
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        network.train()
        train_acc = 0
        for images, labels in tqdm(train_loader, desc=f'Train epoch {epoch + 1}'):
            images = images.to(device)
            labels = labels.to(device)
            outputs = network(images)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, pred = torch.max(outputs, 1)
            num_correct = (pred == labels).sum().item()
            train_acc += num_correct / batch_size
        scheduler.step()
        print(f'Train Loss: {loss.item():.4f}, Train Accuracy: {train_acc / len(train_loader):.4f}')
