from datasets import make_leaves_dl

base_path = r'E:\AIdata\kaggle-classify-leaves'
train_loader, valid_loader, test_loader = make_leaves_dl(base_path, batch_size=32)

# 使用 DataLoader
for X, y in train_loader:
    print(X.shape, y.shape)
    break