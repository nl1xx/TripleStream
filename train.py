import torch
import argparse
from torch.utils.data import DataLoader
from DataProcess.build_dataset import ATTMicroDataset
from Model.triple_stream import TripleStreamNet
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.cuda.amp import autocast, GradScaler


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str,
                        default="E:\PLAYGROUND\PytorchExercise\PytorchProject\MY_TripleStreamNet_Rebuild\Data\CASME2_Preprocessed")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seq_length", type=int, default=5)
    return parser.parse_args()


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化数据集
    train_set = ATTMicroDataset(args.data_root, "train", args.seq_length)
    val_set = ATTMicroDataset(args.data_root, "val", args.seq_length)

    # 初始化模型
    model = TripleStreamNet(num_classes=train_set.num_classes)
    model = model.to(device)

    # 优化器和损失函数
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    criterion = CrossEntropyLoss()
    scaler = GradScaler()  # 初始化梯度缩放器

    # 训练循环
    for epoch in range(args.epochs):
        model.train()
        for batch_idx, (seqs, labels) in enumerate(DataLoader(train_set, batch_size=args.batch_size)):
            seqs = seqs.to(device)  # [B, C, T, H, W]
            labels = labels.to(device)

            # 混合精度前向传播
            with autocast():
                outputs = model(seqs)
                loss = criterion(outputs, labels)

            # 反向传播与优化
            optimizer.zero_grad()
            scaler.scale(loss).backward()  # 缩放梯度
            scaler.step(optimizer)  # 更新参数
            scaler.update()  # 调整缩放因子

            # 打印日志
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch + 1}/{args.epochs} | Batch {batch_idx} | Loss: {loss.item():.4f}")

        # 验证阶段
        model.eval()
        total_correct = 0
        with torch.no_grad():
            for seqs, labels in DataLoader(val_set, batch_size=args.batch_size):
                # 验证时也可使用混合精度节省显存
                with autocast():
                    outputs = model(seqs.to(device))
                preds = outputs.argmax(dim=1)
                total_correct += (preds == labels.to(device)).sum().item()

        acc = total_correct / len(val_set)
        print(f"Validation Acc: {acc:.2%}")


if __name__ == "__main__":
    main()
