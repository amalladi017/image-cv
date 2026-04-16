import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

#Preproessing and pathing for iamges
class FolderCIFAR10(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.class_names = sorted([p.name for p in self.root.iterdir() if p.is_dir()])
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        self.samples = []

        valid_exts = {".png", ".jpg", ".jpeg", ".bmp"}
        for class_name in self.class_names:
            class_dir = self.root / class_name
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() in valid_exts:
                    self.samples.append((img_path, self.class_to_idx[class_name]))

        if len(self.samples) == 0:
            raise ValueError(f"No images found in {self.root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

#Convolution block functions
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act_layer):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.depthwise = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels, bias=False)
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.act = act_layer()

    def forward(self, x):
        x = self.bn(x)
        x = self.act(self.pointwise(x))
        x = self.act(self.depthwise(x))
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return x


#CNN
class SimpleFeatureCNN(nn.Module):
    def __init__(self, num_classes=10, channels=(32, 64, 128, 256), mlp_hidden=256):
        super().__init__()
        act_layer = nn.ReLU
        c1, c2, c3, c4 = channels

        self.stem = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=3, padding=1, bias=False),
            act_layer(),
            nn.Conv2d(c1, c1, kernel_size=3, padding=1, bias=False),
            act_layer(),
        )

        #main blocks for convolutions

        self.block1 = ConvBlock(c1, c2, act_layer)
        self.maxpool = nn.MaxPool2d(2)
        self.block2 = ConvBlock(c2, c3, act_layer)
        self.downsample = nn.Sequential(
            nn.Conv2d(c3, c3, kernel_size=3, stride=2, padding=1, bias=False),
            act_layer(),
        )
        self.block3 = ConvBlock(c3, c4, act_layer)

        #pooling and classifier head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c4, mlp_hidden),
            act_layer(),
            nn.Dropout(0.3),
            nn.Linear(mlp_hidden, num_classes),
        )

    #STEM
    def forward_features(self, x):
        x = self.stem(x)
        feat1 = self.block1(x)
        x = self.maxpool(feat1)
        feat2 = self.block2(x)
        x = self.downsample(feat2)
        feat3 = self.block3(x)
        return feat1, feat2, feat3

    def forward(self, x):
        _, _, x = self.forward_features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_seen += labels.size(0)

    return total_loss / total_seen, 100.0 * total_correct / total_seen


@torch.no_grad()
def evaluate(model, loader, device, class_names):
    model.eval()
    total_correct = 0
    total_seen = 0
    true_labels = []
    pred_labels = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)

        total_correct += (preds == labels).sum().item()
        total_seen += labels.size(0)
        true_labels.extend(labels.cpu().tolist())
        pred_labels.extend(preds.cpu().tolist())

    overall_acc = 100.0 * total_correct / total_seen
    cm = confusion_matrix(true_labels, pred_labels, labels=list(range(len(class_names))))

    per_class_acc = {}
    for i, class_name in enumerate(class_names):
        class_total = cm[i].sum()
        if class_total == 0:
            per_class_acc[class_name] = 0.0
        else:
            per_class_acc[class_name] = 100.0 * cm[i, i] / class_total

    return overall_acc, per_class_acc, cm


@torch.no_grad()
def get_prediction_records(model, loader, device):
    model.eval()
    records = []
    sample_index = 0

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)

        for i in range(images.size(0)):
            pred = preds[i].item()
            records.append({
                "index": sample_index,
                "image": images[i].cpu(),
                "true": labels[i].item(),
                "pred": pred,
                "confidence": probs[i, pred].item(),
            })
            sample_index += 1

    return records


def choose_examples(records):
    correct = [r for r in records if r["true"] == r["pred"]]
    wrong = [r for r in records if r["true"] != r["pred"]]

    if len(correct) == 0:
        raise ValueError("No correct test predictions were found.")
    if len(wrong) == 0:
        raise ValueError("No misclassified test image was found. Reduce epochs or pick one manually.")

    easy_correct = max(correct, key=lambda x: x["confidence"])
    hard_correct = min(correct, key=lambda x: x["confidence"])
    wrong_example = max(wrong, key=lambda x: x["confidence"])

    return {
        "easy_correct": easy_correct,
        "hard_correct": hard_correct,
        "misclassified": wrong_example,
    }


def denormalize(img_tensor, mean, std):
    img = img_tensor.clone()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    img = img.clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


def save_confusion_matrix(cm, class_names, save_path):
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def save_one_block(feature_tensor, title, save_path, true_name, pred_name, conf):
    feature_tensor = feature_tensor.squeeze(0).cpu()
    channels = feature_tensor.shape[0]
    cols = int(np.ceil(np.sqrt(channels)))
    rows = int(np.ceil(channels / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = np.array(axes).reshape(-1)

    for i in range(channels):
        axes[i].imshow(feature_tensor[i].numpy(), cmap="viridis")
        axes[i].axis("off")
        axes[i].set_title(f"C{i}", fontsize=8)

    for i in range(channels, len(axes)):
        axes[i].axis("off")

    fig.suptitle(f"{title} | true={true_name} | pred={pred_name} | conf={conf:.3f}", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)

# ignore gradients
@torch.no_grad()
def save_feature_maps(model, example, class_names, mean, std, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    image = example["image"].unsqueeze(0)
    feat1, feat2, feat3 = model.forward_features(image)

    true_name = class_names[example["true"]]
    pred_name = class_names[example["pred"]]
    conf = example["confidence"]

    img = denormalize(example["image"], mean, std)
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(f"Input\ntrue={true_name}\npred={pred_name}")
    plt.tight_layout()
    plt.savefig(save_dir / "input_image.png", dpi=200)
    plt.close(fig)

    save_one_block(feat1, "Block 1", save_dir / "block1_maps.png", true_name, pred_name, conf)
    save_one_block(feat2, "Block 2", save_dir / "block2_maps.png", true_name, pred_name, conf)
    save_one_block(feat3, "Block 3", save_dir / "block3_maps.png", true_name, pred_name, conf)


def save_metrics(metrics, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]

    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_data = FolderCIFAR10(Path(args.data_dir) / "train", transform=train_tf)
    test_data = FolderCIFAR10(Path(args.data_dir) / "test", transform=test_tf)
    class_names = train_data.class_names

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = SimpleFeatureCNN(num_classes=len(class_names), channels=(32, 64, 128, 256), mlp_hidden=256).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    history = []
    best_acc = 0.0
    best_model_path = out_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        test_acc, _, _ = evaluate(model, test_loader, device, class_names)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
        })

        print(f"Epoch {epoch:02d}/{args.epochs} | loss={train_loss:.4f} | train_acc={train_acc:.2f}% | test_acc={test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), best_model_path)

    model.load_state_dict(torch.load(best_model_path, map_location=device))

    overall_acc, per_class_acc, cm = evaluate(model, test_loader, device, class_names)
    save_confusion_matrix(cm, class_names, out_dir / "confusion_matrix.png")

    records = get_prediction_records(model, test_loader, device)
    chosen = choose_examples(records)

    selected_examples = {}
    for tag, example in chosen.items():
        save_dir = out_dir / "feature_maps" / tag
        save_feature_maps(model, example, class_names, mean, std, save_dir)
        selected_examples[tag] = {
            "index": example["index"],
            "true_label": class_names[example["true"]],
            "pred_label": class_names[example["pred"]],
            "confidence": example["confidence"],
            "folder": str(save_dir),
        }

    metrics = {
        "device": str(device),
        "channels": [32, 64, 128, 256],
        "activation": "ReLU",
        "mlp_hidden": 256,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "best_test_accuracy": best_acc,
        "overall_test_accuracy": overall_acc,
        "per_class_accuracy": per_class_acc,
        "history": history,
        "selected_examples": selected_examples,
    }
    save_metrics(metrics, out_dir / "metrics.json")

    print("\nFinal overall accuracy:", f"{overall_acc:.2f}%")
    print("Per-class accuracy:")
    for class_name, acc in per_class_acc.items():
        print(f"  {class_name:<12} {acc:.2f}%")
    print(f"Saved best model: {best_model_path}")
    print(f"Saved metrics: {out_dir / 'metrics.json'}")
    print(f"Saved confusion matrix: {out_dir / 'confusion_matrix.png'}")
    print(f"Saved feature maps in: {out_dir / 'feature_maps'}")


if __name__ == "__main__":
    main()
