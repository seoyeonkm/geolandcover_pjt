"""Training pipeline for land cover classification."""

from pathlib import Path

import torch
from torch import nn, optim

from src.data.preprocess import build_eurosat_dataloaders
from src.models.landcover_model import SimpleLandCoverCNN


def build_cnn_model(num_classes: int) -> nn.Module:
    """Build simple CNN model for land cover classification."""
    return SimpleLandCoverCNN(num_classes=num_classes)


def evaluate(model: nn.Module, data_loader, criterion: nn.Module, device: torch.device):
    """Evaluate model on validation set and return loss/accuracy."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def train(
    num_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    output_dir: str = "outputs/checkpoints",
    seed: int = 42,
) -> None:
    """Train simple CNN model and save best checkpoint."""
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloaders, id_to_label = build_eurosat_dataloaders(
        batch_size=batch_size,
        image_size=224,
        num_workers=0,
        seed=seed,
    )

    model = build_cnn_model(num_classes=len(id_to_label)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_val_acc = 0.0
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    best_model_path = out_path / "best_cnn_eurosat.pt"

    print(f"Device: {device}")
    print(f"Classes: {id_to_label}")

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_samples = 0

        for images, labels in dataloaders["train"]:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            running_samples += labels.size(0)

        train_loss = running_loss / running_samples
        train_acc = running_correct / running_samples

        val_loss, val_acc = evaluate(
            model=model,
            data_loader=dataloaders["val"],
            criterion=criterion,
            device=device,
        )

        print(
            f"Epoch [{epoch}/{num_epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_acc": best_val_acc,
                    "class_map": id_to_label,
                },
                best_model_path,
            )
            print(f"Best model saved: {best_model_path} (val_acc={best_val_acc:.4f})")

    print(f"Training finished. Best val_acc={best_val_acc:.4f}")


if __name__ == "__main__":
    train()
