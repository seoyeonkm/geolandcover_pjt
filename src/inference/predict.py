"""Inference utilities for land cover classification."""

from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from torchvision import models, transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEFAULT_CLASSES = ["도시", "농지", "산림", "바다", "황무지"]


def _build_eval_transform(image_size: int = 224) -> transforms.Compose:
    """Create evaluation transform pipeline used during inference."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def load_model_and_classes(
    checkpoint_path: str = "outputs/checkpoints/best_resnet18_eurosat.pt",
    device: str | None = None,
) -> Tuple[torch.nn.Module, List[str], torch.device]:
    """Load finetuned ResNet18 checkpoint and class names."""
    model_device = torch.device(
        device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Please train and save model first."
        )

    checkpoint = torch.load(ckpt_path, map_location=model_device)
    class_map = checkpoint.get("class_map", None)

    if isinstance(class_map, dict) and len(class_map) > 0:
        ordered_classes = [class_map[i] for i in sorted(class_map.keys())]
    else:
        ordered_classes = DEFAULT_CLASSES

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(ordered_classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(model_device)
    model.eval()

    return model, ordered_classes, model_device


def predict_proba(
    image: Image.Image,
    checkpoint_path: str = "outputs/checkpoints/best_resnet18_eurosat.pt",
) -> Tuple[str, Dict[str, float]]:
    """Predict class and class-wise probabilities for one PIL image."""
    model, class_names, device = load_model_and_classes(checkpoint_path=checkpoint_path)
    transform = _build_eval_transform(image_size=224)

    rgb_image = image.convert("RGB")
    tensor = transform(rgb_image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()

    prob_by_class = {
        class_name: float(prob)
        for class_name, prob in zip(class_names, probs)
    }
    pred_class = max(prob_by_class, key=prob_by_class.get)

    return pred_class, prob_by_class
