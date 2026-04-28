"""
CNN Classifier for Defect Detection

Implements:
1. Transfer learning with EfficientNet/ResNet
2. Binary classification (Good vs Defect)
3. Training and evaluation
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
import logging
from pathlib import Path

from config import (
    NUM_CLASSES, CLASSES, IMAGE_SIZE, EPOCHS, LEARNING_RATE,
    IMAGENET_MEAN, IMAGENET_STD, MODELS_DIR, CONFIDENCE_THRESHOLD
)

logger = logging.getLogger(__name__)

# Check for PyTorch availability
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import torchvision.models as models
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Using simple classifier fallback.")


class SimpleClassifier:
    """
    Simple baseline classifier using color histogram features.
    Used when PyTorch is not available.
    """
    
    def __init__(self):
        self.threshold = 0.3  # Defect detection threshold
        self.is_fitted = True
    
    def predict(self, image: np.ndarray, threshold: float = 0.5) -> Tuple[str, float]:
        """
        Defect detection based on dark pixel ratio and color variance.
        Defect images contain significantly more very dark or off-color pixels.
        """
        if isinstance(image, np.ndarray) and image.ndim == 3:
            gray = np.mean(image, axis=2)
            
            # Dark pixel ratio: scratches/cracks are dark (0-40) on a light (160-200) base
            # Use threshold of 130 to catch all defect types reliably
            dark_ratio = np.mean(gray < 130)
            
            # Color channel deviation: rust/contamination = high R, low B
            r = image[:, :, 0].astype(float)
            g = image[:, :, 1].astype(float)
            b = image[:, :, 2].astype(float)
            color_dev = np.mean(np.abs(r - b)) / 255.0
            
            # Pixel std normalized
            pixel_std = np.std(image) / 128.0
            
            # Minimum brightness factor (defects are significantly darker than base)
            min_brightness = np.min(gray)
            min_pixel_factor = max(0, (120 - min_brightness) / 120.0)
            
            # Weighted combination — dark ratio and min pixel are strongest for scratches/cracks
            defect_score = min(dark_ratio * 5.0 + min_pixel_factor * 0.4 + color_dev * 3.0 + pixel_std * 0.5, 1.0)
        else:
            defect_score = 0.3
        
        if defect_score >= threshold:
            return "defect", min(defect_score, 0.99)
        else:
            return "good", min(1.0 - defect_score, 0.99)
    
    def predict_proba(self, image: np.ndarray) -> np.ndarray:
        """Return class probabilities."""
        _, score = self.predict(image)
        if score > 0.5:
            return np.array([1 - score, score])
        else:
            return np.array([score, 1 - score])


if TORCH_AVAILABLE:
    class DefectDataset(Dataset):
        """PyTorch Dataset for defect images."""
        
        def __init__(
            self,
            root_dir: str,
            transform=None
        ):
            self.root_dir = Path(root_dir)
            self.transform = transform
            self.samples = []
            
            # Load good images
            good_dir = self.root_dir / "good"
            if good_dir.exists():
                for img_path in good_dir.glob("*.png"):
                    self.samples.append((str(img_path), 0))
                for img_path in good_dir.glob("*.jpg"):
                    self.samples.append((str(img_path), 0))
            
            # Load defect images
            defect_dir = self.root_dir / "defect"
            if defect_dir.exists():
                for img_path in defect_dir.glob("*.png"):
                    self.samples.append((str(img_path), 1))
                for img_path in defect_dir.glob("*.jpg"):
                    self.samples.append((str(img_path), 1))
            
            logger.info(f"Loaded {len(self.samples)} images from {root_dir}")
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            from PIL import Image
            
            img_path, label = self.samples[idx]
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
    
    
    class DefectClassifier(nn.Module):
        """
        CNN for defect classification using transfer learning.
        """
        
        def __init__(
            self,
            model_name: str = "resnet18",
            num_classes: int = NUM_CLASSES,
            pretrained: bool = True
        ):
            super().__init__()
            
            self.model_name = model_name
            
            if model_name == "resnet18":
                self.backbone = models.resnet18(
                    weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
                )
                num_features = self.backbone.fc.in_features
                self.backbone.fc = nn.Linear(num_features, num_classes)
                
            elif model_name == "efficientnet_b0":
                self.backbone = models.efficientnet_b0(
                    weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
                )
                num_features = self.backbone.classifier[1].in_features
                self.backbone.classifier[1] = nn.Linear(num_features, num_classes)
            
            else:
                raise ValueError(f"Unknown model: {model_name}")
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.backbone(x)
        
        def get_features_layer(self):
            """Get the layer to use for Grad-CAM."""
            if self.model_name == "resnet18":
                return self.backbone.layer4
            elif self.model_name == "efficientnet_b0":
                return self.backbone.features[-1]


class DefectDetector:
    """
    High-level wrapper for defect detection.
    
    Handles model loading, inference, and training.
    """
    
    def __init__(
        self,
        model_name: str = "resnet18",
        pretrained: bool = True,
        device: str = None
    ):
        self.model_name = model_name
        self.is_fitted = False
        
        if TORCH_AVAILABLE:
            self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = DefectClassifier(model_name, pretrained=pretrained)
            self.model = self.model.to(self.device)
            
            self.transform = transforms.Compose([
                transforms.Resize(IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
            
            self.train_transform = transforms.Compose([
                transforms.Resize(IMAGE_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
        else:
            self.model = SimpleClassifier()
            self.is_fitted = True
    
    def train(
        self,
        train_dir: str,
        val_dir: str = None,
        epochs: int = EPOCHS,
        learning_rate: float = LEARNING_RATE,
        batch_size: int = 32
    ) -> Dict[str, List[float]]:
        """
        Train the model on the given dataset.
        
        Returns:
            Training history
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available. Using pretrained fallback.")
            return {"loss": [], "accuracy": []}
        
        # Create datasets
        train_dataset = DefectDataset(train_dir, transform=self.train_transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if val_dir:
            val_dataset = DefectDataset(val_dir, transform=self.transform)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        history = {"loss": [], "accuracy": []}
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            epoch_loss = total_loss / len(train_loader)
            epoch_acc = correct / total
            history["loss"].append(epoch_loss)
            history["accuracy"].append(epoch_acc)
            
            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")
        
        self.is_fitted = True
        return history
    
    def predict(
        self,
        image,
        return_proba: bool = False,
        threshold: float = 0.5
    ) -> Tuple[str, float]:
        """
        Predict defect status for an image.
        
        Args:
            image: PIL Image or numpy array
            return_proba: If True, return class probabilities
            threshold: Minimum confidence to classify as defect
            
        Returns:
            (class_name, confidence) or probabilities
        """
        from PIL import Image
        
        if not TORCH_AVAILABLE:
            if isinstance(image, Image.Image):
                image = np.array(image)
            return self.model.predict(image, threshold=threshold)
        
        self.model.eval()
        
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))
        
        # Preprocess
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(img_tensor)
            proba = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        
        if return_proba:
            return proba
        
        defect_prob = float(proba[1])
        if defect_prob >= threshold:
            pred_class = 1
            confidence = defect_prob
        else:
            pred_class = 0
            confidence = float(proba[0])
        
        return CLASSES[pred_class], confidence
    
    def save(self, path: str = None):
        """Save model weights."""
        if not TORCH_AVAILABLE:
            return
        
        path = path or str(MODELS_DIR / "defect_detector.pth")
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str = None):
        """Load model weights."""
        if not TORCH_AVAILABLE:
            return
        
        path = path or str(MODELS_DIR / "defect_detector.pth")
        if Path(path).exists():
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.is_fitted = True
            logger.info(f"Model loaded from {path}")
