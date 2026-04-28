"""
Grad-CAM for Defect Localization

Gradient-weighted Class Activation Mapping to visualize
which image regions triggered the defect detection.
"""

import numpy as np
from PIL import Image
from typing import Tuple, Optional
import logging

from config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, HEATMAP_ALPHA

logger = logging.getLogger(__name__)

# Check for PyTorch
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def apply_colormap(heatmap: np.ndarray) -> np.ndarray:
    """
    Convert grayscale heatmap to RGB colormap.
    
    Args:
        heatmap: Normalized heatmap (0-1)
        
    Returns:
        RGB colormap array
    """
    # Jet-like colormap
    heatmap = np.clip(heatmap, 0, 1)
    
    # Simple jet colormap implementation
    r = np.clip(4 * heatmap - 1.5, 0, 1)
    g = np.clip(1 - 4 * np.abs(heatmap - 0.5), 0, 1)
    b = np.clip(1.5 - 4 * heatmap, 0, 1)
    
    colormap = np.stack([r, g, b], axis=-1)
    return (colormap * 255).astype(np.uint8)


class SimpleGradCAM:
    """
    Simplified Grad-CAM that works with simple heuristics.
    Used when PyTorch is not available.
    """
    
    def generate(self, image: np.ndarray, prediction: str) -> np.ndarray:
        """
        Generate a heatmap highlighting anomalous regions (pure numpy, no scipy).
        """
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2).astype(np.float32)
        else:
            gray = image.astype(np.float32)
        
        # Dark pixel map — defects are dark lines/spots on a light background
        dark_map = np.clip((150.0 - gray) / 150.0, 0, 1)
        
        # Color deviation map — contamination has off-colors
        if len(image.shape) == 3:
            r = image[:, :, 0].astype(np.float32)
            g = image[:, :, 1].astype(np.float32)
            b = image[:, :, 2].astype(np.float32)
            color_dev = np.abs(r - b) / 255.0
        else:
            color_dev = np.zeros_like(dark_map)
        
        # Combine: dark pixels + color deviations = defect signature
        heatmap = np.clip(dark_map * 0.7 + color_dev * 0.3, 0, 1)
        
        # Normalize to 0-1
        hmin, hmax = heatmap.min(), heatmap.max()
        if hmax > hmin:
            heatmap = (heatmap - hmin) / (hmax - hmin)
        
        return heatmap


if TORCH_AVAILABLE:
    class GradCAM:
        """
        Grad-CAM implementation for CNN models.
        
        Generates heatmaps showing which regions contributed most
        to the classification decision.
        """
        
        def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
            self.model = model
            self.target_layer = target_layer
            
            self.gradients = None
            self.activations = None
            
            # Register hooks
            self._register_hooks()
        
        def _register_hooks(self):
            """Register forward and backward hooks."""
            
            def forward_hook(module, input, output):
                self.activations = output.detach()
            
            def backward_hook(module, grad_input, grad_output):
                self.gradients = grad_output[0].detach()
            
            self.target_layer.register_forward_hook(forward_hook)
            self.target_layer.register_full_backward_hook(backward_hook)
        
        def generate(
            self,
            input_tensor: torch.Tensor,
            target_class: int = None
        ) -> np.ndarray:
            """
            Generate Grad-CAM heatmap.
            
            Args:
                input_tensor: Preprocessed input (1, C, H, W)
                target_class: Class to generate heatmap for (default: predicted)
                
            Returns:
                Heatmap as numpy array (H, W)
            """
            self.model.eval()
            
            # Forward pass
            output = self.model(input_tensor)
            
            if target_class is None:
                target_class = output.argmax(dim=1).item()
            
            # Backward pass
            self.model.zero_grad()
            target = output[0, target_class]
            target.backward()
            
            # Compute weights
            gradients = self.gradients[0]  # (C, H, W)
            activations = self.activations[0]  # (C, H, W)
            
            weights = gradients.mean(dim=(1, 2))  # (C,)
            
            # Weighted combination
            cam = torch.zeros(activations.shape[1:], device=activations.device)
            for i, w in enumerate(weights):
                cam += w * activations[i]
            
            # ReLU and normalize
            cam = F.relu(cam)
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)
            
            # Resize to input size
            cam = cam.unsqueeze(0).unsqueeze(0)
            cam = F.interpolate(cam, size=IMAGE_SIZE, mode='bilinear', align_corners=False)
            cam = cam.squeeze().cpu().numpy()
            
            return cam


def create_heatmap_overlay(
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = HEATMAP_ALPHA
) -> Image.Image:
    """
    Overlay heatmap on original image.
    
    Args:
        image: Original PIL Image
        heatmap: Normalized heatmap (0-1)
        alpha: Overlay transparency
        
    Returns:
        Combined image with heatmap overlay
    """
    # Resize heatmap to match image
    img_array = np.array(image)
    
    if heatmap.shape[:2] != img_array.shape[:2]:
        heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8))
        heatmap_img = heatmap_img.resize(image.size, Image.BILINEAR)
        heatmap = np.array(heatmap_img) / 255.0
    
    # Apply colormap
    colormap = apply_colormap(heatmap)
    
    # Blend
    overlay = (1 - alpha) * img_array + alpha * colormap
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    return Image.fromarray(overlay)


class DefectLocalizer:
    """
    High-level interface for defect localization using Grad-CAM.
    """
    
    def __init__(self, detector):
        """
        Args:
            detector: DefectDetector instance
        """
        self.detector = detector
        
        if TORCH_AVAILABLE and hasattr(detector, 'model') and hasattr(detector.model, 'get_features_layer'):
            target_layer = detector.model.get_features_layer()
            self.gradcam = GradCAM(detector.model, target_layer)
        else:
            self.gradcam = SimpleGradCAM()
    
    def localize(
        self,
        image: Image.Image,
        target_class: int = None
    ) -> Tuple[np.ndarray, Image.Image]:
        """
        Generate defect localization heatmap.
        
        Returns:
            (heatmap, overlay_image)
        """
        if TORCH_AVAILABLE and hasattr(self.detector, 'transform'):
            # Prepare input
            img_tensor = self.detector.transform(image).unsqueeze(0)
            img_tensor = img_tensor.to(self.detector.device)
            
            # Generate heatmap
            heatmap = self.gradcam.generate(img_tensor, target_class)
        else:
            img_array = np.array(image)
            prediction, _ = self.detector.predict(image)
            heatmap = self.gradcam.generate(img_array, prediction)
        
        # Create overlay
        overlay = create_heatmap_overlay(image, heatmap)
        
        return heatmap, overlay
