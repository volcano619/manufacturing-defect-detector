"""
Image Preprocessing and Data Generation

Handles:
1. Synthetic defect image generation
2. Image preprocessing and augmentation
3. Data loading utilities
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import random
from pathlib import Path
from typing import Tuple, List
import logging

from config import (
    TRAIN_DIR, TEST_DIR, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    SYNTHETIC_TRAIN_SAMPLES, SYNTHETIC_TEST_SAMPLES, SYNTHETIC_IMAGE_SIZE,
    DEFECT_TYPES
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_base_surface(size: int = SYNTHETIC_IMAGE_SIZE) -> Image.Image:
    """Create a synthetic metal/plastic surface."""
    # Random base color (metallic gray to silver)
    base_color = random.randint(160, 200)
    img = Image.new('RGB', (size, size), (base_color, base_color, base_color))
    
    # Add subtle texture
    pixels = np.array(img)
    noise = np.random.normal(0, 5, pixels.shape).astype(np.int16)
    pixels = np.clip(pixels.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(pixels)
    
    # Add slight gradient for realism
    draw = ImageDraw.Draw(img)
    for i in range(0, size, 10):
        alpha = int(10 * np.sin(i / size * np.pi))
        draw.line([(0, i), (size, i)], fill=(base_color + alpha, base_color + alpha, base_color + alpha))
    
    return img


def add_scratch(img: Image.Image) -> Image.Image:
    """Add scratch defect to image."""
    draw = ImageDraw.Draw(img)
    size = img.size[0]
    
    # Random scratch parameters
    num_scratches = random.randint(1, 3)
    
    for _ in range(num_scratches):
        x1 = random.randint(10, size - 10)
        y1 = random.randint(10, size - 10)
        
        # Scratch direction and length
        angle = random.uniform(0, 2 * np.pi)
        length = random.randint(30, 100)
        
        x2 = int(x1 + length * np.cos(angle))
        y2 = int(y1 + length * np.sin(angle))
        
        # Dark scratch line
        scratch_color = random.randint(30, 80)
        width = random.randint(1, 3)
        draw.line([(x1, y1), (x2, y2)], fill=(scratch_color, scratch_color, scratch_color), width=width)
    
    return img


def add_crack(img: Image.Image) -> Image.Image:
    """Add crack defect to image."""
    draw = ImageDraw.Draw(img)
    size = img.size[0]
    
    # Start point
    x, y = random.randint(20, size - 20), random.randint(20, size - 20)
    
    # Generate crack path (random walk)
    points = [(x, y)]
    for _ in range(random.randint(5, 10)):
        dx = random.randint(-20, 20)
        dy = random.randint(-20, 20)
        x = max(5, min(size - 5, x + dx))
        y = max(5, min(size - 5, y + dy))
        points.append((x, y))
    
    # Draw crack
    crack_color = random.randint(20, 60)
    draw.line(points, fill=(crack_color, crack_color, crack_color), width=1)
    
    # Add branches
    for i in range(0, len(points) - 1, 2):
        if random.random() > 0.5:
            bx = points[i][0] + random.randint(-15, 15)
            by = points[i][1] + random.randint(-15, 15)
            draw.line([points[i], (bx, by)], fill=(crack_color, crack_color, crack_color), width=1)
    
    return img


def add_contamination(img: Image.Image) -> Image.Image:
    """Add contamination spots to image."""
    draw = ImageDraw.Draw(img)
    size = img.size[0]
    
    # Random spots
    num_spots = random.randint(2, 6)
    
    for _ in range(num_spots):
        cx = random.randint(20, size - 20)
        cy = random.randint(20, size - 20)
        radius = random.randint(5, 20)
        
        # Dark or discolored spot
        if random.random() > 0.5:
            spot_color = (random.randint(40, 80), random.randint(40, 80), random.randint(40, 80))
        else:
            spot_color = (random.randint(100, 140), random.randint(80, 100), random.randint(60, 80))
        
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=spot_color)
    
    return img


def add_dent(img: Image.Image) -> Image.Image:
    """Add dent mark to image."""
    draw = ImageDraw.Draw(img)
    size = img.size[0]
    
    # Dent center
    cx = random.randint(30, size - 30)
    cy = random.randint(30, size - 30)
    radius = random.randint(15, 40)
    
    # Create dent effect (lighter center, darker edge)
    for r in range(radius, 0, -2):
        alpha = int(30 * (1 - r / radius))
        color = 180 + alpha
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(color, color, color))
    
    # Dark shadow on one side
    shadow_offset = 5
    shadow_color = 120
    draw.arc(
        [cx - radius + shadow_offset, cy - radius + shadow_offset, 
         cx + radius + shadow_offset, cy + radius + shadow_offset],
        start=45, end=225, fill=(shadow_color, shadow_color, shadow_color), width=2
    )
    
    return img


def generate_defect_image(defect_type: str = None) -> Image.Image:
    """Generate a synthetic defect image."""
    img = create_base_surface()
    
    if defect_type is None:
        defect_type = random.choice(DEFECT_TYPES)
    
    if defect_type == "scratch":
        img = add_scratch(img)
    elif defect_type == "crack":
        img = add_crack(img)
    elif defect_type == "contamination":
        img = add_contamination(img)
    elif defect_type == "dent":
        img = add_dent(img)
    
    return img


def generate_good_image() -> Image.Image:
    """Generate a synthetic non-defective image."""
    return create_base_surface()


def generate_synthetic_dataset(
    train_samples: int = SYNTHETIC_TRAIN_SAMPLES,
    test_samples: int = SYNTHETIC_TEST_SAMPLES
) -> None:
    """Generate full synthetic dataset."""
    
    logger.info("Generating synthetic defect detection dataset...")
    
    # Training data
    train_good = train_samples // 2
    train_defect = train_samples // 2
    
    # Generate training good images
    good_dir = TRAIN_DIR / "good"
    good_dir.mkdir(parents=True, exist_ok=True)
    for i in range(train_good):
        img = generate_good_image()
        img.save(good_dir / f"good_{i:04d}.png")
    logger.info(f"Generated {train_good} training good images")
    
    # Generate training defect images
    defect_dir = TRAIN_DIR / "defect"
    defect_dir.mkdir(parents=True, exist_ok=True)
    for i in range(train_defect):
        defect_type = DEFECT_TYPES[i % len(DEFECT_TYPES)]
        img = generate_defect_image(defect_type)
        img.save(defect_dir / f"defect_{defect_type}_{i:04d}.png")
    logger.info(f"Generated {train_defect} training defect images")
    
    # Test data
    test_good = test_samples // 2
    test_defect = test_samples // 2
    
    # Generate test good images
    test_good_dir = TEST_DIR / "good"
    test_good_dir.mkdir(parents=True, exist_ok=True)
    for i in range(test_good):
        img = generate_good_image()
        img.save(test_good_dir / f"good_{i:04d}.png")
    logger.info(f"Generated {test_good} test good images")
    
    # Generate test defect images
    test_defect_dir = TEST_DIR / "defect"
    test_defect_dir.mkdir(parents=True, exist_ok=True)
    for i in range(test_defect):
        defect_type = DEFECT_TYPES[i % len(DEFECT_TYPES)]
        img = generate_defect_image(defect_type)
        img.save(test_defect_dir / f"defect_{defect_type}_{i:04d}.png")
    logger.info(f"Generated {test_defect} test defect images")
    
    logger.info("Dataset generation complete!")


def preprocess_image(
    image: Image.Image,
    size: Tuple[int, int] = IMAGE_SIZE
) -> np.ndarray:
    """
    Preprocess image for model input.
    
    Returns:
        Normalized numpy array (C, H, W)
    """
    # Resize
    img = image.resize(size, Image.BILINEAR)
    
    # Convert to numpy
    img_array = np.array(img).astype(np.float32) / 255.0
    
    # Normalize with ImageNet stats
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)
    img_array = (img_array - mean) / std
    
    # HWC to CHW
    img_array = np.transpose(img_array, (2, 0, 1))
    
    return img_array


def load_image(path: str) -> Image.Image:
    """Load an image from path."""
    return Image.open(path).convert('RGB')


if __name__ == "__main__":
    generate_synthetic_dataset()
