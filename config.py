"""
Configuration Module for Manufacturing Defect Detection System

Centralizes all configuration parameters.
"""

from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
MODELS_DIR = PROJECT_ROOT / "saved_models"

for d in [DATA_DIR, TRAIN_DIR, TEST_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================================
# IMAGE CONFIGURATION
# ============================================================================
IMAGE_SIZE = (224, 224)  # Standard for EfficientNet/ResNet
CHANNELS = 3
BATCH_SIZE = 32

# Normalization (ImageNet stats for transfer learning)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ============================================================================
# CLASS LABELS
# ============================================================================
CLASSES = ["good", "defect"]
NUM_CLASSES = 2

# Defect subtypes (for synthetic data generation)
DEFECT_TYPES = ["scratch", "crack", "contamination", "dent"]

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
MODEL_NAME = "efficientnet_b0"  # Options: resnet18, efficientnet_b0
PRETRAINED = True
FREEZE_BACKBONE = False  # Fine-tune all layers

# Training hyperparameters
EPOCHS = 20
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 5

# ============================================================================
# INFERENCE CONFIGURATION
# ============================================================================
CONFIDENCE_THRESHOLD = 0.5  # Above this = defect
HIGH_CONFIDENCE_THRESHOLD = 0.8  # High confidence detection

# ============================================================================
# GRAD-CAM CONFIGURATION
# ============================================================================
GRADCAM_LAYER = "features"  # Target layer for activation maps
HEATMAP_ALPHA = 0.4  # Overlay transparency

# ============================================================================
# EVALUATION METRICS
# ============================================================================
ACCURACY_TARGET = 0.95
RECALL_TARGET = 0.98  # Critical: don't miss defects
PRECISION_TARGET = 0.90  # Minimize false alarms

# ============================================================================
# APPLICATION CONFIGURATION
# ============================================================================
APP_TITLE = "🔍 Manufacturing Defect Detection"
APP_LAYOUT = "wide"
DEBUG_MODE = True

# ============================================================================
# SYNTHETIC DATA CONFIGURATION
# ============================================================================
SYNTHETIC_TRAIN_SAMPLES = 400  # 200 good + 200 defect
SYNTHETIC_TEST_SAMPLES = 100   # 50 good + 50 defect
SYNTHETIC_IMAGE_SIZE = 224
