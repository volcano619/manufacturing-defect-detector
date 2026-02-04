"""
Evaluation Metrics for Defect Detection

Standard classification metrics:
- Accuracy
- Precision (minimize false alarms)
- Recall (don't miss defects)
- F1 Score
- Confusion Matrix
"""

import numpy as np
from typing import Dict, List, Tuple
from collections import Counter
import logging

from config import CLASSES, ACCURACY_TARGET, PRECISION_TARGET, RECALL_TARGET

logger = logging.getLogger(__name__)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate accuracy."""
    return np.mean(y_true == y_pred)


def precision(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int = 1) -> float:
    """
    Precision = TP / (TP + FP)
    
    For defect detection: minimize false alarms
    """
    tp = np.sum((y_pred == pos_label) & (y_true == pos_label))
    fp = np.sum((y_pred == pos_label) & (y_true != pos_label))
    
    if tp + fp == 0:
        return 0.0
    
    return tp / (tp + fp)


def recall(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int = 1) -> float:
    """
    Recall = TP / (TP + FN)
    
    For defect detection: don't miss defects (critical!)
    """
    tp = np.sum((y_pred == pos_label) & (y_true == pos_label))
    fn = np.sum((y_pred != pos_label) & (y_true == pos_label))
    
    if tp + fn == 0:
        return 0.0
    
    return tp / (tp + fn)


def f1_score(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int = 1) -> float:
    """F1 = 2 * (Precision * Recall) / (Precision + Recall)"""
    p = precision(y_true, y_pred, pos_label)
    r = recall(y_true, y_pred, pos_label)
    
    if p + r == 0:
        return 0.0
    
    return 2 * (p * r) / (p + r)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Build confusion matrix.
    
    Returns:
        2x2 matrix [[TN, FP], [FN, TP]]
    """
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tp = np.sum((y_pred == 1) & (y_true == 1))
    
    return np.array([[tn, fp], [fn, tp]])


def specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Specificity = TN / (TN + FP)
    
    True negative rate (correctly identifying good items)
    """
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    
    if tn + fp == 0:
        return 0.0
    
    return tn / (tn + fp)


def false_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """FPR = FP / (FP + TN) = 1 - Specificity"""
    return 1 - specificity(y_true, y_pred)


def false_negative_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """FNR = FN / (FN + TP) = 1 - Recall"""
    return 1 - recall(y_true, y_pred)


def evaluate_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray = None
) -> Dict[str, float]:
    """
    Calculate all metrics for a classifier.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional, for AUC)
        
    Returns:
        Dictionary of metric names to values
    """
    metrics = {
        'Accuracy': accuracy(y_true, y_pred),
        'Precision': precision(y_true, y_pred),
        'Recall': recall(y_true, y_pred),
        'F1 Score': f1_score(y_true, y_pred),
        'Specificity': specificity(y_true, y_pred),
        'False Positive Rate': false_positive_rate(y_true, y_pred),
        'False Negative Rate': false_negative_rate(y_true, y_pred)
    }
    
    return metrics


def check_targets(metrics: Dict[str, float]) -> Dict[str, Tuple[bool, float, float]]:
    """
    Check if metrics meet targets.
    
    Returns:
        Dict[metric_name, (passed, actual, target)]
    """
    targets = {
        'Accuracy': ACCURACY_TARGET,
        'Precision': PRECISION_TARGET,
        'Recall': RECALL_TARGET
    }
    
    results = {}
    for metric, target in targets.items():
        actual = metrics.get(metric, 0)
        passed = actual >= target
        results[metric] = (passed, actual, target)
    
    return results


class DefectEvaluator:
    """Evaluate defect detection model."""
    
    def __init__(self):
        self.results = {}
        self.predictions = []
        self.ground_truth = []
    
    def add_prediction(self, y_true: int, y_pred: int):
        """Add a single prediction."""
        self.ground_truth.append(y_true)
        self.predictions.append(y_pred)
    
    def add_batch(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Add batch predictions."""
        self.ground_truth.extend(y_true.tolist())
        self.predictions.extend(y_pred.tolist())
    
    def evaluate(self) -> Dict[str, float]:
        """Calculate metrics from accumulated predictions."""
        y_true = np.array(self.ground_truth)
        y_pred = np.array(self.predictions)
        
        self.results = evaluate_classifier(y_true, y_pred)
        return self.results
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix."""
        y_true = np.array(self.ground_truth)
        y_pred = np.array(self.predictions)
        return confusion_matrix(y_true, y_pred)
    
    def summary(self) -> str:
        """Generate summary report."""
        if not self.results:
            self.evaluate()
        
        lines = ["Defect Detection Evaluation", "=" * 40]
        
        for metric, value in self.results.items():
            lines.append(f"{metric}: {value:.4f}")
        
        # Target check
        lines.append("\nTarget Check:")
        targets = check_targets(self.results)
        for metric, (passed, actual, target) in targets.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            lines.append(f"  {metric}: {actual:.2%} vs {target:.2%} {status}")
        
        # Confusion matrix
        cm = self.get_confusion_matrix()
        lines.append(f"\nConfusion Matrix:")
        lines.append(f"  TN: {cm[0,0]}, FP: {cm[0,1]}")
        lines.append(f"  FN: {cm[1,0]}, TP: {cm[1,1]}")
        
        return "\n".join(lines)
    
    def reset(self):
        """Reset evaluator."""
        self.results = {}
        self.predictions = []
        self.ground_truth = []
