import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.colors import ListedColormap
import pandas as pd
import seaborn as sns

def visualize_batch(images, masks, predictions=None, num_samples=5, save_path=None):
    """
    Visualize a batch of images, masks, and optionally predictions.

    Args:
        images: Batch of images (B, C, H, W)
        masks: Batch of ground truth masks (B, 1, H, W)
        predictions: Optional batch of predicted masks (B, 1, H, W)
        num_samples: Number of samples to visualize
        save_path: Path to save the visualization
    """
    batch_size = min(len(images), num_samples)
    if predictions is not None:
        fig, axes = plt.subplots(batch_size, 3, figsize=(15, 5*batch_size))
    else:
        fig, axes = plt.subplots(batch_size, 2, figsize=(10, 5*batch_size))

    if batch_size == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(batch_size):
        # Original image
        img = images[i].cpu().permute(1, 2, 0).numpy()
        # Denormalize
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)

        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')

        # Ground truth mask
        axes[i, 1].imshow(masks[i].cpu().squeeze(), cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')

        # Prediction (if provided)
        if predictions is not None:
            axes[i, 2].imshow(predictions[i].cpu().squeeze() > 0.5, cmap='gray')
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def create_error_map(prediction, ground_truth, save_path=None):
    """
    Create an error visualization map showing true positives, false positives, and false negatives.

    Args:
        prediction: Binary prediction mask
        ground_truth: Binary ground truth mask
        save_path: Path to save the visualization
    """
    # Convert tensors to numpy if needed
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.cpu().numpy() > 0.5
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.cpu().numpy() > 0.5

    # Ensure binary masks and correct shape
    prediction = prediction.astype(bool)
    ground_truth = ground_truth.astype(bool)

    # Ensure 2D shapes by squeezing
    if prediction.ndim > 2:
        prediction = np.squeeze(prediction)
    if ground_truth.ndim > 2:
        ground_truth = np.squeeze(ground_truth)

    # Create error map
    error_map = np.zeros_like(ground_truth, dtype=np.uint8)

    # True Positive (1): Both prediction and ground truth are True
    error_map[np.logical_and(prediction, ground_truth)] = 1

    # False Positive (2): Prediction is True but ground truth is False
    error_map[np.logical_and(prediction, np.logical_not(ground_truth))] = 2

    # False Negative (3): Prediction is False but ground truth is True
    error_map[np.logical_and(np.logical_not(prediction), ground_truth)] = 3

    # Define colormap: background (black), TP (green), FP (red), FN (blue)
    colors = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (0, 0, 1)]
    cmap = ListedColormap(colors)

    # Plot
    plt.figure(figsize=(8, 8))
    plt.imshow(error_map, cmap=cmap)
    plt.title('Segmentation Error Map')

    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', edgecolor='w', label='True Positive'),
        Patch(facecolor='red', edgecolor='w', label='False Positive'),
        Patch(facecolor='blue', edgecolor='w', label='False Negative')
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    plt.axis('off')
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_metric_comparison(model1_metrics, model2_metrics, model1_name, model2_name, save_path):
    """Create box plots comparing the two models across metrics"""
    metrics = list(model1_metrics.keys())

    for metric in metrics:
        plt.figure(figsize=(10, 6))

        data = {
            model1_name: model1_metrics[metric],
            model2_name: model2_metrics[metric]
        }

        df = pd.DataFrame(data)

        # Create box plot
        sns.boxplot(data=df)
        plt.title(f'Comparison of {metric.capitalize()} Score')
        plt.ylabel(f'{metric.capitalize()} Score')
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'comparison_{metric}.png'), dpi=300)
        plt.close()
