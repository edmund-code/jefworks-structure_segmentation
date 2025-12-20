#!/usr/bin/env python3
"""
Nuclei Segmentation using Meta's Segment Anything Model (SAM)
This script uses SAM to segment nuclei in H&E stained histopathology images.
"""

import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import os
from urllib.request import urlretrieve

try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
except ImportError:
    print("segment-anything not installed. Installing...")
    import subprocess
    subprocess.check_call(["pip", "install", "git+https://github.com/facebookresearch/segment-anything.git"])
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

from skimage import measure, morphology
from skimage.color import rgb2hed


def download_sam_checkpoint(model_type='vit_h', checkpoint_dir='checkpoints'):
    """
    Download SAM model checkpoint if not already present.
    
    Parameters:
    -----------
    model_type : str
        Model type: 'vit_h' (largest, best), 'vit_l', or 'vit_b' (smallest, fastest)
    checkpoint_dir : str
        Directory to save checkpoints
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_urls = {
        'vit_h': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
        'vit_l': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
        'vit_b': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth'
    }
    
    checkpoint_names = {
        'vit_h': 'sam_vit_h_4b8939.pth',
        'vit_l': 'sam_vit_l_0b3195.pth',
        'vit_b': 'sam_vit_b_01ec64.pth'
    }
    
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_names[model_type])
    
    if not os.path.exists(checkpoint_path):
        print(f"Downloading SAM {model_type} checkpoint...")
        print("This may take a few minutes...")
        urlretrieve(checkpoint_urls[model_type], checkpoint_path)
        print(f"Checkpoint downloaded to {checkpoint_path}")
    else:
        print(f"Using existing checkpoint: {checkpoint_path}")
    
    return checkpoint_path


def load_sam_model(model_type='vit_b', device=None):
    """
    Load SAM model.
    
    Parameters:
    -----------
    model_type : str
        Model type: 'vit_h', 'vit_l', or 'vit_b'
    device : str
        Device to run on ('cuda' or 'cpu')
    
    Returns:
    --------
    model : SAM model
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading SAM model ({model_type}) on {device}...")
    
    checkpoint_dir = 'checkpoints'
    checkpoint_path = download_sam_checkpoint(model_type, checkpoint_dir)
    
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    
    print("SAM model loaded successfully!")
    return sam, device


def generate_nuclei_prompts(image_rgb, num_points=100):
    """
    Generate prompt points for nuclei using color deconvolution.
    This helps SAM focus on nuclei regions.
    
    Parameters:
    -----------
    image_rgb : numpy array
        RGB image
    num_points : int
        Number of prompt points to generate
    
    Returns:
    --------
    points : numpy array
        Array of [x, y] coordinates
    labels : numpy array
        Array of 1s (foreground points)
    """
    # Convert to HED color space
    hed = rgb2hed(image_rgb)
    hematoxylin = hed[:, :, 0]
    
    # Normalize and threshold
    hematoxylin_norm = (hematoxylin - hematoxylin.min()) / (hematoxylin.max() - hematoxylin.min() + 1e-10)
    threshold = np.percentile(hematoxylin_norm, 85)  # Top 15% are likely nuclei
    binary = hematoxylin_norm > threshold
    
    # Find centroids of connected components
    labeled = measure.label(binary)
    regions = measure.regionprops(labeled)
    
    points = []
    for region in regions:
        if 20 <= region.area <= 500:  # Filter by size
            y, x = region.centroid
            points.append([int(x), int(y)])
    
    # If we have fewer points than requested, sample more from high-intensity regions
    if len(points) < num_points:
        # Sample additional points from high-intensity regions
        coords = np.column_stack(np.where(hematoxylin_norm > threshold))
        if len(coords) > 0:
            indices = np.random.choice(len(coords), min(num_points - len(points), len(coords)), replace=False)
            additional_points = [[int(coords[i][1]), int(coords[i][0])] for i in indices]
            points.extend(additional_points)
    
    # Limit to num_points
    if len(points) > num_points:
        points = points[:num_points]
    
    points = np.array(points)
    labels = np.ones(len(points), dtype=np.int32)
    
    return points, labels


def segment_nuclei_sam(image_path, model_type='vit_b', use_auto_mask=True, min_nucleus_size=20, max_nucleus_size=1000):
    """
    Segment nuclei using Meta's Segment Anything Model.
    
    Parameters:
    -----------
    image_path : str
        Path to the H&E stained image
    model_type : str
        SAM model type: 'vit_h' (best), 'vit_l', or 'vit_b' (fastest)
    use_auto_mask : bool
        If True, use automatic mask generation. If False, use prompt-based.
    min_nucleus_size : int
        Minimum size (in pixels) for a valid nucleus
    max_nucleus_size : int
        Maximum size (in pixels) for a valid nucleus
    
    Returns:
    --------
    cell_count : int
        Number of detected cells
    nuclei_mask : numpy array
        Binary mask of segmented nuclei
    labeled_nuclei : numpy array
        Labeled image where each nucleus has a unique ID
    all_masks : list
        List of all SAM masks
    """
    # Load the image
    print(f"Loading image from {image_path}...")
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image_rgb.shape[:2]
    
    # Load SAM model
    sam, device = load_sam_model(model_type)
    
    if use_auto_mask:
        # Use automatic mask generation
        print("Generating masks with SAM automatic mask generator...")
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=min_nucleus_size,
        )
        
        masks = mask_generator.generate(image_rgb)
        print(f"SAM generated {len(masks)} candidate masks")
        
    else:
        # Use prompt-based segmentation
        print("Generating prompt points for nuclei...")
        points, labels = generate_nuclei_prompts(image_rgb, num_points=200)
        print(f"Generated {len(points)} prompt points")
        
        predictor = SamPredictor(sam)
        predictor.set_image(image_rgb)
        
        masks = []
        # Process points in batches
        batch_size = 50
        for i in range(0, len(points), batch_size):
            batch_points = points[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            
            mask, scores, logits = predictor.predict(
                point_coords=batch_points,
                point_labels=batch_labels,
                multimask_output=True,
            )
            
            # Take the best mask for each point
            for j in range(len(batch_points)):
                best_mask_idx = np.argmax(scores[j])
                masks.append({
                    'segmentation': mask[j][best_mask_idx],
                    'score': scores[j][best_mask_idx],
                    'area': np.sum(mask[j][best_mask_idx])
                })
    
    # Filter masks by size and merge overlapping masks
    print("Filtering and processing masks...")
    filtered_masks = []
    for mask_data in masks:
        area = mask_data['area'] if 'area' in mask_data else np.sum(mask_data['segmentation'])
        if min_nucleus_size <= area <= max_nucleus_size:
            filtered_masks.append(mask_data)
    
    print(f"After size filtering: {len(filtered_masks)} masks")
    
    # Create combined mask
    nuclei_mask = np.zeros((height, width), dtype=bool)
    mask_areas = []
    
    for mask_data in filtered_masks:
        mask = mask_data['segmentation']
        area = np.sum(mask)
        
        # Additional filtering: check if mask overlaps too much with existing masks
        overlap = np.sum(nuclei_mask & mask) / area if area > 0 else 0
        
        # Only add if overlap is less than 50% (to avoid duplicates)
        if overlap < 0.5:
            nuclei_mask |= mask
            mask_areas.append(area)
    
    # Label connected components
    labeled_nuclei = measure.label(nuclei_mask)
    
    # Filter labeled regions by size again
    regions = measure.regionprops(labeled_nuclei)
    final_mask = np.zeros_like(nuclei_mask, dtype=bool)
    
    for region in regions:
        if min_nucleus_size <= region.area <= max_nucleus_size:
            final_mask[labeled_nuclei == region.label] = True
    
    # Re-label
    labeled_nuclei = measure.label(final_mask)
    cell_count = labeled_nuclei.max()
    
    print(f"\nSegmentation complete!")
    print(f"Number of detected cells: {cell_count}")
    
    return cell_count, final_mask, labeled_nuclei, filtered_masks, image_rgb


def visualize_results_sam(image_rgb, nuclei_mask, labeled_nuclei, cell_count, masks=None, save_path=None):
    """
    Visualize the SAM segmentation results.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    # Original image
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title('Original H&E Image', fontsize=12)
    axes[0, 0].axis('off')
    
    # Show some SAM masks (if available)
    if masks is not None and len(masks) > 0:
        mask_overlay = image_rgb.copy()
        # Show first 50 masks with different colors
        for i, mask_data in enumerate(masks[:50]):
            mask = mask_data['segmentation']
            color = np.random.rand(3)
            mask_overlay[mask] = mask_overlay[mask] * 0.6 + np.array(color) * 255 * 0.4
        
        axes[0, 1].imshow(mask_overlay.astype(np.uint8))
        axes[0, 1].set_title(f'SAM Masks Overlay\n({min(50, len(masks))} masks shown)', fontsize=12)
    else:
        axes[0, 1].imshow(image_rgb)
        axes[0, 1].set_title('Original Image', fontsize=12)
    axes[0, 1].axis('off')
    
    # Binary mask
    axes[1, 0].imshow(nuclei_mask, cmap='gray')
    axes[1, 0].set_title(f'Final Segmented Nuclei Mask\n({cell_count} cells detected)', fontsize=12)
    axes[1, 0].axis('off')
    
    # Overlay on original image
    overlay = image_rgb.copy()
    nuclei_colored = np.zeros_like(image_rgb)
    nuclei_colored[nuclei_mask] = [255, 0, 0]  # Red overlay
    overlay = cv2.addWeighted(overlay, 0.7, nuclei_colored, 0.3, 0)
    
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title(f'Overlay: Original + Segmented Nuclei\n({cell_count} cells)', fontsize=12)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to {save_path}")
    
    plt.close()


def main():
    # Path to the image
    image_path = Path("data/S23_14614_sub2.jpg")
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    # Segment nuclei using SAM
    # Use 'vit_b' for faster processing, 'vit_l' for better quality, 'vit_h' for best quality
    cell_count, nuclei_mask, labeled_nuclei, masks, image_rgb = segment_nuclei_sam(
        image_path,
        model_type='vit_b',  # Change to 'vit_l' or 'vit_h' for better results (slower)
        use_auto_mask=True,  # Use automatic mask generation
        min_nucleus_size=20,
        max_nucleus_size=1000
    )
    
    # Visualize results
    output_path = Path("nuclei_segmentation_sam_results.png")
    visualize_results_sam(image_rgb, nuclei_mask, labeled_nuclei, cell_count, masks, save_path=output_path)
    
    # Print summary statistics
    print("\n" + "="*50)
    print("SAM SEGMENTATION SUMMARY")
    print("="*50)
    print(f"Total cells detected: {cell_count}")
    
    # Calculate statistics
    regions = measure.regionprops(labeled_nuclei)
    if regions:
        areas = [r.area for r in regions]
        print(f"Average nucleus area: {np.mean(areas):.2f} pixels")
        print(f"Median nucleus area: {np.median(areas):.2f} pixels")
        print(f"Min nucleus area: {np.min(areas)} pixels")
        print(f"Max nucleus area: {np.max(areas)} pixels")
    
    print("="*50)


if __name__ == "__main__":
    main()

