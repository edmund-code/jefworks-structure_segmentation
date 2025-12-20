#!/usr/bin/env python3
"""
Glomeruli Segmentation using Meta's Segment Anything Model (SAM)
This script uses SAM to segment glomeruli in H&E stained histopathology images of kidney tissue.
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
from skimage import io
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
import functools

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


def load_sam_model(model_type='vit_b', device=None, num_threads=5):
    """
    Load SAM model.
    
    Parameters:
    -----------
    model_type : str
        Model type: 'vit_h', 'vit_l', or 'vit_b'
    device : str
        Device to run on ('cuda' or 'cpu')
    num_threads : int
        Number of CPU threads to use for PyTorch operations
    
    Returns:
    --------
    model : SAM model
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Set number of threads for PyTorch CPU operations
    if device == "cpu":
        torch.set_num_threads(num_threads)
        print(f"Using {num_threads} CPU threads for PyTorch operations")
    
    print(f"Loading SAM model ({model_type}) on {device}...")
    
    checkpoint_dir = 'checkpoints'
    checkpoint_path = download_sam_checkpoint(model_type, checkpoint_dir)
    
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    
    print("SAM model loaded successfully!")
    return sam, device


def generate_glomeruli_prompts(image_rgb, num_points=50):
    """
    Generate prompt points for glomeruli using color deconvolution.
    Glomeruli typically appear as circular/oval structures with distinct boundaries.
    
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
    eosin = hed[:, :, 1]
    
    # Glomeruli often have distinct boundaries and appear as darker regions
    # Combine hematoxylin and eosin channels to identify glomerular structures
    combined = (hematoxylin + eosin) / 2
    
    # Normalize
    combined_norm = (combined - combined.min()) / (combined.max() - combined.min() + 1e-10)
    
    # Use a threshold to find potential glomerular regions
    # Glomeruli are typically medium to high intensity structures
    threshold = np.percentile(combined_norm, 60)  # Middle to high intensity
    binary = combined_norm > threshold
    
    # Apply morphological operations to find larger structures
    kernel = morphology.disk(5)
    binary = morphology.binary_closing(binary, kernel)
    binary = morphology.binary_opening(binary, kernel)
    
    # Find centroids of connected components (potential glomeruli)
    labeled = measure.label(binary)
    regions = measure.regionprops(labeled)
    
    points = []
    for region in regions:
        # Glomeruli are typically larger structures (1000-50000 pixels depending on magnification)
        if 1000 <= region.area <= 50000:
            y, x = region.centroid
            points.append([int(x), int(y)])
    
    # If we have fewer points than requested, sample more from high-intensity regions
    if len(points) < num_points:
        coords = np.column_stack(np.where(combined_norm > threshold))
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


def _filter_mask_by_size(mask_data, min_size, max_size):
    """Helper function for parallel mask filtering."""
    area = mask_data['area'] if 'area' in mask_data else np.sum(mask_data['segmentation'])
    if min_size <= area <= max_size:
        return mask_data
    return None


def _check_mask_overlap(args):
    """Helper function for parallel overlap checking."""
    mask_data, existing_mask, overlap_threshold = args
    mask = mask_data['segmentation']
    area = np.sum(mask)
    if area == 0:
        return None, 0
    
    overlap = np.sum(existing_mask & mask) / area
    if overlap < overlap_threshold:
        return mask_data, area
    return None, 0


def _filter_region(region, min_size, max_size):
    """Helper function for parallel region filtering."""
    if min_size <= region.area <= max_size:
        if region.major_axis_length > 0:
            circularity = 4 * np.pi * region.area / (region.perimeter ** 2 + 1e-10)
            if circularity > 0.2:
                return region.label
    return None


def segment_glomeruli_sam(image_path, model_type='vit_b', use_auto_mask=True, min_glomerulus_size=1000, max_glomerulus_size=50000, num_workers=5):
    """
    Segment glomeruli using Meta's Segment Anything Model.
    
    Parameters:
    -----------
    image_path : str
        Path to the H&E stained image
    model_type : str
        SAM model type: 'vit_h' (best), 'vit_l', or 'vit_b' (fastest)
    use_auto_mask : bool
        If True, use automatic mask generation. If False, use prompt-based.
    min_glomerulus_size : int
        Minimum size (in pixels) for a valid glomerulus
    max_glomerulus_size : int
        Maximum size (in pixels) for a valid glomerulus
    
    Returns:
    --------
    glomerulus_count : int
        Number of detected glomeruli
    glomeruli_mask : numpy array
        Binary mask of segmented glomeruli
    labeled_glomeruli : numpy array
        Labeled image where each glomerulus has a unique ID
    all_masks : list
        List of all SAM masks
    """
    # Load the image (handle both .tif and other formats)
    print(f"Loading image from {image_path}...")
    image_path_str = str(image_path)
    
    if image_path_str.lower().endswith('.tif') or image_path_str.lower().endswith('.tiff'):
        # Use skimage for TIFF files
        image = io.imread(image_path_str)
        # Handle different TIFF formats (grayscale, RGB, RGBA)
        if len(image.shape) == 2:
            # Grayscale, convert to RGB
            image_rgb = np.stack([image, image, image], axis=-1)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            # RGBA, convert to RGB
            image_rgb = image[:, :, :3]
        elif len(image.shape) == 3 and image.shape[2] == 3:
            # Already RGB
            image_rgb = image
        else:
            raise ValueError(f"Unsupported image format with shape {image.shape}")
        
        # Ensure uint8
        if image_rgb.dtype != np.uint8:
            if image_rgb.max() <= 1.0:
                image_rgb = (image_rgb * 255).astype(np.uint8)
            else:
                image_rgb = image_rgb.astype(np.uint8)
    else:
        # Use OpenCV for other formats
        image = cv2.imread(image_path_str)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    height, width = image_rgb.shape[:2]
    print(f"Image dimensions: {width}x{height}")
    
    # Limit number of workers to available CPUs
    num_workers = min(num_workers, cpu_count())
    print(f"Using {num_workers} parallel workers")
    
    # Load SAM model
    sam, device = load_sam_model(model_type, num_threads=num_workers)
    
    if use_auto_mask:
        # Use automatic mask generation with parameters optimized for larger objects
        print("Generating masks with SAM automatic mask generator...")
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=32,
            pred_iou_thresh=0.88,  # Higher threshold for better quality
            stability_score_thresh=0.95,  # Higher threshold for stability
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=min_glomerulus_size,  # Filter small objects
        )
        
        masks = mask_generator.generate(image_rgb)
        print(f"SAM generated {len(masks)} candidate masks")
        
    else:
        # Use prompt-based segmentation
        print("Generating prompt points for glomeruli...")
        points, labels = generate_glomeruli_prompts(image_rgb, num_points=100)
        print(f"Generated {len(points)} prompt points")
        
        predictor = SamPredictor(sam)
        predictor.set_image(image_rgb)
        
        masks = []
        # Process points in batches
        batch_size = 20  # Smaller batches for larger objects
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
    
    # Filter masks by size in parallel
    print("Filtering and processing masks (parallel)...")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        filter_func = functools.partial(_filter_mask_by_size, 
                                       min_size=min_glomerulus_size, 
                                       max_size=max_glomerulus_size)
        filtered_results = list(executor.map(filter_func, masks))
    
    filtered_masks = [m for m in filtered_results if m is not None]
    print(f"After size filtering: {len(filtered_masks)} masks")
    
    # Create combined mask with parallel overlap checking
    # Sort masks by area (largest first) to prioritize larger glomeruli
    filtered_masks.sort(key=lambda x: x['area'] if 'area' in x else np.sum(x['segmentation']), reverse=True)
    
    glomeruli_mask = np.zeros((height, width), dtype=bool)
    mask_areas = []
    
    # Process masks in batches to check overlap
    batch_size = max(1, len(filtered_masks) // num_workers)
    for i in range(0, len(filtered_masks), batch_size):
        batch = filtered_masks[i:i+batch_size]
        
        for mask_data in batch:
            mask = mask_data['segmentation']
            area = np.sum(mask)
            
            # Check overlap with existing mask
            overlap = np.sum(glomeruli_mask & mask) / area if area > 0 else 0
            
            # Only add if overlap is less than 40% (to avoid duplicates)
            if overlap < 0.4:
                glomeruli_mask |= mask
                mask_areas.append(area)
    
    # Label connected components
    labeled_glomeruli = measure.label(glomeruli_mask)
    
    # Filter labeled regions by size again (parallel)
    print("Filtering regions by size and circularity (parallel)...")
    regions = measure.regionprops(labeled_glomeruli)
    
    if len(regions) > 0:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            filter_func = functools.partial(_filter_region,
                                          min_size=min_glomerulus_size,
                                          max_size=max_glomerulus_size)
            valid_labels = list(executor.map(filter_func, regions))
        
        valid_labels = [l for l in valid_labels if l is not None]
        final_mask = np.zeros_like(glomeruli_mask, dtype=bool)
        for label in valid_labels:
            final_mask[labeled_glomeruli == label] = True
    else:
        final_mask = np.zeros_like(glomeruli_mask, dtype=bool)
    
    # Re-label
    labeled_glomeruli = measure.label(final_mask)
    glomerulus_count = labeled_glomeruli.max()
    
    print(f"\nSegmentation complete!")
    print(f"Number of detected glomeruli: {glomerulus_count}")
    
    return glomerulus_count, final_mask, labeled_glomeruli, filtered_masks, image_rgb


def visualize_results_sam(image_rgb, glomeruli_mask, labeled_glomeruli, glomerulus_count, masks=None, save_path=None):
    """
    Visualize the SAM segmentation results for glomeruli.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    # Original image
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title('Original H&E Kidney Image', fontsize=12)
    axes[0, 0].axis('off')
    
    # Show some SAM masks (if available)
    if masks is not None and len(masks) > 0:
        mask_overlay = image_rgb.copy()
        # Show first 30 masks with different colors
        for i, mask_data in enumerate(masks[:30]):
            mask = mask_data['segmentation']
            color = np.random.rand(3)
            mask_overlay[mask] = mask_overlay[mask] * 0.6 + np.array(color) * 255 * 0.4
        
        axes[0, 1].imshow(mask_overlay.astype(np.uint8))
        axes[0, 1].set_title(f'SAM Masks Overlay\n({min(30, len(masks))} masks shown)', fontsize=12)
    else:
        axes[0, 1].imshow(image_rgb)
        axes[0, 1].set_title('Original Image', fontsize=12)
    axes[0, 1].axis('off')
    
    # Binary mask
    axes[1, 0].imshow(glomeruli_mask, cmap='gray')
    axes[1, 0].set_title(f'Final Segmented Glomeruli Mask\n({glomerulus_count} glomeruli detected)', fontsize=12)
    axes[1, 0].axis('off')
    
    # Overlay on original image
    overlay = image_rgb.copy()
    glomeruli_colored = np.zeros_like(image_rgb)
    glomeruli_colored[glomeruli_mask] = [0, 255, 0]  # Green overlay for glomeruli
    overlay = cv2.addWeighted(overlay, 0.7, glomeruli_colored, 0.3, 0)
    
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title(f'Overlay: Original + Segmented Glomeruli\n({glomerulus_count} glomeruli)', fontsize=12)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to {save_path}")
    
    plt.close()


def main():
    # Path to the image
    image_path = Path("data/Ctrl_1A2_sub.tif")
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    # Segment glomeruli using SAM
    # Use 'vit_b' for faster processing, 'vit_l' for better quality, 'vit_h' for best quality
    glomerulus_count, glomeruli_mask, labeled_glomeruli, masks, image_rgb = segment_glomeruli_sam(
        image_path,
        model_type='vit_b',  # Change to 'vit_l' or 'vit_h' for better results (slower)
        use_auto_mask=True,  # Use automatic mask generation
        min_glomerulus_size=1000,   # Minimum glomerulus size in pixels
        max_glomerulus_size=50000,  # Maximum glomerulus size in pixels
        num_workers=5  # Number of parallel workers (up to 5 cores)
    )
    
    # Visualize results
    output_path = Path("glomeruli_segmentation_sam_results.png")
    visualize_results_sam(image_rgb, glomeruli_mask, labeled_glomeruli, glomerulus_count, masks, save_path=output_path)
    
    # Print summary statistics
    print("\n" + "="*50)
    print("SAM GLOMERULI SEGMENTATION SUMMARY")
    print("="*50)
    print(f"Total glomeruli detected: {glomerulus_count}")
    
    # Calculate statistics
    regions = measure.regionprops(labeled_glomeruli)
    if regions:
        areas = [r.area for r in regions]
        print(f"Average glomerulus area: {np.mean(areas):.2f} pixels")
        print(f"Median glomerulus area: {np.median(areas):.2f} pixels")
        print(f"Min glomerulus area: {np.min(areas)} pixels")
        print(f"Max glomerulus area: {np.max(areas)} pixels")
        
        # Calculate circularity statistics
        circularities = []
        for r in regions:
            if r.perimeter > 0:
                circ = 4 * np.pi * r.area / (r.perimeter ** 2)
                circularities.append(circ)
        if circularities:
            print(f"Average circularity: {np.mean(circularities):.3f}")
            print(f"Median circularity: {np.median(circularities):.3f}")
    
    print("="*50)


if __name__ == "__main__":
    main()

