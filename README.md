# jefworks-structure_segmentation

This repository contains pipelines for the semantic segmentation of kidney structures, specifically glomeruli and erythrocytes (RBCs), utilizing high-resolution whole slide images (WSI) and spatial transcriptomics data. The project leverages deep learning models including DINOv2 and SAM (Segment Anything Model) adapters.

## Repository Structure

### 1. `glomeruli_segmentation/`
This directory contains the primary workflow for segmenting glomeruli using a multi-stage approach guided by gene expression markers.

*   **`final_glomeruli_segmentation.ipynb`**: **[Core Pipeline]** An end-to-end notebook that performs:
    *   Alignment of Visium spatial transcriptomics data with WSI (NDPI/TIFF).
    *   Identification of glomerular candidates using specific gene markers (e.g., `NPHS1`, `PODXL`).
    *   Classification of candidates using a linear model on top of DINOv2 embeddings.
    *   Final segmentation refinement using the Medical SAM Adapter.
*   **`train_glom_dinov2_linear.py`**: A Python script to train the linear classification head used in the pipeline to filter false positive glomeruli candidates.
*   **`label_glomeruli_app.py`**: A GUI application to manually label glomeruli bounding boxes for training or validation.
*   **`medsam_adaptor.ipynb`**: Notebook for experiments and fine-tuning of the MedSAM adapter for renal structures.
*   **`out/`**: Directory containing pipeline outputs, including intermediate CSVs, GeoJSONs, and validation scores.

### 2. `rbc_pipeline/`
A separate pipeline dedicated to the segmentation of Red Blood Cells (RBCs).

*   **`extract_tile.ipynb`**: Notebook for preprocessing Whole Slide Images (WSI) and extracting image tiles for analysis.
*   **`segment_rbcs.ipynb`**: Notebook implementing the segmentation logic for identifying RBCs within the extracted tiles.

### 3. `Medical-SAM-Adapter/`
Contains the implementation of the parameter-efficient adapter for the Segment Anything Model (SAM), customized for medical image segmentation tasks.

### 4. `checkpoints/`
Stores model weights and checkpoints:
*   `best_dice_checkpoint.pth`: Best performing SAM adapter checkpoint.
*   `glom_dinov2_linear.pt`: Trained linear head for DINOv2 glomeruli classification.

### 5. `data/` & `Qupath_Data/`
*   **`data/`**: Stores input data including Visium spatial data and raw NDPI/TIFF slides.
*   **`Qupath_Data/`**: Project files for QuPath integration and visualization.

---

## Usage

To use the glomeruli segmentation pipeline:

1.  **Environment**: Ensure all dependencies (`torch`, `openslide`, `squidpy`, `segment_anything`) are installed.
2.  **Configuration**: Open `glomeruli_segmentation/final_glomeruli_segmentation.ipynb`.
3.  **Setup**: Edit the first code cell to set the `SAMPLE_NAME`, `BASE_DIR`, and relevant paths for your local environment.
4.  **Execution**: Run the notebook cells in order. The pipeline reads aligned points and WSI data to produce segmented GeoJSON overlays.