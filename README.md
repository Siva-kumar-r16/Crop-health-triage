# Crop Health Triage

This repository contains a complete machine learning pipeline for detecting and classifying crop diseases from leaf images. The backend is structured to handle everything from initial dataset analysis to model training, evaluation, and inference.

## 📁 Repository Structure

The core logic is located in the `backend/` directory, which contains a series of numbered Python scripts representing the end-to-end ML pipeline:

*   **`01_dataset_analysis.py`**: Analyzes the raw crop image dataset to verify class distribution and data integrity.
*   **`02_prepare_dataset.py`**: Preprocesses images, applies augmentations, and splits the data into training, validation, and testing sets.
*   **`03_train_leaf_classifier.py`**: Trains a preliminary model (`leaf_classifier.pth`) to identify leaf types or filter out non-leaf images.
*   **`04_train_disease_model.py`**: Trains the primary deep learning model (`best_model.pth`) to detect specific crop diseases.
*   **`05_evaluate.py`**: Tests the trained models against the validation/test sets and generates performance metrics.
*   **`06_predict.py`**: Contains the inference script to run the models on new, unseen crop images.
*   **`07_test_pipeline.py`**: Runs an end-to-end sanity check of the entire data preparation and inference pipeline.
*   **`disease_info.json`**: Stores metadata, treatment recommendations, or triage severity levels mapped to specific disease classifications.

### Generated Model Artifacts (`backend/model/`)
After running the training scripts, the following artifacts are saved:
*   `best_model.pth`: The weights for the primary crop disease classification model.
*   `leaf_classifier.pth`: The weights for the leaf detection/classification model.
*   `class_names.json`: The class index mapping for the disease model.
*   `leaf_class_names.json`: The class index mapping for the leaf classifier.

## ⚙️ Setup and Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/Siva-kumar-r16/Crop-health-triage.git](https://github.com/Siva-kumar-r16/Crop-health-triage.git)
   cd Crop-health-triage/backend
