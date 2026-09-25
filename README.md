# Crop Health Triage 🌱

AI-powered crop disease triage from leaf images, developed as a team hackathon project.

## Overview

Crop Health Triage is an end-to-end system that accepts a crop-leaf image and uses lightweight deep learning to identify the likely crop disease. The project combines a machine-learning inference backend, Android mobile application, and web application.

```text
Crop-health-triage/
├── backend/       # ML pipeline, trained models and API
├── mobile-app/    # Android application
└── web-app/       # Web application
```

## How It Works

```text
User
  │
  ├── Android App
  │
  └── Web App
        │
        ▼
   FastAPI Backend
        │
        ▼
 Leaf / Non-Leaf Check
        │
        ▼
 Crop Disease Classifier
        │
        ▼
 Disease Information
        │
        ▼
 Prediction + Confidence + Guidance
```

The backend uses a two-stage inference pipeline:
1. A lightweight leaf classifier checks whether the submitted image is suitable for leaf-based analysis.
2. A disease classification model predicts the crop/disease class.
3. The API returns the prediction, confidence, top predictions, and available disease information.

## Machine Learning

The backend contains the end-to-end ML workflow:
- `01_dataset_analysis.py` — dataset analysis and class distribution checks
- `02_prepare_dataset.py` — preprocessing, augmentation, and dataset preparation
- `03_train_leaf_classifier.py` — leaf classification model training
- `04_train_disease_model.py` — crop disease model training
- `05_evaluate.py` — model evaluation
- `06_predict.py` — inference
- `07_test_pipeline.py` — end-to-end pipeline testing
- `disease_info.json` — disease-related information

Generated model artifacts include:
- `best_model.pth`
- `leaf_classifier.pth`
- `class_names.json`
- `leaf_class_names.json`

The project uses a lightweight MobileNetV3-Small-based approach for image classification.

## Applications

### Android Mobile App
The Android application provides a mobile interface for capturing or selecting a crop image and sending it to the backend for analysis.

### Web Application
The web application provides a browser-based interface for submitting crop images and viewing the returned analysis.

### Backend
The backend provides the inference API and model-serving layer.

## Dataset

The model training workflow uses the PlantVillage Full dataset:

https://huggingface.co/datasets/geraldmc/plantvillage-full

The raw dataset is intentionally not included in this repository. The repository contains the training pipeline and model artifacts rather than a copy of the full image dataset.

## API

Typical backend endpoints include:

```text
GET  /
GET  /health
GET  /diseases
GET  /disease/{disease_name}
POST /predict
```

The prediction endpoint accepts an image upload and returns structured JSON containing the analysis.

> API URLs should be configured for each deployment environment. Never commit private credentials or secret tokens.

## Installation

### Backend

```bash
cd backend
pip install -r requirements.txt
python server.py
```

### Android

Open `mobile-app/` in Android Studio and configure the backend API URL for the environment being used.

Do not commit `local.properties`.

### Web

Open `web-app/` and follow the project-specific setup instructions contained there.

## Security

This is a public repository. Never commit API keys, access tokens, passwords, cloud credentials, private certificates, Android signing keys, `.env` files containing secrets, `local.properties`, or private deployment credentials.

See [`SECURITY.md`](SECURITY.md).

## Limitations

This is a research/hackathon prototype and should not be treated as a guaranteed agricultural diagnosis. Image quality, lighting, crop variety, disease stage, background, and differences between controlled datasets and real-world field images can affect predictions.

Users should use the result as a triage/decision-support signal and consult qualified agricultural professionals when the diagnosis or treatment decision is uncertain.

## Team

| Member | Role |
|---|---|
| **Siva Kumar R** | Team Lead & Android App Developer |
| **Seprin Smith S** | Server / Backend Developer |
| **Santhana Kevin P** | Web Application Developer |
| **Saminathan S M** | Testing & Debugging |
| **Sharan P** | R&D & Resource Collection |

### Contact

- Siva Kumar R — sirsivakumar@outlook.com
- Seprin Smith S — seprinsmith4@gmail.com
- Santhana Kevin P — kevin0412it@gmail.com
- Saminathan S M — saminathanmurugu2007@gmail.com
- Sharan P — sharan12318@gmail.com

## Hackathon Project

Crop Health Triage was developed collaboratively as a hackathon project, with the team responsible for machine learning, backend integration, Android development, web development, testing, debugging, research, and resource collection.

## Future Improvements

- improve real-world field-image robustness
- expand crop and disease coverage
- improve non-leaf rejection
- strengthen uncertainty handling
- optimize inference for lower-resource deployment
- expand multilingual farmer-facing guidance
- add field-image testing
- improve model explainability

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE).

## Citation

If you use this project in academic or technical work, see [`CITATION.cff`](CITATION.cff).
