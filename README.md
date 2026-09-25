# 🌱 Crop Health Triage

AI-powered crop disease triage system that analyzes crop leaf images using lightweight deep learning and provides disease predictions and related information through a mobile application and web application.

## 🚀 Overview

Crop Health Triage is an end-to-end AI solution developed as a team hackathon project.

The system allows a user to upload or capture a crop leaf image through the Android or web application. The image is sent to the backend, where a lightweight deep learning pipeline analyzes the image and returns the predicted crop disease, confidence, and relevant disease information.

## 🏗️ System Architecture

```text
                    ┌──────────────────┐
                    │      User        │
                    └────────┬─────────┘
                             │
                 ┌───────────┴───────────┐
                 │                       │
                 ▼                       ▼
        ┌────────────────┐      ┌────────────────┐
        │  Android App   │      │    Web App     │
        └───────┬────────┘      └───────┬────────┘
                │                       │
                └───────────┬───────────┘
                            │
                            ▼
                  ┌──────────────────┐
                  │   FastAPI Server │
                  └────────┬─────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Leaf / Non-Leaf     │
                │     Classifier      │
                └─────────┬───────────┘
                          │
                     Leaf Image
                          │
                          ▼
                ┌─────────────────────┐
                │ Disease Classifier │
                └─────────┬───────────┘
                          │
                          ▼
                ┌─────────────────────┐
                │ Disease Information│
                └─────────┬───────────┘
                          │
                          ▼
                ┌─────────────────────┐
                │ JSON API Response  │
                └─────────────────────┘
````

## ✨ Key Features

* 📷 Crop leaf image upload
* 🤖 AI-based crop disease detection
* 🌿 Leaf / non-leaf image verification
* 🧠 Lightweight MobileNetV3-Small models
* 📊 Prediction confidence
* 🔝 Top prediction results
* 📚 Disease information
* 📱 Android mobile application
* 🌐 Web application
* ⚡ FastAPI inference backend
* 🔌 REST API integration
* 🧩 Modular backend, mobile, and web architecture

## 🤖 Machine Learning Pipeline

The project uses a two-stage image classification approach.

### Stage 1 — Leaf Detection

The first model determines whether the submitted image is a suitable crop leaf image.

```text
Input Image
     ↓
Leaf Classifier
     ↓
LEAF / NOT_LEAF
```

### Stage 2 — Disease Classification

If the image is classified as a leaf, it is passed to the disease classifier.

```text
Leaf Image
    ↓
Disease Classifier
    ↓
Crop / Disease Class
    ↓
Confidence
    ↓
Disease Information
```

### Model

The project uses **MobileNetV3-Small** as the lightweight image classification architecture.

This approach was selected to keep the inference model relatively lightweight while allowing the project to run through a server-based API architecture.

## 🌾 Dataset

The training workflow uses the **PlantVillage Full** dataset.

Dataset source:

[https://huggingface.co/datasets/geraldmc/plantvillage-full](https://huggingface.co/datasets/geraldmc/plantvillage-full)

The raw dataset is not included in this repository.

The repository contains the training code and trained model artifacts required by the application.

## 🧪 Supported Classes

The disease classifier is based on the 38 classes represented by the project's `class_names.json`.

The classes cover crops including:

* Apple
* Blueberry
* Cherry
* Corn
* Grape
* Orange
* Peach
* Pepper
* Potato
* Raspberry
* Soybean
* Squash
* Strawberry
* Tomato

The exact class mapping used by the application is stored in:

```text
backend/model/class_names.json
```

## 📁 Project Structure

```text
Crop-health-triage/
│
├── backend/
│   ├── model/
│   │   ├── best_model.pth
│   │   ├── leaf_classifier.pth
│   │   ├── class_names.json
│   │   └── leaf_class_names.json
│   │
│   ├── server.py
│   ├── disease_info.json
│   └── requirements.txt
│
├── mobile-app/
│   └── Android application
│
├── web-app/
│   └── Web application
│
├── training/
│   └── Machine learning training and evaluation scripts
│
├── README.md
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── SECURITY.md
├── CITATION.cff
└── LICENSE
```

## 🔧 Backend

The backend is responsible for:

* receiving image uploads
* validating images
* running the leaf classifier
* running the disease classifier
* calculating prediction confidence
* retrieving disease information
* returning structured JSON responses

### API Endpoints

```text
GET  /
GET  /health
GET  /diseases
GET  /disease/{disease_name}
POST /predict
```

### Example Prediction Flow

```text
POST /predict
      │
      ▼
  Image Upload
      │
      ▼
Leaf / Non-Leaf Check
      │
      ├── NOT_LEAF
      │      ↓
      │   Request Clear Leaf Image
      │
      └── LEAF
             ↓
      Disease Classification
             ↓
      Prediction + Confidence
             ↓
      Disease Information
```

## 📱 Android Application

The Android application provides the mobile interface for the Crop Health Triage system.

Users can:

1. Capture or select a crop image.
2. Send the image to the backend API.
3. Receive the prediction.
4. View disease information and confidence.

The Android application is located in:

```text
mobile-app/
```

The backend API URL should be configured according to the deployment environment.

Do not commit private credentials or local Android configuration files.

## 🌐 Web Application

The web application provides a browser-based interface for crop disease analysis.

Users can:

1. Upload a crop image.
2. Send the image to the backend.
3. View the prediction.
4. View confidence and disease information.

The web application is located in:

```text
web-app/
```

## ⚙️ Backend Installation

Clone the repository:

```bash
git clone https://github.com/Siva-kumar-r16/Crop-health-triage.git
```

Move into the backend:

```bash
cd Crop-health-triage/backend
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the server:

```bash
python server.py
```

The server can then be accessed through the configured local or deployment URL.

## 🔌 API Example

### Health Check

```http
GET /health
```

Example response:

```json
{
  "success": true,
  "status": "healthy"
}
```

### Prediction

```http
POST /predict
```

Send the image as multipart form data:

```text
file = crop_leaf.jpg
```

The response contains information such as:

```json
{
  "success": true,
  "status": "diseased",
  "prediction": "Tomato___Early_blight",
  "confidence": 0.91,
  "top_predictions": []
}
```

The exact response fields may change as the application evolves.

## 📊 Model Evaluation

The project includes scripts for evaluating the trained models.

Model evaluation should be interpreted in the context of the dataset and validation/test setup used during development. Performance on controlled datasets may differ from performance on real-world field photographs.

## 🔐 Security

This repository is public.

Do not commit:

```text
.env
API keys
Access tokens
Passwords
Cloud credentials
Private keys
Android signing keys
local.properties
Private certificates
Deployment credentials
```

See [`SECURITY.md`](SECURITY.md) for the project's security policy.

## ⚠️ Limitations

Crop Health Triage is a hackathon/research prototype.

Predictions can be affected by:

* image quality
* lighting
* camera quality
* background interference
* crop variety
* disease stage
* leaf orientation
* symptoms that are visually similar
* differences between controlled training images and real-world field images

The prediction should therefore be treated as a **triage and decision-support result**, not as a guaranteed agricultural diagnosis.

For important crop-management decisions, users should consult qualified agricultural professionals.

## 🔮 Future Improvements

Potential future improvements include:

* improved real-world field-image robustness
* additional crops and diseases
* stronger non-leaf rejection
* improved uncertainty detection
* model optimization for low-resource environments
* multilingual farmer guidance
* larger field-image evaluation datasets
* improved explainability
* offline or edge inference
* improved treatment and prevention knowledge base

# 👥 Team

Crop Health Triage was developed collaboratively by a five-member team during a hackathon.

| Member               | Role                              | Email                                                                   | GitHub                                                  | Portfolio                                                      |
| -------------------- | --------------------------------- | ----------------------------------------------------------------------- | ------------------------------------------------------- | -------------------------------------------------------------- |
| **Siva Kumar R**     | Team Lead & Android App Developer | [sirsivakumar@outlook.com](mailto:sirsivakumar@outlook.com)             | [Siva-kumar-r16](https://github.com/Siva-kumar-r16)     | [Portfolio](https://sivakumars-portfolio.netlify.app/)         |
| **Seprin Smith S**   | Server / Backend Developer        | [seprinsmith4@gmail.com](mailto:seprinsmith4@gmail.com)                 | [Seprin-Smith-Git](https://github.com/Seprin-Smith-Git) | —                                                              |
| **Santhana Kevin P** | Web Application Developer         | [kevin0412it@gmail.com](mailto:kevin0412it@gmail.com)                   | [Kevin-Stark47](https://github.com/Kevin-Stark47)       | —                                                              |
| **Saminathan S M**   | Testing & Debugging               | [saminathanmurugu2007@gmail.com](mailto:saminathanmurugu2007@gmail.com) | [saminathan2007](https://github.com/saminathan2007)     | [Portfolio](https://saminathanportfolio.studylen.workers.dev/) |
| **Sharan P**         | R&D & Resource Collection         | [sharan12318@gmail.com](mailto:sharan12318@gmail.com)                   | [Sharan242007](https://github.com/Sharan242007)         | [Portfolio](https://sharan-portfolio-2407.netlify.app/)        |

## 🏆 Hackathon Project

Crop Health Triage was developed as a collaborative hackathon project.

The team worked across:

* Machine Learning
* Backend Development
* Android Development
* Web Development
* Testing & Debugging
* Research & Development
* Resource Collection
* System Integration

## 📚 Documentation

Additional project documentation:

* [`CONTRIBUTING.md`](CONTRIBUTING.md) — Contribution guidelines
* [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) — Community guidelines
* [`SECURITY.md`](SECURITY.md) — Security policy
* [`CITATION.cff`](CITATION.cff) — Citation information
* [`LICENSE`](LICENSE) — MIT License

## 📄 License

This project is licensed under the MIT License.

See [`LICENSE`](LICENSE) for details.

## 📌 Disclaimer

Crop Health Triage is provided as a software and machine-learning prototype for crop disease triage and experimentation.

The system does not guarantee disease identification or treatment outcomes. Users should verify important agricultural decisions with appropriate agricultural experts.

---

**Crop Health Triage — AI-assisted crop disease triage for smarter agricultural decision support.** 🌱

```
```
