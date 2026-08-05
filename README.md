# Crop Disease Detection & Care Assistant

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange?logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.60-red?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-Not%20Specified-lightgrey)

A Streamlit app that detects plant leaf diseases from an uploaded image using a locally trained TensorFlow/Keras model, and gives a treatment suggestion based on the diagnosis. All predictions run locally — no external API is used for the core detection flow.

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Model & Dataset](#model--dataset)
- [Installation](#installation)
- [Virtual Environment](#virtual-environment)
- [Installing Dependencies](#installing-dependencies)
- [Running the App](#running-the-app)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

## Overview

The user uploads a photo of a Pepper, Potato, or Tomato leaf. The app predicts whether the leaf is healthy or affected by a specific disease, and shows a matching care recommendation.

## How It Works

1. The app loads `model/plant_disease_model.h5` using `tf.keras.models.load_model()`.
2. The user uploads a `.jpg`, `.jpeg`, or `.png` image from the sidebar.
3. The image is resized to 224x224, converted to RGB, and normalized (0–1 range).
4. The model predicts a class, and the highest-probability class is picked.
5. The predicted class is mapped to a readable disease name.
6. A matching treatment tip is looked up from a local dictionary (no external API call).
7. The image, diagnosis, confidence score, and treatment plan are displayed side by side.

## Features

- Image upload (jpg, jpeg, png) via sidebar
- Offline disease classification using a local Keras model
- Confidence score for each prediction
- Local treatment recommendations for 15 classes (Pepper, Potato, Tomato)
- Two-column layout: image on one side, results on the other
- Error message shown if the model file is missing
- Model is cached so it loads only once per session

The repo also includes `test_env.py`, a script that tests connection to the Google Gemini API. It is not connected to `app.py` and is not part of the disease detection feature.

## Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.10+ |
| Web Framework | Streamlit 1.60.0 |
| ML Framework | TensorFlow 2.21.0 (Keras 3.15.1) |
| Image Handling | Pillow 12.3.0 |
| Numerical Computing | NumPy 2.4.6 |
| Dataset Retrieval | KaggleHub (used in `download_data.py`, not listed in requirements.txt) |
| Data Handling | Pandas 3.0.5 (in requirements.txt, not used directly in app.py) |

## Project Structure

```
crop-disease-detector/
├── model/
│   └── plant_disease_model.h5   # Trained model file (required, not included)
├── app.py                       # Main Streamlit app
├── download_data.py             # Downloads the PlantVillage dataset
├── train_model.py               # Model training script
├── test_env.py                  # Gemini API test (not used by app.py)
├── requirements.txt
├── .gitignore
└── README.md
```

## Model & Dataset

- Dataset: PlantVillage dataset (`emmarex/plantdisease` on Kaggle), downloaded via `download_data.py`.
- Classes: 15 classes covering Pepper, Potato, and Tomato (healthy and diseased).
- Model file: expected at `model/plant_disease_model.h5`. This file is not included in the repo and must be trained or added manually before running the app.
- Architecture: referred to as MobileNetV2 in a code comment; not confirmed against the training script.

## Installation

Clone the repo:

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

**Windows**
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**Linux**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**macOS**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Virtual Environment

Create it:
```bash
python -m venv venv        # Windows
python3 -m venv venv       # Linux / macOS
```

Activate it:
- Windows: `venv\Scripts\activate`
- Linux / macOS: `source venv/bin/activate`

Deactivate it:
```bash
deactivate
```

## Installing Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

To run `download_data.py`, also install kagglehub:
```bash
pip install kagglehub
```

## Running the App

```bash
streamlit run app.py
```

Open the local URL shown in the terminal (usually `http://localhost:8501`).

## Usage

1. Run `streamlit run app.py`.
2. Click "Choose a leaf image..." in the sidebar.
3. Select a jpg, jpeg, or png photo of a leaf.
4. Wait for the analysis to finish.
5. View the diagnosis, confidence score, and treatment plan.

## Troubleshooting

- **ModuleNotFoundError** — activate your virtual environment and run `pip install -r requirements.txt` again.
- **Model file not found** — place `plant_disease_model.h5` inside the `model/` folder.
- **kagglehub not found** — run `pip install kagglehub`.
- **Kaggle authentication error** — set up a Kaggle API token as described in Kaggle's documentation.
- **TensorFlow install issues on Apple Silicon** — try `tensorflow-macos` instead.
- **Port already in use** — run `streamlit run app.py --server.port 8502`.

## Future Improvements

- Add `kagglehub` to `requirements.txt`.
- Add validation for uploaded images.
- Improve the UI.
- Add bilingual support (Kannada and English).
- Add a PDF report download after prediction.

## Contributing

1. Fork the repo.
2. Create a branch: `git checkout -b feature/your-feature`
3. Commit your changes.
4. Push and open a Pull Request.

## License

No LICENSE file is included in this repository yet.