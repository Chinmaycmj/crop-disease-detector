# Contributing to Crop Disease Detector

Thank you for your interest in contributing to the Crop Disease Detector project! Whether you're helping with technical code, UI design, documentation, or testing, we welcome your contributions.

---

## 🛠️ How to Get Started

### 1. Set Up Your Local Development Environment
1. Fork or clone this repository:
   git clone https://github.com/Chinmaycmj/crop-disease-detector.git
   cd crop-disease-detector

2. Create and activate a Python 3.11 virtual environment:
   py -3.11 -m venv venv
   .\venv\Scripts\activate

3. Install dependencies:
   pip install -r requirements.txt

4. Run the Streamlit app:
   python -m streamlit run app.py

---

## 🚀 Ways to Contribute

### 💻 Technical Tasks
* Model Accuracy & Optimization: Help tune hyperparameters, fine-tune MobileNetV2, or expand dataset coverage.
* Feature Enhancements: Integrate export capabilities, database logging, or REST API endpoints.
* Bug Fixes: Check the GitHub Issues tab for open bugs or UI glitches.

### 🎨 Non-Technical & Support Tasks
* Documentation & README: Improve setup instructions, add visual diagrams, or document code functions.
* Dataset Collection & Testing: Gather and organize labeled leaf images for validation across various lighting conditions.
* UI & UX Testing: Provide feedback on application flow, layout, and PDF report styling.

---

## 🔀 Workflow & Pull Requests

To keep the project main branch stable, please follow this branch-based workflow:

1. Create a Feature Branch:
   git checkout -b feature/your-feature-name

2. Make & Test Your Changes locally.

3. Commit & Push:
   git add .
   git commit -m "Description of changes made"
   git push origin feature/your-feature-name

4. Open a Pull Request (PR):
   * Go to the main GitHub repository.
   * Open a PR targeting the main branch with a short summary of your work.

---

## 📜 Code & Collaboration Standards
* Keep commit messages clear and descriptive.
* Test all Streamlit changes locally prior to submitting a Pull Request.
* Ensure no temporary virtual environment files (venv/) are committed.
*