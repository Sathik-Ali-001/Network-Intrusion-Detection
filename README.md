# Network Intrusion Detection System Using Ensemble Machine Learning & Feature Selection

A robust **Network Intrusion Detection System (NIDS)** built with Ensemble Learning (Random Forest + XGBoost) and Feature Selection techniques using the **NSL-KDD** dataset.

---

## 🚀 Features

- **Ensemble Learning**: Combines Random Forest and XGBoost using Soft Voting Classifier
- **Feature Selection**: SelectKBest (ANOVA F-value) to select top 25 most relevant features
- **High Performance**: Achieves excellent accuracy on NSL-KDD dataset
- **Preprocessing**: Handles categorical encoding, scaling, and cleaning
- **Easy Prediction**: Ready-to-use prediction script for new network flows
- **Model Persistence**: Saves trained model, scaler, and selector for deployment

---


---

## 🛠️ Installation & Setup

### 1. Clone or Download the Project
```bash
git clone https://github.com/Sathik-Ali-001/Network-Intrusion-Detection
cd Network-Intrusion-Detection
```
# Install Dependencies
pip install pandas numpy scikit-learn xgboost joblib


# How to Run
Training the Model
python main.py


# Making Predictions
python predict.py
