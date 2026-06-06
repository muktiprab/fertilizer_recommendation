<div align="center">

# 🌱 Fertilizer Recommendation System

![Preview](https://raw.githubusercontent.com/muktiprab/fertilizer_recommendation/source-code/Preview%20Fertilizer%20Recommendation.png)

[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B?logo=streamlit&logoColor=white)](https://fertilizer-recommendation-project.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-F7931E?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0-189fdd?logo=xgboost&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-deployed-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

A machine learning web application that recommends the most suitable fertilizer for crops based on soil conditions, environmental factors, and nutrient levels — powered by Random Forest and XGBoost.

**[🚀 Live Demo](https://fertilizer-recommendation-project.streamlit.app/)**

</div>

---

## 📌 About

Agriculture is one of the world's most important industries. Choosing the right fertilizer can significantly improve crop yield and soil health. This project builds a fertilizer recommendation system using machine learning, trained on data from Maharashtra, India, covering various soil types, crops, and environmental conditions.

The application compares two classification models — **Random Forest** and **XGBoost** — and presents predictions with confidence scores, probability charts, and detailed model evaluation metrics.

---

## ✨ Features

- **Fertilizer Prediction** — Input soil and environmental data to get fertilizer recommendations from both models simultaneously
- **Confidence Score** — Each prediction comes with a probability score so users can gauge model certainty
- **Top-5 Probability Chart** — Visual breakdown of the top 5 fertilizer candidates per model
- **Model Comparison** — Side-by-side evaluation of Random Forest vs XGBoost (F1-Macro, ROC-AUC, 5-Fold Cross-Validation)
- **Per-class F1 Analysis** — Detailed precision, recall, and F1-score for each fertilizer class
- **Data Analysis** — Distribution charts for fertilizer classes, numerical features, correlation matrix, and crop types

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.11 |
| Machine Learning | scikit-learn, XGBoost |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Web App | Streamlit |
| Model Storage | Google Drive + gdown |
| Deployment | Streamlit Community Cloud |
| Version Control | Git, GitHub |

---

## 📊 Dataset

- **Source:** [Crop and Fertilizer Dataset](https://raw.githubusercontent.com/MutiaraCR/Dataset/refs/heads/main/Crop%20and%20fertilizer%20dataset.csv)
- **Context:** Agricultural data from districts in Maharashtra, West India
- **Records:** 4,513 rows
- **Target:** 19 fertilizer classes
- **Features:** District, Soil Color, Nitrogen, Phosphorus, Potassium, pH, Rainfall, Temperature, Crop

---

## 🤖 Models

| Model | Macro F1 | ROC-AUC |
|---|---|---|
| Random Forest | 0.9303 | 0.9976 |
| XGBoost | 0.8493 | 0.9981 |

Both models are trained with an 80/20 train-test split (stratified) and evaluated using 5-fold cross-validation.

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/muktiprab/fertilizer_recommendation.git
cd fertilizer_recommendation
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

> Model files (`rf_model.pkl`, `xgb_model.pkl`, `label_encoder.pkl`) are automatically downloaded from Google Drive on first run.

---

## 📁 Project Structure

```
fertilizer_recommendation/
├── app.py                  # Streamlit application
├── requirements.txt        # Python dependencies
├── .gitignore              # Excludes pkl files from Git
├── .python-version         # Python version config
└── README.md
```

---

## 👥 Team

This is a collaborative project between two Computer Science students at Universitas Lampung.

<table>
  <tr>
    <td align="center">
      <b>Mutiara Cintia Rainy</b><br>
      <i>Data Analysis & Modelling</i><br>
      EDA, feature engineering, model training (RF, XGBoost, CatBoost, MLP), and evaluation<br><br>
      <a href="https://github.com/MutiaraCR">
        <img src="https://img.shields.io/badge/GitHub-MutiaraCR-181717?logo=github" />
      </a>
      <a href="https://linkedin.com/in/mutiara-cintia-rainy-3a5969266">
        <img src="https://img.shields.io/badge/LinkedIn-Mutiara-0A66C2?logo=linkedin" />
      </a>
    </td>
    <td align="center">
      <b>Mukti Prabowo</b><br>
      <i>Deployment & Web App</i><br>
      Streamlit app development, model integration, UI/UX, and deployment to Streamlit Community Cloud<br><br>
      <a href="https://github.com/muktiprab">
        <img src="https://img.shields.io/badge/GitHub-muktiprab-181717?logo=github" />
      </a>
      <a href="https://linkedin.com/in/muktiprabowo">
        <img src="https://img.shields.io/badge/LinkedIn-Mukti-0A66C2?logo=linkedin" />
      </a>
    </td>
  </tr>
</table>

---

## 📄 License

This project is licensed under the MIT License.
