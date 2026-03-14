# 🌍 Climate Trend & Extreme Weather Analysis using NASA POWER Data (2000–2024)

A Machine Learning and Deep Learning based climate analytics project that analyzes long-term climate trends and detects extreme weather events in Eastern India using NASA POWER satellite climate data.

This project combines **data science, climate analytics, machine learning, deep learning, anomaly detection, and interactive dashboards** to extract meaningful insights from 24 years of meteorological data.

---

## 🚀 Project Overview

Climate change is causing increasing temperature variability and extreme weather events across many regions.
This project analyzes historical climate patterns and builds predictive models to understand these changes.

Using NASA POWER daily meteorological data, this project performs:

* Climate trend analysis
* Extreme weather event detection
* Machine learning rainfall prediction
* Deep learning temperature forecasting
* Climate anomaly detection
* Interactive data visualization dashboard

---

## 📊 Dataset

Source: NASA POWER Climate Data

Dataset contains daily climate observations from **2000–2024**.

Features include:

| Variable    | Description               |
| ----------- | ------------------------- |
| T2M         | Average temperature at 2m |
| T2M_MAX     | Maximum daily temperature |
| T2M_MIN     | Minimum daily temperature |
| RH2M        | Relative humidity         |
| WS10M       | Wind speed at 10 meters   |
| PRECTOTCORR | Daily precipitation       |

---

## 🧠 Project Pipeline

```
NASA POWER Climate Data
        │
        ▼
Data Cleaning & Preprocessing
        │
        ▼
Feature Engineering
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Climate Trend Detection
        │
        ▼
Extreme Weather Event Analysis
        │
        ▼
Machine Learning Models
        │
        ▼
Deep Learning Forecasting (LSTM)
        │
        ▼
Climate Anomaly Detection
        │
        ▼
Interactive Streamlit Dashboard
```

---

## 🤖 Machine Learning Models

The following models are implemented:

* Random Forest Regression
* Gradient Boosting
* Deep Learning LSTM for temperature forecasting
* Isolation Forest for anomaly detection

Evaluation metrics used:

* RMSE
* R² Score

---

## 📈 Key Analysis Performed

* Long-term temperature trend analysis
* Rainfall variability study
* Heatwave detection
* Extreme rainfall detection
* Climate anomaly detection
* Predictive modeling of precipitation

---

## 📊 Dashboard

An interactive dashboard built using **Streamlit** allows visualization of:

* Temperature trends
* Rainfall trends
* Extreme weather events
* Model predictions
* Climate anomalies

Run dashboard locally:

```
streamlit run dashboard/app.py
```

---

## 🛠 Tech Stack

Languages & Libraries

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* TensorFlow / Keras
* Streamlit
* SHAP (Explainable AI)

Tools

* Git
* GitHub
* VS Code

---

## 📂 Project Structure

```
climate-trend-analysis-ml
│
├── data
│   └── nasa_power_dataset.csv
│
├── notebooks
│   └── climate_analysis.ipynb
│
├── src
│   ├── preprocessing.py
│   ├── eda.py
│   ├── ml_model.py
│   ├── lstm_model.py
│   └── anomaly_detection.py
│
├── dashboard
│   └── app.py
│
├── requirements.txt
└── README.md
```

---

## 📌 Future Improvements

* Transformer-based time series forecasting
* Climate risk prediction models
* Multi-location climate analysis
* Advanced geospatial visualization

---

## 📜 License

This project is for educational and research purposes.

---

## ⭐ If you found this project interesting

Give the repository a ⭐ and feel free to contribute!
