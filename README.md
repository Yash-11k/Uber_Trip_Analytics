

A professional Machine Learning application designed to predict Uber trip distances (miles) using high-fidelity predictive modeling. This project transitions from raw data exploration to a live, interactive dashboard.

## 🚀 Live Demo
https://ubertripanalytics.streamlit.app/

---

## 📌 Project Overview
The objective of this project is to analyze Uber trip patterns and build a regression model capable of estimating trip mileage. By analyzing temporal features (time, day, month) and categorical trip metadata (purpose, category), the model provides accurate distance forecasts.

### Key Highlights:
- **Algorithm:** XGBoost Regressor
- **Validation Accuracy:** 82.1% (R² Score)
- **Deployment:** Streamlit Community Cloud
- **Dataset:** Uber Trip Data (Categorical & Temporal features)

---

## 🛠️ Tech Stack
- **Languages:** Python (Pandas, NumPy)
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-Learn, XGBoost
- **Web Framework:** Streamlit
- **Environment:** Google Colab / VS Code

---

## 📊 Analysis & Insights
- **Outlier Management:** Implemented IQR (Interquartile Range) capping to handle extreme distance values, ensuring model stability.
- **Feature Engineering:** Extracted granular temporal features including `Day Name`, `Month`, and `Time Windows` (Morning/Evening/etc.).
- **Model Comparison:** Evaluated multiple algorithms (Linear Regression, Decision Trees, Random Forest). XGBoost emerged as the superior model with an R² score of **0.821**.




