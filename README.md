# YIELD-PREDICTIVE-ANALYSIS
# 🌾 Crop Yield Prediction Using Machine Learning

## 📌 Project Overview
This project focuses on predicting **crop yield** using machine learning techniques based on agricultural and environmental factors such as rainfall, fertilizer usage, pesticide usage, crop type, season, and area.

The goal is to analyze different regression models and identify the best-performing algorithm for accurate yield prediction.

---

## 🎯 Objectives
- Predict crop yield using machine learning
- Apply multiple regression models
- Compare model performance using evaluation metrics
- Identify the best model based on accuracy

---

## 🛠 Tools & Technologies Used
- Python
- Jupyter Notebook
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Matplotlib, Seaborn
- Git & GitHub

---

## 📊 Dataset Description
**Source:** Kaggle – Crop Yield in Indian States Dataset  

### Features:
- State
- Annual Rainfall
- Fertilizer Used
- Pesticide Used
- Crop
- Production
- Season
- Area
- Year
- Yield (Target Variable)

---

## ⚙️ Project Workflow
1. Data Collection
2. Data Cleaning & Preprocessing
3. Feature Selection
4. Model Training
5. Model Evaluation
6. Performance Comparison
7. Result Analysis

---

## 🤖 Machine Learning Models Used
- Decision Tree Regressor
- Random Forest Regressor
- XGBoost Regressor

---

## 📈 Model Evaluation Metrics
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

---

## 🏆 Results

| Model | R² Score | MAE | RMSE |
|------|---------|-----|------|
| Decision Tree | 0.950 | 41784 | 189.54 |
| Random Forest | 0.953 | 13.32 | 231.77 |
| XGBoost | **0.968** | 14.05 | 189.54 |

✅ **XGBoost Regressor performed best**

---

## 📌 Conclusion
Among all tested models, **XGBoost Regressor** achieved the highest accuracy with an **R² score of 0.968**, making it the most reliable model for crop yield prediction.

---

## 🚀 Future Enhancements
- Add deep learning models
- Deploy model using Flask or Streamlit
- Add real-time weather API
- Improve feature engineering

---

## 📚 Reference
Dataset:  
https://www.kaggle.com/datasets/akshatgupta7/crop-yield-in-indian-states-dataset

---

## 👤 Author
**Jestin John**  
Data Analytics & Machine Learning Enthusiast
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost

__pycache__/
.ipynb_checkpoints/
.env

