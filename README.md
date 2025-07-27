# 🩺 Diabetes Prediction using Logistic Regression

## 📌 Project Summary
A binary classification model predicting the likelihood of diabetes based on health indicators using the Pima Indians dataset and logistic regression.

## 📊 Dataset
- Source: [Kaggle Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- 768 rows, 8 features + target

## ⚙️ ML Workflow
- Data preprocessing (null values, scaling)
- Logistic regression model
- Evaluation: Confusion matrix, ROC-AUC, precision/recall

## 📈 Results
- Accuracy: 0.77%
- AUC Score: 0.74

## 🧪 Tools Used
`pandas`, `scikit-learn`, `matplotlib`

## 🚀 Run
```bash
conda env create -f environment.yml
conda activate diabetes-ml
