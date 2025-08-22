# 📊 Amazon Alexa Reviews Sentiment Analysis (PySpark MLlib)

## 🔹 Overview
This project performs **sentiment analysis** on Amazon Alexa reviews using **PySpark MLlib**.  
The pipeline covers **data preprocessing, feature engineering, and model training** with multiple ML algorithms, followed by performance comparison across metrics like accuracy, precision, recall, F1-score, and AUC.

---

## 🛠 Tools & Libraries
- **Big Data & ML**: PySpark (Spark MLlib), Spark SQL  
- **NLP**: spaCy (lemmatization), PySpark Tokenizer, StopWordsRemover  
- **Visualization**: Matplotlib, Seaborn  
- **Evaluation**: MulticlassClassificationEvaluator, BinaryClassificationEvaluator, RegressionEvaluator  

---

## 📂 Dataset
- Source: [Amazon Alexa Reviews Dataset](https://www.kaggle.com/sid321axn/amazon-alexa-reviews)  
- Size: ~3.1K reviews  
- Columns: `verified_reviews`, `rating`, `feedback`, `variation`, etc.  
- Target: **Rating Category** (binary sentiment → Negative [0–2], Positive [3–5])

---

## ⚙️ Workflow
1. **Data Cleaning & Preprocessing**
   - Removed nulls, dropped irrelevant columns
   - Tokenization, stopword removal, lemmatization
   - Removed punctuation & normalized text  

2. **Feature Engineering**
   - Converted reviews → TF-IDF vectors using HashingTF + IDF  

3. **Modeling**
   - Trained & tuned multiple models using CrossValidator:
     - Decision Tree Classifier  
     - Logistic Regression  
     - Gradient Boosted Tree (GBT)  
     - Naive Bayes  
     - Multi-Layer Perceptron (MLP)  
     - Random Forest  

4. **Evaluation**
   - Accuracy, Precision, Recall, F1-score  
   - AUC (ROC & PR curves)  
   - Regression-style metrics (MSE, RMSE, MAE, R², MAPE)  
   - Visual comparison of models with Matplotlib  

---

## 📈 Results
- **Best Performing Model:** Logistic Regression & Gradient Boosting  
- **Top Metrics Achieved:**
  - Accuracy: ~98%  
  - F1-score: ~99%  
  - ROC-AUC: ~0.99  

---
