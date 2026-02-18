# 📩 SMS Spam Detection App

A Machine Learning based **Email/SMS Spam Classifier** built using  
Scikit-learn, NLTK, TF-IDF, and deployed with Streamlit.

---

## 🚀 Features
- Text preprocessing using:
  - Lowercasing
  - Tokenization (NLTK)
  - Stopword removal
  - Punctuation removal
  - Stemming (PorterStemmer)
- TF-IDF vectorization
- Trained and evaluated using multiple classifiers
- Interactive Streamlit UI for predictions

---

## 🤖 Model Selection

Multiple classifiers were tested, including:

- Support Vector Machine (SVC)
- K-Nearest Neighbors
- Decision Tree
- Logistic Regression
- Random Forest
- AdaBoost
- Gradient Boosting
- XGBoost
- Multinomial Naive Bayes

After comparison using **Accuracy and Precision**,  
**Multinomial Naive Bayes was selected** because it achieved:

✅ **Precision = 1.0**

Since spam detection requires minimizing false positives  
(real messages incorrectly marked as spam),  
Naive Bayes performed best for this task.

---

## 📂 Project Workflow
1. Load and clean `spam.csv`
2. Encode labels (`ham = 0`, `spam = 1`)
3. Preprocess text
4. Convert text to TF-IDF features
5. Train/Test split (80/20)
6. Train and compare multiple models
7. Select best model (Naive Bayes)
8. Save:
   - `vectorizer.pkl`
   - `model.pkl`
9. Use Streamlit to predict Spam / Not Spam

---


.venv\Scripts\activate



