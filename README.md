# 📩 SMS Spam Detection (Streamlit + Scikit-Learn + NLTK)

A simple **Email/SMS Spam Classifier** built using:
- **Text preprocessing** (NLTK tokenization, stopword removal, punctuation removal, stemming)
- **TF-IDF vectorization**
- Multiple **machine learning classifiers** (Naive Bayes, SVC, Logistic Regression, Random Forest, etc.)
- A clean **Streamlit UI** for prediction

---

## ✅ Project Features
- Loads and cleans the popular `spam.csv` dataset
- Converts labels (`ham/spam`) into numeric format using **LabelEncoder**
- Preprocesses messages using:
  - lowercasing
  - tokenization
  - removing non-alphanumeric tokens
  - removing stopwords & punctuation
  - stemming (PorterStemmer)
- Converts text to numeric features using **TF-IDF**
- Trains multiple ML models and compares them using:
  - **Accuracy**
  - **Precision** (important for spam detection)
- Saves the trained **vectorizer** + **model** as `.pkl`
- Predicts spam/ham through a **Streamlit app**

---

## 📂 Dataset
This project uses a CSV dataset named:

- `spam.csv`

Typical columns in this dataset:
- `v1` → label (`ham` or `spam`)
- `v2` → text message
- extra columns like `Unnamed: 2`, `Unnamed: 3`, `Unnamed: 4` (removed)

---

## 🧹 Data Cleaning Steps
1. Load CSV using correct encoding:
   - `latin-1` (common for this dataset)
2. Drop unused columns:
   - `Unnamed: 2`, `Unnamed: 3`, `Unnamed: 4`
3. Rename:
   - `v1 → target`
   - `v2 → text`
4. Encode labels:
   - `ham = 0`
   - `spam = 1`

---

## 🧠 Text Preprocessing (Transform Function)

The function used converts raw message into cleaned/stemmed words:

### Steps:
1. Convert to lowercase
2. Tokenize words (`nltk.word_tokenize`)
3. Keep only alphanumeric tokens
4. Remove stopwords (English) + punctuation
5. Apply stemming using **PorterStemmer**
6. Join back into a cleaned sentence

---

## 🔢 Vectorization (TF-IDF)
The project uses:

- `TfidfVectorizer(max_features=3000)`

This converts text into TF-IDF numeric vectors for model training.

---

## ✂️ Train/Test Split
Data is split into:

- `80% training`
- `20% testing`

Using:

- `train_test_split(test_size=0.2, random_state=2)`

---

## 🤖 Models Used
Multiple classifiers were tested (examples):

- `SVC(kernel='sigmoid', gamma=1.0)`
- `KNeighborsClassifier()`
- `MultinomialNB()`
- `DecisionTreeClassifier(max_depth=5)`
- `LogisticRegression(solver='liblinear', penalty='l1')`
- `RandomForestClassifier(n_estimators=50, random_state=2)`
- `AdaBoostClassifier(n_estimators=50, random_state=2)`
- `BaggingClassifier(n_estimators=50, random_state=2)`
- `ExtraTreesClassifier(n_estimators=50, random_state=2)`
- `GradientBoostingClassifier(n_estimators=50, random_state=2)`
- `XGBClassifier(n_estimators=50, random_state=2)` *(optional)*

---

## 📊 Evaluation Metrics
Each model was evaluated using:

- **Accuracy**
- **Precision**

Precision is important because:
- False positive = real message marked as spam (bad)
So higher precision means more reliable spam detection.

---

## 💾 Saving Model + Vectorizer
The notebook saves both files using pickle:

- `vectorizer.pkl` → TF-IDF object
- `model.pkl` → trained ML model (example: MultinomialNB)

✅ **Important:** Always save a **FITTED** model (trained with `.fit()`)

Example saving:

```python
import pickle
pickle.dump(tfidf, open('vectorizer.pkl','wb'))
pickle.dump(model, open('model.pkl','wb'))  # model must be fitted
