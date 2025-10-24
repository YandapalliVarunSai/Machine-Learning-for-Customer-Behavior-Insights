# 🧠 Data Mining and Machine Learning Project

> **End-to-end ML & Data Mining workflow** applying the **KDD framework** across three real-world domains — **Retail & Transportation**, **Music Analytics**, and **Social Media Sentiment Analysis**.  
>  
> Built with **Python**, **scikit-learn**, **XGBoost**, and **NLP (VADER)** to demonstrate predictive modeling, regression analysis, and text mining for actionable business insights.

---

## 🚀 Project Overview

This project applies **Machine Learning (ML)** techniques to analyze three distinct datasets across multiple domains:

- 🛣️ **Retail & Transportation:** Predict customer coupon usage (classification).  
- 🎵 **Music Analytics:** Predict song popularity using audio features (regression).  
- 🍲 **Social Media Analysis:** Perform sentiment analysis on recipe reviews (text analytics).

It demonstrates:
- The **KDD process** — from data cleaning to evaluation.  
- **Feature engineering** and **hyperparameter tuning**.  
- Multi-domain **predictive modeling** and **insight generation**.

---

## 🧠 Architecture Overview

```
Raw Datasets
   │
   ▼
 Data Preprocessing (Cleaning, Encoding, Scaling)
   │
   ▼
 Feature Engineering & Selection
   │
   ▼
 Model Training & Hyperparameter Tuning
   │
   ▼
 Evaluation & Visualization
   │
   ▼
 Insights & Recommendations
```

---

## ⚙️ Tools & Technologies

| Layer | Tool / Library | Purpose |
|-------|----------------|----------|
| **Programming** | Python | Core language |
| **Data Processing** | pandas, numpy | Cleaning, transformation |
| **Modeling** | scikit-learn, XGBoost | Classification & regression |
| **Text Analytics** | NLTK, VADER | Sentiment detection |
| **Visualization** | matplotlib, seaborn | Exploratory plots & metrics |
| **Evaluation** | GridSearchCV, RandomizedSearchCV | Hyperparameter tuning |
| **Environment** | Jupyter Notebook | Interactive development |

---

## 🧱 Project Components

### 🛣️ 1️⃣ In-Vehicle Coupon Recommendation (Classification)

| Detail | Description |
|--------|--------------|
| **Goal** | Predict whether a customer will use a given in-vehicle coupon |
| **Models** | Logistic Regression, Random Forest, XGBoost |
| **Best Model** | 🥇 XGBoost – *Accuracy: 75.6%* |
| **Techniques** | Encoding, scaling, and GridSearchCV tuning |
| **Insights** | Passenger type, destination, and time strongly influence coupon usage |

---

### 🎵 2️⃣ Song Popularity Prediction (Regression)

| Detail | Description |
|--------|--------------|
| **Goal** | Predict a song’s popularity (0–100) using audio features |
| **Models** | Random Forest Regressor, Bagging Regressor |
| **Best Model** | 🥇 Random Forest – *R² = 0.40, RMSE = 17.02* |
| **Techniques** | Skewness correction (Yeo-Johnson), PCA, VIF |
| **Insights** | Loudness, energy, and danceability drive song popularity |

---

### 🍲 3️⃣ Recipe Reviews Sentiment Analysis (Text Analytics)

| Detail | Description |
|--------|--------------|
| **Goal** | Identify sentiment (positive, neutral, negative) in recipe reviews |
| **Models** | Random Forest, Logistic Regression, SVM, Decision Tree |
| **Best Model** | 🥇 Random Forest – *Accuracy: 78.97%* |
| **Techniques** | Tokenization, stopword removal, lemmatization, VADER |
| **Insights** | Positive reviews highlight flavor & ease of preparation; negative ones cite missing ingredients |

---

## 📊 Results Summary

| Domain | Task Type | Best Model | Metric | Result |
|---------|------------|-------------|---------|---------|
| Retail & Transportation | Classification | XGBoost | Accuracy | **75.6%** |
| Music Analytics | Regression | Random Forest | R² | **0.40** |
| Social Media | Sentiment Analysis | Random Forest | Accuracy | **78.97%** |

---

## 📦 Repository Structure

```
data-mining-ml-project/
│
├── data/
│   ├── retail_coupon.csv
│   ├── songs_data.csv
│   └── recipe_reviews.csv
│
├── notebooks/
│   ├── 1_coupon_recommendation.ipynb
│   ├── 2_song_popularity.ipynb
│   └── 3_sentiment_analysis.ipynb
│
├── models/
│   ├── xgboost_coupon.pkl
│   ├── random_forest_song.pkl
│   └── vader_sentiment.pkl
│
├── results/
│   ├── confusion_matrices.png
│   ├── feature_importance.png
│   └── metrics_summary.csv
│
├── README.md
└── requirements.txt
```

---

## 🧰 Setup Instructions

### 1️⃣ Clone Repository
```bash
git clone https://github.com/<your_username>/data-mining-ml-project.git
cd data-mining-ml-project
```

### 2️⃣ Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # (Windows: venv\Scripts\activate)
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run Jupyter Notebooks
```bash
jupyter notebook
```

Then open the desired `.ipynb` files inside the `notebooks/` folder.

---

## 📊 Example Visualizations

| Visualization | Description |
|----------------|-------------|
| Heatmaps | Feature correlation matrices |
| Confusion Matrices | Classification performance |
| Word Clouds | Frequent terms in review text |
| Scatter Plots | Audio feature relationships |

---

## 🧩 Key Learnings

- 🧠 KDD process ensures **structured and explainable ML workflows**.  
- 🎯 **Feature engineering & scaling** boost model accuracy.  
- 🔍 **Hyperparameter tuning** (GridSearchCV, RandomizedSearchCV) refines model performance.  
- 💬 **Text preprocessing** crucially improves NLP-based sentiment classification.  
- 🌐 **Random Forest** generalizes effectively across domains.

---

## 🔮 Future Work

- [ ] Integrate **transformer-based models (BERT, RoBERTa)** for advanced NLP.  
- [ ] Develop **Flask/Streamlit dashboards** for deployment.  
- [ ] Address **class imbalance** using SMOTE or class weights.  
- [ ] Add **cross-validation and ensemble stacking**.  
- [ ] Incorporate **SHAP/LIME** for interpretability.

---

## 👤 Author

**Varun Sai Yandapalli**  
