# 💳 Credit Card Fraud Detection using Machine Learning

## 📌 Project Overview

This project focuses on detecting fraudulent credit card transactions using **unsupervised machine learning techniques**. Due to the highly imbalanced nature of the dataset, traditional classification models are less effective.

👉 Therefore, this project treats fraud detection as an **anomaly detection problem** and uses clustering techniques to identify suspicious transactions.

Additionally, an **interactive web application** is built using **Streamlit** to visualize and analyze fraud detection results in real time.

---

## 📥 Dataset Download

Due to GitHub file size limitations, the dataset is not included in this repository.

👉 You can download it from Kaggle:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

---

## 📊 Dataset Information

* Total Records: **284,807**
* Features: **31**
* Target Variable: `Class`

  * `0` → Normal Transaction
  * `1` → Fraudulent Transaction

⚠️ **Note:** Dataset is highly imbalanced with very few fraud cases.

---

## 🎯 Objectives

* Detect fraudulent transactions from large-scale financial data
* Handle extreme class imbalance
* Apply clustering techniques to identify anomalies
* Perform feature selection and dimensionality reduction
* Build an interactive dashboard for real-time insights

---

## 🛠️ Technologies Used

* Python
* Pandas, NumPy
* Matplotlib, Seaborn
* Scikit-learn
* Streamlit (for web app)

---

## 🚀 Project Workflow

### 1️⃣ Data Preprocessing

* Removed duplicate records
* Checked and handled missing values
* Cleaned dataset for analysis

---

### 2️⃣ Exploratory Data Analysis (EDA)

* Correlation heatmap
* Feature importance analysis
* Distribution visualization

---

### 3️⃣ Feature Engineering

* Removed low-correlation features (noise reduction)
* Improved model efficiency

---

### 4️⃣ Data Scaling

* Applied **MinMaxScaler**
* Normalized features for distance-based algorithms

---

### 5️⃣ Model Building

* Implemented **K-Means Clustering**
* Used **Elbow Method** to find optimal clusters (k = 2)

---

### 6️⃣ Model Evaluation

* Accuracy
* Precision
* Recall
* RMSE
* Correlation Score

---

### 7️⃣ Dimensionality Reduction

* Applied **PCA (Principal Component Analysis)**
* Visualized fraud as outliers in reduced dimensions

---

## 💻 Streamlit Web Application

An interactive web application was developed using **Streamlit** to make the model more practical and user-friendly.

### ✨ Features:

* Upload transaction dataset
* Visualize fraud vs normal transactions
* Display correlation heatmap
* Show clustering results
* Interactive PCA visualization
* Real-time fraud detection insights

---

### ▶️ How to Run the App

```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
```

---

### 🌐 App Preview

(Add screenshots here of your Streamlit dashboard)

---

## 📈 Results

* Successfully detected fraud patterns using clustering
* Fraud transactions appear as **anomalies/outliers**
* Demonstrated effectiveness of **unsupervised learning**

---

## 🚀 Key Insights

* Fraud detection can be treated as an **anomaly detection problem**
* Most features have low correlation → complex pattern detection
* Feature selection significantly improves performance
* Clustering effectively separates normal and fraudulent behavior

---

## ⚠️ Limitations

* K-Means assumes spherical clusters
* Sensitive to feature scaling
* Not ideal for highly complex fraud patterns

---

## 🔮 Future Improvements

* Isolation Forest (better anomaly detection)
* XGBoost / Random Forest (supervised approach)
* Deep Learning (Autoencoders)
* Real-time fraud detection system with APIs

---

## 📂 Project Structure

```
├── Credit_Card_Fraud_Detection.ipynb
├── app.py                  # Streamlit application
├── Figures/
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🧠 Key Highlights (For Recruiters)

* Solved **real-world fintech problem**
* Used **unsupervised learning (K-Means)**
* Handled **imbalanced dataset effectively**
* Implemented **feature selection + PCA**
* Built **interactive Streamlit dashboard**

---

## 🤝 Contribution

Feel free to fork this repository and improve the model or experiment with advanced techniques!

---

## ⭐ Acknowledgment

Dataset sourced from Kaggle (Credit Card Fraud Detection dataset).

---

## 📞 Contact

**Kavit Patel**  
📧 Email: [Contact by mail](mailto:kavitpatel1574.kp@gmail.com)  
🔗 GitHub: [Kavit-shadow](https://github.com/Kavit-shadow)  
💼 LinkedIn: [My Linkedin Profile](https://www.linkedin.com/in/kavit-patel-84597a26b?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)

📍 Location: Ahmedabad, Gujarat
