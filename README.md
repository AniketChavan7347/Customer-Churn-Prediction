📊 Customer Churn Prediction System
📌 Overview

This project is a machine learning-based system that predicts whether a telecom customer will stay or leave (churn) based on their usage and billing data.
It helps understand customer behavior and supports retention strategies.

🎯 Objective

To build a classification model that predicts:

0 → Customer will stay
1 → Customer will churn
📂 Dataset

The dataset contains telecom customer information such as:

Tenure
Monthly Charges
Total Charges
Contract Type
Internet Service
Payment Method
Online Security
Tech Support
Senior Citizen status
🛠️ Technologies Used
Python 🐍
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
Pickle
🤖 Machine Learning Model
Algorithm: Random Forest Classifier
Workflow:
Data Cleaning
Handling Missing Values
Feature Encoding
Train-Test Split
Model Training
Evaluation
📊 Model Evaluation
Accuracy: (79)
Metrics used:
Confusion Matrix
Precision
Recall
F1-score
📁 Project Structure
customer-churn-project/
│
├── churn_model.ipynb     # Model training notebook
├── app.py                # Streamlit application
├── model.pkl             # Trained ML model
├── columns.pkl           # Feature columns
├── requirements.txt      # Dependencies
└── README.md             # Project documentation
🚀 How to Run
1️⃣ Install dependencies
pip install -r requirements.txt
2️⃣ Run Streamlit app
streamlit run app.py
💡 Key Learnings
Data preprocessing techniques
Encoding categorical variables
Building classification models
Model evaluation methods
Basic deployment using Streamlit
📈 Future Improvements
Try advanced models like XGBoost / LightGBM
Hyperparameter tuning
Feature engineering
Deploy on cloud (Streamlit Cloud / Render)
👨‍💻 Author

Aniket Chavan
Data Science Enthusiast | Machine Learning | Python Developer
