Employee Satisfaction Prediction using Deep Learning
📌 Project Overview

This project builds and deploys a deep learning regression model to predict the Employee Satisfaction Score based on various employee-related features such as demographics, education, department, job title, and performance data.

By applying data preprocessing, feature encoding, scaling, and neural network modeling, this system helps organizations better understand employee satisfaction levels and take proactive HR measures.

⚙️ Features

Preprocessing pipeline:

Ordinal Encoding for education levels

One-Hot Encoding for categorical features (Department, Gender, Job Title)

Standard Scaling for numerical features

Deep Learning Model:

Multi-layer Neural Network with dropout for regularization

Early stopping to avoid overfitting

Mean Absolute Error (MAE) as evaluation metric

5-Fold Cross Validation for robust performance estimation

Model Deployment:

Saves trained model (models.pkl) and preprocessing objects (scale.pkl, ohe.pkl, oe.pkl)

Ready for deployment with FastAPI

📊 Dataset

The dataset used: Extended Employee Performance and Productivity Data

Includes employee demographic details, job information, and performance indicators

Target variable: Employee_Satisfaction_Score

🛠️ Tech Stack

Python 3.9+

Pandas, NumPy – Data handling

Scikit-learn – Encoding, scaling, cross-validation

TensorFlow / Keras – Deep learning model

Pickle – Saving model & preprocessing objects

FastAPI (for deployment)

🔑 How It Works

Load the dataset

Clean and preprocess the data (drop unnecessary columns, encode categorical data, scale numerical features)

Train a deep learning regression model using cross-validation

Evaluate the model using Mean Absolute Error (MAE)

Save the trained model and preprocessing objects for deployment

Deploy using FastAPI for real-time employee satisfaction prediction

📈 Model Performance

Evaluation Metric: Mean Absolute Error (MAE)

Cross-Validation Results:

Average Train MAE: ~X.XXX

Average Test MAE: ~X.XXX

(Replace with your actual results after training)

🚀 Deployment

The model and preprocessing objects are saved as:

models.pkl → Trained Neural Network

scale.pkl → Standard Scaler

ohe.pkl → One-Hot Encoder

oe.pkl → Ordinal Encoder

These files can be loaded inside a FastAPI application for real-time predictions.