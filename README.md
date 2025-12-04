# Customer Churn Prediction using Artificial Neural Networks (ANN)

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 📌 Project Overview
Customer churn (customer attrition) is a critical metric for businesses, particularly in the banking and finance sector. This project utilizes an **Artificial Neural Network (ANN)** to predict whether a bank customer is likely to leave (churn) or stay based on their demographic and financial activity.

Moving beyond simple analysis, this project implements a full Deep Learning pipeline including data preprocessing, feature scaling, model training with **TensorFlow/Keras**, and deployment readiness using **Streamlit**.

## 📂 Dataset
The dataset used for this model is the `Churn_Modelling.csv`. It contains 10,000 rows of customer data with the following features:

* **CreditScore**: The credit score of the customer.
* **Geography**: Country of residence (France, Spain, Germany).
* **Gender**: Male or Female.
* **Age**: Customer's age.
* **Tenure**: Number of years the customer has been with the bank.
* **Balance**: Account balance.
* **NumOfProducts**: Number of bank products the customer uses.
* **HasCrCard**: Whether the customer has a credit card (1=Yes, 0=No).
* **IsActiveMember**: Active membership status (1=Yes, 0=No).
* **EstimatedSalary**: The estimated annual salary.
* **Exited**: Target variable (1 = Churned, 0 = Stayed).

## 🛠 Tech Stack
* **Language:** Python
* **Deep Learning:** TensorFlow (Keras API)
* **Data Manipulation:** Pandas, NumPy
* **Preprocessing:** Scikit-Learn (OneHotEncoder, LabelEncoder, StandardScaler)
* **Visualization:** Matplotlib, TensorBoard
* **Deployment Interface:** Streamlit

## ⚙️ Model Architecture
[cite_start]The model is a Sequential Artificial Neural Network built with TensorFlow:

1.  **Input Layer**: Accepts 12 standardized features.
2.  **Hidden Layer 1**: 64 Neurons, `ReLU` activation function.
3.  **Hidden Layer 2**: 32 Neurons, `ReLU` activation function.
4.  **Output Layer**: 1 Neuron, `Sigmoid` activation function (outputs probability between 0 and 1).

**Optimizer**: Adam (Learning Rate = 0.01)
**Loss Function**: Binary Crossentropy

## 📊 Performance
[cite_start]The model was trained for 100 epochs with Early Stopping implemented to prevent overfitting.
* **Training Accuracy:** ~86.9%
* **Validation Accuracy:** ~86.0%

