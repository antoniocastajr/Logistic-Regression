# 🚀 Binary Logistic Regression From Scratch  

This repository presents a **custom implementation of Binary Logistic Regression**, built from scratch without using Scikit-Learn's `LogisticRegression`. The goal is to **understand the fundamentals** of logistic regression, including probability estimation, gradient descent, and cost function minimization.

---

## 📌 Features & Concepts Covered  
✔ Development of **Binary Logistic Regression from scratch**, implementing probability calculations manually.  
✔ Implementation of **Sigmoid function** to convert linear outputs into probabilities.  
✔ Explanation of **Maximum Likelihood Estimation (MLE)** and **Negative Log-Likelihood (NLL)** as cost functions.  
✔ Computation of **gradient of the log-likelihood loss function** to optimize model weights.  
✔ **Vectorized implementation** to improve computational efficiency.  
✔ Application of the model to the **Titanic dataset** for binary classification.  
✔ **Comparison of custom implementation vs. Scikit-Learn's logistic regression**, ensuring correctness.  

---

## 📂 Repository Structure  
📜 **titanic_classifier.ipynb** → Jupyter Notebook where the **custom logistic regression** is implemented, trained, and evaluated.  
📖 **Logistic Regression.pdf** → Theoretical background covering **logistic regression, cost function derivation, and optimization techniques**.  
📊 **Titanic.csv** → Dataset used for training.  
---

## 📘 Theory Overview (From Logistic Regression.pdf)  
The **Logistic Regression model** is a probabilistic classification model used for **binary and multiclass classification**. The document explains:  
- **Difference between generative vs. discriminative models** and where logistic regression fits.  
- **Why logistic regression is needed instead of linear regression for classification problems.**  
- **Sigmoid function:** Converts linear model output into probability values.  
- **Decision boundary:** How logistic regression separates classes using probabilities.  
- **Negative Log-Likelihood (NLL):** Loss function derived from Maximum Likelihood Estimation.  
- **Gradient Descent Optimization:** How we compute the gradient of the loss function to update model weights.  
- **Matrix form representation of the cost function and its gradient.**  

This document serves as a **comprehensive introduction** to the mathematical foundations of logistic regression.

---

## 📊 Practical Implementation (From titanic_classifier.ipynb)  
The Jupyter Notebook covers:

1️⃣ **Data Preprocessing**: Handling missing values, feature encoding, and standardization.  
2️⃣ **Custom Logistic Regression Model**: Implementing:
   - Sigmoid function
   - Cost function (Negative Log-Likelihood)
   - Gradient computation
   - Weight updates using Gradient Descent  
3️⃣ **Evaluation Metrics**: Accuracy, Precision, Recall, and F1-score.  
4️⃣ **Comparison with Scikit-Learn's Logistic Regression**: Ensuring correct implementation.  
5️⃣ **Titanic Dataset Application**: Predicting survival based on available features.  

---

## 🚀 How to Use the Repository  
1️⃣ Clone the repository:  
```bash
 git clone https://github.com/your-repo/logistic-regression-scratch.git
```
2️⃣ Open **titanic_classifier.ipynb** in Jupyter Notebook.  
3️⃣ Run all cells to see the model in action 

---


