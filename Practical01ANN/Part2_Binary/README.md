#  Part 2 – Titanic Survival Prediction using Artificial Neural Network (ANN)

## Overview

This project implements an **Artificial Neural Network (ANN)** using **TensorFlow/Keras** to predict whether a passenger survived the Titanic disaster. It is a **Binary Classification** problem where the model classifies passengers into two categories: **Survived (1)** or **Did Not Survive (0)**.

This practical is a part of **Practical 01 – ANN** in the **Advanced Data Science Laboratory** course.

---

## Objective

* Perform data preprocessing and feature engineering.
* Handle missing values in the dataset.
* Encode categorical variables.
* Build and train an ANN model for binary classification.
* Evaluate the model using classification metrics.
* Predict the survival probability of new passengers.

---

## Technologies Used

* Python
* TensorFlow / Keras
* Pandas
* NumPy
* Scikit-learn
* Visual Studio Code

---

## Dataset

The project uses the **Titanic Dataset**, which contains passenger information such as:

* Passenger Class (Pclass)
* Gender (Sex)
* Age
* Number of Siblings/Spouses (SibSp)
* Number of Parents/Children (Parch)
* Fare
* Embarked Port

**Target Variable:**

* `Survived`

  * `1` → Passenger Survived
  * `0` → Passenger Did Not Survive

---

## Data Preprocessing

The following preprocessing steps were performed:

* Handled missing values in the **Age** column using the median value.
* Filled missing values in the **Embarked** column using the mode.
* Removed unnecessary columns:

  * PassengerId
  * Name
  * Ticket
  * Cabin
* Converted categorical features using **Label Encoding**.
* Applied **Feature Scaling** using `StandardScaler`.
* Used **Class Weights** to handle class imbalance.

---

## ANN Architecture

| Layer          | Configuration            |
| -------------- | ------------------------ |
| Input Layer    | Number of input features |
| Hidden Layer 1 | 128 Neurons (ReLU)       |
| Hidden Layer 2 | 64 Neurons (ReLU)        |
| Hidden Layer 3 | 32 Neurons (ReLU)        |
| Output Layer   | 1 Neuron (Sigmoid)       |

---

## Model Compilation

| Parameter     | Value               |
| ------------- | ------------------- |
| Optimizer     | Adam                |
| Loss Function | Binary Crossentropy |
| Metric        | Accuracy            |

---

## Training Configuration

* Epochs: 100
* Batch Size: 16
* Validation Split: 20%
* Early Stopping: Enabled
* Class Weighting: Enabled

---

## Evaluation Metrics

The model performance was evaluated using:

* Accuracy Score
* Confusion Matrix
* Precision
* Recall
* F1-Score
* Classification Report

---

## Sample Output

The program displays:

* Dataset information
* ANN model summary
* Training and validation accuracy
* Test accuracy
* Confusion matrix
* Classification report
* Actual vs Predicted values
* Survival probability of a new passenger

---

## Project Structure

```text
Practical 01 - ANN/
├── Part2_Binary/
│   ├── Binary_classification.py
│   ├── titanic.csv
│   ├── output.png
│   └── README.md
```

---

## Learning Outcomes

After completing this practical, I was able to:

* Understand ANN for binary classification problems.
* Handle missing values and preprocess real-world datasets.
* Encode categorical features.
* Apply feature scaling techniques.
* Build ANN models using TensorFlow/Keras.
* Evaluate classification models using multiple performance metrics.
* Predict outcomes using deep learning techniques.

---

## Conclusion

This practical demonstrates the implementation of an Artificial Neural Network (ANN) for solving a binary classification problem using the Titanic dataset. The model successfully predicts passenger survival by learning patterns from demographic and travel-related features.

---

## Author

**Unnati Dabhade**
B.Tech Computer Engineering
Advanced Data Science Laboratory

