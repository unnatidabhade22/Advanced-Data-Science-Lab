# Practical 01 – House Price Prediction using Artificial Neural Network (ANN)

## Overview

This project implements an **Artificial Neural Network (ANN)** using **TensorFlow/Keras** to predict house prices based on various property features. The objective is to understand the complete deep learning workflow, including data preprocessing, model building, training, evaluation, and prediction.

This practical is part of the **Advanced Data Science Laboratory** course.

---

## Objective

* Perform data preprocessing and feature engineering.
* Build an ANN model using TensorFlow/Keras.
* Train the model for house price prediction.
* Evaluate the model using regression metrics.
* Predict the price of a new house using the trained model.

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

The dataset contains information about residential properties, including features such as:

* Bedrooms
* Bathrooms
* Living Area
* Lot Size
* Floors
* Waterfront
* View
* Condition
* Above Ground Area
* Basement Area
* Year Built
* Year Renovated

**Target Variable:**

* Price

---

## Project Workflow

1. Load the dataset.
2. Handle missing values.
3. Remove unnecessary text-based columns.
4. Split features and target variable.
5. Divide the dataset into training and testing sets.
6. Normalize features using `StandardScaler`.
7. Build an Artificial Neural Network using the Keras Sequential API.
8. Compile the model using the Adam optimizer and Mean Squared Error (MSE) loss function.
9. Train the model with Early Stopping to reduce overfitting.
10. Evaluate the model using MAE and RMSE.
11. Compare actual and predicted house prices.
12. Predict the price of a new house.

---

## ANN Architecture

| Layer          | Configuration                |
| -------------- | ---------------------------- |
| Input Layer    | Number of input features     |
| Hidden Layer 1 | 64 neurons (ReLU)            |
| Hidden Layer 2 | 32 neurons (ReLU)            |
| Hidden Layer 3 | 16 neurons (ReLU)            |
| Output Layer   | 1 neuron (Linear Activation) |

---

## Evaluation Metrics

The model is evaluated using:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)

These metrics help measure the prediction accuracy of the regression model.

---

## Sample Output

The program displays:

* Dataset information
* Model summary
* Training and validation loss
* Test loss and MAE
* RMSE
* Actual vs Predicted house prices
* Predicted price for a new house

---

## Learning Outcomes

After completing this practical, I was able to:

* Understand the working of Artificial Neural Networks.
* Perform data preprocessing using Pandas.
* Normalize data using StandardScaler.
* Build and train ANN models using TensorFlow/Keras.
* Apply regression evaluation metrics.
* Predict continuous values using deep learning techniques.

---

## Repository Structure

```text
Practical 01 - ANN/
├── Part1_Linear/
   ├── house_price.py
   ├── data.csv
   ├── README.md
   └── Output Screenshots
```

---

## Author

**Unnati Dabhade**

B.Tech Computer Engineering
Advanced Data Science Laboratory

