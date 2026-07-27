
# Practical 02 – Optimization Techniques and Regularization using ANN

## Objective

The objective of this practical is to implement an Artificial Neural Network (ANN) for Customer Churn Prediction using TensorFlow/Keras. The model applies optimization and regularization techniques such as Adam optimizer, Batch Normalization, Dropout, and Early Stopping to improve learning performance and reduce overfitting. The model is evaluated using standard classification metrics and visualization graphs.

## Technologies Used

* Python 3.13
* TensorFlow / Keras
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Visual Studio Code

## Dataset

* Dataset: Telco Customer Churn Dataset
* Records: 7043
* Features: 21 (including target column)
* Target Variable: Churn

## Features Implemented

* Data loading and exploration
* Data cleaning and preprocessing
* Handling missing values
* Label Encoding of categorical features
* Feature scaling using StandardScaler
* Train-Test Split
* Artificial Neural Network (ANN)
* Batch Normalization
* Dropout Regularization
* Adam Optimizer
* Early Stopping
* Model Evaluation
* Confusion Matrix
* Accuracy and Loss Visualization

## Project Structure

```
Practical 02 - Optimization and Regularization
│
├── churn.csv
├── Optimization_Regularization.py
├── README.md
├── requirements.txt
│
├── outputs
│   ├── accuracy.png
│   ├── loss.png
│   ├── confusion_matrix.png
│   ├── output_1.png
│   ├── output_2.png
│   ├── output_3.png
│   └── output_4.png
```

## Model Architecture

* Input Layer
* Dense Layer (64 neurons, ReLU)
* Batch Normalization
* Dropout (0.3)
* Dense Layer (32 neurons, ReLU)
* Batch Normalization
* Dropout (0.3)
* Output Layer (1 neuron, Sigmoid)

## Optimizer Used

* Adam Optimizer
* Binary Crossentropy Loss Function
* Accuracy Metric
* Early Stopping Callback

## Evaluation Metrics

The model performance is evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

## Results

The ANN model successfully predicts customer churn with good classification performance. Batch Normalization and Dropout help improve model generalization by reducing overfitting, while the Adam optimizer provides faster and more stable convergence during training. The generated graphs visualize the training and validation accuracy, loss, and confusion matrix.

## How to Run

1. Open the project folder in Visual Studio Code.
2. Ensure all required libraries are installed.
3. Place `churn.csv` in the project directory.
4. Run the following command:

```bash
python Optimization_Regularization.py
```

5. After execution, the generated graphs will be saved inside the `outputs` folder.

## Learning Outcomes

* Understood the importance of optimization algorithms in deep learning.
* Implemented an ANN using TensorFlow/Keras.
* Applied Batch Normalization and Dropout for regularization.
* Used Early Stopping to prevent overfitting.
* Evaluated the model using multiple performance metrics.
* Visualized model performance through accuracy, loss, and confusion matrix plots.

## Conclusion

This practical demonstrates the implementation of an optimized and regularized Artificial Neural Network for Customer Churn Prediction. The use of Adam optimizer, Batch Normalization, Dropout, and Early Stopping improved the model's performance and generalization capability. The model was successfully evaluated using classification metrics and graphical visualizations, providing a complete deep learning workflow for binary classification.
