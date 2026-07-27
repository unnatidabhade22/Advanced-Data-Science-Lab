# ==========================================
# Practical 02
# Optimization Techniques and Regularization
# Customer Churn Prediction using ANN
# ==========================================

# Import required libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping
# ==========================================
# Step 1: Load Dataset
# ==========================================

df = pd.read_csv("churn.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

# ==========================================
# Step 2: Explore Dataset
# ==========================================

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nTarget Variable Distribution:")
print(df["Churn"].value_counts())
# ==========================================
# Step 3: Data Cleaning
# ==========================================

# Remove CustomerID column
df.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Fill missing values with median
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# ==========================================
# Step 4: Encode Categorical Variables
# ==========================================

encoder = LabelEncoder()

for column in df.columns:
    if df[column].dtype == "object":
        df[column] = encoder.fit_transform(df[column])

print("\nEncoded Dataset:")
print(df.head())
# ==========================================
# Step 5: Feature and Target Split
# ==========================================

X = df.drop("Churn", axis=1)
y = df["Churn"]

print("\nFeature Shape:", X.shape)
print("Target Shape:", y.shape)

# ==========================================
# Step 6: Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])

# ==========================================
# Step 7: Feature Scaling
# ==========================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nFeature Scaling Completed Successfully.")
# ==========================================
# Step 8: Build the ANN Model
# ==========================================

model = Sequential()

# Input Layer
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))

# Batch Normalization
model.add(BatchNormalization())

# Dropout
model.add(Dropout(0.3))

# Hidden Layer
model.add(Dense(32, activation='relu'))

# Batch Normalization
model.add(BatchNormalization())

# Dropout
model.add(Dropout(0.3))

# Output Layer
model.add(Dense(1, activation='sigmoid'))

print("\nANN Model Created Successfully.")
# ==========================================
# Step 9: Compile Model
# ==========================================

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\nModel Compiled Successfully.")

# ==========================================
# Step 10: Early Stopping
# ==========================================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# ==========================================
# Step 11: Train Model
# ==========================================

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# ==========================================
# Step 12: Model Summary
# ==========================================

print("\nModel Summary:")
model.summary()

# ==========================================
# Step 13: Model Evaluation
# ==========================================

# Predict on test data
y_pred = model.predict(X_test)

# Convert probabilities to 0 and 1
y_pred = (y_pred > 0.5).astype(int)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nTest Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# ==========================================
# Step 14: Confusion Matrix Plot
# ==========================================

disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot(cmap="Blues")

plt.title("Confusion Matrix")
# Create outputs folder if it doesn't exist
os.makedirs("outputs", exist_ok=True)

plt.savefig("outputs/confusion_matrix.png")

plt.show()

# ==========================================
# Step 15: Accuracy Graph
# ==========================================

plt.figure(figsize=(8,5))

plt.plot(history.history['accuracy'], label='Training Accuracy')

plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title("Training vs Validation Accuracy")

plt.legend()

plt.grid(True)

plt.savefig("outputs/accuracy.png")

plt.show()

# ==========================================
# Step 16: Loss Graph
# ==========================================

plt.figure(figsize=(8,5))

plt.plot(history.history['loss'], label='Training Loss')

plt.plot(history.history['val_loss'], label='Validation Loss')

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("Training vs Validation Loss")

plt.legend()

plt.grid(True)

plt.savefig("outputs/loss.png")

plt.show()

