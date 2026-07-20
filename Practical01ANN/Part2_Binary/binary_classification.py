# -*- coding: utf-8 -*-
"""
Titanic Survival Prediction using ANN (Binary Classification)
"""

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

tf.keras.backend.clear_session()

# Load Dataset
df = pd.read_csv("titanic.csv")

print(df.head())
print(df.info())

# Handle Missing Values
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

if "Cabin" in df.columns:
    df.drop("Cabin", axis=1, inplace=True)

# ==========================
# Drop Unnecessary Columns
# ==========================
for col in ["PassengerId", "Name", "Ticket"]:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)

# Encode Categorical Columns
le = LabelEncoder()

for col in ["Sex", "Embarked"]:
    if col in df.columns:
        df[col] = le.fit_transform(df[col])

# Features and Target
X = df.drop("Survived", axis=1)
y = df["Survived"]


# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Class Weights
weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(weights))

# Build ANN
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(128, activation="relu"),
    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

# Train
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stop],
    class_weight=class_weights,
    verbose=1
)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print("\nTest Loss:", loss)
print("Test Accuracy:", accuracy)


# Prediction
probabilities = model.predict(X_test)

predictions = (probabilities >= 0.5).astype(int)

print("\nAccuracy:", accuracy_score(y_test, predictions))
print("\nConfusion Matrix")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report")
print(classification_report(
    y_test,
    predictions,
    zero_division=0
))

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": predictions.flatten(),
    "Probability": probabilities.flatten()
})

print("\nActual vs Predicted")
print(results.head(10))

# Predict New Passenger
new_passenger = pd.DataFrame({
    "Pclass":[3],
    "Sex":[1],      # male=1, female=0 (check encoder mapping)
    "Age":[25],
    "SibSp":[0],
    "Parch":[0],
    "Fare":[7.25],
    "Embarked":[2]  # adjust according to encoder mapping
})

new_scaled = scaler.transform(new_passenger)

survival_prob = model.predict(new_scaled)

print("\nSurvival Probability:", survival_prob[0][0])

if survival_prob[0][0] >= 0.5:
    print("Prediction: Survived")
else:
    print("Prediction: Did Not Survive")