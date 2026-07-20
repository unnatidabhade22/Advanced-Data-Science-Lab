# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 11:54:29 2026

@author: shrad

House Price Prediction using Artificial Neural Network (ANN)
"""


# Import Libraries

import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

# Clear previous TensorFlow session
tf.keras.backend.clear_session()

# Step 1: Load Dataset

df = pd.read_csv("data.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:", df.shape)

# Step 2: Remove Missing Values

df = df.dropna()

# Step 3: Remove String Columns

df = df.drop(columns=[
    'date',
    'street',
    'city',
    'statezip',
    'country'
])


# Step 4: Features and Target
X = df.drop('price', axis=1)
y = df['price']


# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# Step 6: Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Step 7: Build ANN
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='linear')
])

# Step 8: Compile Model 

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

print("\nModel Summary")
model.summary()

# Early Stopping
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Step 9: Train Model
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)


# Step 10: Evaluate
loss, mae = model.evaluate(X_test, y_test, verbose=0)

print("\nTest Loss :", loss)
print("Test MAE :", mae)


# Step 11: Predict
predictions = model.predict(X_test)


# Step 12: Performance Metrics
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("\nMean Absolute Error :", mae)
print("Root Mean Square Error :", rmse)


# Step 13: Actual vs Predicted
results = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": predictions.flatten()
})

print("\nActual vs Predicted")
print(results.head(10))

# Step 14: Predict New House
new_house = pd.DataFrame({
    'bedrooms': [3],
    'bathrooms': [2],
    'sqft_living': [1800],
    'sqft_lot': [5000],
    'floors': [2],
    'waterfront': [0],
    'view': [1],
    'condition': [4],
    'sqft_above': [1800],
    'sqft_basement': [0],
    'yr_built': [2005],
    'yr_renovated': [0]
})

new_house = scaler.transform(new_house)

predicted_price = model.predict(new_house)

print("\nPredicted House Price = ${:,.2f}".format(predicted_price[0][0]))