import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.utils import class_weight

def train_conv_lstm(data, features, target, window_size=24, learning_rate=0.001, epochs=50, batch_size=32):
    # Normalize the features
    scaler = MinMaxScaler()
    data[features] = scaler.fit_transform(data[features])

    # Create sequences for Conv-LSTM
    def create_sequences(data, features, target, window_size):
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[features].iloc[i:i + window_size].values)
            y.append(data[target].iloc[i + window_size])
        return np.array(X), np.array(y)

    # Prepare training and testing datasets
    X, y = create_sequences(data, features, target, window_size)

    # Split the data into training and testing sets
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Calculate class weights to handle class imbalance
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = dict(enumerate(class_weights))

    # Model Definition
    model = Sequential([
        Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(window_size, len(features))),
        BatchNormalization(),
        Dropout(0.2),
        LSTM(50, activation='tanh', return_sequences=False),
        Dropout(0.2),
        Dense(1, activation='sigmoid')  # Sigmoid activation for binary classification
    ])

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), class_weight=class_weights_dict)

    # Evaluate the model
    eval_results = model.evaluate(X_test, y_test)
    print(f'Test Loss: {eval_results[0]}, Test Accuracy: {eval_results[1]}')

    return model, history, eval_results

# Example usage
# model, history, eval_results = train_conv_lstm(data, features=['close', 'ma_signal', 'macd_signal', 'bband_signal', 'rsi_signal', 'volatility', 'volume'], target='returns_direction', window_size=24, learning_rate=0.001, epochs=50, batch_size=32)