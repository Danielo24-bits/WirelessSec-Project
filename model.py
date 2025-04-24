# Model: Vanilla Random Forest from sklearn
# Main point of this code is to show how to use GridSearchCV with RandomForestClassifier
# Goal: Find the best hyperparameters for RandomForestClassifier (between a few options)

import os
import json
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

from dataset_download import download_dataset
from dataset_preprocessing import (
    load_dataset, 
    clean_dataset, 
    extract_features_and_labels, 
    balance_dataset,
)
from constants import RANDOM_SEED, STORED_MODEL_PATH
from log import _logger

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(RANDOM_SEED)

def build_dense_model(labels_encoded):
    model = Sequential([
        Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.2),
        Dense(258, activation='relu'),
        Dropout(0.2),
        Dense(512, activation='relu'),
        Dropout(0.2),
        Dense(258, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(labels_encoded.shape[1], activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model
    
if __name__ == '__main__':
    # 1) Download, load & clean dataset
    download_dataset()
    dataset = load_dataset(verbose=False)
    cleaned = clean_dataset(dataset, attacks='all')

    # 2) Extract features & labels
    features, labels = extract_features_and_labels(cleaned)
    
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    encoder = OneHotEncoder(sparse_output=False)
    labels_encoded = encoder.fit_transform(labels.values.reshape(-1, 1))
    
    # 3) Optional: balance only training data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels_encoded, test_size=0.2, random_state=RANDOM_SEED
    )
    X_train_bal, y_train_bal = balance_dataset(X_train, y_train, ratio=0.25)

    model = build_dense_model(labels_encoded)
    model.fit(X_train, y_train, epochs=6, batch_size=32)
    
    if os.path.exists(STORED_MODEL_PATH):
        _logger.info(f"Model already exists at {STORED_MODEL_PATH}. Overwriting...")
    else:
        os.makedirs(os.path.dirname(STORED_MODEL_PATH), exist_ok=True)
        _logger.info(f"Storing model at {STORED_MODEL_PATH}...")
        
    model.save(STORED_MODEL_PATH)
    
    loss, accuracy = model.evaluate(X_test, y_test)
    _logger.info(f"Loss: {loss}, Accuracy: {accuracy}")