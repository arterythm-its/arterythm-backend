import pandas as pd
import numpy as np
import glob
import os
import shutil
from datetime import datetime as dt
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, log_loss
from sklearn.model_selection import train_test_split
from datetime import timedelta
from collections import Counter

df = pd.read_csv(r'telunjuk kanan 1m-2025-10-08.csv')
# gelang = pd.read_csv(r'tes data\gelang kanan 1m - 2025-10-08.csv')
df

df.dropna(axis=1, inplace=True)
df.drop(index=[0, 1, 2], inplace=True)
df.rename(columns={'#!':'ppg'}, inplace=True)
df.reset_index(drop=True, inplace=True)
df

scaler = MinMaxScaler()
scaler.fit(df)

df = scaler.transform(df)
df = pd.DataFrame(df)
df

print(len(df))
n_windows = len(df) // 125
n_windows

sampling_rate = 12
window_size = sampling_rate
ppg_signal = df.iloc[:, 0].values
n_windows = len(ppg_signal) // window_size

telunjuk_windows = []
for i in range(n_windows):
    start = i * window_size
    end = start + window_size
    window = ppg_signal[start:end]
    telunjuk_windows.append(window)

x = pd.DataFrame(telunjuk_windows)
x

sample = df.rename(columns={0:'ppg'})
print(len(sample))
sample

start, stop = 0, 50
wave = sample['ppg'].iloc[start:stop]
plt.figure(figsize=(12, 6))
plt.plot(range(len(wave)), wave)
plt.title('PPG Signal Visualization', fontsize=14, fontweight='bold')
plt.xlabel('Sample Index')
plt.ylabel('PPG Value')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

best_model_file = r'model_05102025_225139.h5'
model = load_model(best_model_file)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Predict
y_pred_prob = model.predict(x).ravel()
y_pred = (y_pred_prob >= 0.5).astype(int)
print(y_pred_prob)
print(y_pred)

y_true = [0] * len(x)
y_true

# Eval metrics
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f'Accuracy: {acc:.4f}')
print(f"Precision: {prec:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")