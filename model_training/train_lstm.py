import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint

# -------------------------
# Load CSV Dataset
# -------------------------
df = pd.read_csv("C:\\Users\\HP\\Downloads\\gen-ai_proj\\dataset\\synthetic_pose_risk_dataset.csv")

X = df.drop("risk", axis=1).values     # first 90 columns
y = df["risk"].values                  # last column

# Reshape: (samples, 30 frames, 3 angles)
X = X.reshape(len(X), 30, 3)

print("Dataset Loaded:")
print("X shape =", X.shape)
print("y shape =", y.shape)

# -------------------------
# Build LSTM Model
# -------------------------
model = Sequential([
    LSTM(64, return_sequences=False, input_shape=(30, 3)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss="mse", optimizer="adam")

checkpoint = ModelCheckpoint("../models/risk_lstm.h5",
                             monitor="loss",
                             save_best_only=True)

model.fit(X, y,
          validation_split=0.2,
          epochs=25,
          batch_size=16,
          callbacks=[checkpoint])

print("Training complete. Model saved to ../models/risk_lstm.h5")
model.save("models/risk_lstm.h5")

