import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Create models directory if it doesn't exist
if not os.path.exists("models"):
    os.makedirs("models")

# Build a minimal LSTM model matching the expected input shape (30 frames, 3 angles)
model = Sequential([
    LSTM(64, return_sequences=False, input_shape=(30, 3)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss="mse", optimizer="adam")

# Create a dummy training example and fit once to initialize weights
X_dummy = np.random.rand(1, 30, 3)
y_dummy = np.array([0.5])

model.fit(X_dummy, y_dummy, epochs=1, verbose=0)

# Save the model
model.save("models/risk_lstm.h5")
print("Dummy LSTM model created and saved to models/risk_lstm.h5")
