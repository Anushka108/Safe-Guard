import numpy as np
from tensorflow.keras.models import load_model

class RiskModel:
    def __init__(self, model_path="C:\\Users\\HP\\Downloads\\gen-ai_proj\\models\\risk_lstm.h5"):
        try:
            self.model = load_model(model_path)
        except Exception as e:
            print(f"Error loading model with default settings: {e}")
            # Try loading with custom_objects parameter
            try:
                self.model = load_model(model_path, custom_objects={"mse": "mse"})
            except:
                # If still fails, try compiling without custom loss
                self.model = load_model(model_path, compile=False)

    def predict_risk(self, angle_seq):
        arr = np.array(angle_seq).reshape(1, 30, 3)
        pred = self.model.predict(arr)[0][0]
        return float(pred * 100)
