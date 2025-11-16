import numpy as np
import pandas as pd
import random
import os

# -------------------------------
# 1) Generate biomechanically realistic sequences
# -------------------------------

def generate_good_sequence():
    """Safe squat-like movement."""
    seq = []
    for _ in range(30):
        hip = np.random.normal(85, 10)        # hip 70–100 degrees
        knee = np.random.normal(90, 8)        # knee 80–100 degrees
        shoulder = np.random.normal(50, 10)   # neutral shoulder
        seq.append([hip, knee, shoulder])
    return np.array(seq)


def generate_bad_knee_sequence():
    """Knee valgus collapse → high risk."""
    seq = []
    for _ in range(30):
        hip = np.random.normal(60, 12)
        knee = np.random.normal(45, 6)        # knee < 60 = very risky
        shoulder = np.random.normal(110, 10)
        seq.append([hip, knee, shoulder])
    return np.array(seq)


def generate_twist_sequence():
    """Torso twist / arm overextension."""
    seq = []
    for _ in range(30):
        hip = np.random.normal(70, 10)
        knee = np.random.normal(85, 10)
        shoulder = np.random.normal(150, 15)   # overextended shoulder
        seq.append([hip, knee, shoulder])
    return np.array(seq)


# -------------------------------
# 2) Risk Calculation
# -------------------------------

def biomechanical_risk(seq):
    """Assign synthetic risk based on final-frame angles."""
    hip, knee, shoulder = seq[-1]

    risk = 0
    if knee < 60:            # knee collapse
        risk += 0.5
    if hip < 50:             # excessive hip rounding
        risk += 0.3
    if shoulder > 140:       # shoulder overextension
        risk += 0.2

    # Add small noise to make labels natural
    risk = min(max(risk + np.random.normal(0.05, 0.02), 0.0), 0.99)
    return float(risk)


# -------------------------------
# 3) Generate Dataset
# -------------------------------

def generate_dataset(samples=500):
    X = []
    y = []
    categories = ["good", "bad", "twist"]

    for _ in range(samples):
        choice = random.choice(categories)

        if choice == "good":
            seq = generate_good_sequence()
        elif choice == "bad":
            seq = generate_bad_knee_sequence()
        else:
            seq = generate_twist_sequence()

        X.append(seq)
        y.append(biomechanical_risk(seq))

    return np.array(X), np.array(y)


# -------------------------------
# 4) Save CSV + Numpy Files
# -------------------------------

def save_dataset(samples=500, outdir="dataset"):
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    X, y = generate_dataset(samples)

    # Save numpy versions
    np.save(f"{outdir}/angles_X.npy", X)
    np.save(f"{outdir}/risk_y.npy", y)

    # Convert to CSV format
    flat_rows = []
    for i in range(len(X)):
        flat = X[i].flatten().tolist()
        flat.append(y[i])
        flat_rows.append(flat)

    columns = [f"f{i}" for i in range(90)] + ["risk"]
    df = pd.DataFrame(flat_rows, columns=columns)

    df.to_csv(f"{outdir}/synthetic_pose_risk_dataset.csv", index=False)

    print("Synthetic dataset generated!")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print(f"Saved to folder: {outdir}/")


# -------------------------------
# Run the generator
# -------------------------------

if __name__ == "__main__":
    save_dataset(samples=500)
