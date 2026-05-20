import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import logging
import json
from datetime import datetime

# Logging — every run recorded with timestamp
logging.basicConfig(
    filename="pipeline.log",
    level=logging.INFO,
    format="%(asctime)s — %(message)s"
)

def load_data():
    logging.info("Loading train.csv")
    df = pd.read_csv("train.csv")
    logging.info(f"Loaded {len(df)} rows")
    return df

def clean_data(df):
    logging.info("Cleaning...")
    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Embarked"].fillna("S", inplace=True)
    df.drop(columns=["Cabin","Name","Ticket","PassengerId"], inplace=True)
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})
    logging.info(f"Clean done. Shape: {df.shape}")
    return df

def train_model(df):
    logging.info("Training...")
    X = df.drop(columns=["Survived"])
    y = df["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    logging.info(f"Accuracy: {acc:.4f}")
    print(f"Accuracy: {acc:.2%}")
    return model, acc

def save_results(model, acc):
    # Save model
    joblib.dump(model, "titanic_model.pkl")
    # Save metrics as JSON — Prometheus will read this
    metrics = {
        "accuracy": round(acc, 4),
        "timestamp": datetime.now().isoformat(),
        "status": "success" if acc >= 0.80 else "low_accuracy"
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics, f)
    logging.info(f"Saved model and metrics. Status: {metrics['status']}")
    print(f"Done. Status: {metrics['status']}")

if __name__ == "__main__":
    logging.info("=== Pipeline started ===")
    df = load_data()
    df = clean_data(df)
    model, acc = train_model(df)
    save_results(model, acc)
    logging.info("=== Pipeline finished ===")