import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import logging
import json
from datetime import datetime

logging.basicConfig(
    filename="pipeline.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

def load_data():
    print("Step 1 - Loading data...")
    df = pd.read_csv("train.csv")
    print(f"Loaded {len(df)} rows")
    logging.info(f"Loaded {len(df)} rows")
    return df

def clean_data(df):
    print("Step 2 - Cleaning data...")
    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Embarked"].fillna("S", inplace=True)
    df.drop(columns=["Cabin","Name","Ticket","PassengerId"], inplace=True)
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})
    print(f"Cleaned. Shape: {df.shape}")
    logging.info(f"Cleaned. Shape: {df.shape}")
    return df

def train_model(df):
    print("Step 3 - Training model...")
    X = df.drop(columns=["Survived"])
    y = df["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Step 4 - Accuracy: {acc:.2%}")
    logging.info(f"Accuracy: {acc:.4f}")
    return model, acc

def save_results(model, acc):
    print("Step 5 - Saving model and metrics...")
    joblib.dump(model, "titanic_model.pkl")
    metrics = {
        "accuracy": round(acc, 4),
        "timestamp": datetime.now().isoformat(),
        "status": "success" if acc >= 0.80 else "low_accuracy"
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics, f)
    logging.info(f"Saved. Status: {metrics['status']}")
    print(f"Done. Status: {metrics['status']}")

if __name__ == "__main__":
    logging.info("=== Pipeline started ===")
    df = load_data()
    df = clean_data(df)
    model, acc = train_model(df)
    save_results(model, acc)
    logging.info("=== Pipeline finished ===")