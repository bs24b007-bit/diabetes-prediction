import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
)

df = pd.read_csv("diabetes.csv")

X = df.drop("Outcome",axis=1)
y = df["Outcome"]

model = joblib.load("diabetes_model.pkl")

pred = model.predict(X)

print("Accuracy")
print(accuracy_score(y,pred))

print(classification_report(y,pred))