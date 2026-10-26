import numpy as np, pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("diabetes.csv")

# 0 is invalid in these columns -> treat as missing
bad = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[bad] = df[bad].replace(0, np.nan)

X, y = df.drop("Outcome", axis=1), df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("clf", RandomForestClassifier(
        n_estimators=300, max_depth=6, class_weight="balanced", random_state=42
    )),
])
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

joblib.dump({"model": model, "features": list(X.columns),
             "accuracy": accuracy_score(y_test, pred)}, "model.pkl")
print("Saved model.pkl")
