import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

os.makedirs("model", exist_ok=True)

df = pd.read_csv("data/student_data.csv")
print("✅ Dataset loaded:", df.shape)

# Use numeric features only
features = ['Study_Hours_per_Week', 'Attendance_Rate', 'Past_Exam_Scores']
X = df[features].copy()
y = df['Pass_Fail']          # 'Pass' / 'Fail'

print("Target distribution:\n", y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Stronger logistic model with class_weight to avoid always predicting majority
model = LogisticRegression(max_iter=2000, class_weight="balanced")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

pickle.dump({'model': model, 'columns': X.columns.tolist()}, open("model/model.pkl", "wb"))
print("🎉 Saved model/model.pkl")
