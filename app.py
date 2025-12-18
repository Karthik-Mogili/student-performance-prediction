import streamlit as st
import pickle
import pandas as pd

st.title("🎓 Student Performance Predictor")

with open("model/model.pkl", "rb") as f:
    saved = pickle.load(f)
model = saved["model"]
columns = saved["columns"]

hours = st.slider("Study Hours/Week", 0, 50, 25)
att = st.slider("Attendance Rate (%)", 0, 100, 80)
score = st.slider("Past Exam Scores", 0, 100, 70)

if st.button("Predict"):
    row = pd.DataFrame([{
        "Study_Hours_per_Week": hours,
        "Attendance_Rate": att,          # NOTE: same scale as training
        "Past_Exam_Scores": score
    }])
    row = row[columns]

    pred = model.predict(row)[0]
    prob = model.predict_proba(row)[0]

    st.write("Classes:", model.classes_)
    st.write("Probabilities:", prob)

    st.success(f"Result: {pred}")
