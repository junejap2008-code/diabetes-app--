import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="centered")

@st.cache_resource
def load():
    return joblib.load("model.pkl")

bundle = load()
model, features = bundle["model"], bundle["features"]

st.title("🩺 Diabetes Risk Predictor")
st.caption("Educational tool only. Not medical advice.")

with st.form("input_form"):
    c1, c2 = st.columns(2)
    with c1:
        preg = st.number_input("Pregnancies", 0, 20, 1)
        glucose = st.slider("Glucose (mg/dL)", 40, 250, 120)
        bp = st.slider("Blood Pressure (mm Hg)", 30, 130, 70)
        skin = st.slider("Skin Thickness (mm)", 5, 100, 20)
    with c2:
        insulin = st.slider("Insulin (µU/mL)", 10, 900, 80)
        bmi = st.slider("BMI", 15.0, 60.0, 25.0, 0.1)
        dpf = st.slider("Diabetes Pedigree Function", 0.05, 2.5, 0.5, 0.01)
        age = st.number_input("Age", 15, 100, 30)
    submit = st.form_submit_button("Predict", use_container_width=True)

if submit:
    row = pd.DataFrame([[preg, glucose, bp, skin, insulin, bmi, dpf, age]], columns=features)
    prob = model.predict_proba(row)[0][1]

    st.subheader("Result")
    st.progress(float(prob))
    st.metric("Estimated risk", f"{prob:.0%}")

    if prob >= 0.6:
        st.error("High risk. Please consult a doctor.")
    elif prob >= 0.35:
        st.warning("Moderate risk. Consider a check-up and lifestyle changes.")
    else:
        st.success("Low risk. Keep a healthy lifestyle.")

    with st.expander("What influenced this?"):
        imp = pd.Series(model.named_steps["clf"].feature_importances_, index=features)
        st.bar_chart(imp.sort_values())

st.sidebar.header("About")
st.sidebar.write(f"Model: Random Forest\n\nTest accuracy: **{bundle['accuracy']:.1%}**")
st.sidebar.write("Dataset: PIMA Indians Diabetes")
