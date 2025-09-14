import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# Load Models and Preprocessors
# =========================
scaler = joblib.load("scaler.pkl")
xgb_model = joblib.load("model_cases.pkl")
rf_disease = joblib.load("model_disease.pkl")
rf_state = joblib.load("model_state.pkl")

# Load feature sets
cases_features = joblib.load("cases_features.pkl")
state_features = joblib.load("state_features.pkl")
disease_features = joblib.load("disease_features.pkl")

# =========================
# Load Dataset for Input Ranges
# =========================
df = pd.read_csv("Cleaned_Final_Data.csv")  # make sure this file is saved

# =========================
# Streamlit UI
# =========================
st.title("🦠 AI Climate Driven Disease Outbreak Prediction System")
st.markdown("Provide climate & time inputs to predict **State, Disease, and Cases**.")

# =========================
# Input fields (side by side)
# =========================
col1, col2 = st.columns(2)

with col1:
    Temp = st.number_input(
        "🌡️ Temperature",
        min_value=float(df["Temp"].min()),
        max_value=float(df["Temp"].max()),
        value=float(df["Temp"].iloc[0]),
        step=0.1
    )

    preci = st.number_input(
        "🌧️ Precipitation",
        min_value=float(df["preci"].min()),
        max_value=float(df["preci"].max()),
        value=float(df["preci"].iloc[0]),
        step=0.1
    )

    LAI = st.number_input(
        "🌿 LAI",
        min_value=float(df["LAI"].min()),
        max_value=float(df["LAI"].max()),
        value=float(df["LAI"].iloc[0]),
        step=0.1
    )

    Latitude = st.number_input(
        "📍 Latitude",
        min_value=float(df["Latitude"].min()),
        max_value=float(df["Latitude"].max()),
        value=float(df["Latitude"].iloc[0]),
        step=0.1
    )

    Longitude = st.number_input(
        "📍 Longitude",
        min_value=float(df["Longitude"].min()),
        max_value=float(df["Longitude"].max()),
        value=float(df["Longitude"].iloc[0]),
        step=0.1
    )

with col2:
    day = st.number_input(
        "📅 Day",
        min_value=int(df["day"].min()),
        max_value=int(df["day"].max()),
        value=int(df["day"].iloc[0])
    )

    mon = st.number_input(
        "📅 Month",
        min_value=int(df["mon"].min()),
        max_value=int(df["mon"].max()),
        value=int(df["mon"].iloc[0])
    )

    year = st.number_input(
        "📅 Year",
        min_value=int(df["year"].min()),
        max_value=int(df["year"].max()),
        value=int(df["year"].iloc[0])
    )

    week_of_outbreak_num = st.number_input(
        "📆 Week of outbreak",
        min_value=int(df["week_of_outbreak_num"].min()),
        max_value=int(df["week_of_outbreak_num"].max()),
        value=int(df["week_of_outbreak_num"].iloc[0])
    )

# =========================
# Prediction
# =========================
if st.button("🔮 Predict Outbreak"):
    # Build dataframe from input
    user_df = pd.DataFrame([{
        "Temp": Temp,
        "preci": preci,
        "LAI": LAI,
        "Latitude": Latitude,
        "Longitude": Longitude,
        "day": day,
        "mon": mon,
        "year": year,
        "week_of_outbreak_num": week_of_outbreak_num
    }])

    # Scale numeric inputs
    scaled = scaler.transform(user_df)
    scaled_df = pd.DataFrame(scaled, columns=user_df.columns)

    # ---------------------------
    # 1. Predict State
    # ---------------------------
    X_state_input = scaled_df.reindex(columns=state_features, fill_value=0)
    state_pred = rf_state.predict(X_state_input)
    predicted_state = state_pred[0]

    # ---------------------------
    # 2. Predict Disease
    # ---------------------------
    X_disease_input = scaled_df.reindex(columns=disease_features, fill_value=0)
    disease_pred = rf_disease.predict(X_disease_input)
    predicted_disease = disease_pred[0]

    # ---------------------------
    # 3. Predict Cases
    # ---------------------------
    X_cases_input = scaled_df.reindex(columns=cases_features, fill_value=0)
    cases_pred_log = xgb_model.predict(X_cases_input)[0]
    predicted_cases = np.expm1(cases_pred_log)  # inverse of log1p

    # =========================
    # Display Results
    # =========================
    st.subheader("✅ Prediction Results")
    st.write(f"🌍 **Predicted State**: {predicted_state}")
    st.write(f"🦠 **Predicted Disease**: {predicted_disease}")
    st.write(f"📊 **Predicted Cases**: {predicted_cases:.0f}")
