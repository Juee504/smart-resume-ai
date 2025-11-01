import streamlit as st
import pandas as pd
import joblib
from PIL import Image
import os;

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------- LOAD MODELS & ASSETS ---------------------- #
MODEL_PATH = os.path.join(BASE_DIR, "4-MODELS", "Attrition_Model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "4-MODELS", "Scaler.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "4-MODELS", "Feature_Names.pkl")

# Load ML model & preprocessors
xgb_model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_names = joblib.load(FEATURES_PATH)

# Load images for prediction results
stay_img = Image.open(os.path.join(BASE_DIR, "2-ASSETS", "Employee_Stay.jpg"))
leave_img = Image.open(os.path.join(BASE_DIR, "2-ASSETS", "Employee_Leave.jpg"))


def render_employee_attrition():
    st.title("Employee Attrition Prediction")
    st.write("Predict whether an employee is likely to **stay** or **leave** the company using machine learning.")

    st.markdown("""
            <style>
                    .st-emotion-cache-17r1dd6 p{ font-size: 18px !important;}
            
            </style>
        """, unsafe_allow_html=True)


    with st.form("attrition_form"):
        st.subheader("Enter Employee Details")

        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", 18, 80, 30)
            distance = st.number_input("Distance From Home (km)", 1, 1000, 5)
            monthly_income = st.number_input("Monthly Income", 1000, 50000000, 5000)
        with col2:
            years_at_company = st.number_input("Years at Company", 0, 40, 3)
            years_since_promotion = st.number_input("Years Since Last Promotion", 0, 15, 1)
            environment_satisfaction = st.slider("Environment Satisfaction (1-10)", 1, 10, 3)
        with col3:
            percent_salary_hike = st.slider("Percent Salary Hike", 1, 200, 15)
            work_life_balance = st.slider("Work Life Balance (1-10)", 1, 10, 3)
            job_satisfaction = st.slider("Job Satisfaction (1-10)", 1, 10, 3)

        overtime = st.selectbox("OverTime", ["Yes", "No"])
        gender = st.selectbox("Gender", ["Male", "Female"])

        submit_btn = st.form_submit_button("🔍 Predict Attrition")

    if submit_btn:
        # ---------------------- PREDICTION LOGIC ---------------------- #
        gender_numeric = 1 if gender == "Male" else 0
        overtime_numeric = 1 if overtime == "Yes" else 0

        input_data = pd.DataFrame({
            "Age": [age],
            "DistanceFromHome": [distance],
            "MonthlyIncome": [monthly_income],
            "YearsAtCompany": [years_at_company],
            "YearsSinceLastPromotion": [years_since_promotion],
            "EnvironmentSatisfaction": [environment_satisfaction],
            "PercentSalaryHike": [percent_salary_hike],
            "WorkLifeBalance": [work_life_balance],
            "JobSatisfaction": [job_satisfaction],
            "OverTime": [overtime_numeric],
            "Gender": [gender_numeric]
        })

        # Align input columns with training data
        input_data = input_data.reindex(columns=feature_names, fill_value=0)

        # Scale input
        input_scaled = scaler.transform(input_data)

        # Predict
        pred = xgb_model.predict(input_scaled)[0]

        st.markdown("---")
        if pred == 1:
            st.error("⚠️ The employee is **likely to leave** the company.")
            st.image(leave_img, caption="Likely to Leave", width=400)
        else:
            st.success("✅ The employee is **likely to stay** in the company.")
            st.image(stay_img, caption="Likely to Stay", width=400)

# Run directly

if __name__ == "__main__":
    render_employee_attrition()
