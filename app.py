import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
import tempfile
import os
import requests

# -------------------------
# Download model if missing
# -------------------------
MODEL_URL = MODEL_URL = "https://drive.google.com/uc?export=download&id=1Fzn6Pq6ifldqjFzxCRbGORxh40gsDaME"
MODEL_PATH = "model/model.pkl"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model file..."):
        os.makedirs("model", exist_ok=True)
        r = requests.get(MODEL_URL)
        with open(MODEL_PATH, 'wb') as f:
            f.write(r.content)
        st.success("Model downloaded successfully!")
model = joblib.load(MODEL_PATH)
# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="Employee Salary Prediction",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# Load model
# -------------------------
model = joblib.load(MODEL_PATH)

# Sidebar styling
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2920/2920277.png", width=80)
st.sidebar.title("Employee Details")
st.sidebar.markdown("\n**🌗 Want to change theme?**\n\nGo to top-right ⋮ menu → Settings → Theme\n")

# Input form
def user_input():
    with st.sidebar.form("input_form"):
        age = st.slider('Age', 17, 75, 30)
        workclass = st.selectbox('Workclass', [
            'Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov',
            'State-gov', 'Without-pay', 'Unknown', 'Others'
        ])
        fnlwgt = st.number_input('Final Weight (fnlwgt)', value=200000)
        edu_num = st.slider('Educational Number', 1, 16, 9)
        marital = st.selectbox('Marital Status', [
            'Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent'
        ])
        occupation = st.selectbox('Occupation', [
            'Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial',
            'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical',
            'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv',
            'Armed-Forces', 'Unknown'
        ])
        relationship = st.selectbox('Relationship', ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'])
        race = st.selectbox('Race', ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])
        gender = st.radio('Gender', ['Male', 'Female'])
        capital_gain = st.number_input('Capital Gain', 0)
        capital_loss = st.number_input('Capital Loss', 0)
        hours_per_week = st.slider('Hours per week', 1, 99, 40)
        country = st.selectbox('Native Country', ['United-States', 'Mexico', 'Philippines', 'Germany', 'Canada', 'India', 'Unknown'])
        submitted = st.form_submit_button("Predict Salary")

    data = {
        'age': age,
        'workclass': workclass,
        'fnlwgt': fnlwgt,
        'educational-num': edu_num,
        'marital-status': marital,
        'occupation': occupation,
        'relationship': relationship,
        'race': race,
        'gender': gender,
        'capital-gain': capital_gain,
        'capital-loss': capital_loss,
        'hours-per-week': hours_per_week,
        'native-country': country
    }
    return pd.DataFrame([data]), submitted

input_df, submitted = user_input()

# Main UI content
st.title("💼 Employee Salary Classification")
st.markdown("""
Welcome to the **Employee Salary Prediction App**. 
This tool predicts whether an employee earns **>50K or <=50K** annually based on demographic and job-related inputs.
""")

st.image("assets/banner.jpg")  # Make sure the banner image exists

if submitted:
    with st.spinner("Analyzing data and predicting salary..."):
        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df).max()

        st.success(f"### ✅ Predicted Salary Group: {prediction}")
        st.info(f"🔍 Confidence Score: {prob * 100:.2f}%")

        # Confidence chart
        st.subheader("📊 Model Confidence Chart")
        fig, ax = plt.subplots()
        ax.bar(['<=50K', '>50K'], model.predict_proba(input_df)[0], color=["skyblue", "lightgreen"])
        ax.set_ylabel("Probability")
        ax.set_title("Prediction Confidence")
        st.pyplot(fig)

        st.markdown("---")
        st.subheader("📊 Input Summary")
        st.dataframe(input_df.T.rename(columns={0: 'Value'}))

        # Generate PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Employee Salary Prediction Report", ln=True, align="C")
        pdf.ln(10)
        pdf.cell(200, 10, txt=f"Predicted Salary Group: {prediction}", ln=True)
        pdf.cell(200, 10, txt=f"Confidence Score: {prob * 100:.2f}%", ln=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
            pdf.output(tmpfile.name)
            with open(tmpfile.name, "rb") as f:
                st.download_button(
                    label="📄 Download Prediction as PDF",
                    data=f,
                    file_name="salary_prediction_report.pdf",
                    mime="application/pdf"
                )

# Footer
st.markdown("""
---
Made with ❤️ using Streamlit, Scikit-learn, and Python.  
Feel free to [connect on LinkedIn](https://www.linkedin.com/in/shreyash-rastogi-04794125a) or [explore the GitHub project](https://github.com/shrey-0907/employee-salary-prediction).
""")
