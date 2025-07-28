import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
import tempfile
import os
import requests
import gdown

# Page config
# -------------------------
st.set_page_config(
    page_title="Employee Salary Prediction",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)
# -------------------------
# Function to download from Google Drive
# -------------------------
def download_file_from_google_drive(file_id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

# -------------------------
# Model download + load
# -------------------------
gdrive_url = f"https://drive.google.com/file/d/1Fzn6Pq6ifldqjFzxCRbGORxh40gsDaME/view?usp=sharing"
MODEL_PATH = "model/model.pkl"
os.makedirs("model", exist_ok=True)

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model from Google Drive..."):
        gdown.download(gdrive_url, MODEL_PATH, quiet=False)
        st.success("✅ Model downloaded!")

# Load model
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)


# Sidebar UI
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2920/2920277.png", width=80)
st.sidebar.title("Employee Details")
st.sidebar.markdown("🌗 Change theme from top-right settings menu")

# Form
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

# -------------------------
# Main content
# -------------------------
st.title("💼 Employee Salary Classification")
st.markdown("""
This app predicts whether an employee earns **>50K or <=50K** annually based on input features.
""")
st.image("assets/banner.jpg")  # Optional

if submitted:
    with st.spinner("Predicting salary group..."):
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

        # Input Summary
        st.markdown("---")
        st.subheader("📊 Input Summary")
        st.dataframe(input_df.T.rename(columns={0: 'Value'}))

        # PDF Download
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

# -------------------------
# Footer
# -------------------------
st.markdown("""
---
Made with ❤️ using Streamlit & Scikit-learn.  
[🔗 LinkedIn](https://www.linkedin.com/in/shreyash-rastogi-04794125a) | [📂 GitHub](https://github.com/shrey-0907/employee-salary-prediction)
""")
