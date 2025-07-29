import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
import tempfile
import os
import cloudpickle as pickle  # safer than pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

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
# Optional: Retrain button
# -------------------------
def train_model():
    df = pd.read_csv("data/adult.csv")
    df = df.replace('?', 'Unknown')
    df = df[df['workclass'] != 'Without-pay']
    df = df[df['workclass'] != 'Never-worked']
    df = df[(df['age'] >= 17) & (df['age'] <= 75)]
    df = df[(df['educational-num'] >= 5) & (df['educational-num'] <= 16)]
    df = df.drop(columns=['education'])

    X = df.drop('income', axis=1)
    y = df['income']

    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

    clf = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf.fit(X_train, y_train)

    os.makedirs("model", exist_ok=True)
    with open("model/model.pkl", "wb") as f:
        pickle.dump(clf, f)
    return clf

# -------------------------
# Load or Train Model
# -------------------------
MODEL_PATH = "model/model.pkl"
if not os.path.exists(MODEL_PATH):
    st.warning("Model not found. Training from scratch...")
    model = train_model()
else:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

# Sidebar UI
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2920/2920277.png", width=80)
st.sidebar.title("Employee Details")

# Optional retrain button
if st.sidebar.button("🔁 Retrain Model on Server"):
    model = train_model()
    st.sidebar.success("✅ Model retrained and reloaded!")

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

# Main UI
st.title("💼 Employee Salary Classification")
st.markdown("Predicts whether an employee earns **>50K or <=50K** annually based on inputs.")

if submitted:
    with st.spinner("Predicting salary group..."):
        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df).max()

        st.success(f"### ✅ Predicted Salary Group: {prediction}")
        st.info(f"🔍 Confidence Score: {prob * 100:.2f}%")

        # Confidence chart
        st.subheader("📊 Confidence Chart")
        fig, ax = plt.subplots()
        ax.bar(['<=50K', '>50K'], model.predict_proba(input_df)[0], color=["skyblue", "lightgreen"])
        st.pyplot(fig)

        st.subheader("📄 Input Summary")
        st.dataframe(input_df.T.rename(columns={0: "Value"}))

        # PDF report
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Employee Salary Prediction Report", ln=True, align="C")
        pdf.ln(10)
        pdf.cell(200, 10, txt=f"Prediction: {prediction}", ln=True)
        pdf.cell(200, 10, txt=f"Confidence: {prob * 100:.2f}%", ln=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
            pdf.output(tmpfile.name)
            with open(tmpfile.name, "rb") as f:
                st.download_button("📥 Download PDF", data=f, file_name="report.pdf", mime="application/pdf")

# Footer
st.markdown("---\nBuilt with ❤️ by Shreyash Rastogi")
