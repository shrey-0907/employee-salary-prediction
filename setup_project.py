import os
import shutil

# Define project structure
base_dir = "employee-salary-prediction"
os.makedirs(f"{base_dir}/model", exist_ok=True)
os.makedirs(f"{base_dir}/data", exist_ok=True)
os.makedirs(f"{base_dir}/notebooks", exist_ok=True)

# Create requirements.txt
requirements = """streamlit
scikit-learn
pandas
matplotlib
fpdf
joblib
"""
with open(f"{base_dir}/requirements.txt", "w", encoding="utf-8") as f:
    f.write(requirements)

# Create README.md
readme = """# 💼 Employee Salary Prediction App

A professional machine learning project to predict whether an employee earns more than $50K per year using demographic and employment data.

## 🔍 Features
- Exploratory Data Analysis
- Data Cleaning and Preprocessing
- Model Training using Random Forest
- Interactive Web App via Streamlit

## 🚀 How to Run

1. Clone this repo:
```bash
git clone https://github.com/your-username/employee-salary-prediction.git
cd employee-salary-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train the model:
```bash
python model/train_model.py
```

4. Launch the web app:
```bash
streamlit run app.py
```

## 🌐 Deploy Online

Push this repo to GitHub and deploy instantly via [Streamlit Cloud](https://streamlit.io/cloud).

---
Made with ❤️ by an aspiring data scientist.
"""
with open(f"{base_dir}/README.md", "w", encoding="utf-8") as f:
    f.write(readme)

print("✅ Project structure created successfully! You can now push it to GitHub.")
