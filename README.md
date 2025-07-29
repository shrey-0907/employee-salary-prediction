# 💼 Employee Salary Prediction App

A professional machine learning project to predict whether an employee earns more than $50K per year using demographic and employment data.

## 🔍 Features

- Clean and preprocessed dataset using pipelines  
- Model trained with Random Forest Classifier  
- Model loaded dynamically from Google Drive  
- Interactive UI with modern design and banner image  
- Predict salary group (>50K or <=50K)  
- Download prediction results as a PDF  
- Fully deployable on Render

---

## 🚀 How to Run Locally

1. **Clone this repo:**

```bash
git clone https://github.com/shrey-0907/employee-salary-prediction.git
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

🌐 Deploy on Render
Push this repo to GitHub

Go to https://render.com/

Connect your GitHub repository

Set your Build Command:
```bash
pip install -r requirements.txt
```
Set your Start Command:
```bash
streamlit run app.py
```
Render will auto-install packages and download the model from Google Drive on first launch.

✅ Model auto-downloads to model/model.pkl using gdown.


## 🚀 Try the App
👉 Launch Live App on Render

Click below to launch the web app instantly on Streamlit Cloud:

[![Open in Render](Available at your primary URL https://employee-salary-prediction-okzt.onrender.com)

👨‍💻 Author
Made with ❤️ by Shreyash Rastogi
GitHub: @shrey-0907

---


