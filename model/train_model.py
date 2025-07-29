import pandas as pd
import cloudpickle as pickle  # safer alternative
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import os

# Load dataset
df = pd.read_csv("data/adult.csv")

# Preprocessing
df = df.replace('?', 'Unknown')
df = df[df['workclass'] != 'Without-pay']
df = df[df['workclass'] != 'Never-worked']
df = df[(df['age'] >= 17) & (df['age'] <= 75)]
df = df[(df['educational-num'] >= 5) & (df['educational-num'] <= 16)]
df = df.drop(columns=['education'])

# Features and target
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

print("✅ Model trained and saved with cloudpickle")
