import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("../data/adult.csv")

# Clean and preprocess
df = df.replace('?', 'Unknown')
df = df[df['workclass'] != 'Without-pay']
df = df[df['workclass'] != 'Never-worked']
df = df[(df['age'] >= 17) & (df['age'] <= 75)]
df = df[(df['educational-num'] >= 5) & (df['educational-num'] <= 16)]
df = df.drop(columns=['education'])  # redundant with 'educational-num'

# Define features and target
X = df.drop('income', axis=1)
y = df['income']

# Identify column types
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# Preprocessing pipeline
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Full pipeline with model
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf.fit(X_train, y_train)

# Save model
joblib.dump(clf, "../model/model.pkl")
print("Model training complete and saved as model.pkl")
