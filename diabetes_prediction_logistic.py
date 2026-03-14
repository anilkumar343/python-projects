import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Sample dataset
data = {
    "Glucose": [85, 89, 78, 120, 140, 95, 160, 110],
    "BMI": [22, 25, 20, 30, 35, 28, 40, 32],
    "Age": [25, 30, 22, 45, 50, 35, 55, 40],
    "Outcome": [0,0,0,1,1,0,1,1]
}

df = pd.DataFrame(data)

# Features
X = df[["Glucose", "BMI", "Age"]]

# Target
y = df["Outcome"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LogisticRegression()

# Train
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# New patient prediction
new_patient = pd.DataFrame({
    "Glucose": [130],
    "BMI": [33],
    "Age": [45]
})

prediction = model.predict(new_patient)

if prediction[0] == 1:
    print("Diabetes Positive")
else:
    print("Diabetes Negative")
