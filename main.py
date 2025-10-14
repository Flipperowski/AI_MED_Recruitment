import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Modele klasyfikacyjne
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Ewaluacja
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

#Importing the data from CSV (80% training, 20% testing)
data = pd.read_csv("task_data.csv")
data.columns = data.columns.str.strip()

numeric_cols = [
    "Heart width", "Lung width", "CTR - Cardiothoracic Ratio", "xx", "yy", "xy", "normalized_diff",
    "Inscribed circle radius", "Polygon Area Ratio", "Heart perimeter", "Heart area", "Lung area"
]

#Repairing data types
for col in numeric_cols:
    data[col] = data[col].astype(str).str.replace(",", ".", regex=True).astype(float)
    
X = data[[
    "Heart width", "Lung width", "CTR - Cardiothoracic Ratio",
    "xx", "yy", "xy", "normalized_diff",
    "Inscribed circle radius", "Polygon Area Ratio",
    "Heart perimeter", "Heart area", "Lung area"
]]

#Selecting targeted column (Cardiomegaly)
y = data["Cardiomegaly"]

#Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Creating a StandardScaler object
scaler = StandardScaler()

#Fitting and applying scaler
X_scaled_train = scaler.fit_transform(X_train)
X_scaled_test = scaler.transform(X_test)

#1st test with baseline Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_scaled_train, y_train)

y_pred = model.predict(X_scaled_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))