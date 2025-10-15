import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, f1_score, roc_auc_score


#Evaluation function
def evaluate_model(model, X, y, cv=5):
    # Accuracy
    acc = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    print("\nAccuracy scores:", np.round(acc, 2))
    print(f"Mean Accuracy: {np.mean(acc):.3f} | Std: {np.std(acc):.3f}")

    # F1
    f1 = cross_val_score(model, X, y, cv=cv, scoring="f1")
    print("\nF1 scores:", np.round(f1, 2))
    print(f"Mean F1: {np.mean(f1):.3f} | Std: {np.std(f1):.3f}")

    # ROC AUC
    auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    print("\nROC AUC scores:", np.round(auc, 2))
    print(f"Mean ROC AUC: {np.mean(auc):.3f} | Std: {np.std(auc):.3f}")

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

#Defining hyperparameters
param_grid = {
    "model__n_neighbors": [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
    "model__weights": ["uniform", "distance"],
    "model__metric": ["minkowski", "manhattan", "euclidean"],
    "model__p": [1, 2]
}

#Setting up the cross-validation strategy
rskf = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=5,
    random_state=42
)

#Applying K-Nearest Neighbors (KNN) Classifier
pipe_knn = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("model", KNeighborsClassifier())
])

#Initializing the Grid Search for the KNN model
grid_search = GridSearchCV(
    estimator=pipe_knn,
    param_grid=param_grid,
    scoring="accuracy",
    cv=rskf,
    verbose=0,
    n_jobs=-1
)

#Training
grid_search.fit(X_train, y_train)
best_knn = grid_search.best_estimator_

#Displaying results
print("General test --- KNN Model \n")
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best accuracy (averaged CV): {grid_search.best_score_:.4f}\n\n")

evaluate_model(best_knn, X_train, y_train)

#Decision Tree
clf_tree = DecisionTreeClassifier(
    max_depth=5,
    criterion='entropy',
    min_samples_split=8,
    min_samples_leaf=8,
    class_weight=None,
    random_state=42
)

#Training
clf_tree.fit(X_train, y_train)

cv_score = np.round(cross_val_score(clf_tree, X_train, y_train),2 )

#Displaying results
print("\n\nGeneral test --- Decision Tree \n")

print(f"Scores of training data cross-validation (each fold):")
list(map(print, cv_score))
print(f"\nCross-validation mean score: {np.mean(cv_score):.3}")
print(f"Standard deviation of CV score: {np.std(cv_score):.3f}\n\n")

evaluate_model(clf_tree, X_train, y_train)
