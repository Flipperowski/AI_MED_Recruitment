import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RepeatedStratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, f1_score, roc_auc_score, precision_recall_curve, average_precision_score


#Evaluation function
#Evaluate one model
def evaluate_single_model(model, X, y, cv=5):
    print(f"\nEvaluating model: {model.__class__.__name__}")
    
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


#Evaluate all models at once
def evaluate_all_models(models, X, y, cv=5):
    results = {}
    for name, model in models.items():
        print(f"\n\n--- Evaluating {name} ---")
        evaluate_single_model(model, X, y, cv=cv)

        scores = cross_validate(
            model, X, y, cv=cv, scoring=["accuracy", "f1", "roc_auc"]
        )
        results[name] = {
            "acc_mean": scores["test_accuracy"].mean(),
            "f1_mean": scores["test_f1"].mean(),
            "roc_mean": scores["test_roc_auc"].mean(),
        }
    return results

#Visualization of Tests
def visualize_model_performance(model, X_test, y_test, model_name="Model"):
    
    y_pred = model.predict(X_test)
    
    # ROC/PR
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        print(f"{model_name} nie obsługuje ROC/PR (brak predict_proba/decision_function).")
        y_score = None

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", cbar=False)
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    if y_score is not None:
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = roc_auc_score(y_test, y_score)
        plt.figure(figsize=(6,5))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], "k--", lw=1)
        plt.title(f"ROC Curve - {model_name}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, y_score)
        ap = average_precision_score(y_test, y_score)
        plt.figure(figsize=(6,5))
        plt.plot(recall, precision, color="blue", lw=2, label=f"AP = {ap:.3f}")
        plt.title(f"Precision-Recall Curve - {model_name}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()



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
grid_search_knn = GridSearchCV(
    estimator=pipe_knn,
    param_grid=param_grid,
    scoring="accuracy",
    cv=rskf,
    verbose=0,
    n_jobs=-1
)

#Training
grid_search_knn.fit(X_train, y_train)
best_knn = grid_search_knn.best_estimator_


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


#Support Vector Machine (SVM)
param_grid_svm = {
    "model__C": [0.1, 1, 3, 10],
    "model__gamma": ["scale", "auto", 0.01, 0.1, 1],
    "model__kernel": ["rbf", "poly", "sigmoid"]
}

#Applying SVM Classifier
pipe_svc = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("model", SVC(probability=True))
])

#Initializing the Grid Search for the SVC model
grid_search_svm = GridSearchCV(
    estimator=pipe_svc,
    param_grid=param_grid_svm,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

#Training
grid_search_svm.fit(X_train, y_train)
best_svm = grid_search_svm.best_estimator_

cv_score = np.round(cross_val_score(best_svm, X_train, y_train), 2)


#Logistic Regression
#Setting up GridSearchCV
param_grid_lr = [
    {
        "model__penalty": ["l1"],
        "model__C": [0.01, 0.1, 1, 3, 10],
        "model__solver": ["liblinear", "saga"]
    },
    {
        "model__penalty": ["l2"],
        "model__C": [0.01, 0.1, 1, 3, 10],
        "model__solver": ["liblinear", "saga", "lbfgs", "newton-cg"]
    },
    {
        "model__penalty": ["elasticnet"],
        "model__C": [0.01, 0.1, 1, 3, 10],
        "model__solver": ["saga"],
        "model__l1_ratio": [0.3, 0.5, 0.7]
    }
]

#Applying Logistic Regression Scaler
pipe_lr = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=50000))
])

#Initializing the Grid Search for the LR
grid_search_lr = GridSearchCV(
    estimator=pipe_lr,
    param_grid=param_grid_lr,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

# Training
grid_search_lr.fit(X_train, y_train)
best_lr = grid_search_lr.best_estimator_

cv_score = np.round(cross_val_score(best_lr, X_train, y_train, cv=5, scoring="accuracy"), 2)

#Random Forest Classifier
clf_rf = RandomForestClassifier(
    max_depth=6,
    min_samples_split=6,
    n_estimators=125,
    min_samples_leaf=2,
    max_features='sqrt'
)

#Training
clf_rf.fit(X_train, y_train)

cv_score = np.round(cross_val_score(clf_rf, X_train, y_train), 2)


#Displaying results
models = {
    "KNN": best_knn,
    "Decision Tree": clf_tree,
    "SVM": best_svm,
    "Logistic Regression": best_lr,
    "Random Forest": clf_rf
}

#Test Set Evaluation
results = evaluate_all_models(models, X_train, y_train)

results = {}

models = {
    "KNN": grid_search_knn.best_estimator_,
    "Decision Tree": clf_tree,
    "SVM": grid_search_svm.best_estimator_,
    "Logistic Regression": grid_search_lr.best_estimator_,
    "Random Forest": clf_rf
}

#Cross validation
for name, model in models.items():
    scores = cross_validate(
        model, X_train, y_train,
        cv=5,
        scoring=["accuracy", "f1", "roc_auc"]
    )
    results[name] = {
        "acc_mean": scores["test_accuracy"].mean(),
        "f1_mean": scores["test_f1"].mean(),
        "roc_mean": scores["test_roc_auc"].mean(),
        "acc_std": scores["test_accuracy"].std(),
        "f1_std": scores["test_f1"].std(),
        "roc_std": scores["test_roc_auc"].std()
    }

models_names = list(results.keys())
acc = [results[m]["acc_mean"] for m in models_names]
f1  = [results[m]["f1_mean"] for m in models_names]
roc = [results[m]["roc_mean"] for m in models_names]

x = np.arange(len(models_names))
width = 0.25

fig, ax = plt.subplots(figsize=(10,6))
ax.bar(x - width, acc, width, label="Accuracy (CV mean)")
ax.bar(x, f1, width, label="F1 (CV mean)")
ax.bar(x + width, roc, width, label="ROC AUC (CV mean)")

ax.set_ylabel("Score")
ax.set_title("Model Comparison - Accuracy, F1, ROC AUC")
ax.set_xticks(x)
ax.set_xticklabels(models_names)
ax.legend()
plt.ylim(0, 1.1)
plt.show()

for name, model in models.items():
    print(f"\n\n=== {name} ===")

    visualize_model_performance(model, X_test, y_test, model_name=name)
