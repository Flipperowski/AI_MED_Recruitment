# ML Classifier Comparison: KNN, Decision Tree, SVM, Logistic Regression, Random Forest

## Project Goal
The aim of this project was to compare the performance of several machine learning classifiers on a medical dataset.
The analysis includes hyperparameter tuning (GridSearchCV), cross-validation, and evaluation using Accuracy, F1-score, and ROC AUC metrics. 
Visualization of model comparison was also performed.

## Models
- K-Nearest Neighbors (KNN)
- Decision Tree
- Support Vector Machine (SVM)
- Logistic Regression
- Random Forest Classifier

## Techniques
- `Pipeline` for scaling and training models
- `GridSearchCV` for hyperparameter tuning
- Cross-validation (CV) for robust evaluation
- Testing on a hold-out dataset

# Machine Learning Model Evaluation

This repository contains the evaluation results of different classification models on the general test dataset. The models were evaluated using cross-validation metrics and test set accuracy.

## Model Performance Summary

| Model                 | Accuracy (CV mean ± std) | F1-score (CV mean ± std) | ROC AUC (CV mean ± std) | Test Accuracy |
|------------------------|-------------------------|--------------------------|-------------------------|---------------|
| KNN                   | 0.900 ± 0.082           | 0.937 ± 0.052           | 0.838 ± 0.200           | 0.625         |
| Decision Tree          | 0.720 ± 0.096           | 0.816 ± 0.063           | 0.735 ± 0.156           | 0.750         |
| SVM (RBF kernel)       | 0.898 ± 0.083           | 0.933 ± 0.054           | 0.785 ± 0.200           | 0.750         |
| Logistic Regression    | 0.827 ± 0.013           | 0.893 ± 0.008           | 0.680 ± 0.194           | 0.750         |
| Random Forest          | 0.867 ± 0.067           | 0.933 ± 0.054           | 0.885 ± 0.102           | 0.625         |

## Notes

- **CV mean ± std**: Mean and standard deviation of cross-validation scores.
- **Test Accuracy**: Accuracy of the model on the unseen test dataset.
- KNN achieved the highest F1-score on cross-validation but had a lower test set accuracy.
- Random Forest showed strong ROC AUC performance, suggesting good class separation, despite moderate test accuracy.

## Visualizations
- Bar charts comparing Accuracy, F1-score, and ROC AUC
- Confusion matrices
- ROC curves

## Technologies
- Python 3.11
- scikit-learn
- numpy, pandas
- matplotlib, seaborn

For detailed code and experiments, please refer to the [Jupyter Notebook](./main.ipynb).

Based on [Artificial-Intelligence-in-Medicine-AGH - AI_MED_Recruitment](https://github.com/Artificial-Intelligence-in-Medicine-AGH/AI_MED_Recruitment/tree/main).
