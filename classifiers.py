from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

# SVM with RandomizedSearchCV
def svm_classifier():
    svm = SVC()
    params = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }

    tuned_svm = RandomizedSearchCV(svm, param_distributions=params, n_iter=5, scoring='accuracy', cv=4, random_state=42, n_jobs=-1)
    return tuned_svm

# Random Forest with RandomizedSearchCV
def rf_classifier():
    rf = RandomForestClassifier(random_state=42)
    params = {
        'n_estimators': [50, 100, 150],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5],
    }
    tuned_rf = RandomizedSearchCV(rf, param_distributions=params, n_iter=5, scoring='accuracy', cv=4, random_state=42, n_jobs=-1)
    return tuned_rf

# XGBoost with RandomizedSearchCV
def gradient_boosting_classifier():
    xgb_classifier = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    params = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5],
        'learning_rate': [0.1, 0.01],
        'subsample': [0.8, 1.0],
    }
    tuned_xgb = RandomizedSearchCV(xgb_classifier, param_distributions=params, n_iter=5, scoring='accuracy', cv=4, random_state=42, n_jobs=-1)
    return tuned_xgb

# k-NN with RandomizedSearchCV
def knn_classifier():
    knn = KNeighborsClassifier()
    params = {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan'],
    }
    tuned_knn = RandomizedSearchCV(knn, param_distributions=params, n_iter=5, scoring='accuracy', cv=4, random_state=42, n_jobs=-1)
    return tuned_knn

# MLP with RandomizedSearchCV
def mlp_classifier():
    mlp = MLPClassifier(max_iter=200, random_state=42)
    params = {
        'hidden_layer_sizes': [(64,), (64, 32)],
        'alpha': [0.0001, 0.001],
        'learning_rate_init': [0.001, 0.01],
    }
    tuned_mlp = RandomizedSearchCV(mlp, param_distributions=params, n_iter=5, scoring='accuracy', cv=4, random_state=42, n_jobs=-1)
    return tuned_mlp