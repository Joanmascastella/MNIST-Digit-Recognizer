import torch.nn as nn
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

class DNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(DNNClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def svm_classifier():
    svm_classifier = SVC(kernel='linear', C=1)
    return svm_classifier

def rf_classifier():
    rf_classifier = RandomForestClassifier(n_estimators=100)
    return rf_classifier

def gradient_boosting_classifier():
    xgb_classifier = xgb.XGBClassifier(n_estimators=100, use_label_encoder=False)
    return xgb_classifier

def knn_classifier():
    knn_classifier = KNeighborsClassifier(n_neighbors=3)
    return knn_classifier

def mlp_classifier():
    mlp_classifier = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500)
    return mlp_classifier