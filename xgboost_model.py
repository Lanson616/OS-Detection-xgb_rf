import logging
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

logging.getLogger("scapy.runtime").setLevel(logging.ERROR)

# Load features and labels from the text file
feature_vectors_loaded = []
with open('28_feature_array.txt', 'r') as file:
    for line in file:
        feature_vector = list(map(float, line.strip().split(',')))
        feature_vectors_loaded.append(feature_vector)

with open('50250_os_system_label.txt', 'r') as file:
    selected_os_loaded = [line.strip() for line in file]

X = np.array(feature_vectors_loaded)
y = np.array(selected_os_loaded)

print(X.shape)
print(y.shape)

# Encode labels to numerical values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# XGBoost model
xgb_model = xgb.XGBClassifier(
    #n_estimators=100,
    #max_depth=6,
    #learning_rate=0.5，
    random_state=42
    )
xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)
print(f"XGBoost Accuracy: {accuracy_score(y_test, y_pred_xgb)}")
